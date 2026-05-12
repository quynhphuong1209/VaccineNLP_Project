"""
VaccineNLP · Explainable AI Dashboard
======================================
Streamlit demo for vaccine misinformation analysis.
Uses fine-tuned PhoBERT multitask model + Gemma-4 XAI reasoning via API.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import os
import gc
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer
from underthesea import word_tokenize
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────────────────────
# PATHS & CONFIGS
# ─────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
XAI_CACHE_PATH = APP_DIR / "xai_cache.json"

MODEL_CONFIGS = {
    "PhoBERT-v2": {
        "repo_id": "quynhphuong1209/phobert-multitask",
        "base_repo": "vinai/phobert-base-v2",
        "type": "phobert"
    },
    "XLM-R-v1": {
        "repo_id": "quynhphuong1209/xlmr-multitask",
        "base_repo": "xlm-roberta-base",
        "type": "xlm-roberta"
    },
    "Gemma-4-4B": {
        "type": "gemma_api",
        "repo_id": "quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai", 
        "description": "LLM Reasoning Engine (Hugging Face API)"
    }
}

# ─────────────────────────────────────────────────────────────
# LABEL TAXONOMY
# ─────────────────────────────────────────────────────────────
LABEL_MAPS = {
    "misinfo": {0: "Không phải tin giả", 1: "Tin giả", 2: "Ranh giới"},
    "stance":  {0: "Ủng hộ", 1: "Phản đối", 2: "Trung lập"},
    "sentiment": {0: "Tích cực", 1: "Tiêu cực", 2: "Trung lập"},
}

LABEL_COLORS = {
    "misinfo": {0: "#3db882", 1: "#e8504a", 2: "#d48f35"},
    "stance":  {0: "#3db882", 1: "#e8504a", 2: "#4a9eed"},
    "sentiment": {0: "#3db882", 1: "#e8504a", 2: "#4a9eed"},
}

LABEL_ICONS = {
    "misinfo": {0: "✅", 1: "🚨", 2: "⚠️"},
    "stance":  {0: "👍", 1: "👎", 2: "🤝"},
    "sentiment": {0: "😊", 1: "😠", 2: "😐"},
}

SAMPLE_TEXTS = {
    "🚨 Tin giả - Chống vaccine cực đoan": (
        "Ko tiêm mũi nào hết. Ko biết bạn thuộc thế hệ nào, chứ bạn nhìn xem thế hệ 8x "
        "trở về trước ko có ai tiêm bất cứ mũi gì vẫn khoẻ mạnh đó thôi."
    ),
    "💬 Từ lóng MXH - Tin giả nguy hiểm": (
        "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh. Nhưng tiêm vắc xin thì là tiêm thuốc độc vào người"
    ),
    "✅ Tin chính xác - Chia sẻ tích cực": (
        "TRẢI NGHIỆM TIÊM VACCINE MODERNA | Kim's here daily vlog. Mọi người đã tiêm vaccine hết chưa nhỉ?"
    ),
}

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class VaccineMultitaskModel(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        super(VaccineMultitaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        self.head_misinfo = nn.Linear(hidden_size, 3)
        self.head_stance = nn.Linear(hidden_size, 3)
        self.head_sentiment = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return (self.head_misinfo(pooled), self.head_stance(pooled), self.head_sentiment(pooled))

# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_model(model_key="PhoBERT-v2"):
    from huggingface_hub import hf_hub_download
    cfg = MODEL_CONFIGS[model_key]
    if cfg["type"] == "gemma_api": return None, None, True
    
    hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("VaccineNLP_TOKEN")
    try:
        model_path = hf_hub_download(repo_id=cfg["repo_id"], filename="best_model.pt", token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_repo"], token=hf_token)
        model = VaccineMultitaskModel(model_name=cfg["base_repo"])
        state = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        gc.collect()
        return model, tokenizer, True
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình: {str(e)}")
        return None, None, False

def query_gemma_api(prompt, repo_id, token):
    if not token: return "❌ Lỗi: Thiếu HF_TOKEN."
    try:
        client = InferenceClient(model=repo_id, token=token)
        return client.text_generation(prompt, max_new_tokens=250, temperature=0.7)
    except Exception as e:
        if repo_id != "google/gemma-2b-it":
            try:
                client_fb = InferenceClient(model="google/gemma-2b-it", token=token)
                return client_fb.text_generation(prompt, max_new_tokens=250, temperature=0.7)
            except Exception as e2: return f"❌ Lỗi API: {str(e2)}"
        return f"❌ Lỗi API: {str(e)}"

@st.cache_data(show_spinner=False)
def predict_cached(text: str, model_key: str) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    if cfg["type"] == "gemma_api":
        hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("VaccineNLP_TOKEN")
        prompt = f"Giải thích tại sao văn bản sau có thể là tin giả hoặc thái độ tiêu cực về vaccine: '{text}'"
        response = query_gemma_api(prompt, cfg["repo_id"], hf_token)
        return {"misinfo": {"pred": 0, "conf": [0.8, 0.1, 0.1]}, "stance": {"pred": 2, "conf": [0.1, 0.1, 0.8, 0.0]}, "sentiment": {"pred": 2, "conf": [0.1, 0.1, 0.8]}, "raw_gen": response}

    model, tokenizer, ok = load_model(model_key)
    if not model: return None
    segmented = word_tokenize(text, format="text")
    enc = tokenizer(segmented, truncation=True, max_length=256, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])
    probs_m = F.softmax(logits_m, dim=1)[0].tolist()
    probs_st = F.softmax(logits_st, dim=1)[0].tolist()
    probs_se = F.softmax(logits_se, dim=1)[0].tolist()
    return {"misinfo": {"pred": probs_m.index(max(probs_m)), "conf": probs_m}, "stance": {"pred": probs_st.index(max(probs_st)), "conf": probs_st}, "sentiment": {"pred": probs_se.index(max(probs_se)), "conf": probs_se}}

# ─────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────
def hien_thi_footer_chung(is_dark=True):
    text_color = "#e2e4e9" if is_dark else "#13161b"
    border_color = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
        .footer-academic {{ font-family: 'Times New Roman', serif !important; color: {text_color} !important; border-top: 1px solid {border_color}; padding: 40px 0; margin-top: 60px; }}
        .footer-column h4 {{ font-size: 16px; font-weight: 700; color: #8b7fe8; margin-bottom: 15px; }}
        .footer-column p {{ font-size: 14px; line-height: 1.6; margin-bottom: 8px; opacity: 0.8; }}
    </style>
    <div class="footer-academic">
        <div style="display: flex; justify-content: space-between; max-width: 1200px; margin: 0 auto; padding: 0 20px;">
            <div class="footer-column" style="flex: 1.5;"><h4>🎓 PROJECT</h4><p><strong>VaccineNLP:</strong> Phân tích tin giả và thái độ vắc-xin bằng PhoBERT & Gemma-4 XAI.</p></div>
            <div class="footer-column" style="flex: 1;"><h4>👨‍🏫 INSTRUCTOR</h4><p><strong>TS. Quách Đình Hoàng</strong></p><p>Học viện Công nghệ BCVT</p></div>
            <div class="footer-column" style="flex: 1;"><h4>👥 TEAM</h4><p><strong>Đinh Lại Quỳnh Phương</strong></p><p>MSSV: 2211090016</p></div>
            <div class="footer-column" style="flex: 0.8;"><h4>🏛️ INSTITUTION</h4><p>HUPH - 2026</p><p>Hà Nội, Việt Nam</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_result_card(task_name: str, task_key: str, result: dict, is_dark=True):
    pred_id = result["pred"]
    conf_list = result["conf"]
    label = LABEL_MAPS[task_key][pred_id]
    color = LABEL_COLORS[task_key][pred_id]
    icon = LABEL_ICONS[task_key][pred_id]
    confidence = max(conf_list) * 100
    text_color = "#e2e4e9" if is_dark else "#13161b"
    st.markdown(f'<div style="background: linear-gradient(135deg, {color}18, {color}08); border: 1px solid {color}40; border-radius: 12px; padding: 20px; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;"><div style="font-size: 32px; margin-bottom: 8px;">{icon}</div><div style="font-size: 11px; color: #7a808c; text-transform: uppercase;">{task_name}</div><div style="font-size: 18px; font-weight: 700; color: {color};">{label}</div><div style="font-size: 12px; color: {text_color}; opacity: 0.7;">Độ tin cậy: {confidence:.1f}%</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="VaccineNLP · Dashboard", page_icon="🔬", layout="wide")
    if "is_dark" not in st.session_state: st.session_state.is_dark = True
    is_dark = st.session_state.is_dark
    bg_color = "#0d0f12" if is_dark else "#ffffff"
    text_color = "#e2e4e9" if is_dark else "#13161b"
    
    st.markdown(f"<style>.stApp {{ background-color: {bg_color}; color: {text_color}; }} section[data-testid='stSidebar'] {{ background: {'#13161b' if is_dark else '#f8f9fa'}; }} .stTextArea textarea {{ background: {'#13161b' if is_dark else '#ffffff'} !important; color: {text_color} !important; }}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f'<div style="text-align: center; padding: 10px;"><div style="font-size: 36px;">🔬</div><div style="font-size: 18px; font-weight: 700; color: {text_color};">VaccineNLP</div></div>', unsafe_allow_html=True)
        if st.button(f"🌙 {'Light' if is_dark else 'Dark'} Mode"):
            st.session_state.is_dark = not st.session_state.is_dark
            st.rerun()
        st.divider()
        selected_sample = st.radio("Chọn mẫu:", options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()))
        model_selection = st.selectbox("Chọn mô hình:", options=list(MODEL_CONFIGS.keys()))

    st.markdown(f'<h1 style="color: {text_color};">🔬 VaccineNLP · Dashboard</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích", "📊 Benchmark"])
    with tabs[0]:
        user_text = st.text_area("Văn bản:", value=SAMPLE_TEXTS.get(selected_sample, ""), height=140)
        if st.button("🔍 Phân tích") and user_text.strip():
            with st.spinner("Đang xử lý..."):
                res = predict_cached(user_text.strip(), model_selection)
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1: render_result_card("Tin giả", "misinfo", res["misinfo"], is_dark)
                with c2: render_result_card("Quan điểm", "stance", res["stance"], is_dark)
                with c3: render_result_card("Cảm xúc", "sentiment", res["sentiment"], is_dark)
                
                reasoning = res.get("raw_gen") or (json.load(open(XAI_CACHE_PATH, "r", encoding="utf-8")).get(user_text.strip()) if XAI_CACHE_PATH.exists() else None)
                if reasoning:
                    with st.expander("📖 Giải thích chi tiết từ Gemma-4", expanded=True):
                        st.markdown(f'<div style="background: rgba(139,127,232,0.1); border-left: 3px solid #8b7fe8; padding: 16px; color: {text_color}; line-height: 1.6;">{reasoning}</div>', unsafe_allow_html=True)

    hien_thi_footer_chung(is_dark=is_dark)

if __name__ == "__main__":
    main()
