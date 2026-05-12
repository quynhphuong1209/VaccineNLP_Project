"""
VaccineNLP · Explainable AI Dashboard
======================================
Streamlit demo for vaccine misinformation analysis.
Uses fine-tuned PhoBERT multitask model + cached XAI reasoning.
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
        "trở về trước ko có ai tiêm bất cứ mũi gì vẫn khoẻ mạnh đó thôi. Cha mẹ thời nay "
        "bị doạ cho sợ hãi, đem con đi tiêm vì bị bóng ma sợ hãi nó đè."
    ),
    "💬 Từ lóng MXH - Tin giả nguy hiểm": (
        "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh \n"
        "Nhưng tiêm vắc xin thì là tiêm thuốc độc vào người"
    ),
    "✅ Tin chính xác - Chia sẻ tích cực": (
        "TRẢI NGHIỆM TIÊM VACCINE MODERNA | Kim's here daily vlog #covid19 #vaccine #moderna. "
        "Mọi người đã tiêm vaccine hết chưa nhỉ?"
    ),
}

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class VaccineMultitaskModel(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2", num_misinfo=3, num_stance=3, num_sentiment=3):
        super(VaccineMultitaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        self.head_misinfo = nn.Linear(hidden_size, num_misinfo)
        self.head_stance = nn.Linear(hidden_size, num_stance)
        self.head_sentiment = nn.Linear(hidden_size, num_sentiment)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return (self.head_misinfo(pooled_output), self.head_stance(pooled_output), self.head_sentiment(pooled_output))

# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_model(model_key="PhoBERT-v2"):
    from huggingface_hub import hf_hub_download
    cfg = MODEL_CONFIGS[model_key]
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

def predict(text: str, model, tokenizer) -> dict:
    segmented = word_tokenize(text, format="text")
    enc = tokenizer(segmented, truncation=True, max_length=256, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])
    probs_m = F.softmax(logits_m, dim=1)[0].tolist()
    probs_st = F.softmax(logits_st, dim=1)[0].tolist()
    probs_se = F.softmax(logits_se, dim=1)[0].tolist()
    return {
        "misinfo": {"pred": probs_m.index(max(probs_m)), "conf": probs_m},
        "stance": {"pred": probs_st.index(max(probs_st)), "conf": probs_st},
        "sentiment": {"pred": probs_se.index(max(probs_se)), "conf": probs_se},
    }

# ─────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────
def hien_thi_footer_chung(is_dark=True):
    """Footer học thuật chuyên nghiệp với Times New Roman."""
    text_color = "#e2e4e9" if is_dark else "#13161b"
    border_color = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"
    bg_color = "rgba(19, 22, 27, 0.8)" if is_dark else "rgba(255, 255, 255, 0.8)"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
        .footer-academic {{
            font-family: 'Times New Roman', serif !important;
            color: {text_color} !important;
            border-top: 1px solid {border_color};
            padding: 40px 0;
            margin-top: 60px;
            background: {bg_color};
            backdrop-filter: blur(10px);
        }}
        .footer-column h4 {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8b7fe8;
        }}
        .footer-column p {{
            font-size: 15px;
            line-height: 1.8;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
    </style>
    <div class="footer-academic">
        <div style="display: flex; justify-content: space-between; max-width: 1200px; margin: 0 auto; padding: 0 20px;">
            <div class="footer-column" style="flex: 1.5;">
                <h4>🎓 PROJECT OVERVIEW</h4>
                <p><strong>VaccineNLP:</strong> Hệ thống AI đa nhiệm phân tích tin giả và thái độ vắc-xin trên mạng xã hội Việt Nam.</p>
                <p>Phát triển dựa trên công nghệ Transformer (PhoBERT) và Distilled Gemma-4 cho khả năng giải thích AI (XAI).</p>
            </div>
            <div class="footer-column" style="flex: 1;">
                <h4>👨‍🏫 INSTRUCTOR</h4>
                <p><strong>TS. Quách Đình Hoàng</strong></p>
                <p>Học viện Công nghệ Bưu chính Viễn thông</p>
                <p><em>Chuyên gia Xử lý Ngôn ngữ Tự nhiên (NLP)</em></p>
            </div>
            <div class="footer-column" style="flex: 1;">
                <h4>👥 RESEARCH TEAM</h4>
                <p><strong>Đinh Lại Quỳnh Phương</strong></p>
                <p>MSSV: 2211090016</p>
                <p>Email: quynhphuong.huph@gmail.com</p>
            </div>
            <div class="footer-column" style="flex: 0.8;">
                <h4>🏛️ INSTITUTION</h4>
                <p>HUPH - 2026</p>
                <p>Khoa Công nghệ Thông tin</p>
                <p>Hà Nội, Việt Nam</p>
            </div>
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

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}18, {color}08); border: 1px solid {color}40; border-radius: 12px; padding: 20px; text-align: center; min-height: 160px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 11px; color: #7a808c; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">{task_name}</div>
        <div style="font-size: 20px; font-weight: 700; color: {color}; margin-bottom: 6px;">{label}</div>
        <div style="font-size: 13px; color: {text_color}; opacity: 0.8;">Độ tin cậy: <strong style="color: {color};">{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="VaccineNLP · Dashboard", page_icon="🔬", layout="wide")
    
    if "is_dark" not in st.session_state: st.session_state.is_dark = True
    is_dark = st.session_state.is_dark
    bg_color = "#0d0f12" if is_dark else "#f8f9fa"
    text_color = "#e2e4e9" if is_dark else "#13161b"
    sidebar_bg = "linear-gradient(180deg, #13161b 0%, #0d0f12 100%)" if is_dark else "#ffffff"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        section[data-testid="stSidebar"] {{ background: {sidebar_bg}; border-right: 1px solid rgba(128,128,128,0.1); }}
        .stTextArea textarea {{ background: {"#13161b" if is_dark else "#ffffff"} !important; color: {text_color} !important; border-radius: 8px !important; border: 1px solid rgba(128,128,128,0.2) !important; }}
        .stButton > button {{ background: linear-gradient(135deg, #8b7fe8, #6366f1) !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; }}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f'<div style="text-align: center; padding: 16px 0;"><div style="font-size: 36px;">🔬</div><div style="font-size: 18px; font-weight: 700; color: {text_color};">VaccineNLP</div></div>', unsafe_allow_html=True)
        st.divider()
        if st.button(f"🌙 {'Light Mode' if is_dark else 'Dark Mode'}"):
            st.session_state.is_dark = not st.session_state.is_dark
            st.rerun()
        st.divider()
        selected_sample = st.radio("Chọn mẫu thử nghiệm:", options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()))
        model_selection = st.selectbox("Chọn mô hình:", options=list(MODEL_CONFIGS.keys()))

    model, tokenizer, ok = load_model(model_selection)
    xai_cache = json.load(open(XAI_CACHE_PATH, "r", encoding="utf-8")) if XAI_CACHE_PATH.exists() else {}

    st.markdown(f'<h1 style="color: {text_color};">🔬 VaccineNLP · Dashboard Phân tích</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích", "📊 Benchmark"])
    
    with tabs[0]:
        input_val = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area("Nhập văn bản:", value=input_val, height=140)
        if st.button("🔍 Phân tích") and user_text.strip():
            with st.spinner("Đang xử lý..."):
                res = predict(user_text.strip(), model, tokenizer)
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1: render_result_card("Tin giả", "misinfo", res["misinfo"], is_dark)
                with col2: render_result_card("Quan điểm", "stance", res["stance"], is_dark)
                with col3: render_result_card("Cảm xúc", "sentiment", res["sentiment"], is_dark)
                reasoning = xai_cache.get(user_text.strip())
                if reasoning:
                    with st.expander("📖 Giải thích chi tiết", expanded=True):
                        st.markdown(f'<div style="background: rgba(139,127,232,0.1); border-left: 3px solid #8b7fe8; padding: 16px; color: {text_color};">{reasoning}</div>', unsafe_allow_html=True)
    
    hien_thi_footer_chung(is_dark=is_dark)

if __name__ == "__main__":
    main()
