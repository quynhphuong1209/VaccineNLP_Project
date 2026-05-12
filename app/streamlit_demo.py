"""
VaccineNLP · Explainable AI Dashboard
======================================
Streamlit demo for vaccine misinformation analysis.
Uses fine-tuned PhoBERT multitask model + cached Gemma-4 XAI reasoning.
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
        "bị doạ cho sợ hãi, đem con đi tiêm vì bị bóng ma sợ hãi nó đè, chứ thực chất chả "
        "có tác dụng gì còn gây hại cho cơ thể nữa. Bao giờ bạn hết sợ hãi thì tự khắc bạn "
        "sẽ hết tiêm. Còn sợ là còn tiêm."
    ),
    "💬 Từ lóng MXH - Tin giả nguy hiểm": (
        "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh \n"
        "Nhưng tiêm vắc xin thì là tiêm thuốc độc vào người \n\n"
        "Càng tiêm nhiều càng bệnh nhiều \n\n"
        "Bạn xem thời xưa có ai phải tiêm đâu sao ai cũng khỏe mạnh\n\n"
        "Muốn thải độc vx , kim loại nặng thì nên cho uống nc lá mùi đun lên \n\n"
        "Muốn hạ sốt ( sốt nóng ) cho con uống nc chanh ấm có đường \n"
        "Lấy chanh xoa toàn thân"
    ),
    "✅ Tin chính xác - Chia sẻ tích cực": (
        "TRẢI NGHIỆM TIÊM VACCINE MODERNA | Kim's here daily vlog #covid19 #vaccine #moderna. "
        "Mọi người đã tiêm vaccine hết chưa nhỉ?\nNhiều lúc mình cũng hơi ức chế vì khu nhà "
        "mình nằm trong vùng đỏ ấy vì vẫn phải giãn cách theo chỉ thị 16, nhưng có thể vì "
        "thế mà tốc độ tiêm của phường được đẩy lên chăng?\nMọi người giữ gìn sức khoẻ nhé! "
        "Mong là Hà Nội sẽ được trở lại trạng thái bình thường mới sớm nè😇\n"
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
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return (
            self.head_misinfo(pooled_output),
            self.head_stance(pooled_output),
            self.head_sentiment(pooled_output),
        )

# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
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
        # Fix keys if they were saved with "head_" prefix instead of "head_misinfo" etc.
        new_state = { (k.replace("head_", "heads.") if k.startswith("head_") and "heads." not in k else k): v for k, v in state.items() }
        # Re-map to match current class names
        final_state = {}
        for k, v in state.items():
            final_state[k] = v
            
        model.load_state_dict(final_state, strict=False)
        model.eval()
        gc.collect()
        return model, tokenizer, True
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình: {str(e)}")
        return None, None, False

@st.cache_data
def load_xai_cache():
    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def predict(text: str, model, tokenizer) -> dict:
    segmented = word_tokenize(text, format="text")
    enc = tokenizer(segmented, truncation=True, max_length=256, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])
    
    probs_m = F.softmax(logits_m, dim=1)[0].tolist()
    probs_st = F.softmax(logits_st, dim=1)[0].tolist()
    probs_se = F.softmax(logits_se, dim=1)[0].tolist()

    return {
        "misinfo":   {"pred": probs_m.index(max(probs_m)),   "conf": probs_m},
        "stance":    {"pred": probs_st.index(max(probs_st)), "conf": probs_st},
        "sentiment": {"pred": probs_se.index(max(probs_se)), "conf": probs_se},
    }

# ─────────────────────────────────────────────────────────────
# UI COMPONENTS (STAYING THE SAME)
# ─────────────────────────────────────────────────────────────
def render_result_card(task_name: str, task_key: str, result: dict):
    pred_id = result["pred"]
    conf_list = result["conf"]
    label = LABEL_MAPS[task_key][pred_id]
    color = LABEL_COLORS[task_key][pred_id]
    icon = LABEL_ICONS[task_key][pred_id]
    confidence = max(conf_list) * 100

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}18, {color}08); border: 1px solid {color}40; border-radius: 12px; padding: 20px; text-align: center; min-height: 160px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 11px; color: #7a808c; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">{task_name}</div>
        <div style="font-size: 20px; font-weight: 700; color: {color}; margin-bottom: 6px;">{label}</div>
        <div style="font-size: 13px; color: #a0a5b0;">Độ tin cậy: <strong style="color: {color};">{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📊 Chi tiết {task_name}", expanded=False):
        for idx, prob in enumerate(conf_list):
            class_label = LABEL_MAPS[task_key][idx]
            class_color = LABEL_COLORS[task_key][idx]
            pct = prob * 100
            st.markdown(f"""
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #a0a5b0; margin-bottom: 2px;">
                    <span>{class_label}</span><span style="color: {class_color};">{pct:.1f}%</span>
                </div>
                <div style="background: #1a1e25; border-radius: 4px; height: 6px;">
                    <div style="background: {class_color}; width: {pct}%; height: 6px; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_benchmark_tab():
    import pandas as pd
    import plotly.graph_objects as go
    st.markdown("### 📊 Thống kê Benchmark")
    data = [
        {"Model": "PhoBERT-v2", "Misinfo": 0.4547, "Stance": 0.6608, "Sentiment": 0.7325},
        {"Model": "XLM-R-v1",   "Misinfo": 0.4572, "Stance": 0.6247, "Sentiment": 0.6918},
    ]
    df = pd.DataFrame(data)
    st.table(df)
    fig = go.Figure()
    for task, color in zip(["Misinfo", "Stance", "Sentiment"], ["#3db882", "#4a9eed", "#e8504a"]):
        fig.add_trace(go.Bar(x=df["Model"], y=df[task], name=task, marker_color=color))
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e2e4e9")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="VaccineNLP · Dashboard", page_icon="🔬", layout="wide")
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0d0f12; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #13161b 0%, #0d0f12 100%); border-right: 1px solid rgba(255,255,255,0.07); }
        .stTextArea textarea { background: #13161b !important; border: 1px solid rgba(255,255,255,0.12) !important; color: #e2e4e9 !important; border-radius: 8px !important; }
        .stButton > button { background: linear-gradient(135deg, #8b7fe8, #6366f1) !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; transition: all 0.3s ease !important; }
        .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(139,127,232,0.4) !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 16px 0;"><div style="font-size: 36px;">🔬</div><div style="font-size: 18px; font-weight: 700; color: #e2e4e9;">VaccineNLP</div><div style="font-size: 11px; color: #7a808c; letter-spacing: 0.1em; margin-top: 4px;">XAI DASHBOARD</div></div>', unsafe_allow_html=True)
        st.divider()
        selected_sample = st.radio("Chọn mẫu thử nghiệm:", options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()))
        st.divider()
        model_selection = st.selectbox("Chọn mô hình dự đoán:", options=list(MODEL_CONFIGS.keys()))
        st.divider()
        st.markdown(f'<div style="font-size: 11px; color: #4a4f5a; line-height: 1.6;"><strong>Hệ thống:</strong><br>• Classifier: {model_selection}<br>• XAI: Cached Reasoning<br>• Tasks: 3 Classification</div>', unsafe_allow_html=True)

    model, tokenizer, ok = load_model(model_selection)
    xai_cache = load_xai_cache()

    st.markdown('<div style="margin-bottom: 24px;"><h1 style="font-size: 28px; font-weight: 700; color: #e2e4e9;">🔬 VaccineNLP · Dashboard Phân tích</h1><p style="font-size: 14px; color: #7a808c;">Phân loại tin giả và thái độ vắc-xin bằng PhoBERT Multitask Model</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích", "📊 Benchmark"])
    
    with tabs[0]:
        input_val = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area("Văn bản cần phân tích:", value=input_val, height=140)
        
        c1, c2, _ = st.columns([1, 1, 4])
        analyze = c1.button("🔍 Phân tích")
        if c2.button("🗑️ Reset"): st.rerun()

        if analyze and user_text.strip():
            with st.spinner("Đang xử lý..."):
                res = predict(user_text.strip(), model, tokenizer)
                st.markdown("---")
                st.markdown('<div style="font-size: 11px; color: #7a808c; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 12px;">📊 KẾT QUẢ DỰ ĐOÁN</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1: render_result_card("Tin giả", "misinfo", res["misinfo"])
                with col2: render_result_card("Quan điểm", "stance", res["stance"])
                with col3: render_result_card("Cảm xúc", "sentiment", res["sentiment"])

                reasoning = xai_cache.get(user_text.strip())
                if reasoning:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div style="font-size: 11px; color: #7a808c; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 12px;">🧪 GIẢI THÍCH (XAI CACHE)</div>', unsafe_allow_html=True)
                    with st.expander("📖 Xem giải thích chi tiết", expanded=True):
                        st.markdown(f'<div style="background: linear-gradient(135deg, #8b7fe818, #6366f108); border-left: 3px solid #8b7fe8; padding: 16px 20px; font-size: 14px; line-height: 1.8; color: #c8cad0;">{reasoning}</div>', unsafe_allow_html=True)
                else:
                    st.info("💡 Không có dữ liệu XAI cache cho văn bản này.")
    
    with tabs[1]: render_benchmark_tab()

    st.markdown('<br><br><div style="border-top: 1px solid rgba(255,255,255,0.07); padding-top: 16px; display: flex; justify-content: space-between; font-size: 11px; color: #4a4f5a;"><span>HUPH · 2026</span><span>VaccineNLP · Explainable AI</span></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
