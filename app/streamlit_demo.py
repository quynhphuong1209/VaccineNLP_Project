"""
VaccineNLP · Explainable AI Dashboard
======================================
Streamlit demo for vaccine misinformation analysis.
Uses fine-tuned PhoBERT multitask model + cached Gemma-4 XAI reasoning.

Run:  streamlit run app/streamlit_demo.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer
# from pyvi import ViTokenizer  <-- Di chuyển vào trong hàm predict để tránh lỗi atexit

# ─────────────────────────────────────────────────────────────
# PATHS & CONFIGS
# ─────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
XAI_CACHE_PATH = APP_DIR / "xai_cache.json"

MODEL_CONFIGS = {
    "PhoBERT-v2": {
        "repo_id": "vinai/phobert-base-v2",
        "path": PROJECT_ROOT / "experiments" / "models" / "phobert_multitask_v2" / "best_model.pt",
        "type": "phobert"
    },
    "XLM-R-v1": {
        "repo_id": "xlm-roberta-base",
        "path": PROJECT_ROOT / "experiments" / "models" / "xlmr_multitask_v1" / "best_model.pt",
        "type": "xlmr"
    }
}

# ─────────────────────────────────────────────────────────────
# LABEL TAXONOMY (matches trained checkpoint)
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

# ─────────────────────────────────────────────────────────────
# SAMPLE TEXTS (selected from benchmark test set, present in XAI cache)
# ─────────────────────────────────────────────────────────────
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
        "Đừng quên like, share và subscribe channel của mình nhé💓\n\n"
        "Link nhạc:\n[No copyright] 1 Hour Relaxing Acoustic Guitar Music Collection: "
        "https://youtu.be/GgGIHkbXh5w\nMii Channel Music: "
        "https://www.youtube.com/watch?v=E9s1ltPGQOo\nSad meme music(no copyright material): "
        "https://youtu.be/0OSIEEM-FzI\nPiano Việt Nam sẽ chiến thắng | Sáng tác: Nguyễn Hải "
        "Phong: https://www.youtube.com/watch?v=kv6XnfuPyII\n --------\n\nĐọc blog mình viết "
        "tại: https://kimisgonnabeanadult.blogspot.com/"
    ),
}


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION (exact copy from training script)
# ─────────────────────────────────────────────────────────────
class VaccineMultitaskModel(nn.Module):
    """Multitask model with shared PhoBERT encoder and task-specific heads."""

    def __init__(self, model_name="vinai/phobert-base-v2",
                 num_misinfo=3, num_stance=3, num_sentiment=3):
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
        # Handle cases where pooler_output might be missing (common in some XLM-R versions)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback to CLS token
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        return (
            self.head_misinfo(pooled_output),
            self.head_stance(pooled_output),
            self.head_sentiment(pooled_output),
        )


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_key="PhoBERT-v2"):
    """Load selected multitask model + tokenizer (cached)."""
    cfg = MODEL_CONFIGS[model_key]
    model = VaccineMultitaskModel(model_name=cfg["repo_id"])
    checkpoint_loaded = False
    
    if cfg["path"].exists():
        state = torch.load(str(cfg["path"]), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        checkpoint_loaded = True
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"])
    return model, tokenizer, checkpoint_loaded


@st.cache_data
def load_xai_cache():
    """Load pre-built XAI reasoning cache (text → reasoning)."""
    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer) -> dict:
    """Run multitask inference with word segmentation + softmax."""
    from pyvi import ViTokenizer # Import tại đây để tránh lỗi Streamlit shutdown
    # Step 1: Word segmentation (mandatory for PhoBERT)
    segmented = ViTokenizer.tokenize(text)

    # Step 2: Tokenize
    enc = tokenizer(
        segmented,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        padding=True,
    )

    # Step 3: Inference
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(
            enc["input_ids"], enc["attention_mask"]
        )

    # Step 4: Softmax → confidence %
    probs_m = F.softmax(logits_m, dim=1)[0].tolist()
    probs_st = F.softmax(logits_st, dim=1)[0].tolist()
    probs_se = F.softmax(logits_se, dim=1)[0].tolist()

    return {
        "misinfo":   {"pred": int(probs_m.index(max(probs_m))),   "conf": probs_m},
        "stance":    {"pred": int(probs_st.index(max(probs_st))), "conf": probs_st},
        "sentiment": {"pred": int(probs_se.index(max(probs_se))), "conf": probs_se},
    }


def find_xai_reasoning(text: str, cache: dict) -> str | None:
    """Look up XAI reasoning by exact text match."""
    return cache.get(text)


# ─────────────────────────────────────────────────────────────
# UI HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────
def render_result_card(task_name: str, task_key: str, result: dict):
    """Render a styled result card for one task."""
    pred_id = result["pred"]
    conf_list = result["conf"]
    label = LABEL_MAPS[task_key][pred_id]
    color = LABEL_COLORS[task_key][pred_id]
    icon = LABEL_ICONS[task_key][pred_id]
    confidence = max(conf_list) * 100

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}18, {color}08);
        border: 1px solid {color}40;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 11px; color: #7a808c; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 4px;">{task_name}</div>
        <div style="font-size: 20px; font-weight: 700; color: {color};
                    margin-bottom: 6px;">{label}</div>
        <div style="font-size: 13px; color: #a0a5b0;">
            Độ tin cậy: <strong style="color: {color};">{confidence:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar for all classes
    with st.expander(f"📊 Chi tiết {task_name}", expanded=False):
        for idx, prob in enumerate(conf_list):
            class_label = LABEL_MAPS[task_key][idx]
            class_color = LABEL_COLORS[task_key][idx]
            pct = prob * 100
            st.markdown(f"""
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between;
                            font-size: 12px; color: #a0a5b0; margin-bottom: 2px;">
                    <span>{class_label}</span>
                    <span style="color: {class_color};">{pct:.1f}%</span>
                </div>
                <div style="background: #1a1e25; border-radius: 4px; height: 6px;">
                    <div style="background: {class_color}; width: {pct}%;
                                height: 6px; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_benchmark_tab():
    """Render the model comparison charts and tables."""
    import pandas as pd
    import plotly.graph_objects as go

    st.markdown("### 📊 Kết quả Benchmark (Gold Test Set)")
    st.caption("So sánh Macro F1-score giữa các mô hình trên tập Test 10% (186 mẫu).")

    # Metrics from JSON files
    data = [
        {"Model": "PhoBERT-v2", "Misinfo": 0.4547, "Stance": 0.6608, "Sentiment": 0.7325, "Type": "Student (Encoder)"},
        {"Model": "XLM-R-v1",   "Misinfo": 0.4572, "Stance": 0.6247, "Sentiment": 0.6918, "Type": "Baseline (Encoder)"},
        {"Model": "Gemma-4-4B", "Misinfo": 0.4400, "Stance": 0.6200, "Sentiment": 0.6600, "Type": "Teacher (Decoder)"},
    ]
    df = pd.DataFrame(data)

    # Table
    st.table(df)

    # Grouped Bar Chart
    fig = go.Figure()
    tasks = ["Misinfo", "Stance", "Sentiment"]
    colors = ["#3db882", "#4a9eed", "#e8504a"]

    for i, task in enumerate(tasks):
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df[task],
            name=task,
            marker_color=colors[i],
            text=df[task].apply(lambda x: f"{x:.2f}"),
            textposition='auto',
        ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#e2e4e9",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Nhận xét:** PhoBERT-v2 đạt hiệu năng tốt nhất ở các tác vụ phân loại Quan điểm và Cảm xúc, vượt qua cả Teacher Model Gemma-4 sau quá trình chưng cất tri thức.")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="VaccineNLP · XAI Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0d0f12; }
        .block-container { padding-top: 2rem; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #13161b 0%, #0d0f12 100%);
            border-right: 1px solid rgba(255,255,255,0.07);
        }
        .stTextArea textarea {
            background: #13161b !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            color: #e2e4e9 !important;
            font-size: 14px !important;
            border-radius: 8px !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #8b7fe8, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(139,127,232,0.4) !important;
        }
        div[data-testid="stExpander"] {
            background: #13161b;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 16px 0;">
            <div style="font-size: 36px; margin-bottom: 8px;">🔬</div>
            <div style="font-size: 18px; font-weight: 700; color: #e2e4e9;">VaccineNLP</div>
            <div style="font-size: 11px; color: #7a808c; letter-spacing: 0.1em;
                        margin-top: 4px;">EXPLAINABLE AI DASHBOARD</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("##### 📋 Mẫu thử nghiệm")
        st.caption("Chọn một mẫu có sẵn để xem phân tích XAI đầy đủ")

        selected_sample = st.radio(
            "Chọn mẫu:",
            options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("##### 🤖 Mô hình Phân loại")
        st.caption("Chọn mô hình Student để thực hiện dự đoán")
        model_selection = st.selectbox(
            "Chọn model:",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown(f"""
        <div style="font-size: 11px; color: #4a4f5a; line-height: 1.6;">
            <strong style="color: #7a808c;">Về hệ thống</strong><br>
            • <strong>Classifier:</strong> {model_selection}<br>
            • <strong>XAI Engine:</strong> Gemma-4 4B (cached)<br>
            • <strong>Tasks:</strong> Misinfo · Stance · Sentiment<br>
            • <strong>Benchmark:</strong> 186 samples
        </div>
        """, unsafe_allow_html=True)

    # ── Load resources ──
    model, tokenizer, checkpoint_loaded = load_model(model_selection)
    xai_cache = load_xai_cache()

    if not checkpoint_loaded:
        st.warning(
            f"⚠️ Không tìm thấy checkpoint cho `{model_selection}` tại đường dẫn dự kiến. "
            "Mô hình đang chạy ở trạng thái chưa fine-tune."
        )

    # ── Header ──
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 700; color: #e2e4e9;
                   margin-bottom: 4px; letter-spacing: -0.02em;">
            🔬 VaccineNLP · Phân tích Tin giả Vắc-xin
        </h1>
        <p style="font-size: 14px; color: #7a808c; margin: 0;">
            Hệ thống AI giải thích được — Phân loại đa chiều thông tin vắc-xin trên mạng xã hội Việt Nam
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Tabs ──
    tab_inference, tab_benchmark = st.tabs(["🔍 Phân tích Real-time", "📊 Thống kê Benchmark"])

    with tab_inference:
        # ── Input area ──
        if selected_sample != "Tự nhập":
            input_text = SAMPLE_TEXTS[selected_sample]
        else:
            input_text = ""

        user_text = st.text_area(
            "Nhập văn bản cần phân tích:",
            value=input_text,
            height=140,
            placeholder="Dán hoặc nhập nội dung bài viết về vắc-xin tại đây...",
            key="input_area"
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            analyze_btn = st.button("🔍 Phân tích", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Xóa / Reset", use_container_width=True):
                st.rerun()

        # ── Analysis ──
        if analyze_btn and user_text.strip():
            with st.spinner(f"🧠 Đang sử dụng {model_selection} để phân tích..."):
                time.sleep(0.5)  # Visual delay for UX
                result = predict(user_text.strip(), model, tokenizer)

            st.markdown("---")

            # ── Result cards ──
            st.markdown("""
            <div style="font-size: 11px; color: #7a808c; text-transform: uppercase;
                        letter-spacing: 0.15em; margin-bottom: 12px;">
                📊 KẾT QUẢ PHÂN LOẠI
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                render_result_card("Tin giả", "misinfo", result["misinfo"])
            with col2:
                render_result_card("Quan điểm", "stance", result["stance"])
            with col3:
                render_result_card("Cảm xúc", "sentiment", result["sentiment"])

            # ── XAI Reasoning section ──
            st.markdown("<br>", unsafe_allow_html=True)

            reasoning = find_xai_reasoning(user_text.strip(), xai_cache)

            if reasoning:
                st.markdown("""
                <div style="font-size: 11px; color: #7a808c; text-transform: uppercase;
                            letter-spacing: 0.15em; margin-bottom: 12px;">
                    🧪 LÝ LUẬN AI (EXPLAINABLE AI — Gemma-4 4B)
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📖 Xem phân tích chi tiết từ Gemma-4", expanded=True):
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #8b7fe818, #6366f108);
                        border: 1px solid rgba(139,127,232,0.2);
                        border-left: 3px solid #8b7fe8;
                        border-radius: 0 8px 8px 0;
                        padding: 16px 20px;
                        font-size: 14px;
                        line-height: 1.8;
                        color: #c8cad0;
                    ">
                        {reasoning}
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption("💡 Lý luận được sinh bởi Gemma-4 4B (QLoRA) qua cơ chế Knowledge Distillation.")
            else:
                st.info(
                    "💡 **Lý luận XAI không khả dụng** cho văn bản này. "
                    "Hệ thống XAI chỉ hỗ trợ 186 mẫu trong tập Benchmark. "
                    "Hãy chọn một mẫu từ thanh bên trái để xem phân tích đầy đủ.",
                    icon="ℹ️"
                )

        elif analyze_btn:
            st.warning("⚠️ Vui lòng nhập văn bản trước khi phân tích.")

    with tab_benchmark:
        render_benchmark_tab()

    # ── Footer ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        border-top: 1px solid rgba(255,255,255,0.07);
        padding-top: 16px;
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #4a4f5a;
    ">
        <span>HUPH · MSSV 2211090016 · 2026</span>
        <span>VaccineNLP · Explainable AI for Public Health</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
