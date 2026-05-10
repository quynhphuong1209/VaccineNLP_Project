"""
VaccineNLP · Explainable AI Dashboard
======================================
Streamlit demo for vaccine misinformation analysis.
Uses fine-tuned PhoBERT multitask model + cached Gemma-4 XAI reasoning.

Run:  streamlit run app/streamlit_demo.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
import json
import time
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để nhận diện 'src'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.common import paths
from src.modeling.phobert_multitask_trainer import VaccineMultitaskModel
from src.modeling.inference import VaccineInferenceAPI

# ─────────────────────────────────────────────────────────────
# CONFIGS
# ─────────────────────────────────────────────────────────────
XAI_CACHE_PATH = paths.APP_DIR / "xai_cache.json"

# Label Taxonomy (mapping match với taxonomy trong inference.py)
LABEL_MAPS = {
    "misinfo":   {1: "Tin giả", 2: "Chính xác", 0: "Không liên quan"},
    "stance":    {1: "Ủng hộ", 2: "Phản đối", 0: "Trung lập", 3: "Không rõ"},
    "sentiment": {2: "Tiêu cực", 0: "Trung tính", 1: "Tích cực"}
}

LABEL_COLORS = {
    "misinfo":   {1: "#e8504a", 2: "#3db882", 0: "#d48f35"},
    "stance":    {1: "#3db882", 2: "#e8504a", 0: "#4a9eed", 3: "#7a808c"},
    "sentiment": {2: "#e8504a", 0: "#4a9eed", 1: "#3db882"},
}

LABEL_ICONS = {
    "misinfo":   {1: "🚨", 2: "✅", 0: "⚠️"},
    "stance":    {1: "👍", 2: "👎", 0: "🤝", 3: "❓"},
    "sentiment": {2: "😠", 0: "😐", 1: "😊"},
}

# Sample texts for Demo
SAMPLE_TEXTS = {
    "🚨 Tin giả - Chống vaccine cực đoan": (
        "Ko tiêm mũi nào hết. Ko biết bạn thuộc thế hệ nào, chứ bạn nhìn xem thế hệ 8x "
        "trở về trước ko có ai tiêm bất cứ mũi gì vẫn khoẻ mạnh đó thôi. Cha mẹ thời nay "
        "bị doạ cho sợ hãi, đem con đi tiêm vì bị bóng ma sợ hãi nó đè."
    ),
    "💬 Từ lóng MXH - Tin giả nguy hiểm": (
        "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh. "
        "Nhưng tiêm vắc xin thì là tiêm thuốc độc vào người. Càng tiêm nhiều càng bệnh nhiều."
    ),
    "✅ Tin chính xác - Chia sẻ tích cực": (
        "Mọi người đã tiêm vaccine hết chưa nhỉ? Đừng quên like, share và subscribe channel của mình nhé💓"
    ),
}

# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_inference_api():
    """Khởi tạo API dự đoán (cached)"""
    return VaccineInferenceAPI(model_version="phobert-multitask-v2")

@st.cache_data
def load_xai_cache():
    """Load pre-built XAI reasoning cache (text → reasoning)."""
    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ─────────────────────────────────────────────────────────────
# UI HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────
def hien_thi_footer_chung():
    """Hiển thị chân trang (footer) chuyên nghiệp tương tự Rehab-AI-Monitor"""
    footer_bg = "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)"
    footer_text = "#ccc"
    border_color = "#007bff"
    title_color = "#007bff"
    label_color = "#eee"
    school_name_color = "#fff"

    footer_html = f"""
    <style>
        .main-footer {{
            background: {footer_bg};
            padding: 50px 20px;
            color: {footer_text};
            font-family: 'Times New Roman', Times, serif !important;
            border-top: 4px solid {border_color};
            box-shadow: 0 -10px 25px rgba(0, 123, 255, 0.15);
            margin-top: 60px;
        }}
        .footer-container {{
            display: flex;
            flex-wrap: nowrap;
            justify-content: space-between;
            max-width: 1200px;
            margin: 0 auto;
            gap: 40px;
        }}
        .footer-col {{
            flex: 1;
            padding: 0 20px;
        }}
        .footer-title {{
            color: {title_color};
            font-weight: bold;
            margin-bottom: 20px;
            font-size: 1.2rem;
            text-transform: uppercase;
            font-family: 'Times New Roman', Times, serif !important;
        }}
        .info-row {{
            margin-bottom: 10px;
            font-size: 1rem;
            line-height: 1.4;
            font-family: 'Times New Roman', Times, serif !important;
        }}
        .footer-bottom {{
            padding-top: 20px;
            margin-top: 30px;
            border-top: 1px solid rgba(255,255,255,0.05);
            font-size: 0.9rem;
            color: #777;
            text-align: center;
        }}
        .school-name {{
            font-weight: bold; 
            color: {school_name_color}; 
            font-size: 1.1rem;
            margin-bottom: 5px;
        }}
    </style>
    <div class="main-footer">
        <div class="footer-container">
            <div class="footer-col">
                <div class="school-name">DỰ ÁN VACCINENLP</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    <p>Phân tích tin giả vắc-xin bằng trí tuệ nhân tạo đa nhiệm và giải thích được (XAI).</p>
                </div>
            </div>
            <div class="footer-col">
                <div class="footer-title">👤 NGHIÊN CỨU VIÊN</div>
                <div class="info-row">Đinh Lê Quỳnh Phương</div>
                <div class="info-row">Email: quynhphuong@studenthuph.edu.vn</div>
            </div>
            <div class="footer-col">
                <div class="footer-title">🏫 ĐƠN VỊ CÔNG TÁC</div>
                <div class="info-row">Khoa Khoa học dữ liệu Y sinh</div>
                <div class="info-row">Trường Đại học Y tế Công cộng</div>
            </div>
        </div>
        <div class="footer-bottom">
            © 2026 VaccineNLP Project | Phát triển bởi Nhóm Nghiên cứu viên HUPH
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
def render_result_card(task_name: str, task_key: str, label_name: str, confidence: float):
    """Render a styled result card for one task with premium aesthetics."""
    pred_id = next((k for k, v in LABEL_MAPS[task_key].items() if v == label_name), 0)
    color = LABEL_COLORS[task_key].get(pred_id, "#7a808c")
    icon = LABEL_ICONS[task_key].get(pred_id, "ℹ️")

    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid {color}80;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        font-family: 'Times New Roman', Times, serif !important;
    ">
        <div style="font-size: 40px; margin-bottom: 12px;">{icon}</div>
        <div style="font-size: 0.85rem; color: #888; text-transform: uppercase;
                    letter-spacing: 0.15em; margin-bottom: 8px;">{task_name}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {color};
                    margin-bottom: 10px; font-family: 'Times New Roman', Times, serif !important;">{label_name}</div>
        <div style="font-size: 1rem; color: #a0a5b0;">
            Độ tin cậy: <strong style="color: {color};">{confidence*100:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="VaccineNLP · XAI Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    # ── Custom CSS (Match Rehab AI Monitor) ──
    st.markdown("""
    <style>
        /* === GLOBAL TYPOGRAPHY (SERIF) === */
        html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6, button {
            font-family: 'Times New Roman', Times, serif !important;
        }

        .stApp { background-color: #0d0f12; }
        
        /* === TABS STYLING === */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px !important;
            background-color: #1a1a2e !important;
            border-radius: 10px 10px 0 0;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0 25px !important;
            font-family: 'Times New Roman', Times, serif !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #007bff !important;
            border: 1px solid #007bff !important;
            font-weight: bold !important;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
        }
        
        .stTextArea textarea { background: #13161b !important; color: #e2e4e9 !important; border-radius: 12px !important; border: 1px solid #2a5298 !important; }
        .stButton > button { background: linear-gradient(135deg, #007bff, #00c6ff) !important; color: white !important; border-radius: 12px !important; font-weight: 600 !important; border: none !important; padding: 10px 25px !important; }
        
        /* Expander styling */
        .stExpander { background: #13161b !important; border: 1px solid #2a5298 !important; border-radius: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🔬 VaccineNLP</h2>", unsafe_allow_html=True)
        st.divider()
        st.markdown("##### 📋 Mẫu thử nghiệm")
        selected_sample = st.radio("Chọn mẫu:", options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()), index=0)
        st.divider()
        st.info("Hệ thống sử dụng PhoBERT để phân loại và Gemma-4 để giải thích lý luận (XAI).")

    # ── Header (Premium Style) ──
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05);">
        <h1 style="color: #FFD700; font-family: 'Times New Roman', Times, serif; font-weight: bold; font-size: 3rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🔬 VaccineNLP · Dashboard</h1>
        <p style="color: white; font-family: 'Times New Roman', Times, serif; font-style: italic; font-size: 1.3rem;">Hệ thống AI giải thích được — Phân loại đa chiều thông tin vắc-xin trên mạng xã hội</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ORGANIZATION ──
    tabs = st.tabs(["🏠 PHÂN TÍCH", "📖 HƯỚNG DẪN", "🌐 CÔNG NGHỆ", "💬 PHẢN HỒI"])
    
    with tabs[0]: # TAB: PHÂN TÍCH
        # ── Load resources ──
        api = load_inference_api()
        xai_cache = load_xai_cache()

        # ── Input area ──
        input_text = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area("Nhập văn bản cần phân tích:", value=input_text, height=140)

        if st.button("🔍 Phân tích"):
            if user_text.strip():
                with st.spinner("🧠 Đang phân tích đa nhiệm..."):
                    result = api.predict(user_text.strip())
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                res = result["results"]
                with col1:
                    render_result_card("Tin giả", "misinfo", res["misinfo"]["label"], res["misinfo"]["confidence"])
                with col2:
                    render_result_card("Quan điểm", "stance", res["stance"]["label"], res["stance"]["confidence"])
                with col3:
                    render_result_card("Cảm xúc", "sentiment", res["sentiment"]["label"], res["sentiment"]["confidence"])

                # ── XAI Reasoning ──
                st.markdown("<br>", unsafe_allow_html=True)
                reasoning = xai_cache.get(user_text.strip())
                if reasoning:
                    with st.expander("📖 Xem giải thích từ Gemma-4 (XAI)", expanded=True):
                        st.info(reasoning)
                else:
                    st.caption("💡 Không tìm thấy lý luận XAI trong cache cho văn bản này.")
            else:
                st.warning("⚠️ Vui lòng nhập văn bản.")

    with tabs[1]: # TAB: HƯỚNG DẪN
        st.markdown("### 📖 HƯỚNG DẪN SỬ DỤNG HỆ THỐNG")
        st.info("Hệ thống VaccineNLP giúp bạn kiểm tra tính xác thực của các bài đăng về vắc-xin.")
        with st.expander("Bước 1: Nhập liệu", expanded=True):
            st.write("Dán nội dung văn bản từ mạng xã hội (Facebook, TikTok, v.v.) vào ô nhập liệu.")
        with st.expander("Bước 2: Phân tích đa nhiệm"):
            st.write("Hệ thống sẽ đồng thời dự đoán: Loại tin (Tin giả/Thật), Quan điểm (Ủng hộ/Phản đối) và Cảm xúc.")
        with st.expander("Bước 3: Xem giải thích XAI"):
            st.write("Sử dụng tính năng XAI để hiểu tại sao AI lại đưa ra kết luận đó thông qua lý luận từ mô hình ngôn ngữ lớn.")

    with tabs[2]: # TAB: CÔNG NGHỆ
        st.markdown("### 🌐 HỆ SINH THÁI CÔNG NGHỆ")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            #### 🤖 PhoBERT Multitask
            Mô hình ngôn ngữ tiếng Việt mạnh mẽ, được tinh chỉnh cho 3 nhiệm vụ phân loại đồng thời.
            """)
        with col_t2:
            st.markdown("""
            #### 🧠 Gemma-4 XAI
            Trí tuệ nhân tạo thế hệ mới cung cấp khả năng lý luận, giúp minh bạch hóa các quyết định của AI.
            """)

    with tabs[3]: # TAB: PHẢN HỒI
        st.markdown("### 💬 Ý KIẾN NGƯỜI DÙNG")
        with st.form("feedback"):
            f_name = st.text_input("Tên của bạn")
            f_msg = st.text_area("Góp ý của bạn")
            if st.form_submit_button("Gửi phản hồi"):
                st.success("Cảm ơn bạn đã đóng góp ý kiến!")

    # ── Footer ──
    hien_thi_footer_chung()

if __name__ == "__main__":
    main()
