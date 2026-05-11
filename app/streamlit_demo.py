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
import os
import sys
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
        "repo_id": "vinai/phobert-base-v2",
        "path": PROJECT_ROOT / "experiments" / "models" / "phobert-multitask-v2" / "pytorch_model.bin",
        "type": "phobert"
    },
    "XLM-R-v1": {
        "repo_id": "xlm-roberta-base",
        "path": PROJECT_ROOT / "experiments" / "models" / "xlm-r-multitask-v1" / "pytorch_model.bin",
        "type": "xlm-roberta"
    },
    "Gemma-4-4B": {
        "repo_id": "google/gemma-2b-it",
        "path": PROJECT_ROOT / "experiments" / "models" / "gemma-4-4B" / "pytorch_model.bin",
        "type": "gemma"
    }
}

# ─────────────────────────────────────────────────────────────
# LABEL TAXONOMY (matches trained checkpoint)
# ─────────────────────────────────────────────────────────────
LABEL_MAPS = {
    "misinfo": {0: "Không phải tin giả", 1: "Tin giả", 2: "Ranh giới"},
    "stance":  {0: "Ủng hộ", 1: "Phản đối", 2: "Trung lập", 3: "Không liên quan"},
    "sentiment": {0: "Tích cực", 1: "Tiêu cực", 2: "Trung lập"},
}

LABEL_COLORS = {
    "misinfo": {0: "#3db882", 1: "#e8504a", 2: "#d48f35"},
    "stance":  {0: "#3db882", 1: "#e8504a", 2: "#4a9eed", 3: "#9e9e9e"},
    "sentiment": {0: "#3db882", 1: "#e8504a", 2: "#4a9eed"},
}

LABEL_ICONS = {
    "misinfo": {0: "✅", 1: "🚨", 2: "⚠️"},
    "stance":  {0: "👍", 1: "👎", 2: "🤝", 3: "⚪"},
    "sentiment": {0: "😊", 1: "😠", 2: "😐"},
}

# ─────────────────────────────────────────────────────────────
# SAMPLE TEXTS
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
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class VaccineMultitaskModel(nn.Module):
    """Multitask model with shared PhoBERT encoder and task-specific heads."""

    def __init__(self, model_name="vinai/phobert-base-v2",
                 num_misinfo=3, num_stance=4, num_sentiment=3, token=None):
        super(VaccineMultitaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, token=token)
        self.encoder = AutoModel.from_pretrained(model_name, token=token)

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
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_key="PhoBERT-v2"):
    """Load selected multitask model + tokenizer (cached)."""
    cfg = MODEL_CONFIGS[model_key]
    checkpoint_loaded = False
    
    # Lấy token từ Secrets của Streamlit (nếu có)
    hf_token = None
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]
    elif "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
    
    try:
        model = VaccineMultitaskModel(model_name=cfg["repo_id"], token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"], token=hf_token)
        
        if cfg["path"].exists():
            state = torch.load(str(cfg["path"]), map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            checkpoint_loaded = True
            
    except Exception as e:
        # Trả về None để hiển thị lỗi minh bạch trên UI
        return None, None, False
        
    model.eval()
    return model, tokenizer, checkpoint_loaded

@st.cache_data
def load_xai_cache():
    """Load pre-built XAI reasoning cache (text → reasoning)."""
    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ─────────────────────────────────────────────────────────────
# INFERENCE & XAI FUNCTIONS
# ─────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer) -> dict:
    """Run multitask inference with word segmentation + softmax."""
    segmented = word_tokenize(text, format="text")
    enc = tokenizer(segmented, truncation=True, max_length=256, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])

    probs_m = F.softmax(logits_m, dim=1)[0].tolist()
    probs_st = F.softmax(logits_st, dim=1)[0].tolist()
    probs_se = F.softmax(logits_se, dim=1)[0].tolist()

    return {
        "misinfo":   {"pred": int(probs_m.index(max(probs_m))),   "conf": probs_m},
        "stance":    {"pred": int(probs_st.index(max(probs_st))), "conf": probs_st},
        "sentiment": {"pred": int(probs_se.index(max(probs_se))), "conf": probs_se},
    }

def find_xai_reasoning(text: str, cache: dict) -> str | None:
    return cache.get(text)

# ─────────────────────────────────────────────────────────────
# UI COMPONENTS (Premium Style)
# ─────────────────────────────────────────────────────────────
def hien_thi_footer_chung(is_dark=True):
    """Hiển thị chân trang (footer) 3 cột chuyên nghiệp cho đồ án VaccineNLP"""
    import base64
    
    # Xác định đường dẫn logo an toàn
    logo_path_local = PROJECT_ROOT / "abc1.png"
    logo_src = "https://huph.edu.vn/uploads/logo/logo-huph.png" # Link dự phòng
    
    try:
        if logo_path_local.exists():
            with open(logo_path_local, "rb") as img_file:
                logo_b64 = base64.b64encode(img_file.read()).decode()
                logo_src = f"data:image/png;base64,{logo_b64}"
    except Exception:
        pass

    # Cấu hình màu sắc theo giao diện Sáng/Tối
    if is_dark:
        footer_bg = "linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)"
        footer_text = "#ccc"
        title_color = "#007bff"
        label_color = "#eee"
        school_name_color = "#fff"
        col_border = "rgba(255,255,255,0.1)"
        bottom_text = "#777"
        project_vi = "#ffd700"
    else:
        footer_bg = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)"
        footer_text = "#444"
        title_color = "#0056b3"
        label_color = "#222"
        school_name_color = "#000"
        col_border = "rgba(0,0,0,0.1)"
        bottom_text = "#666"
        project_vi = "#b8860b"

    border_color = "#007bff"

    footer_html = f"""<style>.main-footer{{background:{footer_bg};padding:50px 20px;color:{footer_text};font-family:'Times New Roman',Times,serif!important;border-top:4px solid {border_color};box-shadow:0 -10px 25px rgba(0,123,255,0.1);margin-top:60px}}.footer-container{{display:flex;flex-wrap:wrap;justify-content:space-between;max-width:none!important;margin:0 auto;gap:20px}}.footer-col{{flex:1;min-width:300px;padding:0 25px;border-right:1px solid {col_border}}}.footer-col:last-child{{border-right:none}}.logo-col{{text-align:center}}.footer-logo-img{{width:100px;margin-bottom:15px;filter:drop-shadow(0 0 8px rgba(0,123,255,0.2))}}.footer-title{{color:{title_color};font-weight:bold;margin-bottom:15px;font-size:1.2rem;text-transform:uppercase;letter-spacing:1px}}.info-row{{margin-bottom:10px;font-size:1rem;line-height:1.4}}.info-label{{font-weight:bold;color:{label_color}}}.school-name{{font-weight:bold;color:{school_name_color};font-size:1.1rem;margin-bottom:5px}}.footer-bottom{{padding-top:20px;margin-top:30px;border-top:1px solid rgba(0,0,0,0.05);font-size:0.9rem;color:{bottom_text};text-align:center}}.project-name-vi{{color:{project_vi};font-weight:bold;font-style:italic;margin-bottom:8px}}.project-name-en{{font-size:0.9rem;opacity:0.8;line-height:1.3}}.footer-link{{color:{title_color}!important;text-decoration:none;transition:opacity 0.2s}}.footer-link:hover{{opacity:0.8;text-decoration:underline}}</style><div class="main-footer"><div class="footer-container"><div class="footer-col logo-col"><img src="{logo_src}" class="footer-logo-img" alt="HUPH Logo"><div class="school-name">TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</div><div style="font-size:0.9rem;opacity:0.8;"><p>📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p><p>🌐 <a href="https://huph.edu.vn/" target="_blank" class="footer-link">huph.edu.vn</a></p></div></div><div class="footer-col"><div class="footer-title">🔬 ĐỀ TÀI ĐỒ ÁN</div><div class="project-name-vi">Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam</div><div class="project-name-en">(Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)</div></div><div class="footer-col"><div class="footer-title">👥 NHÓM THỰC HIỆN</div><div class="info-row"><b>1. Kim Mạnh Hưng</b><br><span style="font-size:0.9rem;opacity:0.8;">MSSV: 2211090016 | Lớp: CNCQ KHDL1-1A<br>📧 <a href="mailto:2211090016@studenthuph.edu.vn" class="footer-link">2211090016@studenthuph.edu.vn</a></span></div><div class="info-row" style="margin-top:15px;"><b>2. Đinh Lê Quỳnh Phương</b><br><span style="font-size:0.9rem;opacity:0.8;">MSSV: 2211090031 | Lớp: CNCQ KHDL1-1A<br>📧 <a href="mailto:2211090031@studenthuph.edu.vn" class="footer-link">2211090031@studenthuph.edu.vn</a></span></div><div class="info-row" style="margin-top:20px;border-top:1px solid {col_border};padding-top:10px;"><span class="info-label">👨‍🏫 Giảng viên hướng dẫn:</span><br><span>TS. Trần Lâm Quân</span></div></div></div><div class="footer-bottom">© 2026 VaccineNLP Project | Đồ án tốt nghiệp chuyên ngành Khoa học Dữ liệu - HUPH</div></div>"""
    st.markdown(footer_html, unsafe_allow_html=True)

def render_result_card(task_name: str, task_key: str, result: dict):
    """Render a styled result card for one task with premium aesthetics."""
    pred_id = result["pred"]
    conf_list = result["conf"]
    label = LABEL_MAPS[task_key][pred_id]
    color = LABEL_COLORS[task_key][pred_id]
    icon = LABEL_ICONS[task_key][pred_id]
    confidence = max(conf_list) * 100

    is_dark = st.session_state.get("theme", "Dark") == "Dark"
    card_bg = "rgba(255, 255, 255, 0.03)" if is_dark else "#ffffff"
    text_color = "#e2e4e9" if is_dark else "#1a1e2e"
    secondary_text = "#888" if is_dark else "#666"
    shadow = "0 10px 20px rgba(0,0,0,0.3)" if is_dark else "0 10px 20px rgba(0,0,0,0.1)"

    st.markdown(f"""
    <div style="
        background: {card_bg};
        border: 1px solid {color}80;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: {shadow};
        font-family: 'Times New Roman', Times, serif !important;
    ">
        <div style="font-size: 40px; margin-bottom: 12px;">{icon}</div>
        <div style="font-size: 0.85rem; color: {secondary_text}; text-transform: uppercase;
                    letter-spacing: 0.15em; margin-bottom: 8px;">{task_name}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {color};
                    margin-bottom: 10px;">{label}</div>
        <div style="font-size: 1rem; color: {secondary_text};">
            Độ tin cậy: <strong style="color: {color};">{confidence:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📊 Chi tiết {task_name}", expanded=False):
        for idx, prob in enumerate(conf_list):
            class_label = LABEL_MAPS[task_key][idx]
            class_color = LABEL_COLORS[task_key][idx]
            pct = prob * 100
            bar_bg = "#262730" if is_dark else "#e6eaf1"
            label_text_color = "#a0a5b0" if is_dark else "#555"
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: {label_text_color};">
                    <span>{class_label}</span>
                    <span style="color: {class_color}; font-weight: bold;">{pct:.1f}%</span>
                </div>
                <div style="background: {bar_bg}; border-radius: 10px; height: 8px; margin-top: 4px;">
                    <div style="background: {class_color}; width: {pct}%; height: 8px; border-radius: 10px; box-shadow: 0 0 10px {class_color}40;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_benchmark_tab():
    import pandas as pd
    import plotly.graph_objects as go
    st.markdown("### 📊 Kết quả Benchmark (Gold Test Set)")
    data = [
        {"Model": "PhoBERT-v2", "Misinfo": 0.4547, "Stance": 0.6608, "Sentiment": 0.7325},
        {"Model": "XLM-R-v1",   "Misinfo": 0.4572, "Stance": 0.6247, "Sentiment": 0.6918},
        {"Model": "Gemma-4-4B", "Misinfo": 0.4400, "Stance": 0.6200, "Sentiment": 0.6600},
    ]
    df = pd.DataFrame(data)
    st.table(df)

    is_dark = st.session_state.get("theme", "Dark") == "Dark"
    chart_font_color = "#e2e4e9" if is_dark else "#111111"
    
    fig = go.Figure()
    tasks = ["Misinfo", "Stance", "Sentiment"]
    colors = ["#3db882", "#4a9eed", "#e8504a"]
    for i, task in enumerate(tasks):
        fig.add_trace(go.Bar(x=df["Model"], y=df[task], name=task, marker_color=colors[i]))
    
    fig.update_layout(
        barmode='group', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color=chart_font_color,
        legend_font_color=chart_font_color,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    # ─────────────────────────────────────────────────────────────
    # THEME STATE & TOGGLE
    # ─────────────────────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🔬 VaccineNLP</h2>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("##### 🎨 Giao diện")
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Tối", use_container_width=True, type="primary" if st.session_state.theme == "Dark" else "secondary"):
                st.session_state.theme = "Dark"
                st.rerun()
        with theme_col2:
            if st.button("☀️ Sáng", use_container_width=True, type="primary" if st.session_state.theme == "Light" else "secondary"):
                st.session_state.theme = "Light"
                st.rerun()
        
        st.divider()
        st.markdown("##### 📋 Mẫu thử nghiệm")
        selected_sample = st.radio("Chọn mẫu:", options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()), index=0)
        st.divider()
        st.markdown("##### 🤖 Mô hình Phân loại")
        st.info("Mô hình này đảm nhiệm việc phân loại nhãn (Tin giả, Quan điểm, Cảm xúc).")
        model_selection = st.selectbox("Chọn model:", options=list(MODEL_CONFIGS.keys()), index=0)

    # ─────────────────────────────────────────────────────────────
    # DYNAMIC CSS BASED ON THEME
    # ─────────────────────────────────────────────────────────────
    is_dark = st.session_state.theme == "Dark"
    bg_color = "#0d0f12" if is_dark else "#ffffff"
    card_bg = "rgba(255, 255, 255, 0.03)" if is_dark else "#fdfdfd"
    text_color = "#e2e4e9" if is_dark else "#111111"
    secondary_text = "#888" if is_dark else "#555"
    border_color = "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.1)"
    sidebar_bg = "#111" if is_dark else "#f8f9fa"
    input_bg = "#13161b" if is_dark else "#ffffff"
    input_border = "#2a5298" if is_dark else "#ced4da"

    st.markdown(f"""
    <style>
        /* PHÁ BỎ MỌI GIỚI HẠN CHIỀU RỘNG (NUCLEAR FULL-WIDTH) */
        
        /* 1. Target toàn bộ container chính */
        [data-testid="stAppViewContainer"], 
        [data-testid="stAppViewBlockContainer"],
        .main, .block-container {{
            max-width: none !important;
            width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}

        /* 2. Target các lớp cache động của Streamlit (thường gây bó hẹp 46rem/60rem) */
        div[class^="st-emotion-cache-"] {{
            max-width: none !important;
        }}

        /* 3. Đảm bảo sidebar không bị ảnh hưởng quá mức (giữ nguyên độ rộng sidebar) */
        [data-testid="stSidebar"] div[class^="st-emotion-cache-"] {{
            max-width: 20rem !important;
        }}

        /* 4. Ép các khối nội dung bên trong dãn 100% */
        .element-container, .stMarkdown, .stVerticalBlock, div[data-testid="stVerticalBlock"] > div {{
            width: 100% !important;
        }}

        /* 5. Loại bỏ nền đen thừa nếu có ở hai bên */
        .stApp {{
            background-color: {bg_color} !important;
        }}
        
        /* Toàn bộ giao diện chính */
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Font chữ toàn cục và màu chữ */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stMarkdown, p, label, li, span, h1, h2, h3, h4, h5, h6, button, input, select, textarea {{
            font-family: 'Times New Roman', Times, serif !important;
            color: {text_color} !important;
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio span {{
            color: {text_color} !important;
        }}
        
        /* BẢO VỆ CÁC ICON (Không đổi màu hoặc font chữ icon) */
        .stIconMaterial, [data-testid="stIconMaterial"], span[data-baseweb="icon"], svg {{
            font-family: 'Material Symbols Rounded' !important;
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab"] {{
            height: 48px !important;
            background-color: {"#1a1a2e" if is_dark else "#e9ecef"} !important;
            border-radius: 10px 10px 0 0;
            color: {text_color} !important;
            border: 1px solid {border_color};
            padding: 0 25px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #007bff !important;
            color: white !important;
            font-weight: bold !important;
        }}

        /* Input area styling */
        .stTextArea textarea {{ 
            background: {input_bg} !important; 
            color: {text_color} !important; 
            border-radius: 12px !important; 
            border: 1px solid {input_border} !important; 
        }}

        /* Button styling */
        .stButton > button {{ 
            border-radius: 12px !important; 
            font-weight: 600 !important;
        }}
        
        /* Expander styling */
        [data-testid="stExpander"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
        }}

        /* Fix Selectbox (Dropdown) colors in Light Mode */
        div[data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: 1px solid {input_border} !important;
        }}
        div[role="listbox"] {{
            background-color: {sidebar_bg} !important;
            color: {text_color} !important;
        }}
        div[role="option"] {{
            background-color: transparent !important;
            color: {text_color} !important;
        }}
        div[role="option"]:hover {{
            background-color: #007bff20 !important;
        }}

        /* Table styling - Thêm viền đen cho bảng Benchmark */
        table {{
            width: 100%;
            border-collapse: collapse !important;
            color: {text_color} !important;
            font-family: 'Times New Roman', Times, serif !important;
            border: 1px solid {"#444" if is_dark else "#000"} !important;
        }}
        th, td {{
            border: 1px solid {"#444" if is_dark else "#000"} !important;
            padding: 12px !important;
            text-align: center !important;
        }}
        th {{
            background-color: {"#1a1a2e" if is_dark else "#f0f2f6"} !important;
            font-weight: bold !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    model, tokenizer, checkpoint_loaded = load_model(model_selection)
    xai_cache = load_xai_cache()

    if model is None:
        st.error(f"❌ Không thể tải mã nguồn gốc của `{model_selection}` từ Hugging Face.")
        st.info("💡 **Lý do phổ biến:** Mô hình này (như Gemma) yêu cầu quyền truy cập (Gated Model). \n\n**Cách khắc phục trên Streamlit Cloud:** Hãy vào phần Settings > Secrets và thêm dòng: `HF_TOKEN = 'your_huggingface_token'`")
        st.stop()

    # Ẩn đi cảnh báo không tìm thấy checkpoint theo yêu cầu
    # if not checkpoint_loaded:
    #     st.warning(f"⚠️ Không tìm thấy checkpoint cho `{model_selection}`. Chế độ chưa fine-tune.")

    is_dark = st.session_state.get("theme", "Dark") == "Dark"
    banner_bg = "rgba(255,255,255,0.02)" if is_dark else "rgba(0,0,0,0.02)"
    banner_border = "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.05)"
    banner_p_color = "white" if is_dark else "#333"
    banner_p_opacity = "0.9" if is_dark else "1.0"

    st.markdown(f"""
    <div style="width: 100%; text-align: center; margin-bottom: 2.5rem; padding: 2.5rem; background: {banner_bg}; border-radius: 20px; border: 1px solid {banner_border}; box-sizing: border-box;">
        <h1 style="color: #FFD700; font-family: 'Times New Roman', Times, serif; font-weight: bold; font-size: 3.5rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔬 VaccineNLP · Dashboard</h1>
        <p style="color: {banner_p_color}; font-family: 'Times New Roman', Times, serif; font-style: italic; font-size: 1.5rem; opacity: {banner_p_opacity};">Hệ thống AI giải thích được — Phân loại đa chiều thông tin vắc-xin trên mạng xã hội</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích Real-time", "📊 Thống kê Benchmark"])

    with tabs[0]:
        input_text = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area("Nhập văn bản cần phân tích:", value=input_text, height=140, placeholder="Dán nội dung bài viết về vắc-xin...")
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            analyze_btn = st.button("🔍 Phân tích", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Reset", use_container_width=True): st.rerun()

        if analyze_btn and user_text.strip():
            with st.spinner(f"🧠 {model_selection} đang xử lý..."):
                time.sleep(0.5)
                result = predict(user_text.strip(), model, tokenizer)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1: render_result_card("Tin giả", "misinfo", result["misinfo"])
            with col2: render_result_card("Quan điểm", "stance", result["stance"])
            with col3: render_result_card("Cảm xúc", "sentiment", result["sentiment"])

            reasoning = find_xai_reasoning(user_text.strip(), xai_cache)
            if reasoning:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🧠 Hệ thống Giải thích (XAI Engine)")
                with st.expander("📖 Xem giải thích chi tiết từ Gemma-4", expanded=True):
                    st.markdown(f"<div style='border-left: 3px solid #007bff; padding-left: 20px; color: {text_color}; opacity: 0.9;'>{reasoning}</div>", unsafe_allow_html=True)
                    st.caption("💡 Đây là mô hình Reasoning Engine (Gemma-4) giải thích lý do cho kết quả phân loại ở trên.")
            else:
                st.info("💡 Lý luận XAI không khả dụng cho văn bản này. Hãy chọn mẫu từ thanh bên.")
        elif analyze_btn:
            st.warning("⚠️ Vui lòng nhập văn bản.")

    with tabs[1]:
        render_benchmark_tab()

    hien_thi_footer_chung(is_dark=is_dark)

if __name__ == "__main__":
    main()
