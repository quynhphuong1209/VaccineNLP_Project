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
from underthesea import word_tokenize
import logging
import os
import gc
import json
import torch
import torch.nn as nn

# Ẩn các cảnh báo
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ẩn các cảnh báo không cần thiết của transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Mô hình mặc định cho hệ thống giải thích (XAI Engine)
XAI_MODEL_REPO = "quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai"

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
        from transformers import AutoConfig, AutoModel
        super(VaccineMultitaskModel, self).__init__()
        import transformers
        AutoConfig = transformers.AutoConfig
        AutoModel = transformers.AutoModel
        
        self.config = AutoConfig.from_pretrained(model_name, token=token, trust_remote_code=True)
        # Nạp encoder với chế độ tiết kiệm RAM tối đa
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            token=token, 
            trust_remote_code=True,
            low_cpu_mem_usage=True # Tối ưu nạp trên CPU
        )

        hidden_size = self.config.hidden_size
        # Cấu trúc linh hoạt: Hỗ trợ cả ModuleDict và các lớp riêng lẻ
        self.heads = nn.ModuleDict({
            "misinfo": nn.Linear(hidden_size, num_misinfo),
            "stance": nn.Linear(hidden_size, num_stance),
            "sentiment": nn.Linear(hidden_size, num_sentiment)
        })
        # Alias để tương thích ngược nếu cần
        self.head_misinfo = self.heads["misinfo"]
        self.head_stance = self.heads["stance"]
        self.head_sentiment = self.heads["sentiment"]
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        return (
            self.heads["misinfo"](pooled_output),
            self.heads["stance"](pooled_output),
            self.heads["sentiment"](pooled_output),
        )

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_model(model_key="PhoBERT-v2"):
    """Load model BERT/XLM-R từ Hugging Face với kỹ thuật mmap."""
    import gc
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    
    # Dọn dẹp RAM trước khi nạp
    gc.collect()
    
    cfg = MODEL_CONFIGS[model_key]
    hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("VaccineNLP_TOKEN")
    
    try:
        # Tải file checkpoint (.pt)
        model_path = hf_hub_download(repo_id=cfg["repo_id"], filename="best_model.pt", token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_repo"], token=hf_token, trust_remote_code=True)
        model = VaccineMultitaskModel(model_name=cfg["base_repo"], token=hf_token)
        
        # Nạp trọng số (mmap=True giúp tiết kiệm RAM)
        state = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        new_state = { (k.replace("head_", "heads.") if k.startswith("head_") and "heads." not in k else k): v for k, v in state.items() }
        model.load_state_dict(new_state, strict=False)
        model.eval()
        
        del state
        gc.collect()
        return model, tokenizer, True
        
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình: {str(e)}")
        return None, None, False

@st.cache_data
def load_xai_cache():
    """Load pre-built XAI reasoning cache (text → reasoning)."""
    if XAI_CACHE_PATH.exists():
        with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def find_xai_reasoning(text: str, cache: dict) -> str | None:
    """Look up XAI reasoning with ultra-robust alphanumeric normalization."""
    if not text or not cache:
        return None
    
    import re
    def normalize(t):
        # Chuyển về chữ thường và chỉ giữ lại ký tự chữ và số
        return re.sub(r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', t.lower())
    
    input_norm = normalize(text)
    
    # 1. Thử tìm kiếm chính xác trước
    if text.strip() in cache:
        return cache[text.strip()]
        
    # 2. Thử tìm kiếm theo nội dung đã chuẩn hóa (bỏ qua dấu câu, emoji, xuống dòng)
    for k, v in cache.items():
        if normalize(k) == input_norm:
            return v
            
    # 3. Thử tìm kiếm mờ (nếu đầu văn bản khớp)
    for k, v in cache.items():
        if input_norm.startswith(normalize(k)[:100]) or normalize(k).startswith(input_norm[:100]):
            return v
            
    return None

def query_gemma_api(prompt, token):
    """Calls Hugging Face Inference API for dynamic reasoning."""
    from huggingface_hub import InferenceClient
    if not token:
        return "❌ Lỗi: Chưa cấu hình HF_TOKEN trong Streamlit Secrets."
        
    try:
        # Sử dụng mô hình fine-tuned chính
        client = InferenceClient(model=XAI_MODEL_REPO, token=token)
        response = client.text_generation(prompt, max_new_tokens=300, temperature=0.7)
        return response
    except Exception as e:
        error_msg = str(e)
        # Dự phòng mô hình Mistral-7B (rất ổn định với text-generation)
        try:
            client_fb = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=token)
            # Format prompt theo Mistral
            mistral_prompt = f"<s>[INST] {prompt} [/INST]"
            return client_fb.text_generation(mistral_prompt, max_new_tokens=300, temperature=0.7)
        except Exception as fallback_e:
            return f"❌ Lỗi API: {error_msg} (Dự phòng cũng lỗi: {str(fallback_e)})"

@st.cache_data(show_spinner=False)
def predict_cached(text: str, model_key: str) -> dict:
    import torch.nn.functional as F
    model, tokenizer, _ = load_model(model_key)
    if model is None: return None

    processed_text = text
    if "phobert" in model_key.lower():
        try:
            from underthesea import word_tokenize
            processed_text = word_tokenize(text, format="text")
        except Exception as e:
            print(f">>> [WARNING] Lỗi tách từ underthesea: {e}")
            
    enc = tokenizer(processed_text, truncation=True, max_length=256, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])

    # Lấy xác suất
    p_mis = F.softmax(logits_m, dim=1).cpu().numpy()[0]
    p_st  = F.softmax(logits_st, dim=1).cpu().numpy()[0]
    p_sen = F.softmax(logits_se, dim=1).cpu().numpy()[0]

    # Tra cứu giải thích (Ưu tiên cache -> Sau đó gọi Gemma API)
    xai_cache = load_xai_cache()
    reasoning = find_xai_reasoning(text, xai_cache)
    
    if not reasoning:
        hf_token = st.secrets.get("HF_TOKEN") or st.secrets.get("VaccineNLP_TOKEN")
        # Rút gọn văn bản nếu quá dài để tránh lỗi API
        short_text = text.strip()[:1000] + "..." if len(text.strip()) > 1000 else text.strip()
        # Prompt trung lập để Gemma giải thích được cả tin đúng và tin sai
        prompt = f"Hãy phân tích nội dung sau về vắc-xin và giải thích tại sao nó được phân loại như vậy: '{short_text}'"
        reasoning = query_gemma_api(prompt, hf_token)

    return {
        "misinfo":   {"pred": int(torch.argmax(logits_m, dim=1)), "conf": p_mis},
        "stance":    {"pred": int(torch.argmax(logits_st, dim=1)), "conf": p_st},
        "sentiment": {"pred": int(torch.argmax(logits_se, dim=1)), "conf": p_sen},
        "reasoning": reasoning
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
        footer_bg = "linear-gradient(135deg, #0a192f 0%, #112240 100%)"
        footer_text = "#a8b2d1"
        title_color = "#007bff"
        label_color = "#ccd6f6"
        school_name_color = "#fff"
        col_border = "rgba(0, 123, 255, 0.2)"
        bottom_text = "#8892b0"
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

    footer_html = f"""<div class="main-footer">
<div class="footer-container">
<!-- Cột 1: Logo & Trường -->
<div class="footer-col logo-col">
<img src="{logo_src}" class="footer-logo-img" alt="HUPH Logo">
<div class="school-name">TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</div>
<div style="font-size:0.9rem; opacity:0.8; margin-top:10px;">
<p>📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
<p>🌐 <a href="https://huph.edu.vn/" target="_blank" class="footer-link">huph.edu.vn</a></p>
</div>
</div>
<!-- Cột 2: Đề tài -->
<div class="footer-col">
<div class="footer-title">🔬 ĐỀ TÀI ĐỒ ÁN</div>
<div class="project-name-vi">Phát hiện Tin giả và Phân tích Thái độ về Vaccine tại Việt Nam</div>
<div class="project-name-en" style="margin-top:10px; font-size:0.9rem; opacity:0.8;">
(Vaccine Misinformation & Attitude Analysis in Vietnam)
</div>
</div>
<!-- Cột 3: Nhóm thực hiện -->
<div class="footer-col">
<div class="footer-title">👥 NHÓM THỰC HIỆN</div>
<div class="info-row">
<b>1. Kim Mạnh Hưng</b><br>
<span style="font-size:0.9rem; opacity:0.8;">
MSSV: 2211090016 | Lớp: CNCQ KHDL1-1A<br>
📧 <a href="mailto:2211090016@studenthuph.edu.vn" class="footer-link">2211090016@studenthuph.edu.vn</a>
</span>
</div>
<div class="info-row" style="margin-top:15px;">
<b>2. Đinh Lê Quỳnh Phương</b><br>
<span style="font-size:0.9rem; opacity:0.8;">
MSSV: 2211090031 | Lớp: CNCQ KHDL1-1A<br>
📧 <a href="mailto:2211090031@studenthuph.edu.vn" class="footer-link">2211090031@studenthuph.edu.vn</a>
</span>
</div>
</div>
<!-- Cột 4: Giảng viên hướng dẫn -->
<div class="footer-col">
<div class="footer-title">👨‍🏫 GIẢNG VIÊN HƯỚNG DẪN</div>
<div class="info-row">
<b style="font-size:1.1rem;">TS. Trần Lâm Quân</b><br>
<div style="margin-top:10px; font-size:0.9rem; opacity:0.8;">
Giảng viên Khoa học dữ liệu<br>
Trường Đại học Y tế Công Cộng<br>
📧 <a href="mailto:tlq@huph.edu.vn" class="footer-link">tlq@huph.edu.vn</a>
</div>
</div>
</div>
</div>
<div class="footer-bottom">
© 2026 VaccineNLP Project | Đồ án tốt nghiệp chuyên ngành Khoa học Dữ liệu - HUPH
</div>
</div>"""
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
            label_text_color = "#a0a5b0" if is_dark else "#000"
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
    chart_font_color = "#e2e4e9" if is_dark else "#000000"
    
    fig = go.Figure()
    tasks = ["Misinfo", "Stance", "Sentiment"]
    colors = ["#3db882", "#4a9eed", "#e8504a"]
    for i, task in enumerate(tasks):
        fig.add_trace(go.Bar(
            x=df["Model"], 
            y=df[task], 
            name=task, 
            marker_color=colors[i],
            text=df[task],
            texttemplate='%{text:.4f}',
            textposition='outside',
            cliponaxis=False
        ))
    
    fig.update_layout(
        barmode='group', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(family='Times New Roman', color=chart_font_color, size=14),
        legend=dict(font=dict(color=chart_font_color)),
        xaxis=dict(tickfont=dict(color=chart_font_color, size=12)),
        yaxis=dict(tickfont=dict(color=chart_font_color, size=12), range=[0, 1.0]),
        margin=dict(l=20, r=20, t=60, b=40),
        uniformtext_minsize=8, 
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    # ─────────────────────────────────────────────────────────────
    # PAGE CONFIGURATION
    # ─────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="VaccineNLP - Phân tích tin giả & Thái độ",
        page_icon="💉",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ─────────────────────────────────────────────────────────────
    # THEME STATE & TOGGLE
    # ─────────────────────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Hàm callback để cập nhật văn bản khi chọn mẫu
    def on_sample_change():
        # Chỉ xóa kết quả cũ khi người dùng chủ động chọn mẫu khác
        st.session_state.last_result = None

    # ─────────────────────────────────────────────────────────────
    # DYNAMIC COLORS BASED ON THEME
    # ─────────────────────────────────────────────────────────────
    is_dark = st.session_state.theme == "Dark"
    bg_color = "#0a192f" if is_dark else "#ffffff"
    text_color = "#ccd6f6" if is_dark else "#111111"
    secondary_text = "#8892b0" if is_dark else "#555"
    border_color = "rgba(0, 123, 255, 0.3)" if is_dark else "rgba(0,0,0,0.1)"
    sidebar_bg = "#0d1b3e" if is_dark else "#f8f9fa"
    input_bg = "#112240" if is_dark else "#ffffff"
    input_border = "#233554" if is_dark else "#ced4da"
    card_bg = "rgba(17, 34, 64, 0.7)" if is_dark else "#fdfdfd"

    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🔬 VaccineNLP</h2>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("##### 🎨 Giao diện")
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Tối", width="stretch", type="primary" if st.session_state.theme == "Dark" else "secondary"):
                st.session_state.theme = "Dark"
                st.rerun()
        with theme_col2:
            if st.button("☀️ Sáng", width="stretch", type="primary" if st.session_state.theme == "Light" else "secondary"):
                st.session_state.theme = "Light"
                st.rerun()
        
        st.divider()
        st.markdown("##### 📋 Mẫu thử nghiệm")
        selected_sample = st.radio(
            "Chọn mẫu:", 
            options=["Tự nhập"] + list(SAMPLE_TEXTS.keys()), 
            index=0, 
            key="sidebar_sample_selector",
            on_change=on_sample_change
        )
        st.divider()
        st.markdown("##### 🤖 Mô hình Phân loại")
        st.info("Mô hình này đảm nhiệm việc phân loại nhãn (Tin giả, Quan điểm, Cảm xúc).")
        model_selection = st.selectbox("Chọn model:", options=list(MODEL_CONFIGS.keys()), index=0)

        # PHẦN THÔNG TIN HỆ THỐNG MỚI THÊM
        st.markdown(f"""
        <div style="margin-top: 15px; padding: 12px; background: {input_bg}; border: 1px solid {border_color}; border-radius: 10px;">
            <div style="font-size: 13px; font-weight: bold; margin-bottom: 8px; color: {text_color};">Về hệ thống</div>
            <div style="font-size: 10.5px; line-height: 1.5; color: {text_color}; opacity: 0.85;">
                • <b>Classifier:</b> PhoBERT-v2<br>
                • <b>XAI Engine:</b> Gemma-4 4B (cached)<br>
                • <b>Tasks:</b> Misinfo · Stance · Sentiment<br>
                • <b>Benchmark:</b> 186 samples
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # DYNAMIC CSS BASED ON THEME
    # ─────────────────────────────────────────────────────────────

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
            background-color: {"#112240" if is_dark else "#e9ecef"} !important;
            border-radius: 10px 10px 0 0;
            color: {text_color} !important;
            border: 1px solid {border_color};
            padding: 0 25px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {"#1a2b4b" if is_dark else "#ffffff"} !important;
            color: {"#007bff" if is_dark else "#000000"} !important;
            font-weight: bold !important;
            border: {"1px solid #007bff" if is_dark else "1px solid #ced4da"} !important;
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
        
        /* Expander styling - Fix Header color */
        [data-testid="stExpander"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}
        [data-testid="stExpander"] summary {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            transition: all 0.2s ease !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background-color: {"rgba(100, 255, 218, 0.05)" if is_dark else "rgba(0,0,0,0.02)"} !important;
            color: #64ffda !important;
        }}
        /* Fix icon trong expander */
        [data-testid="stExpander"] summary svg {{
            fill: {text_color} !important;
        }}

        /* Fix Button colors (Phân tích, Reset, etc.) */
        .stButton > button {{ 
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }}
        .stButton > button:hover {{
            border-color: #64ffda !important;
            color: #64ffda !important;
            background-color: {"#172a45" if is_dark else "#f0f7ff"} !important;
            box-shadow: 0 4px 12px rgba(100, 255, 218, 0.2) !important;
        }}

        /* Fix Selectbox (Dropdown) colors - SIÊU ÉP BUỘC */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSidebar"] [data-testid="stSelectbox"] button,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: none !important; /* Xóa viền nội bộ */
        }}
        
        /* Chỉ giữ lại viền ngoài cùng của ô Selectbox */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {{
            border: 1px solid {input_border} !important;
            border-radius: 8px !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stSelectbox"] p,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] span {{
            color: {text_color} !important;
        }}
        
        /* Cực kỳ quan trọng: Fix phần danh sách thả xuống (Popover) - PHIÊN BẢN CUỐI CÙNG */
        div[data-baseweb="popover"], 
        div[role="listbox"], 
        ul[role="listbox"],
        div[role="option"],
        li[role="option"] {{
            background-color: {sidebar_bg} !important;
            color: {text_color} !important;
        }}
        
        /* Viền cho danh sách */
        div[data-baseweb="popover"] {{
            border: 1px solid {input_border} !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }}

        [data-baseweb="popover"] li:hover, 
        [role="option"]:hover,
        div[data-baseweb="popover"] div:hover {{
            background-color: {"rgba(100, 255, 218, 0.1)" if is_dark else "rgba(0,0,123,0.05)"} !important;
            color: #64ffda !important;
        }}

        /* Fix Radio Button styling (Mẫu thử nghiệm) */
        div[data-testid="stRadio"] {{
            background-color: transparent !important;
        }}
        div[data-testid="stRadio"] label {{
            color: {text_color} !important;
        }}
        /* Style các lựa chọn trong radio */
        div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            padding: 8px 15px !important;
            border-radius: 10px !important;
            margin-bottom: 5px !important;
            width: 100% !important;
        }}

        /* Fix Info/Warning/Success boxes */
        div[data-testid="stNotification"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}

        /* Fix Divider (st.divider) visibility */
        hr {{
            border-top: 1px solid {"rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.2)"} !important;
            margin: 1rem 0 !important;
        }}

        /* Fix Textarea Placeholder color */
        textarea::placeholder {{
            color: {"#666" if is_dark else "#888"} !important;
            opacity: 1 !important;
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
            background-color: {"#112240" if is_dark else "#f0f2f6"} !important;
            font-weight: bold !important;
        }}

        /* Footer Unified Styling */
        .main-footer {{
            background: {"linear-gradient(135deg, #0a192f 0%, #112240 100%)" if is_dark else "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)"};
            padding: 50px 20px;
            color: {"#ccc" if is_dark else "#444"};
            font-family: 'Times New Roman', Times, serif !important;
            border-top: 4px solid #007bff;
            box-shadow: 0 -10px 25px rgba(0, 123, 255, 0.1);
            margin-top: 60px;
        }}
        .footer-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            max-width: none !important;
            margin: 0 auto;
            gap: 20px;
            font-family: 'Times New Roman', Times, serif !important;
        }}
        .footer-col {{
            flex: 1;
            min-width: 300px;
            padding: 0 25px;
            border-right: 1px solid {"rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"};
        }}
        .footer-col:last-child {{ border-right: none; }}
        .logo-col {{
            text-align: center;
        }}
        .footer-logo-img {{
            width: 100px;
            margin-bottom: 15px;
            filter: drop-shadow(0 0 8px rgba(0,123,255,0.2));
        }}
        .footer-title {{
            color: {"#007bff" if is_dark else "#0056b3"};
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 1.2rem;
            text-transform: uppercase;
        }}
        .school-name {{
            font-weight: bold;
            color: {"#fff" if is_dark else "#000"};
            font-size: 1.1rem;
        }}
        .project-name-vi {{
            color: {"#ffd700" if is_dark else "#b8860b"};
            font-weight: bold;
            font-style: italic;
        }}
        .footer-bottom {{
            padding-top: 20px;
            margin-top: 30px;
            border-top: 1px solid rgba(0,0,0,0.05);
            font-size: 0.9rem;
            color: {"#777" if is_dark else "#666"};
            text-align: center;
        }}
        .footer-link {{
            color: #007bff !important;
            text-decoration: none;
        }}
    </style>
    """, unsafe_allow_html=True)

    model, tokenizer, checkpoint_loaded = load_model(model_selection)
    
    # Ở phiên bản mới, reasoning được lấy trực tiếp từ hàm predict_cached

    # Kiểm tra sẵn sàng: Với gemma_api thì model sẽ None nhưng vẫn OK
    is_api = MODEL_CONFIGS[model_selection]["type"] == "gemma4"
    if model is None and not is_api:
        st.warning(f"⚠️ Mô hình `{model_selection}` chưa sẵn sàng. Vui lòng chọn mô hình khác.")
    
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
        <h1 style="color: #FFD700; font-family: 'Times New Roman', Times, serif; font-weight: bold; font-size: 2.2rem; margin-bottom: 0.8rem; line-height: 1.3; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔬 PHÁT HIỆN TIN GIẢ VÀ PHÂN TÍCH THÁI ĐỘ VỀ VACCINE TẠI VIỆT NAM 💉</h1>
        <p style="color: {banner_p_color}; font-family: 'Times New Roman', Times, serif; font-style: italic; font-size: 1.1rem; opacity: {banner_p_opacity}; line-height: 1.4;">(Vaccine Misinformation & Attitude Analysis in Vietnam)</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích Real-time", "📊 Thống kê Benchmark"])

    with tabs[0]:
        input_text = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area(
            "Nhập văn bản cần phân tích:", 
            value=input_text, 
            height=140, 
            placeholder="Dán nội dung bài viết về vắc-xin..."
        )
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            analyze_btn = st.button("🔍 Phân tích", width="stretch")
        with col_btn2:
            if st.button("🗑️ Reset", width="stretch"):
                st.session_state.last_result = None
                st.rerun()

        if analyze_btn and user_text.strip():
            with st.spinner(f"🧠 {model_selection} đang xử lý..."):
                # Thực hiện dự đoán và lấy cả nhãn lẫn lời giải thích
                result = predict_cached(user_text.strip(), model_selection)
                reasoning = result.get("reasoning") if result else None
                
                # Lưu vào session state
                st.session_state.last_result = {
                    "text": user_text.strip(),
                    "result": result,
                    "reasoning": reasoning
                }

        # Hiển thị kết quả: Chỉ cần có kết quả trong bộ nhớ và văn bản hiện tại KHÔNG rỗng
        if st.session_state.last_result and user_text.strip():
            # Nếu văn bản trong ô nhập liệu khớp với kết quả đã lưu, hiển thị nó
            if st.session_state.last_result["text"] == user_text.strip():
                saved = st.session_state.last_result
                result = saved["result"]
                reasoning = saved["reasoning"]

                st.markdown("---")
                if result:
                    col1, col2, col3 = st.columns(3)
                    with col1: render_result_card("Tin giả", "misinfo", result["misinfo"])
                    with col2: render_result_card("Quan điểm", "stance", result["stance"])
                    with col3: render_result_card("Cảm xúc", "sentiment", result["sentiment"])
                else:
                    st.error("❌ Không thể phân tích văn bản này. Vui lòng kiểm tra lại mô hình đã chọn.")

                if reasoning:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### 🧠 Hệ thống Giải thích (XAI Engine)")
                    with st.expander("📖 Xem giải thích chi tiết từ Gemma-4 XAI Engine", expanded=True):
                        st.markdown(f"<div style='border-left: 3px solid #64ffda; padding-left: 20px; color: {text_color}; opacity: 0.9;'>{reasoning}</div>", unsafe_allow_html=True)
                        st.caption("💡 Giải thích được tạo tự động bởi mô hình Gemma-4 Reasoning Engine.")
                else:
                    st.info("💡 Lý luận XAI không khả dụng cho văn bản này. Hãy chọn mẫu từ thanh bên.")
        elif analyze_btn:
            st.warning("⚠️ Vui lòng nhập văn bản.")

    with tabs[1]:
        render_benchmark_tab()

    hien_thi_footer_chung(is_dark=is_dark)

if __name__ == "__main__":
    main()
2px 4px rgba(0,0,0,0.3);">🔬 PHÁT HIỆN TIN GIẢ VÀ PHÂN TÍCH THÁI ĐỘ VỀ VACCINE TẠI VIỆT NAM 💉</h1>
        <p style="color: {banner_p_color}; font-family: 'Times New Roman', Times, serif; font-style: italic; font-size: 1.1rem; opacity: {banner_p_opacity}; line-height: 1.4;">(Vaccine Misinformation & Attitude Analysis in Vietnam)</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Phân tích Real-time", "📊 Thống kê Benchmark"])

    with tabs[0]:
        input_text = SAMPLE_TEXTS[selected_sample] if selected_sample != "Tự nhập" else ""
        user_text = st.text_area(
            "Nhập văn bản cần phân tích:", 
            value=input_text, 
            height=140, 
            placeholder="Dán nội dung bài viết về vắc-xin..."
        )
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            analyze_btn = st.button("🔍 Phân tích", width="stretch")
        with col_btn2:
            if st.button("🗑️ Reset", width="stretch"):
                st.session_state.last_result = None
                st.rerun()

        if analyze_btn and user_text.strip():
            with st.spinner(f"🧠 {model_selection} đang xử lý..."):
                # Thực hiện dự đoán và lấy cả nhãn lẫn lời giải thích
                result = predict_cached(user_text.strip(), model_selection)
                reasoning = result.get("reasoning") if result else None
                
                # Lưu vào session state
                st.session_state.last_result = {
                    "text": user_text.strip(),
                    "result": result,
                    "reasoning": reasoning
                }

        # Hiển thị kết quả: Chỉ cần có kết quả trong bộ nhớ và văn bản hiện tại KHÔNG rỗng
        if st.session_state.last_result and user_text.strip():
            # Nếu văn bản trong ô nhập liệu khớp với kết quả đã lưu, hiển thị nó
            if st.session_state.last_result["text"] == user_text.strip():
                saved = st.session_state.last_result
                result = saved["result"]
                reasoning = saved["reasoning"]

                st.markdown("---")
                if result:
                    col1, col2, col3 = st.columns(3)
                    with col1: render_result_card("Tin giả", "misinfo", result["misinfo"])
                    with col2: render_result_card("Quan điểm", "stance", result["stance"])
                    with col3: render_result_card("Cảm xúc", "sentiment", result["sentiment"])
                else:
                    st.error("❌ Không thể phân tích văn bản này. Vui lòng kiểm tra lại mô hình đã chọn.")

                if reasoning:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### 🧠 Hệ thống Giải thích (XAI Engine)")
                    with st.expander("📖 Xem giải thích chi tiết từ Gemma-4 XAI Engine", expanded=True):
                        st.markdown(f"<div style='border-left: 3px solid #64ffda; padding-left: 20px; color: {text_color}; opacity: 0.9;'>{reasoning}</div>", unsafe_allow_html=True)
                        st.caption("💡 Giải thích được tạo tự động bởi mô hình Gemma-4 Reasoning Engine.")
                else:
                    st.info("💡 Lý luận XAI không khả dụng cho văn bản này. Hãy chọn mẫu từ thanh bên.")
        elif analyze_btn:
            st.warning("⚠️ Vui lòng nhập văn bản.")

    with tabs[1]:
        render_benchmark_tab()

    hien_thi_footer_chung(is_dark=is_dark)

if __name__ == "__main__":
    main()
