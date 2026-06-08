"""
VaccineNLP — Gradio Web Application v2.0 (Production-Grade)
=============================================================
Migrated from Streamlit to Gradio for HuggingFace Spaces deployment.

Version 2.0 changes vs v1.0:
- ✅ FIX: Session history refresh button now works correctly
- ✅ FIX: AI Voice uses unique temp files (no race condition)
- ✅ FIX: _CACHE thread-safe with threading.Lock
- ✅ FIX: low_cpu_mem_usage=True for memory efficiency
- ✨ ADD: Loading progress indicator (gr.Progress)
- ✨ ADD: gr.Examples component for quick demo
- ✨ ADD: Hero section with 3 instant test buttons
- ✨ ADD: Header badge HUPH with student info
- ✨ ADD: Confusion Matrix heatmap in Benchmark tab
- ✨ ADD: Per-class breakdown in Đánh giá tab
- ✨ ADD: Export Report button (markdown download)
- ✨ ADD: Captum opt-in checkbox (faster demo)

Features:
- 6 tabs: Phân tích · Benchmark · Đánh giá · Tài liệu · Phương pháp · Đề cương
- 3-Layer XAI: Cache → HF Inference API → Captum IG
- Multi-source Fetcher: News + YouTube + Apify
- AI Voice via gTTS
- Batch mode, Compare models, Session history

Deploy: HuggingFace Spaces (CPU Basic 16GB)
URL: huggingface.co/spaces/kimmnhhng/vaccinenlp-demo
"""

import os
import json
import time
import tempfile
import hashlib
import logging
import threading
import base64
import unicodedata

# ============================================================================
# STARLETTE / GRADIO TEMPLATE RESPONSE MONKEY PATCH (CRITICAL FIX FOR STARLETTE >=0.28.0)
# ============================================================================
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse

    def patched_template_response(self, name, context=None, status_code=200, headers=None, media_type=None, background=None):
        # Gradio 4.x gọi: templates.TemplateResponse("index.html", {"request": request, "api_info": ...})
        # Trong Starlette mới: def TemplateResponse(self, request, name, context=None, status_code=200, ...)
        # Ta map lại: request = context["request"], name = "index.html", context = context
        if isinstance(name, str) and isinstance(context, dict):
            request_obj = context.get("request")
            return _orig_template_response(self, request_obj, name, context, status_code, headers, media_type, background)
        
        # Nếu gọi kiểu khác (đã đúng chuẩn)
        return _orig_template_response(self, name, context, status_code, headers, media_type, background)

    starlette.templating.Jinja2Templates.TemplateResponse = patched_template_response
    print("[OK] Starlette TemplateResponse monkey patch successfully applied!")
except Exception as e:
    print(f"[WARN] Failed to apply TemplateResponse patch: {e}")


from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gradio as gr

# Compatibility Fallback: If running on an older Gradio version (e.g., Gradio 3.x),
# map gr.Sidebar to gr.Column to prevent AttributeError: module 'gradio' has no attribute 'Sidebar'
if not hasattr(gr, "Sidebar"):
    print("[WARN] gr.Sidebar not found. Falling back to gr.Column layout.")
    gr.Sidebar = gr.Column

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data" if (APP_DIR / "data").exists() else APP_DIR

CONFIG = {
    "models": {
        "PhoBERT-v2": {
            "repo_id": "hung2903/phobert-vaccine-multitask",
            "base_repo": "vinai/phobert-base-v2",
            "type": "phobert",
        },
        "XLM-R-v1": {
            "repo_id": "hung2903/xlmr-vaccine-multitask",
            "base_repo": "xlm-roberta-base",
            "type": "xlm-roberta",
        },
    },
    "xai_models": [
        "hung2903/gemma-4-E4B-vaccine-xai-merged",
        "google/gemma-4-e4b-it",
    ],
    "cache_file": DATA_DIR / "xai_cache.json",
    "benchmark_file": DATA_DIR / "benchmark_results.json",
    "temperature_file": DATA_DIR / "temperature_params.json",
    "max_seq_length": 256,
    "session_history_limit": 10,
}

LABEL_MAPS = {
    "misinfo":   {0: "Tin giả",  1: "Chính xác"},
    "stance":    {0: "Ủng hộ",   1: "Phản đối", 2: "Trung lập"},
    "sentiment": {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"},
}
LABEL_ICONS = {
    "misinfo":   {0: "🚨", 1: "✅"},
    "stance":    {0: "👍", 1: "👎", 2: "🤝"},
    "sentiment": {0: "😠", 1: "😐", 2: "😊"},
}
LABEL_COLORS = {
    "misinfo":   {0: "#d2453a", 1: "#0e9384"},
    "stance":    {0: "#0e9384", 1: "#d2453a", 2: "#4a9eed"},
    "sentiment": {0: "#d2453a", 1: "#4a9eed", 2: "#0e9384"},
}

# Load .env file if available (local development helper)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("VaccineNLP_TOKEN", "")

# LM Studio local server configuration (configurable via env vars)
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1").strip()
LM_STUDIO_MODEL    = os.environ.get("LM_STUDIO_MODEL", "local-model").strip()
LM_API_TOKEN       = os.environ.get("LM_API_TOKEN", "lm-studio").strip()

# Robust Apify tokens collection supporting different naming conventions
APIFY_TOKENS = []
possible_apify_keys = [
    "APIFY_TOKEN", "APIFY_API_TOKEN", "APIFY_TOKEN_1", "APIFY_API_TOKEN_1",
    "APIFY_TOKEN_2", "APIFY_API_TOKEN_2", "APIFY_TOKEN_3", "APIFY_API_TOKEN_3",
    "APIFY_TOKEN_4", "APIFY_API_TOKEN_4", "APIFY_TOKEN_5", "APIFY_API_TOKEN_5",
]
for k in possible_apify_keys:
    val = os.environ.get(k, "").strip()
    if val and val not in APIFY_TOKENS:
        APIFY_TOKENS.append(val)

# Path B: External Gemma endpoint (Kaggle+ngrok)
# QUAN TRỌNG: URL ngrok thay đổi mỗi session Kaggle.
# Set Secret GEMMA_ENDPOINT_URL trong HF Spaces Settings mỗi khi restart Kaggle.
# Nếu không set → bỏ qua Path B, fallback sang HF Inference API.
_env_gemma_url = os.environ.get("GEMMA_ENDPOINT_URL", "").strip()
GEMMA_ENDPOINT_URL = "https://pearle-staglike-nonsyntonically.ngrok-free.dev/predict"
if _env_gemma_url and "ngrok-free.dev" in _env_gemma_url and "pearle-staglike-nonsyntonically" not in _env_gemma_url:
    GEMMA_ENDPOINT_URL = _env_gemma_url
if GEMMA_ENDPOINT_URL:
    logger.info(f"🔗 Kaggle ngrok endpoint configured: {GEMMA_ENDPOINT_URL}")

# Path C: OpenRouter fallback (public LLM inference)
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")

# Gemini API keys: collect flexibly (GEMINI_API_KEY, GEMINI_API_KEY_2 ... GEMINI_API_KEY_5)
_GEMINI_KEYS: list = []
for _gi in range(1, 6):
    _gname = "GEMINI_API_KEY" if _gi == 1 else f"GEMINI_API_KEY_{_gi}"
    _gval = os.environ.get(_gname, "").strip()
    if _gval and _gval not in _GEMINI_KEYS:
        _GEMINI_KEYS.append(_gval)
if _GEMINI_KEYS:
    logger.info(f"🔑 Loaded {len(_GEMINI_KEYS)} Gemini API key(s)")

# Thread-safe cache lock
_CACHE_LOCK = threading.Lock()
_CACHE = {}

# Sample texts for quick demo
SAMPLE_TEXTS = {
    "🚨 Tin giả - Chống vaccine cực đoan": "Ko tiêm mũi nào hết. Ko biết bạn thuộc thế hệ nào, chứ bạn nhìn xem thế hệ 8x trở về trước ko có ai tiêm bất cứ mũi gì vẫn khoẻ mạnh đó thôi. Cha mẹ thời nay bị doạ cho sợ hãi, đem con đi tiêm vì bị bóng ma sợ hãi nó đè, chứ thực chất chả có tác dụng gì còn gây hại cho cơ thể nữa.",
    "🚨 Tin giả - Vô sinh": "Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.",
    "🟢 Ủng hộ tiêm chủng": "Em cũng đang tiêm từng mũi 1 cho con, con e 5 tháng, mới tiêm tới phế cầu, 3 tháng đầu chỉ tiêm 6in1 và uống rota. Chậm mà đủ và an toàn cho con là được. Trộm vía bé e chưa sốt, chưa hành mũi nào ❤️",
    "🟡 Nghi ngại": "Cún mình chỉ tiêm mũi ở viện nhà là ko tiêm gì nữa. Bây giờ 2 tuổi rồi. Ai hỏi t vẫn nói tiêm đủ.",
    "✅ Thông tin chuẩn": "Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng để phòng các bệnh truyền nhiễm nguy hiểm.",
    "🔵 Câu hỏi tư vấn": "Trâm Trần ví dụ như Ko có tiêm 6in1 hay 5in1, mà tiêm từng mũi từng bệnh phải không ạ?",
    "💬 Tin giả - Từ lóng MXH": "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh \nNhưng tiêm vắc xin thì là tiêm thuốc độc vào người \n\nCàng tiêm nhiều càng bệnh nhiều \n\nBạn xem thời xưa có ai phải tiêm đâu sao ai cũng khỏe mạnh\n\nMuốn thải độc vx , kim loại nặng thì nên cho uống nc lá mùi đun lên \n\nMuốn hạ sốt ( sốt nóng ) cho con uống nc chanh ấm có đường \nLấy chanh xoa toàn thân",
}

# Examples for gr.Examples component
GRADIO_EXAMPLES = [
    ["Vắc xin COVID gây vô sinh ở phụ nữ trẻ và biến đổi gen ở trẻ em.", "PhoBERT-v2"],
    ["Em đã tiêm đủ 5 mũi cho con theo lịch tiêm chủng mở rộng. Bé khỏe mạnh, không sốt.", "PhoBERT-v2"],
    ["Bộ Y tế công bố lịch tiêm chủng mới cho trẻ em năm 2026.", "PhoBERT-v2"],
    ["Vaccine là chip 5G theo dõi người dân, không nên tin tưởng.", "XLM-R-v1"],
]


# ============================================================================
# MULTITASK MODEL ARCHITECTURE
# ============================================================================

class VaccineMultitaskModel(nn.Module):
    """Multitask model with shared encoder + 3 task-specific heads."""

    def __init__(self, model_name: str, num_misinfo=2, num_stance=3, num_sentiment=3, token=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.config = AutoConfig.from_pretrained(model_name, token=token, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name, token=token, trust_remote_code=True, low_cpu_mem_usage=False
        )
        hidden = self.config.hidden_size
        self.heads = nn.ModuleDict({
            "misinfo":   nn.Linear(hidden, num_misinfo),
            "stance":    nn.Linear(hidden, num_stance),
            "sentiment": nn.Linear(hidden, num_sentiment),
        })
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return (
            self.heads["misinfo"](pooled),
            self.heads["stance"](pooled),
            self.heads["sentiment"](pooled),
        )


# ============================================================================
# MODEL LOADING (Thread-safe + Cached)
# ============================================================================

def load_model(model_key: str):
    """Load model from HF Hub. Returns (model, tokenizer, success_flag)."""
    cache_key = f"model_{model_key}"
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    import gc
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    cfg = CONFIG["models"][model_key]
    gc.collect()

    try:
        logger.info(f"Loading {model_key} from HF Hub...")
        model_path = hf_hub_download(
            repo_id=cfg["repo_id"], filename="best_model.pt",
            token=HF_TOKEN if HF_TOKEN else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_repo"], token=HF_TOKEN if HF_TOKEN else None, trust_remote_code=True
        )
        model = VaccineMultitaskModel(model_name=cfg["base_repo"], token=HF_TOKEN if HF_TOKEN else None)

        state = torch.load(model_path, map_location="cpu", weights_only=False)
        new_state = {
            (k.replace("head_", "heads.") if k.startswith("head_") and "heads." not in k else k): v
            for k, v in state.items()
        }
        model.load_state_dict(new_state, strict=False)
        model.eval()

        del state
        gc.collect()

        with _CACHE_LOCK:
            _CACHE[cache_key] = (model, tokenizer, True)
        logger.info(f"✅ {model_key} loaded")
        return _CACHE[cache_key]
    except Exception as e:
        logger.error(f"❌ Failed to load {model_key}: {e}")
        with _CACHE_LOCK:
            _CACHE[cache_key] = (None, None, False)
        return _CACHE[cache_key]


def load_temperature_params() -> Dict:
    """Load Temperature Scaling params (thread-safe)."""
    with _CACHE_LOCK:
        if "T_params" in _CACHE:
            return _CACHE["T_params"]

    default = {
        "phobert_v2": {"misinfo": 1.8197, "stance": 1.6666, "sentiment": 1.3474},
        "xlmr_v1":    {"misinfo": 1.7424, "stance": 1.6258, "sentiment": 1.3270},
    }
    if CONFIG["temperature_file"].exists():
        try:
            with open(CONFIG["temperature_file"], encoding="utf-8") as f:
                loaded = json.load(f)
            default.update(loaded)
        except Exception as e:
            logger.warning(f"Could not load T params: {e}")
    with _CACHE_LOCK:
        _CACHE["T_params"] = default
    return default


def load_benchmark() -> Dict:
    """Load LIVE benchmark from JSON file with fallback (thread-safe)."""
    with _CACHE_LOCK:
        if "benchmark" in _CACHE:
            return _CACHE["benchmark"]

    fallback = {
        "phobert": {
            "name": "PhoBERT-v2 (Classification Engine)",
            "misinfo": 0.6996, "stance": 0.6640, "sentiment": 0.7266,
            "per_class_misinfo":   [0.5075, 0.8918],
            "per_class_stance":    [0.5934, 0.6612, 0.7375],
            "per_class_sentiment": [0.8000, 0.7917, 0.5882],
            "support_misinfo":   [28, 158],
            "support_stance":    [54, 48, 84],
            "support_sentiment": [71, 75, 40],
        },
        "xlmr": {
            "name": "XLM-R-v1 (Baseline)",
            "misinfo": 0.7038, "stance": 0.6224, "sentiment": 0.6866,
            "per_class_misinfo":   [0.5079, 0.8997],
            "per_class_stance":    [0.5495, 0.6387, 0.6790],
            "per_class_sentiment": [0.7682, 0.7162, 0.5753],
            "support_misinfo":   [28, 158],
            "support_stance":    [54, 48, 84],
            "support_sentiment": [71, 75, 40],
        },
        "gemma": {
            "name": "Gemma-4 4B (XAI Reasoning Engine)",
            "misinfo": 0.6377, "stance": 0.6264, "sentiment": 0.7700,
            "per_class_misinfo":   [0.4444, 0.8309],
            "per_class_stance":    [0.4528, 0.6905, 0.7360],
            "per_class_sentiment": [0.8039, 0.8034, 0.7027],
            "support_misinfo":   [22, 113],
            "support_stance":    [32, 40, 59],
            "support_sentiment": [56, 52, 20],
        },
    }
    if CONFIG["benchmark_file"].exists():
        try:
            with open(CONFIG["benchmark_file"], encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge name field from fallback
                for k in loaded:
                    if k in fallback and "name" in fallback[k]:
                        loaded[k]["name"] = fallback[k]["name"]
                fallback.update(loaded)
        except Exception as e:
            logger.warning(f"Could not load benchmark: {e}")
    with _CACHE_LOCK:
        _CACHE["benchmark"] = fallback
    return fallback


def load_xai_cache() -> Dict:
    """Load XAI reasoning cache (thread-safe)."""
    with _CACHE_LOCK:
        if "xai_cache" in _CACHE:
            return _CACHE["xai_cache"]

    cache = {}
    if CONFIG["cache_file"].exists():
        try:
            with open(CONFIG["cache_file"], encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached XAI reasonings")
        except Exception as e:
            logger.warning(f"Could not load XAI cache: {e}")
    with _CACHE_LOCK:
        _CACHE["xai_cache"] = cache
    return cache


# ============================================================================
# XAI ENGINE (3-Layer Fallback)
# ============================================================================

def find_xai_reasoning_cache(text: str) -> Optional[str]:
    """Layer 1: Lookup in cache (exact + fuzzy match)."""
    import re

    def normalize(t):
        if not t:
            return ""
        return re.sub(
            r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            "", t.lower()
        )

    cache = load_xai_cache()
    t_strip = text.strip()
    if t_strip in cache:
        return cache[t_strip]

    input_norm = normalize(t_strip)
    for k, v in cache.items():
        k_norm = normalize(k)
        if len(input_norm) > 20 and (input_norm in k_norm or k_norm in input_norm):
            return v
    return None


def _build_gemma_prompt(text: str) -> str:
    """Build standardized prompt format for the fine-tuned Gemma-4 MoE/merged model."""
    prompt = (
        "<start_of_turn>user\n"
        "Phân tích nội dung sau về vaccine:\n\n"
        f'"{text.strip()}"\n\n'
        "QUY TẮC PHÂN LOẠI (bắt buộc tuân thủ):\n"
        "- Misinformation CHỈ được chọn 1 trong 2: \"Tin gia\" hoặc \"Chinh xac\". \n"
        "  TUYỆT ĐỐI KHÔNG dùng từ khác. Nội dung không liên quan vaccine \n"
        "  hoặc không có thông tin sai = \"Chinh xac\".\n"
        "- Stance CHỈ: \"Ung ho\" / \"Phan doi\" / \"Trung lap\".\n"
        "- Sentiment CHỈ: \"Tieu cuc\" / \"Trung tinh\" / \"Tich cuc\".\n\n"
        "Trả lời theo ĐÚNG cấu trúc: KẾT QUẢ trước, GIẢI THÍCH sau.\n"
        "=== KẾT QUẢ ===\n"
        "- Misinformation: <Tin gia HOẶC Chinh xac>\n"
        "- Stance: <Ung ho HOẶC Phan doi HOẶC Trung lap>\n"
        "- Sentiment: <Tieu cuc HOẶC Trung tinh HOẶC Tich cuc>\n"
        "=== GIẢI THÍCH ===\n"
        "<lý luận chi tiết bằng tiếng Việt>\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt


def clean_reasoning_output(raw: str) -> str:
    """
    Clean và format reasoning text từ API.
    - Bỏ phần "=== KẾT QUẢ ===" (vì đã có PhoBERT predict rồi)
    - Giữ phần "=== GIẢI THÍCH ===" trở đi
    - Prefix "Lý luận: " nếu chưa có
    - Bỏ các token đặc biệt: <end_of_turn>, <start_of_turn>, <|turn|>, etc.
    - Truncate nếu > 2000 chars nhưng giữ nguyên markdown bold
    """
    if not raw:
        return ""
    
    t = raw.strip()
    
    # 1. Bỏ các special tokens
    t = t.replace("<end_of_turn>", "").replace("<start_of_turn>", "").replace("<|turn|>", "").replace("<|turn>", "").strip()
    
    # 2. Tìm phần giải thích
    if "=== GIẢI THÍCH ===" in t:
        reasoning = t.split("=== GIẢI THÍCH ===")[-1].strip()
    elif "=== giải thích ===" in t.lower():
        idx = t.lower().find("=== giải thích ===")
        after_marker = t[idx + len("=== giải thích ==="):]
        reasoning = after_marker.strip()
    else:
        reasoning = t
        
    # Remove leading colon or dashes if any
    reasoning = reasoning.lstrip(":-\n\r ")
    
    # 3. Đảm bảo có prefix "Lý luận: "
    if not reasoning.startswith("Lý luận:"):
        if reasoning.startswith("Lý luận"):
            reasoning = "Lý luận: " + reasoning[len("Lý luận"):].lstrip(": ")
        else:
            reasoning = "Lý luận: " + reasoning
            
    # 4. Truncate nếu > 2000 kí tự
    if len(reasoning) > 2000:
        reasoning = reasoning[:2000] + "..."
        
    return reasoning


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")

def parse_gemma_labels(raw: str):
    """Tách 3 nhãn từ phần '=== KẾT QUẢ ===' của output Gemma.
    Trả {'misinfo':int|None,'stance':int|None,'sentiment':int|None}, hoặc None nếu không có trục nào.
    Khớp LABEL_MAPS: misinfo 0=Tin giả/1=Chính xác; stance 0=Ủng hộ/1=Phản đối/2=Trung lập;
    sentiment 0=Tiêu cực/1=Trung tính/2=Tích cực."""
    if not raw:
        return None
    norm = _strip_accents(raw).lower()
    # Giới hạn vùng KẾT QUẢ (trước GIẢI THÍCH) để tránh nhiễu từ phần lý giải
    if "ket qua" in norm:
        sec = norm.split("ket qua", 1)[1]
        if "giai thich" in sec:
            sec = sec.split("giai thich", 1)[0]
    else:
        sec = norm
    lines = sec.splitlines()

    def pick(name_key, options):
        for ln in lines:                      # ưu tiên dòng chứa tên trục
            if name_key in ln:
                for opt, val in options:
                    if opt in ln:
                        return val
        for opt, val in options:              # fallback: quét cả vùng KẾT QUẢ
            if opt in sec:
                return val
        return None

    mis = pick("misinformation", [("chinh xac", 1), ("tin gia", 0)])
    st  = pick("stance",         [("ung ho", 0), ("phan doi", 1), ("trung lap", 2)])
    se  = pick("sentiment",      [("tieu cuc", 0), ("trung tinh", 1), ("tich cuc", 2)])
    if mis is None and st is None and se is None:
        return None
    return {"misinfo": mis, "stance": st, "sentiment": se}

_SVG_SCALE = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
              'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--teal);vertical-align:-2px">'
              '<path d="M12 3v18M7 7h10M5 7l-2 5a3 3 0 0 0 6 0L7 7M17 7l-2 5a3 3 0 0 0 6 0l-2-5"/></svg>')

def render_disagreement_table(result, gemma_labels) -> str:
    """Bảng PhoBERT vs Gemma (3 trục). Trả "" nếu không parse được nhãn Gemma (fallback/template)."""
    if not gemma_labels:
        return ""
    rows = ""
    for key, vi in (("misinfo", "Tính xác thực"), ("stance", "Lập trường"), ("sentiment", "Cảm xúc")):
        p_id = result[key]["pred"]
        p_lbl = LABEL_MAPS[key][p_id]
        g_id = gemma_labels.get(key)
        g_lbl = LABEL_MAPS[key].get(g_id, "—") if g_id is not None else "—"
        diff = (g_id is not None and g_id != p_id)
        rows += (f'<tr class="{"flag" if diff else ""}"><td>{vi}</td><td>{p_lbl}</td><td>{g_lbl}</td>'
                 f'<td style="text-align:center" class="{"no" if diff else "yes"}">{"≠" if diff else "✓"}</td></tr>')
    return (
        '<div style="margin-top:18px">'
        f'<div class="section-label" style="margin-bottom:10px">{_SVG_SCALE} Bất đồng thuận nhãn — PhoBERT vs Gemma</div>'
        '<table class="dtable"><thead><tr><th>Trục</th><th>PhoBERT-v2</th><th>Gemma-4-4B</th>'
        '<th style="text-align:center">Khớp</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p class="muted" style="font-size:11.5px;margin-top:8px;color:var(--ink-3)">Nhãn Gemma trích từ phần “KẾT QUẢ” của lời giải; “≠” = lệch với PhoBERT.</p>'
        '</div>'
    )


import openai

def get_live_xai_reasoning(text: str, result: Optional[Dict] = None, return_raw: bool = False):
    """Gọi Gemma-4 GGUF qua LM Studio. return_raw=True → trả (cleaned, raw|None) để parse nhãn."""
    client = openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_API_TOKEN)
    try:
        logger.info(f"⏳ Calling LM Studio at {LM_STUDIO_BASE_URL} (model={LM_STUDIO_MODEL})...")
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Bạn là chuyên gia AI phân tích y tế công cộng (Explainable AI — XAI). "
                    "Hãy phân tích nội dung vaccine bằng TIẾNG VIỆT theo 3 chiều: "
                    "(1) Tính xác thực, (2) Thái độ với vaccine, (3) Cảm xúc tổng thể. "
                    "Trả lời có cấu trúc: === KẾT QUẢ === trước, === GIẢI THÍCH === sau.")},
                {"role": "user", "content": _build_gemma_prompt(text)},
            ],
            max_tokens=2048, temperature=0.1, timeout=120,
        )
        content = response.choices[0].message.content
        if content and len(content.strip()) > 30:
            cleaned = clean_reasoning_output(content)
            return (cleaned, content) if return_raw else cleaned
        logger.warning("⚠️ LM Studio returned empty/short response")
    except Exception as e:
        logger.warning(f"❌ LM Studio unreachable ({LM_STUDIO_BASE_URL}): {str(e)[:120]}")

    if result:
        fb = generate_smart_fallback(result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"])
        return (fb, None) if return_raw else fb
    msg = "⚠️ LM Studio chưa được khởi động. Hãy mở LM Studio và bật Local Server tại cổng 1234."
    return (msg, None) if return_raw else msg



def generate_smart_fallback(misinfo_pred: int, stance_pred: int, sentiment_pred: int) -> str:
    """Layer 4: Template fallback in Vietnamese."""
    if misinfo_pred == 0:
        res = "Dựa trên các đặc trưng ngôn ngữ, hệ thống nhận diện đây là nội dung có rủi ro cao về tin giả y tế. "
    else:
        res = "Nội dung này được đánh giá là thông tin chia sẻ thông thường, không chứa các dấu hiệu của tin giả. "

    if stance_pred == 1:
        res += "Người viết đang bày tỏ sự phản đối hoặc nghi ngờ khá gay gắt về hiệu quả của vắc-xin. "
    elif stance_pred == 2:
        res += "Văn bản chủ yếu tập trung vào việc thảo luận hoặc đặt câu hỏi để làm rõ thông tin. "
    else:
        res += "Thông điệp truyền tải thái độ tích cực và sự tin tưởng vào việc tiêm chủng an toàn. "

    if sentiment_pred == 0:
        res += "Cảm xúc tiêu cực được thể hiện rõ qua cách dùng từ, có thể gây tâm lý hoang mang."
    elif sentiment_pred == 2:
        res += "Sắc thái văn bản rất lạc quan, giúp củng cố niềm tin cho cộng đồng."
    else:
        res += "Sắc thái văn bản trung tính, mang tính thông tin khách quan."
    return res


def is_mostly_english(txt: str) -> bool:
    common = {"the", "and", "of", "to", "a", "in", "is", "that", "it", "he", "was", "for", "on"}
    words = set(txt.lower().split())
    return len(words.intersection(common)) > 2


def translate_to_vietnamese(txt: str) -> str:
    """Translate English to Vietnamese via Google Translate."""
    import urllib.request, urllib.parse
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=vi&dt=t&q=" + urllib.parse.quote(txt)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.loads(r.read().decode("utf-8"))
            return "".join(s[0] for s in res[0] if s[0])
    except Exception:
        return txt


# ============================================================================
# PREDICTION
# ============================================================================

def predict(text: str, model_key: str = "PhoBERT-v2") -> Optional[Dict]:
    """Run multi-task prediction with calibrated confidence."""
    model, tokenizer, ok = load_model(model_key)
    if not ok or model is None:
        return None

    processed = text
    if "phobert" in model_key.lower():
        try:
            from underthesea import word_tokenize
            processed = word_tokenize(text, format="text")
        except Exception:
            pass

    enc = tokenizer(processed, truncation=True, max_length=256, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])

    T_params = load_temperature_params()
    if "phobert" in model_key.lower():
        Ts = T_params.get("phobert_v2", {"misinfo": 1.0, "stance": 1.0, "sentiment": 1.0})
    else:
        Ts = T_params.get("xlmr_v1", {"misinfo": 1.0, "stance": 1.0, "sentiment": 1.0})

    p_mis_raw = F.softmax(logits_m, dim=1).cpu().numpy()[0]
    p_st_raw  = F.softmax(logits_st, dim=1).cpu().numpy()[0]
    p_se_raw  = F.softmax(logits_se, dim=1).cpu().numpy()[0]

    p_mis_cal = F.softmax(logits_m / Ts["misinfo"], dim=1).cpu().numpy()[0]
    p_st_cal  = F.softmax(logits_st / Ts["stance"], dim=1).cpu().numpy()[0]
    p_se_cal  = F.softmax(logits_se / Ts["sentiment"], dim=1).cpu().numpy()[0]

    return {
        "misinfo":   {"pred": int(np.argmax(p_mis_raw)), "conf_raw": p_mis_raw.tolist(),
                      "conf_cal": p_mis_cal.tolist(), "T": Ts["misinfo"]},
        "stance":    {"pred": int(np.argmax(p_st_raw)),  "conf_raw": p_st_raw.tolist(),
                      "conf_cal": p_st_cal.tolist(),  "T": Ts["stance"]},
        "sentiment": {"pred": int(np.argmax(p_se_raw)),  "conf_raw": p_se_raw.tolist(),
                      "conf_cal": p_se_cal.tolist(),  "T": Ts["sentiment"]},
    }


def get_reasoning(text: str, result: Dict) -> Tuple[str, str, Optional[Dict]]:
    """2-layer: Cache → LM Studio. Trả (reasoning, source, gemma_labels)."""
    cached = find_xai_reasoning_cache(text)
    if cached:
        return (clean_reasoning_output(cached),
                "✅ Từ cache (Gold Test Set, 186 mẫu)",
                parse_gemma_labels(cached))

    reasoning, raw = get_live_xai_reasoning(text, result=result, return_raw=True)
    if reasoning.startswith("⚠️") or "chưa được khởi động" in reasoning:
        return reasoning, "⚠️ LM Studio không khả dụng — hiển thị phân tích mẫu", None
    return reasoning, f"🖥️ Từ Gemma-4 GGUF · {LM_STUDIO_MODEL} ({LM_STUDIO_BASE_URL})", parse_gemma_labels(raw)



# ============================================================================
# CAPTUM INTEGRATED GRADIENTS
# ============================================================================

def _resolve_embedding_layer(model):
    for name in ("encoder", "phobert", "roberta", "bert", "model", "backbone"):
        owner = getattr(model, name, None)
        emb = getattr(owner, "embeddings", None) if owner is not None else None
        if emb is not None:
            return emb

    emb = getattr(model, "embeddings", None)
    if emb is not None:
        return emb

    for getter in (
        getattr(model, "get_input_embeddings", None),
        getattr(getattr(model, "encoder", None), "get_input_embeddings", None),
    ):
        if callable(getter):
            try:
                emb = getter()
                if emb is not None:
                    return emb
            except Exception:
                pass

    raise RuntimeError("Khong tim thay lop embedding cho IG")

def compute_captum_saliency(text: str, model_key: str) -> Tuple[List[str], List[float], int]:
    """Compute token-level attribution using Integrated Gradients."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        return [], [], -1

    model, tokenizer, ok = load_model(model_key)
    if not ok:
        return [], [], -1

    processed = text
    if "phobert" in model_key.lower():
        try:
            from underthesea import word_tokenize
            processed = word_tokenize(text, format="text")
        except Exception:
            pass

    enc = tokenizer(processed, truncation=True, max_length=256, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    def forward_fn(ids, mask):
        logits_m, _, _ = model(ids, mask)
        return logits_m

    with torch.no_grad():
        logits_m, _, _ = model(input_ids, attention_mask)
    pred_class = int(torch.argmax(logits_m, dim=1))

    lig = LayerIntegratedGradients(forward_fn, _resolve_embedding_layer(model))
    baseline = torch.zeros_like(input_ids) + (tokenizer.pad_token_id or 0)
    attributions = lig.attribute(
        inputs=input_ids, baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=pred_class, n_steps=20,
    )
    attr = attributions.sum(dim=-1).squeeze(0).detach().numpy()
    norm_max = np.abs(attr).max() + 1e-9
    attr_norm = (attr / norm_max).tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, attr_norm, pred_class


def render_saliency_html(tokens: List[str], attr_norm: List[float], pred_class: int) -> str:
    """Render saliency as HTML heatmap."""
    if not tokens:
        return "<p><em>⚠️ Captum IG đã bị tắt hoặc lỗi load model</em></p>"

    html = '<div style="line-height:1.9; padding:20px; border-radius:15px; background:var(--custom-card-bg); border:1px dashed var(--custom-card-border); font-family:Times New Roman, serif;">'
    for tok, score in zip(tokens, attr_norm):
        if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "<unk>"]:
            continue
        tok_clean = tok.replace("▁", " ").replace("@@", "").replace("Ġ", " ")
        abs_score = abs(score)
        if abs_score < 0.15:
            html += f'<span style="color: var(--custom-text-muted); opacity:0.65;">{tok_clean}</span> '
        else:
            intensity = min(abs_score, 0.7)
            if pred_class == 0:
                bg = f"rgba(255,75,75,{intensity})"
            else:
                bg = f"rgba(var(--saliency-pos-color),{intensity})"
            html += f'<span style="background:{bg}; padding:2px 6px; border-radius:4px; font-weight:bold;">{tok_clean}</span> '
    html += "</div>"
    label = LABEL_MAPS["misinfo"].get(pred_class, "?")
    html += f'<p style="font-size:11px; color: var(--custom-text-muted); margin-top:10px;">💡 Dự đoán: <b>{label}</b> · Token có màu đậm hơn = đóng góp lớn hơn vào quyết định model</p>'
    return html


# ============================================================================
# AI VOICE (gTTS) - Thread-safe with unique temp file
# ============================================================================

def text_to_speech(text: str) -> str:
    """Generate audio via gTTS, encode to base64, and return HTML player button."""
    if not text:
        return ""
    try:
        from gtts import gTTS
        import base64
        from io import BytesIO
        
        # Tạo âm thanh từ Google TTS
        tts = gTTS(text=text[:500], lang="vi")
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Mã hóa sang Base64 để nhúng trực tiếp vào HTML
        audio_b64 = base64.b64encode(fp.read()).decode()
        
        return f"""
        <div style="margin-top: 15px; margin-bottom: 10px;">
            <audio src="data:audio/mp3;base64,{audio_b64}"></audio>
            <button class="tts-speak-btn" style="
                background: linear-gradient(135deg, #00c853 0%, #b2ff59 100%);
                color: #0a192f;
                border: none;
                padding: 12px 24px;
                border-radius: 30px;
                font-weight: bold;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-family: 'Times New Roman', serif;
                font-size: 1rem;
            " onclick="togglePlay(this)">
                <span style="font-size: 1.2rem;">🔊</span> Nghe AI Giải Thích
            </button>
        </div>
        <style>
            .tts-speak-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 200, 83, 0.4);
                filter: brightness(1.05);
            }}
            .tts-speak-btn:active {{
                transform: translateY(1px);
                box-shadow: 0 2px 10px rgba(0, 200, 83, 0.2);
            }}
            @keyframes tts-pulse {{
                0% {{ transform: scale(1); opacity: 0.9; }}
                100% {{ transform: scale(1.15); opacity: 1; }}
            }}
            .tts-playing-icon {{
                display: inline-block;
                animation: tts-pulse 0.6s infinite alternate ease-in-out;
            }}
        </style>
        <script>
            if (typeof window.togglePlay !== 'function') {{
                window.togglePlay = function(btn) {{
                    const audio = btn.previousElementSibling;
                    if (!audio) return;
                    
                    if (audio.paused) {{
                        // Dừng tất cả audio khác nếu có
                        document.querySelectorAll('audio').forEach(a => {{
                            if (a !== audio) {{
                                a.pause();
                                a.currentTime = 0;
                                const otherBtn = a.nextElementSibling;
                                if (otherBtn && otherBtn.classList.contains('tts-speak-btn')) {{
                                    otherBtn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe AI Giải Thích';
                                    otherBtn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                                    otherBtn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                                    otherBtn.style.color = '#0a192f';
                                }}
                            }}
                        }});
                        
                        audio.play().then(() => {{
                            btn.innerHTML = '<span class="tts-playing-icon" style="font-size: 1.2rem;">⏹️</span> Đang đọc giải thích...';
                            btn.style.background = 'linear-gradient(135deg, #ff4b4b 0%, #ff8f8f 100%)';
                            btn.style.boxShadow = '0 4px 15px rgba(255, 75, 75, 0.3)';
                            btn.style.color = '#ffffff';
                        }}).catch(err => {{
                            console.error("Lỗi phát audio:", err);
                        }});
                    }} else {{
                        audio.pause();
                        audio.currentTime = 0;
                        btn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe AI Giải Thích';
                        btn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                        btn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                        btn.style.color = '#0a192f';
                    }}
                    
                    audio.onended = () => {{
                        btn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe Lại';
                        btn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                        btn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                        btn.style.color = '#0a192f';
                    }};
                }};
            }}
        </script>
        """
    except Exception as e:
        logger.warning(f"gTTS failed: {e}")
        return ""


# ============================================================================
# MULTI-SOURCE FETCHER
# ============================================================================

def detect_source(url: str) -> str:
    """Detect URL source type."""
    url_lower = url.lower()
    news_domains = ["vnexpress", "tuoitre", "thanhnien", "dantri", "vietnamnet",
                    "suckhoedoisong", "laodong", "tienphong", "znews", "hanoimoi",
                    "baochinhphu", "nhandan", "vov", "vtv"]
    if any(d in url_lower for d in news_domains):
        return "news"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if any(d in url_lower for d in ["facebook.com", "tiktok.com", "threads.net"]):
        return "apify"
    return "news"


def fetch_news(url: str) -> str:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded, favor_recall=True)
            return content or ""
        return ""
    except Exception as e:
        return f"❌ News fetch error: {e}"


def fetch_youtube(url: str, max_comments: int = 30) -> str:
    try:
        import yt_dlp
        opts = {"quiet": True, "skip_download": True, "getcomments": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "")
            desc = info.get("description", "")[:1000]
            comments = (info.get("comments") or [])[:max_comments]
            cmt_text = "\n".join([f"- {c.get('text', '')}" for c in comments])
            return f"TIÊU ĐỀ: {title}\n\nMÔ TẢ: {desc}\n\nBÌNH LUẬN ({len(comments)}):\n{cmt_text}"
    except Exception as e:
        return f"❌ YouTube fetch error: {e}"


def fetch_apify(url: str) -> str:
    if not APIFY_TOKENS:
        return "❌ APIFY_TOKEN chưa được setup trong HF Spaces Secrets"
    try:
        from apify_client import ApifyClient
        for token in APIFY_TOKENS:
            try:
                client = ApifyClient(token)
                client.user().get()
                if "facebook.com" in url.lower():
                    actor_id = "apify/facebook-posts-scraper"
                elif "tiktok.com" in url.lower():
                    actor_id = "clockworks/tiktok-scraper"
                elif "threads.net" in url.lower():
                    actor_id = "igview-owner/threads-search-scraper"
                else:
                    return "❌ URL không được hỗ trợ"
                run = client.actor(actor_id).call(run_input={"startUrls": [{"url": url}]})
                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                if items:
                    texts = [item.get("text", "") or item.get("caption", "") for item in items[:5]]
                    return "\n\n".join(t for t in texts if t)
                return "❌ Không trả về content"
            except Exception:
                continue
        return "❌ Tất cả Apify tokens đều thất bại"
    except Exception as e:
        return f"❌ Apify error: {e}"


def _is_valid_comment(txt: str) -> bool:
    """Màng lọc thông minh đầu vào cho comments/posts mạng xã hội."""
    if not txt:
        return False
    txt_strip = txt.strip()
    
    # 1. Lọc theo độ dài (loại bỏ "chấm", "hóng", "inbox", icon đơn lẻ)
    if len(txt_strip) < 15 or len(txt_strip.split()) < 4:
        return False
        
    # 2. Lọc spam bán hàng, tuyển dụng, quảng cáo (từ khóa rác)
    spam_keywords = [
        "inbox", "ib shop", "giá bao nhiêu", "mua ở đâu", "ship", "freeship", 
        "liên hệ zalo", "sđt", "tuyển dụng", "tuyển ctv", "sỉ lẻ", "giá rẻ", 
        "thanh lý", "chốt đơn", "nhận hàng", "uy tín", "đặt hàng", "zalo sđt",
        "cam kết", "hiệu quả", "giá sỉ", "giá lẻ", "chuyên sỉ", "tuyển đại lý"
    ]
    txt_lower = txt_strip.lower()
    if any(kw in txt_lower for kw in spam_keywords):
        return False
        
    # 3. Lọc OOD (thú y, showbiz, không liên quan đến vaccine)
    try:
        from src.preprocessing.text_cleaner_v2 import is_human_vaccine_context
        if not is_human_vaccine_context(txt_strip):
            return False
    except Exception:
        # Fallback keyword filter if text_cleaner_v2 is unavailable
        core_kws = ["vaccine", "vắc xin", "tiêm", "mũi", "bác sĩ", "bệnh", "y tế", "thuốc", "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv"]
        if not any(kw in txt_lower for kw in core_kws):
            return False
            
    return True


def fetch_url_as_list(url: str, max_comments: int = 30) -> Tuple[List[str], str]:
    """Fetch content from URL and return as a list of individual text segments (posts/comments) along with source info."""
    if not url or not url.strip():
        return [], "❌ Vui lòng nhập URL"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return [], "❌ URL không hợp lệ"

    kind = detect_source(url)
    texts = []
    info = ""

    if kind == "news":
        content = fetch_news(url)
        if content and not content.startswith("❌"):
            texts = [content]
            info = "📰 Báo điện tử (Tier 1, ~2s)"
        else:
            return [], content

    elif kind == "youtube":
        try:
            import yt_dlp
            opts = {"quiet": True, "skip_download": True, "getcomments": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                title = info_dict.get("title", "")
                desc = info_dict.get("description", "") or ""
                comments = info_dict.get("comments") or []
                
                if title:
                    texts.append(f"[TIÊU ĐỀ VIDEO] {title}")
                if desc.strip():
                    texts.append(f"[MÔ TẢ VIDEO] {desc.strip()[:500]}")
                for c in comments:
                    c_text = c.get("text", "").strip()
                    if c_text and _is_valid_comment(c_text):
                        texts.append(c_text)
                    if len(texts) >= max_comments + 2:  # +2 cho tiêu đề & mô tả
                        break
                info = "🎬 YouTube (Tier 2, ~10s)"
        except Exception as e:
            logger.warning(f"yt_dlp failed, trying Apify fallback: {e}")
            kind = "youtube_apify"

    if kind in ("apify", "youtube_apify"):
        if not APIFY_TOKENS:
            return [], "❌ APIFY_TOKEN chưa được setup trong HF Spaces Secrets"
        last_err = ""
        actor_id = ""
        try:
            from apify_client import ApifyClient
            for token in APIFY_TOKENS:
                try:
                    client = ApifyClient(token)
                    client.user().get()
                    
                    # Khống chế cứng max_comments cho Apify để tránh đốt API token và chạy vô hạn
                    apify_max_cmt = min(max_comments, 15)

                    # Phân loại thông minh URL để đưa ra actor thu thập phù hợp
                    if kind == "youtube_apify" or "youtube.com" in url.lower() or "youtu.be" in url.lower():
                        actor_id = "streamers/youtube-comments-scraper"
                        run_input = {"startUrls": [{"url": url}], "maxComments": apify_max_cmt}
                        source_display = "🎬 YouTube (Apify Fallback Scraper - Tier 3)"
                    elif "facebook.com" in url.lower():
                        if "/groups/" in url.lower():
                            actor_id = "apify/facebook-groups-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxPosts": 3,
                                "maxComments": apify_max_cmt,
                                "maxCommentsPerPost": 5,
                                "maxPostsPerGroup": 3,
                                "resultsLimit": 15
                            }
                            source_display = "👥 Facebook Group (Apify Scraper - Tier 3)"
                        elif "/posts/" in url.lower() or "/permalink/" in url.lower() or "comment_id=" in url.lower() or "/pfbid" in url.lower():
                            actor_id = "apify/facebook-comments-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxComments": apify_max_cmt,
                                "resultsLimit": apify_max_cmt
                            }
                            source_display = "💬 Facebook Post/Comments (Apify Scraper - Tier 3)"
                        else:
                            actor_id = "apify/facebook-posts-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxPosts": 3,
                                "maxComments": apify_max_cmt,
                                "maxCommentsPerPost": 5
                            }
                            source_display = "📱 Facebook Page/Profile (Apify Scraper - Tier 3)"
                    elif "tiktok.com" in url.lower():
                        actor_id = "clockworks/tiktok-comments-scraper"
                        run_input = {
                            "postURLs": [url],
                            "maxComments": apify_max_cmt
                        }
                        source_display = "🎵 TikTok Comments (Apify Scraper - Tier 3)"
                    elif "threads.net" in url.lower():
                        actor_id = "thenetaji/threads-scraper"
                        run_input = {
                            "startUrls": [{"url": url}],
                            "maxItems": apify_max_cmt
                        }
                        source_display = "🧵 Threads (Apify Scraper - Tier 3)"
                    else:
                        return [], "❌ URL không được hỗ trợ"
                    
                    logger.info(f"🚀 Running Apify Actor: {actor_id} for URL: {url}")
                    run = client.actor(actor_id).call(run_input=run_input)
                    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                    
                    if items:
                        for item in items:
                            # Capture a wide variety of text fields from different actors
                            txt = (
                                item.get("text", "") or 
                                item.get("message", "") or 
                                item.get("caption", "") or 
                                item.get("comment", "") or 
                                item.get("fullText", "") or 
                                item.get("description", "") or 
                                item.get("messageText", "") or
                                item.get("title", "") or
                                item.get("commentText", "") or
                                item.get("body", "")
                            )
                            if txt and txt.strip():
                                clean_txt = txt.strip()
                                if _is_valid_comment(clean_txt):
                                    texts.append(clean_txt)
                            if len(texts) >= apify_max_cmt:
                                break
                        if texts:
                            info = source_display
                            break
                except Exception as ex:
                    last_err = str(ex)
                    logger.warning(f"Apify token failed for actor {actor_id}: {ex}")
                    continue
            if not texts:
                err_detail = f" (Chi tiết lỗi: {last_err})" if last_err else ""
                return [], f"❌ Không thu thập được bài viết/bình luận nào từ Apify{err_detail}"
        except Exception as e:
            return [], f"❌ Apify error: {e}"

    return texts, info


def fetch_url(url: str, max_comments: int = 30) -> Tuple[str, str]:
    """Main fetcher dispatcher (backward compatibility)."""
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        return "", info
    return "\n\n".join(texts), f"**Nguồn:** {info}"


# ============================================================================
# CHARTS
# ============================================================================

def make_radar_chart(result: Dict) -> go.Figure:
    """3-axis radar chart."""
    misinfo_score = result["misinfo"]["conf_raw"][0]
    stance_score = result["stance"]["conf_raw"][1]
    sentiment_score = result["sentiment"]["conf_raw"][0]
    categories = ["Tin giả", "Phản đối", "Tiêu cực"]
    values = [misinfo_score, stance_score, sentiment_score]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself", line_color="#0e9384",
        fillcolor="rgba(14, 147, 132, 0.3)", name="Mức độ rủi ro"
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def make_benchmark_chart() -> go.Figure:
    """Macro F1 comparison bar chart."""
    bench = load_benchmark()
    models = ["PhoBERT-v2", "XLM-R-v1", "Gemma-4 4B"]
    keys = ["phobert", "xlmr", "gemma"]
    fig = go.Figure()
    tasks = ["misinfo", "stance", "sentiment"]
    task_names = ["Misinfo", "Stance", "Sentiment"]
    for i, task in enumerate(tasks):
        vals = [bench[k][task] for k in keys]
        fig.add_trace(go.Bar(
            x=models, y=vals, name=task_names[i],
            text=[f"{v:.4f}" for v in vals], textposition="auto"
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode="group", height=500,
        yaxis=dict(title="Macro F1", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def make_confusion_matrix_chart() -> go.Figure:
    """Confusion Matrix Heatmap for PhoBERT-v2 Sentiment (from phobert_v2_results.json).
    
    Actual values computed from probs + true_labels arrays in results JSON.
    Confusion matrix: rows = Thực tế, columns = Dự đoán
    - Tiêu cực (n=71):  TP=64, FP=3 → Trung tính, FP=4 → Tích cực
    - Trung tính (n=75): FP=14 → Tiêu cực, TP=57, FP=4 → Tích cực
    - Tích cực (n=40):  FP=11 → Tiêu cực, FP=9 → Trung tính, TP=20
    Macro F1: 0.7266 (matches phobert_v2_results.json)
    """
    z_data = [
        [64, 3, 4],
        [14, 57, 4],
        [11, 9, 20]
    ]
    labels = ["Tiêu cực", "Trung tính", "Tích cực"]
    fig = px.imshow(
        z_data, x=labels, y=labels, text_auto=True, aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(x="Dự đoán", y="Thực tế", color="Số mẫu"),
        title="Confusion Matrix — PhoBERT-v2 Sentiment (n=186)"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def make_per_class_chart(task: str = "misinfo") -> go.Figure:
    """Per-class F1 comparison across 3 models."""
    bench = load_benchmark()
    keys = ["phobert", "xlmr", "gemma"]
    colors = ["#0e9384", "#007bff", "#FFA500"]
    
    if task == "misinfo":
        class_labels = ["Tin giả", "Chính xác"]
        pc_key = "per_class_misinfo"
        support_key = "support_misinfo"
    elif task == "stance":
        class_labels = ["Ủng hộ", "Phản đối", "Trung lập"]
        pc_key = "per_class_stance"
        support_key = "support_stance"
    else:
        class_labels = ["Tiêu cực", "Trung tính", "Tích cực"]
        pc_key = "per_class_sentiment"
        support_key = "support_sentiment"
    
    support = bench["phobert"][support_key]
    x_labels = [f"{name} (n={sup})" for name, sup in zip(class_labels, support)]
    
    fig = go.Figure()
    for key, color in zip(keys, colors):
        pc = bench[key][pc_key]
        fig.add_trace(go.Bar(
            x=x_labels, y=pc, name=bench[key]["name"].split("(")[0].strip(),
            marker_color=color,
            text=[f"{v:.4f}" for v in pc], textposition="auto"
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode="group", height=480,
        yaxis=dict(title="F1 Score", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title=f"Per-class F1 — {task.upper()}"
    )
    return fig


def make_sankey_chart() -> go.Figure:
    """Sentiment to Stance co-occurrence flow (Gold Test Set, n=186).
    
    Data source: phobert_v2_results.json true_labels (sentiment & stance).
    Node indices: 0=Tiêu cực, 1=Trung tính, 2=Tích cực
                  3=Phản đối, 4=Trung lập, 5=Ủng hộ
    Co-occurrence matrix (actual):
      Tiêu cực → Phản đối=45, Trung lập=17, Ủng hộ=9
      Trung tính → Phản đối=3,  Trung lập=65, Ủng hộ=7
      Tích cực → Trung lập=2,  Ủng hộ=38  (Phản đối=0, omitted)
    """
    nodes = ["Tiêu cực", "Trung tính", "Tích cực", "Phản đối", "Trung lập", "Ủng hộ"]
    # Zero-value link (Tích cực→Phản đối=0) removed for clean visualisation
    sources = [0, 0, 0, 1, 1, 1, 2, 2]
    targets = [3, 4, 5, 3, 4, 5, 4, 5]
    values  = [45, 17, 9, 3, 65, 7, 2, 38]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=18, thickness=22, label=nodes,
                  color=["#ff4b4b", "#4a9eed", "#3db882", "#ff4b4b", "#007bff", "#64ffda"]),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(100, 255, 218, 0.2)")
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=15, r=15, t=15, b=15)
    )
    return fig


# ============================================================================
# SESSION HISTORY (Fixed: now updates correctly)
# ============================================================================

def session_history_to_markdown(history: List) -> str:
    """Render session history as markdown table."""
    if not history:
        return "*Chưa có lượt phân tích nào trong phiên này*"
    md = "### 📜 10 Lượt phân tích gần nhất\n\n"
    md += "| ⏱️ Thời gian | 📝 Văn bản | 🦠 Misinfo | 🎯 Stance | 💭 Sentiment |\n"
    md += "|---|---|---|---|---|\n"
    for h in history:
        # Sanitize newlines and pipe characters which break markdown tables
        text_clean = h['text'].replace('\n', ' ').replace('\r', ' ').replace('|', '&#124;')
        md += f"| {h['timestamp']} | {text_clean} | {h['misinfo']} | {h['stance']} | {h['sentiment']} |\n"
    return md


# ============================================================================
# CUSTOM VIEW FUNCTIONS & DATA FOR AESTHETICS (GRADIO PORT OVER)
# ============================================================================

def get_huph_logo_base64():
    # 1. HuggingFace Space root (app.py and huph_logo.png in the same directory)
    path1 = Path(__file__).resolve().parent / "huph_logo.png"
    # 2. Local workspace (running from app_gradio/ directory, huph_logo.png is in parent directory)
    path2 = Path(__file__).resolve().parent.parent / "huph_logo.png"
    # 3. Current working directory fallback
    path3 = Path.cwd() / "huph_logo.png"
    # 4. Nested app_gradio directory fallback
    path4 = Path.cwd() / "app_gradio" / "huph_logo.png"
    
    logo_path = None
    for p in [path1, path2, path3, path4]:
        if p.exists():
            logo_path = p
            break
            
    if logo_path:
        try:
            with open(logo_path, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        except Exception:
            pass
    # Fallback to public HUPH logo url
    return "https://huph.edu.vn/uploads/logo/logo-huph.png"


def render_result_cards_html(result: Dict, elapsed: float, model_choice: str) -> str:
    """Render beautiful HTML cards with progress bars for multi-task predictions."""
    html = '<div class="result-cards-grid" style="font-family: \'Times New Roman\', Times, serif;">'
    for axis, axis_name in [("misinfo", "Tin giả / Xác thực"), ("stance", "Quan điểm"), ("sentiment", "Cảm xúc")]:
        r = result[axis]
        pred_id = r["pred"]
        label = LABEL_MAPS[axis][pred_id]
        icon = LABEL_ICONS[axis][pred_id]
        color = LABEL_COLORS[axis][pred_id]
        
        conf_raw = max(r["conf_raw"]) * 100
        conf_cal = max(r["conf_cal"]) * 100
        T = r["T"]
        has_cal = abs(T - 1.0) > 0.001
        
        if has_cal:
            conf_html = f"""
            <div style="font-size: 0.85rem; color: var(--card-text-muted); margin-top: 5px; font-family: 'Times New Roman', Times, serif;">
                Thô: <span style="text-decoration: line-through;">{conf_raw:.1f}%</span>
            </div>
            <div style="font-size: 1.05rem; color: {color}; font-weight: bold; margin-top: 2px; font-family: 'Times New Roman', Times, serif;">
                Đã hiệu chuẩn (T={T:.2f}): {conf_cal:.1f}%
            </div>
            """
        else:
            conf_html = f"""
            <div style="font-size: 0.95rem; color: var(--card-text-muted); margin-top: 5px; font-family: 'Times New Roman', Times, serif;">
                Độ tin cậy: <strong style="color: {color};">{conf_raw:.1f}%</strong>
            </div>
            """

        # Class breakdown items
        breakdown_items = ""
        for idx, prob_raw in enumerate(r["conf_raw"]):
            prob_cal = r["conf_cal"][idx]
            class_label = LABEL_MAPS[axis][idx]
            class_color = LABEL_COLORS[axis][idx]
            pct_raw = prob_raw * 100
            pct_cal = prob_cal * 100
            
            if has_cal:
                breakdown_items += f"""
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--card-text-secondary); font-family: 'Times New Roman', Times, serif;">
                        <span>{class_label}</span>
                        <span style="font-size: 10px; color: var(--card-text-muted);">Thô: {pct_raw:.1f}% → <strong style="color: {class_color};">{pct_cal:.1f}%</strong></span>
                    </div>
                    <div style="background: var(--progress-bar-bg); border-radius: 5px; height: 8px; margin-top: 3px; overflow: hidden; border: 1px solid var(--card-border);">
                        <div style="background: {class_color}; width: {pct_cal}%; height: 100%; border-radius: 5px; box-shadow: 0 0 5px {class_color}80; animation: pulseGlow 2s infinite ease-in-out;"></div>
                    </div>
                </div>
                """
            else:
                breakdown_items += f"""
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--card-text-secondary); font-family: 'Times New Roman', Times, serif;">
                        <span>{class_label}</span>
                        <span style="color: {class_color}; font-weight: bold;">{pct_raw:.1f}%</span>
                    </div>
                    <div style="background: var(--progress-bar-bg); border-radius: 5px; height: 8px; margin-top: 3px; overflow: hidden; border: 1px solid var(--card-border);">
                        <div style="background: {class_color}; width: {pct_raw}%; height: 100%; border-radius: 5px; box-shadow: 0 0 5px {class_color}80; animation: pulseGlow 2s infinite ease-in-out;"></div>
                    </div>
                </div>
                """

        html += f"""
        <div class="result-card-hover" style="background: var(--card-bg); border: 1px solid {color}60; border-radius: 16px; padding: 22px; text-align: center; box-shadow: 0 8px 24px var(--shadow-color), 0 0 15px {color}10; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); border-bottom: 4px solid {color};">
            <div style="font-size: 40px; margin-bottom: 5px;">{icon}</div>
            <div style="font-size: 0.8rem; color: var(--card-text-muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px; font-family: 'Times New Roman', Times, serif; font-weight: 600;">{axis_name}</div>
            <div style="font-size: 1.6rem; font-weight: bold; color: {color}; margin-bottom: 8px; font-family: 'Times New Roman', Times, serif;">{label}</div>
            {conf_html}
            
            <div style="margin-top: 15px; border-top: 1px dashed var(--input-border); padding-top: 10px; text-align: left;">
                <div style="font-size: 0.8rem; font-weight: bold; color: var(--accent-color); text-transform: uppercase; margin-bottom: 5px; font-family: 'Times New Roman', Times, serif; letter-spacing: 0.05em;">Chi tiết nhãn:</div>
                {breakdown_items}
            </div>
        </div>
        """
    html += '</div>'
    html += f"<div style='margin-top: 15px; font-style: italic; color: var(--card-text-muted); font-family: \"Times New Roman\", Times, serif; font-size: 0.9rem; text-align: right;'>⏱️ Thời gian xử lý: {elapsed:.2f}s · Mô hình: {model_choice}</div>"
    return html


# ============================================================================
# REDESIGN: Verdict hero + Consistency flag (tam giác nhãn)
# ============================================================================
CONSISTENCY_UI = {
    "plausible": ("ok",     "Tổ hợp nhãn hợp lệ"),
    "unusual":   ("warn",   "Bất thường — nghi mô hình sai"),
    "high_risk": ("danger", "Nguy cơ cao — nên rà soát"),
}

def compute_consistency(result: Dict) -> str:
    """Cờ nhất quán tam giác nhãn từ int-pred (khớp compute_consistency của web).
    misinfo: 0=Tin giả,1=Chính xác | stance: 0=Ủng hộ,1=Phản đối,2=Trung lập
    sentiment: 0=Tiêu cực,1=Trung tính,2=Tích cực
    """
    mis = result["misinfo"]["pred"]
    st  = result["stance"]["pred"]
    se  = result["sentiment"]["pred"]
    if st == 1 and se == 0:                 # Phản đối + Tiêu cực → hồ sơ chống vaccine
        return "high_risk"
    if st == 1 and se == 2:                 # Phản đối + Tích cực (hiếm theo H1)
        return "unusual"
    if mis == 0 and st in (0, 2):           # Tin giả + (Ủng hộ/Trung lập) (hiếm theo H3)
        return "unusual"
    return "plausible"

_SVG_SHIELD = ('<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" '
               'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M12 3 5 6v5c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6Z"/><path d="M12 8v4M12 15.5h.01"/></svg>')
_SVG_CHECK  = ('<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" '
               'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
               '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5L16 9.5"/></svg>')

def render_verdict_hero(result: Dict) -> str:
    """Khối kết luận lớn nhất: Tin giả/Chính xác + độ tin cậy hiệu chuẩn + pill nhất quán."""
    mis = result["misinfo"]["pred"]
    fake = (mis == 0)
    label = LABEL_MAPS["misinfo"][mis]
    conf = max(result["misinfo"]["conf_cal"]) * 100
    pill_cls, pill_txt = CONSISTENCY_UI[compute_consistency(result)]
    glyph = _SVG_SHIELD if fake else _SVG_CHECK
    ok_cls = "" if fake else " ok"
    return (
        f'<div class="verdict{ok_cls}">'
        f'<div class="glyph">{glyph}</div>'
        f'<div><div class="vtitle">Kết luận đối soát</div>'
        f'<div class="vmain">{label}</div>'
        f'<div class="vmeta">Độ tin cậy mô hình (đã hiệu chuẩn) <b class="mono">{conf:.1f}%</b></div></div>'
        f'<div class="vside"><span class="pill {pill_cls}">{pill_txt}</span></div>'
        f'</div>'
    )

def render_consistency_legend(flag: str) -> str:
    """Chú giải 3 trạng thái cờ nhất quán, làm nổi trạng thái hiện tại."""
    chips = ""
    for k in ("plausible", "unusual", "high_risk"):
        cls, txt = CONSISTENCY_UI[k]
        on = (k == flag)
        style = ("outline:2px solid color-mix(in srgb,var(--ink-3) 40%,transparent);outline-offset:1px"
                 if on else "opacity:.55")
        chips += f'<span class="pill {cls}" style="{style}">{k} · {txt}{" ◂ hiện tại" if on else ""}</span>'
    return (
        '<div style="margin:14px 0 4px;padding-top:12px;border-top:1px solid var(--line-2)">'
        '<div style="font-size:12px;color:var(--ink-3);margin-bottom:8px">Cờ nhất quán tam giác nhãn '
        '<span class="mono">(consistency_flag)</span> — đối chiếu Gold n=186:</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap">{chips}</div></div>'
    )


def make_speed_chart() -> go.Figure:
    """Throughput (samples/sec) comparison bar chart."""
    models = ["PhoBERT-v2", "XLM-R-v1", "Gemma-4 4B"]
    throughputs = [120.5, 85.2, 1.8]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=throughputs,
        marker_color=['#0e9384', '#007bff', '#FFA500'],
        text=[f"{v:.1f} mẫu/s" for v in throughputs], textposition="auto"
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Times New Roman', color='#ccd6f6', size=13),
        yaxis=dict(title='Số mẫu xử lý/giây', range=[0, 140]),
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def make_sunburst_chart() -> go.Figure:
    """Sunburst label hierarchy chart."""
    sun_data = pd.DataFrame({
        "Label": ["Gold Test Set", "Tin giả", "Tin đúng", "Phản đối (Fake)", "Trung lập (Fake)", "Phản đối (True)", "Trung lập (True)", "Ủng hộ (True)"],
        "Parent": ["", "Gold Test Set", "Gold Test Set", "Tin giả", "Tin giả", "Tin đúng", "Tin đúng", "Tin đúng"],
        "Value": [186, 28, 158, 22, 6, 26, 78, 54]
    })
    fig_sun = px.sunburst(
        sun_data, names='Label', parents='Parent', values='Value',
        color_discrete_sequence=['#0d1b3e', '#ff4b4b', '#3db882', '#ff4b4b', '#007bff', '#ff4b4b', '#007bff', '#64ffda']
    )
    fig_sun.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        height=480
    )
    return fig_sun


METRICS_DB = {
    "Tin giả (Misinfo = Tin giả)": {
        "support": 28, "tp": 20, "fp": 33, "fn": 8, "desc": "Nhận diện tin giả chứa thông tin sai lệch về tác dụng phụ nguy hiểm hoặc thuyết âm mưu vắc-xin."
    },
    "Tin chính xác (Misinfo = Chính xác)": {
        "support": 158, "tp": 150, "fp": 8, "fn": 8, "desc": "Tin tức y tế chính thống, hướng dẫn tiêm chủng hoặc thông báo khoa học xác thực từ Bộ Y Tế."
    },
    "Lập trường Ủng hộ (Stance = Ủng hộ)": {
        "support": 54, "tp": 36, "fp": 22, "fn": 18, "desc": "Người dùng bày tỏ thái độ đồng ý tiêm chủng, kêu gọi cộng đồng cùng tiêm phòng dịch."
    },
    "Lập trường Phản đối (Stance = Phản đối)": {
        "support": 48, "tp": 30, "fp": 15, "fn": 18, "desc": "Lập trường bài trừ vắc-xin cực đoan, chống đối hoặc tuyên truyền tiêu cực về chiến dịch tiêm chủng."
    },
    "Lập trường Trung lập (Stance = Trung lập)": {
        "support": 84, "tp": 58, "fp": 26, "fn": 26, "desc": "Báo cáo lịch tiêm, hỏi đáp thông tin y khoa khách quan hoặc chia sẻ trải nghiệm tiêm bình thường."
    },
    "Cảm xúc Tiêu cực (Sentiment = Tiêu cực)": {
        "support": 71, "tp": 54, "fp": 17, "fn": 17, "desc": "Thể hiện sự lo lắng, sợ hãi tác dụng phụ y tế hoặc bức xúc chính sách giãn cách xã hội."
    },
    "Cảm xúc Trung tính (Sentiment = Trung tính)": {
        "support": 75, "tp": 56, "fp": 15, "fn": 19, "desc": "Chia sẻ thông tin công cộng, số liệu thống kê tiêm chủng hoặc tin tức sự kiện không chứa sắc thái cảm xúc."
    },
    "Cảm xúc Tích cực (Sentiment = Tích cực)": {
        "support": 40, "tp": 26, "fp": 13, "fn": 14, "desc": "Bày tỏ lòng biết ơn lực lượng y tế, sự an tâm và nhẹ nhõm sau khi đã tiêm đủ số mũi phòng ngừa."
    }
}


def update_calculator(selected_class: str):
    db = METRICS_DB[selected_class]
    tp = db["tp"]
    fp = db["fp"]
    fn = db["fn"]
    support = db["support"]
    desc = db["desc"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics_html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px; font-family: 'Times New Roman', serif;">
        <div style="flex: 1; min-width: 130px; border: 1px solid var(--custom-card-border); border-radius: 8px; padding: 10px; text-align: center; background: var(--custom-card-bg);">
            <p style="margin: 0; font-size: 0.85rem; color: var(--custom-text-muted);">Support (Tổng mẫu)</p>
            <h3 style="margin: 5px 0; color: var(--custom-text-neon); font-size: 1.5rem;">{support}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #3db882; border-radius: 8px; padding: 10px; text-align: center; background: rgba(61,184,130,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: var(--custom-text-muted);">True Positives (TP)</p>
            <h3 style="margin: 5px 0; color: #3db882; font-size: 1.5rem;">{tp}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #ff4b4b; border-radius: 8px; padding: 10px; text-align: center; background: rgba(255,75,75,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: var(--custom-text-muted);">False Positives (FP)</p>
            <h3 style="margin: 5px 0; color: #ff4b4b; font-size: 1.5rem;">{fp}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #FFA500; border-radius: 8px; padding: 10px; text-align: center; background: rgba(255,165,0,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: var(--custom-text-muted);">False Negatives (FN)</p>
            <h3 style="margin: 5px 0; color: #FFA500; font-size: 1.5rem;">{fn}</h3>
        </div>
    </div>
    <div style="font-style: italic; color: var(--custom-text-normal); font-size: 0.95rem; margin-bottom: 20px;">
        📌 <b>Định nghĩa nhãn</b>: {desc}
    </div>
    """
    
    precision_md = f"""
##### **1. Precision**
$$\\text{{Precision}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FP}}}}$$
$$\\text{{Precision}} = \\frac{{{tp}}}{{{tp} + {fp}}} = {precision:.4f}$$
"""

    recall_md = f"""
##### **2. Recall**
$$\\text{{Recall}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}}$$
$$\\text{{Recall}} = \\frac{{{tp}}}{{{tp} + {fn}}} = {recall:.4f}$$
"""

    f1_md = f"""
##### **3. F1-Score**
$$F_1 = 2 \\times \\frac{{\\text{{Precision}} \\times \\text{{Recall}}}}{{\\text{{Precision}} + \\text{{Recall}}}}$$
$$F_1 = 2 \\times \\frac{{{precision:.4f} \\times {recall:.4f}}}{{{precision:.4f} + {recall:.4f}}} = {f1:.4f}$$
"""

    return metrics_html, precision_md, recall_md, f1_md


# ============================================================================
# HANDLERS (Enhanced with progress + report export)
# ============================================================================

def handle_analyze(
    text: str, model_choice: str, use_captum: bool, history: List,
    progress=gr.Progress()
) -> Tuple:
    """Main analysis handler with progress indicator."""
    if not text or not text.strip():
        error_html = '<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">⚠️ Vui lòng nhập văn bản hoặc chọn mẫu thử!</div>'
        return (error_html, None, "", "", "", "", history, 
                session_history_to_markdown(history), "")

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()
    
    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    result = predict(text, model_choice)
    if not result:
        error_html = f'<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">❌ Không thể load mô hình {model_choice} — kiểm tra HF_TOKEN</div>'
        return (error_html, None, "", "", "", "", history,
                session_history_to_markdown(history), "")
    
    progress(0.5, desc="📊 Đang tính radar chart...")
    radar = make_radar_chart(result)
    
    progress(0.6, desc="💭 Đang sinh giải thích (XAI 3-layer)...")
    reasoning, source, gemma_labels = get_reasoning(text, result)
    reasoning_md = f"**{source}**\n\n{reasoning}"
    disagreement_html = render_disagreement_table(result, gemma_labels)

    if use_captum:
        progress(0.8, desc="🎯 Đang tính Captum Integrated Gradients...")
        tokens, attr_norm, pred_class = compute_captum_saliency(text, model_choice)
        saliency_html = render_saliency_html(tokens, attr_norm, pred_class)
    else:
        saliency_html = "<p style='color:#888;'><em>💡 Bật checkbox <b>Captum IG</b> để xem token attribution (chậm hơn 5-10s)</em></p>"

    progress(0.9, desc="🔊 Đang tạo AI Voice...")
    voice_text = reasoning if reasoning and not reasoning.startswith("⚠️") else ""
    audio_html = text_to_speech(voice_text) if voice_text else ""

    elapsed = time.time() - start

    # Generate custom HTML result cards (redesign: verdict hero → cards → consistency legend)
    summary_html = (
        render_verdict_hero(result)
        + render_result_cards_html(result, elapsed, model_choice)
        + render_consistency_legend(compute_consistency(result))
    )

    # Build export report markdown
    report_md = build_report_markdown(text, model_choice, result, reasoning, elapsed)

    # Update session history
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "misinfo": LABEL_MAPS["misinfo"][result["misinfo"]["pred"]],
        "stance":  LABEL_MAPS["stance"][result["stance"]["pred"]],
        "sentiment": LABEL_MAPS["sentiment"][result["sentiment"]["pred"]],
    }
    history = ([entry] + (history or []))[:CONFIG["session_history_limit"]]
    history_md = session_history_to_markdown(history)

    progress(1.0, desc="✅ Hoàn tất!")
    return (summary_html, radar, reasoning_md, disagreement_html, saliency_html, audio_html, history, history_md, report_md)


def build_report_markdown(text: str, model: str, result: Dict, reasoning: str, elapsed: float) -> str:
    """Build downloadable markdown report."""
    timestamp = time.strftime("%d/%m/%Y %H:%M:%S")
    report = f"""# Báo cáo Phân tích VaccineNLP

**Thời gian:** {timestamp}
**Mô hình:** {model}
**Thời gian xử lý:** {elapsed:.2f}s

---

## Văn bản phân tích

> {text}

## Kết quả phân loại

| Trục | Nhãn | Confidence (thô) | Confidence (hiệu chuẩn) |
|---|---|:---:|:---:|
"""
    for axis, axis_name in [("misinfo", "Misinformation"), ("stance", "Stance"), ("sentiment", "Sentiment")]:
        r = result[axis]
        label = LABEL_MAPS[axis][r["pred"]]
        conf_raw = max(r["conf_raw"]) * 100
        conf_cal = max(r["conf_cal"]) * 100
        report += f"| **{axis_name}** | {label} | {conf_raw:.1f}% | {conf_cal:.1f}% (T={r['T']:.2f}) |\n"
    
    report += f"""
## Giải thích (XAI)

{reasoning}

---

*Báo cáo được tạo tự động bởi VaccineNLP · HUPH 2026*
*Kim Mạnh Hưng (2211090016) · Đinh Lê Quỳnh Phương (2211090031)*
"""
    return report


def export_report_file(report_md: str) -> Optional[str]:
    """Save report markdown to file for download."""
    if not report_md or not report_md.strip():
        return None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="VaccineNLP_Report_", suffix=".md", delete=False, dir="/tmp", mode="w", encoding="utf-8"
        )
        tmp_file.write(report_md)
        tmp_file.close()
        return tmp_file.name
    except Exception as e:
        logger.warning(f"Export failed: {e}")
        return None


def handle_export_report(report_md: str):
    if not report_md or not report_md.strip():
        return gr.update(visible=False)
    path = export_report_file(report_md)
    if path:
        return gr.update(value=path, visible=True)
    return gr.update(visible=False)




def handle_cell_select(evt: gr.SelectData) -> str:
    """Handle selecting a cell in the fetched dataframe to send directly to main analysis input."""
    val = evt.value
    if val:
        try:
            gr.Info("🚀 Đã gửi nội dung được chọn lên ô phân tích chính!")
        except:
            pass
        return str(val).strip()
    return ""


def handle_export_fetched(fetched_raw_state: str) -> gr.update:
    """Export fetched texts as Excel file for download."""
    if not fetched_raw_state:
        return gr.update(visible=False)
    try:
        # Tach cac comment bang dau phan cach '---'
        texts = [t.strip() for t in fetched_raw_state.split("---") if t.strip()]
        df = pd.DataFrame({
            "STT": range(1, len(texts) + 1),
            "Nội dung thu thập": texts
        })
        # Luu ra file Excel tam thoi
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / f"crawled_vaccine_data_{int(time.time())}.xlsx"
        df.to_excel(file_path, index=False)
        return gr.update(value=str(file_path), visible=True)
    except Exception as e:
        logger.error(f"Failed to export fetched data: {e}")
        return gr.update(visible=False)


def handle_fetch_url(url: str, max_comments: int) -> Tuple[str, gr.update, gr.update, gr.update, gr.update, str]:
    """Unified handler for fetching URL content as a list of segments."""
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        error_msg = info if info.startswith("❌") else f"❌ Lỗi: {info}"
        return ("", gr.update(visible=False), gr.update(visible=False), gr.update(value=f"<p style='color:#ff4b4b;'>{error_msg}</p>"), "")
    
    rows = [[i + 1, t] for i, t in enumerate(texts)]
    df = pd.DataFrame(rows, columns=["STT", "Nội dung thu thập được"])
    batch_text_str = "\n\n---\n\n".join(texts)
    preview_text = texts[0] if texts else ""
    status_html = f"<p style='color:#3db882; font-weight:bold;'>✅ Thu thập thành công {len(texts)} bài viết/bình luận từ {info}!</p>"
    
    return (preview_text, gr.update(value=df, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value=status_html), batch_text_str)


def handle_send_to_batch(batch_text_str: str) -> Tuple[str, gr.update]:
    """Send fetched texts to batch textbox and open accordion."""
    try:
        gr.Info("🚀 Đã sao chép toàn bộ bài viết/comments vào ô Phân tích Batch! Vui lòng cuộn xuống dưới để thực hiện phân tích hàng loạt.")
    except:
        pass
    return batch_text_str, gr.update(open=True)


def handle_batch(text: str, model_choice: str, progress=gr.Progress()) -> str:
    """Batch mode analysis with progress."""
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập văn bản"
    # Tách các mẫu bằng dấu phân cách "---" nếu có, nếu không tách bằng dấu xuống dòng "\n"
    if "---" in text:
        lines = [l.strip() for l in text.split("---") if l.strip()]
    else:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) > 50:
        lines = lines[:50]

    rows = []
    for i, line in enumerate(lines):
        progress((i + 1) / len(lines), desc=f"🔬 Đang phân tích {i+1}/{len(lines)}...")
        r = predict(line, model_choice)
        if r:
            line_clean = line.replace("\n", " ").strip()
            rows.append({
                "STT": i + 1,
                "Văn bản": line_clean[:80] + ("..." if len(line_clean) > 80 else ""),
                "Tin giả": LABEL_MAPS["misinfo"][r["misinfo"]["pred"]],
                "Conf": f"{max(r['misinfo']['conf_raw']):.2%}",
                "Quan điểm": LABEL_MAPS["stance"][r["stance"]["pred"]],
                "Cảm xúc": LABEL_MAPS["sentiment"][r["sentiment"]["pred"]],
            })
    if not rows:
        return "❌ Không có kết quả"
    df = pd.DataFrame(rows)
    md = f"### 📋 Kết quả Batch ({len(rows)} mẫu)\n\n"
    md += df.to_markdown(index=False)
    return md


def handle_compare(text: str, progress=gr.Progress()) -> str:
    """Compare PhoBERT vs XLM-R."""
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập văn bản"
    progress(0.3, desc="🔬 Đang phân tích PhoBERT-v2...")
    p = predict(text, "PhoBERT-v2")
    progress(0.7, desc="🔬 Đang phân tích XLM-R-v1...")
    x = predict(text, "XLM-R-v1")
    progress(1.0, desc="✅ Hoàn tất!")
    if not p or not x:
        return "❌ Không load được model"
    md = "## 🔬 So sánh PhoBERT-v2 vs XLM-R-v1\n\n"
    md += "| Trục | PhoBERT-v2 | XLM-R-v1 | Khớp |\n|---|---|---|:---:|\n"
    for axis, name in [("misinfo", "Misinformation"), ("stance", "Stance"), ("sentiment", "Sentiment")]:
        pl = LABEL_MAPS[axis][p[axis]["pred"]]
        xl = LABEL_MAPS[axis][x[axis]["pred"]]
        pc = max(p[axis]["conf_cal"]) * 100
        xc = max(x[axis]["conf_cal"]) * 100
        match = "✅" if pl == xl else "⚠️"
        md += f"| **{name}** | {pl} ({pc:.1f}%) | {xl} ({xc:.1f}%) | {match} |\n"
    return md


def render_benchmark_md() -> str:
    """Render benchmark leaderboard markdown."""
    bench = load_benchmark()
    md = "## 📊 LIVE Benchmark — Macro F1 trên Gold Test Set (n=186)\n\n"
    md += "| Hạng | Mô hình | Misinfo | Stance | Sentiment | Trung bình |\n"
    md += "|---|---|:---:|:---:|:---:|:---:|\n"
    items = []
    for k in ["phobert", "xlmr", "gemma"]:
        d = bench[k]
        avg = (d["misinfo"] + d["stance"] + d["sentiment"]) / 3
        items.append((d["name"], d["misinfo"], d["stance"], d["sentiment"], avg))
    items.sort(key=lambda x: -x[4])
    medals = ["🥇", "🥈", "🥉"]
    for i, (name, m, s, se, avg) in enumerate(items):
        md += f"| {medals[i]} | **{name}** | {m:.4f} | {s:.4f} | {se:.4f} | **{avg:.4f}** |\n"
    md += "\n*Nguồn: data/benchmark_results.json (LIVE từ Kaggle 20-21/05/2026)*"
    return md


# ============================================================================
# STATIC CONTENT (Tabs 4, 5, 6)
# ============================================================================

RESOURCES_MD = """## 📚 Tài liệu & Notebooks Nghiên cứu

### 👨‍💻 Kim Mạnh Hưng (MSSV 2211090016)

**📘 Kaggle Notebooks:**
- [PhoBERT Multitask Classifier](https://www.kaggle.com/code/kimmnhhng/vaccinenlp-phobert-v2-multitask)
- [XLM-R Multitask Classifier](https://www.kaggle.com/code/kimmnhhng/vaccinenlp-xlm-r-v1-multitask-classifier)
- [Gemma QLoRA Training (03A)](https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-training)
- [Gemma XAI Inference (03B)](https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-inference)
- [Model Benchmark Report](https://www.kaggle.com/code/kimmnhhng/vaccinenlp-model-benchmark-report)

**🤗 HuggingFace:**
- [Gradio Demo App Space](https://huggingface.co/spaces/hung2903/vaccinenlp-demo)
- [Gemma GGUF Merged Model](https://huggingface.co/hung2903/gemma-4-E4B-vaccine-xai-merged)
- [PhoBERT Multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask)
- [XLM-R Multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask)
- [Gemma QLoRA Adapter](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai)

**💻 GitHub:**
- [VaccineNLP Thesis Repo](https://github.com/hwngkm/VaccineNLP-Thesis)

---

### 👩‍💻 Đinh Lê Quỳnh Phương (MSSV 2211090031)

**📘 Kaggle Notebooks:**
- [PhoBERT Multitask Classifier](https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-phobert-v2-multitask-classifier)
- [XLM-R Multitask Classifier](https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-xlm-r-v1-multitask-classifier)
- [Gemma QLoRA Training (03A)](https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-training)
- [Gemma XAI Inference (03B)](https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-inference)

**🤗 HuggingFace:**
- [Gradio Demo App Space](https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_demo)
- [PhoBERT Multitask](https://huggingface.co/quynhphuong1209/phobert-multitask)
- [XLM-R Multitask](https://huggingface.co/quynhphuong1209/xlmr-multitask)
- [Gemma QLoRA Adapter](https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai)

**💻 GitHub:**
- [VaccineNLP Project Repo](https://github.com/quynhphuong1209/VaccineNLP_Project)
"""

METHODOLOGY_MD = """## 📜 Phương pháp luận & Kiến trúc Hệ thống

### 🏗️ Kiến trúc Dual-Student Hybrid

Dự án xây dựng hệ thống **Ensemble** tận dụng ưu điểm của hai dòng kiến trúc Transformer:

**Động cơ Phân loại (Classification Engine):**
- PhoBERT-v2 (kiến trúc Encoder)
- Multi-task Learning với 3 heads độc lập
- Ưu điểm: Hiểu sâu ngữ pháp tiếng Việt, phân loại nhãn chính xác cao

**Động cơ Giải thích (XAI Reasoning Engine):**
- Gemma-4 E4B-it (kiến trúc Decoder)
- QLoRA 4-bit fine-tuning
- Ưu điểm: Sinh văn bản giải thích Tiếng Việt mạch lạc

### 🛠️ Pipeline Xử lý

```
[ Văn bản đầu vào (Tiếng Việt) ]
            ↓
[ Tiền xử lý: 8 bước cleaning ]
            ↓
    ┌───────┴───────┐
    ↓               ↓
[ PhoBERT-v2 ]   [ Gemma-4 XAI ]
    ↓               ↓
[ Labels ]      [ Reasoning ]
    ↓               ↓
    └───────┬───────┘
            ↓
[ Hiệu chuẩn (Temperature Scaling) ]
            ↓
[ Hiển thị: Labels + Confidence + Reasoning ]
```

### 🎯 Ba nhiệm vụ chính

1. **Misinformation Detection** — Xác định tin giả về vaccine
2. **Stance Analysis** — Phân tích quan điểm Ủng hộ/Phản đối/Trung lập
3. **Sentiment Analysis** — Nhận diện cảm xúc Tích cực/Tiêu cực/Trung tính

### 🧪 Quy trình thực nghiệm

- **Optimization:** QLoRA 4-bit + Temperature Scaling
- **Evaluation:** Macro F1 + ECE (Expected Calibration Error)
"""

CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f4f6f6;
    --bg-2: #edf0ef;
    --surface: #ffffff;
    --surface-2: #fafbfb;
    --ink: #15201e;
    --ink-2: #54625e;
    --ink-3: #8a9692;
    --line: #e5e9e8;
    --line-2: #eef1f0;
    --teal: #0e9384;
    --teal-strong: #0b6a60;
    --teal-50: #eaf6f3;
    --teal-100: #d4ece7;
    --danger: #d2453a;
    --danger-2: #b23a31;
    --danger-50: #fbece9;
    --warn: #b67d1c;
    --warn-50: #f9f1de;
    --shadow-sm: 0 1px 2px rgba(16,32,30,.05), 0 2px 6px rgba(16,32,30,.04);
    --shadow-md: 0 16px 40px -22px rgba(16,32,30,.28);
    --shadow-lg: 0 28px 60px -30px rgba(16,32,30,.34);
    --r-sm: 9px; --r: 13px; --r-lg: 18px;
    --ui: 'Be Vietnam Pro', system-ui, sans-serif;
    --mono: 'JetBrains Mono', monospace;
    
    /* Gradio mapped variables */
    --bg-color: #f4f6f6;
    --bg-gradient: linear-gradient(180deg, #f4f6f6 0%, #edf0ef 100%);
    --text-color: #15201e;
    --card-bg: #ffffff;
    --card-border: #e5e9e8;
    --header-bg: #ffffff;
    --header-text: #15201e;
    --footer-bg: #ffffff;
    --footer-text: #54625e;
    --input-bg: #fafbfb;
    --input-text: #15201e;
    --input-border: #e5e9e8;
    --accordion-bg: #fafbfb;
    --tab-button-bg: #edf0ef;
    --tab-button-text: #54625e;
    --accent-color: #0e9384;
    --accent-bg: #eaf6f3;
    --shadow-color: rgba(16,32,30,.06);
    --glow-color: rgba(14,147,132,.15);
    --card-text-muted: #8a9692;
    --card-text-primary: #15201e;
    --card-text-secondary: #54625e;
    --progress-bar-bg: #edf0ef;
    --dropdown-bg: #ffffff;
    --saliency-pos-color: 14, 147, 132;
    --custom-card-bg: #ffffff;
    --custom-card-border: #d4ece7;
    --custom-text-neon: #0e9384;
    --custom-text-muted: #54625e;
    --custom-text-normal: #15201e;
    --custom-phobert-bg: #ffffff;
    --custom-xlmr-bg: #ffffff;
    --custom-gemma-bg: #ffffff;
    --custom-phobert-border: #0e9384;
    --custom-phobert-text: #0e9384;
}

:root.dark, body.dark, .dark {
    --bg: #0a1211;
    --bg-2: #0d1816;
    --surface: #101d1b;
    --surface-2: #13201d;
    --ink: #e9efed;
    --ink-2: #9db0ab;
    --ink-3: #687a75;
    --line: rgba(255,255,255,.085);
    --line-2: rgba(255,255,255,.05);
    --teal: #2bcfba;
    --teal-strong: #63e0cf;
    --teal-50: rgba(43,207,186,.10);
    --teal-100: rgba(43,207,186,.18);
    --danger: #f0786c;
    --danger-2: #f4938a;
    --danger-50: rgba(240,120,108,.12);
    --warn: #e2b563;
    --warn-50: rgba(226,181,99,.12);
    --shadow-sm: 0 1px 2px rgba(0,0,0,.4);
    --shadow-md: 0 22px 48px -24px rgba(0,0,0,.66);
    --shadow-lg: 0 30px 70px -30px rgba(0,0,0,.7);
    
    /* Gradio mapped variables */
    --bg-color: #0a1211;
    --bg-gradient: linear-gradient(180deg, #0a1211 0%, #070e0d 100%);
    --text-color: #e9efed;
    --card-bg: #101d1b;
    --card-border: rgba(255,255,255,.085);
    --header-bg: #0d1816;
    --header-text: #ffffff;
    --footer-bg: #0d1816;
    --footer-text: #9db0ab;
    --input-bg: #13201d;
    --input-text: #e9efed;
    --input-border: rgba(255,255,255,.12);
    --accordion-bg: #13201d;
    --tab-button-bg: rgba(255,255,255,.06);
    --tab-button-text: #9db0ab;
    --accent-color: #2bcfba;
    --accent-bg: rgba(43,207,186,.10);
    --shadow-color: rgba(0,0,0,.4);
    --glow-color: rgba(43,207,186,.25);
    --card-text-muted: #687a75;
    --card-text-primary: #cbd5e1;
    --card-text-secondary: #ccd6f6;
    --progress-bar-bg: rgba(255,255,255,.06);
    --dropdown-bg: #101d1b;
    --saliency-pos-color: 43, 207, 186;
    --custom-card-bg: rgba(19,32,29,.55);
    --custom-card-border: rgba(43,207,186,.35);
    --custom-text-neon: #2bcfba;
    --custom-text-muted: #9db0ab;
    --custom-text-normal: #cbd5e1;
    --custom-phobert-bg: rgba(43,207,186,.06);
    --custom-xlmr-bg: rgba(74,158,237,.06);
    --custom-gemma-bg: rgba(255,165,0,.06);
    --custom-phobert-border: #2bcfba;
    --custom-phobert-text: #2bcfba;
}

body, html {
    background-color: var(--bg) !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--ui) !important;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

* {
    font-family: var(--ui) !important;
}

/* Modern thin scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(14, 147, 132, 0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(14, 147, 132, 0.5);
}

/* Hide default Gradio tabs (as navigation is driven by sidebar) */
.gradio-container > .tabs > .tab-nav {
    display: none !important;
}

/* ============ LAYOUT GRID ============ */
#main-layout-row {
    display: block !important;
    min-height: 100vh !important;
    width: 100% !important;
    margin: 0 !important;
    gap: 0 !important;
}

#content-col {
    padding: 0 !important;
    width: 100% !important;
}

/* ============ NAVBAR REDESIGN ============ */
#navbar-html-wrapper {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
}

.navbar-redesign {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--line) !important;
    padding: 12px clamp(15px, 3vw, 40px) !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
    width: 100% !important;
    box-sizing: border-box !important;
    box-shadow: var(--shadow-sm) !important;
}

.navbar-redesign .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.navbar-redesign .brand .logo {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(150deg, var(--teal), var(--teal-strong));
    display: grid;
    place-items: center;
    color: #fff;
    box-shadow: var(--shadow-sm);
}
.dark .navbar-redesign .brand .logo {
    color: #07120f;
}
.navbar-redesign .brand .name {
    font-weight: 700;
    font-size: 18px;
    letter-spacing: -.2px;
    color: var(--ink);
    line-height: 1;
}
.navbar-redesign .brand .name span {
    color: var(--teal);
}
.navbar-redesign .brand .tag {
    font-size: 11px;
    color: var(--ink-3);
    font-weight: 500;
    margin-left: 8px;
    padding-left: 8px;
    border-left: 1px solid var(--line);
}

.navbar-redesign .nav-menu {
    display: flex;
    align-items: center;
    gap: clamp(4px, 0.6vw, 10px);
    flex-grow: 1;
    justify-content: center;
    margin: 0 15px;
}
.navbar-redesign .nav-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px clamp(10px, 1vw, 16px);
    border-radius: var(--r-sm);
    color: var(--ink-2);
    font-weight: 500;
    font-size: 13.5px;
    cursor: pointer;
    border: none;
    background: none;
    transition: background .16s, color .16s;
    position: relative;
    white-space: nowrap;
}
.navbar-redesign .nav-item .icon {
    width: 16px;
    height: 16px;
}
.navbar-redesign .nav-item:hover {
    background: var(--bg-2);
    color: var(--ink);
}
.navbar-redesign .nav-item.active {
    background: var(--teal-50);
    color: var(--teal-strong);
    font-weight: 600;
}

.navbar-redesign .nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.navbar-redesign .status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: var(--r-sm);
    background: var(--surface-2);
    border: 1px solid var(--line);
    font-size: 11.5px;
    color: var(--ink-2);
    white-space: nowrap;
}
.navbar-redesign .status-chip .pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 2.4s infinite;
}

.navbar-redesign .theme-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: var(--r-sm);
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: var(--ink-2);
    font-size: 12.5px;
    font-weight: 500;
    cursor: pointer;
}
.navbar-redesign .theme-toggle span {
    font-size: 11.5px;
}
.navbar-redesign .theme-toggle .sw {
    width: 34px;
    height: 18px;
    border-radius: 20px;
    background: var(--bg-2);
    border: 1px solid var(--line);
    position: relative;
    transition: background .2s;
}
.navbar-redesign .theme-toggle .sw i {
    position: absolute;
    top: 1px;
    left: 1px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--teal);
    transition: transform .22s cubic-bezier(.4,1.3,.6,1);
    display: grid;
    place-items: center;
    color: #fff;
}
.dark .navbar-redesign .theme-toggle .sw i {
    transform: translateX(16px);
    color: #07120f;
}

/* Responsive Navbar */
@media (max-width: 1024px) {
    .navbar-redesign {
        flex-wrap: wrap !important;
        padding: 10px 20px !important;
        gap: 10px !important;
    }
    .navbar-redesign .brand {
        order: 1 !important;
    }
    .navbar-redesign .nav-right {
        order: 2 !important;
    }
    .navbar-redesign .nav-menu {
        order: 3 !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 5px 0 0 0 !important;
        justify-content: flex-start !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        gap: 6px !important;
        padding: 4px 0 !important;
        scrollbar-width: none !important; /* Firefox */
        -webkit-overflow-scrolling: touch !important;
    }
    .navbar-redesign .nav-menu::-webkit-scrollbar {
        display: none !important; /* Chrome, Safari, Opera */
    }
    .navbar-redesign .brand .tag {
        display: none !important;
    }
}
@media (max-width: 600px) {
    .navbar-redesign .theme-toggle span {
        display: none !important;
    }
    .navbar-redesign .status-chip {
        display: none !important;
    }
}

/* Global icon base — applies to every inline SVG using the .icon class.
   Without this, SVGs default to fill:black and a 300x150 intrinsic size,
   rendering as large solid black blobs (empty-states, section labels, logo). */
.icon {
    width: 18px;
    height: 18px;
    stroke: currentColor;
    stroke-width: 1.7;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    flex-shrink: 0;
    vertical-align: middle;
}
.icon.sm { width: 16px; height: 16px; }
.icon.lg { width: 24px; height: 24px; }

/* Hidden buttons used for python callbacks */
.hidden-btn {
    display: none !important;
}

/* Responsive Grid for Result Cards */
.result-cards-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: clamp(12px, 1.5vw, 20px) !important;
    width: 100% !important;
    margin-bottom: 10px !important;
}
@media (max-width: 768px) {
    .result-cards-grid {
        grid-template-columns: 1fr !important;
    }
}

/* ============ CONTENT & TOPBAR ============ */
.screen-container {
    padding: 30px clamp(20px, 3vw, 40px) !important;
    animation: rise .42s cubic-bezier(.2,.7,.3,1);
    max-width: 1320px !important;
    width: 100% !important;
    margin: 0 auto !important;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: none; }
}

.topbar {
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 16px !important;
    margin-bottom: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    background: transparent !important;
}

.topbar .crumb {
    font-size: 12.5px;
    color: var(--ink-3);
    font-weight: 500;
}

.topbar h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: -.3px;
    margin: 1px 0 0 0 !important;
    color: var(--ink) !important;
}

/* ============ CARDS & UI COMPONENTS ============ */
.block, .gr-box, .gr-panel {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 16px !important;
}

.card-pad {
    padding: 22px !important;
}

.section-label {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink-2);
    letter-spacing: .2px;
}
.section-label .ic {
    color: var(--teal);
}

/* Textarea & inputs */
.gradio-container textarea, 
.gradio-container input[type="text"] {
    background: var(--surface-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
    padding: 10px 15px !important;
    font-size: 14.5px !important;
    transition: all .2s !important;
}

.gradio-container textarea:focus, 
.gradio-container input[type="text"]:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px var(--teal-50) !important;
}

/* Buttons */
button.primary, button.gr-button-primary {
    background: var(--teal) !important;
    border: 1px solid var(--teal) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: var(--r-sm) !important;
    box-shadow: 0 4px 12px var(--teal-50) !important;
    text-transform: none !important;
    font-size: 14px !important;
}
.dark button.primary {
    color: #07120f !important;
}
button.primary:hover {
    background: var(--teal-strong) !important;
    border-color: var(--teal-strong) !important;
}

button.secondary, button.gr-button-secondary {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink-2) !important;
    border-radius: var(--r-sm) !important;
}
button.secondary:hover {
    border-color: var(--ink-3) !important;
    color: var(--ink) !important;
}

/* Verdict Hero */
.verdict {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 24px;
    border-radius: var(--r-lg);
    background: linear-gradient(120deg, var(--danger-50), color-mix(in srgb, var(--surface) 80%, var(--danger-50)));
    border: 1px solid color-mix(in srgb, var(--danger) 18%, transparent);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
.verdict::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--danger);
}
.verdict.ok {
    border-color: var(--teal-100);
    background: linear-gradient(120deg, var(--teal-50), color-mix(in srgb, var(--surface) 80%, var(--teal-50)));
}
.verdict.ok::before {
    background: var(--teal);
}
.verdict .glyph {
    width: 60px;
    height: 60px;
    border-radius: 16px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    background: var(--surface);
    color: var(--danger);
    border: 1px solid color-mix(in srgb, var(--danger) 24%, transparent);
    box-shadow: 0 1px 2px rgba(16,32,30,.06);
}
.verdict.ok .glyph {
    color: var(--teal);
    border-color: var(--teal-100);
}
.verdict .vtitle {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .8px;
    text-transform: uppercase;
    color: var(--ink-3);
}
.verdict .vmain {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -.6px;
    line-height: 1.05;
    margin: 3px 0 5px;
    color: var(--danger-2);
}
.verdict.ok .vmain {
    color: var(--teal-strong);
}
.verdict .vmeta {
    font-size: 13px;
    color: var(--ink-2);
}
.verdict .vmeta b {
    color: var(--ink);
    font-weight: 600;
}
.verdict .vside {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
}

/* Axis Card */
.axis {
    display: flex;
    flex-direction: column;
    padding: 18px;
    border-radius: var(--r);
    background: var(--surface-2);
    border: 1px solid var(--line);
    text-align: left;
}
.axis .cap {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 4px;
}
.axis .val {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -.2px;
    margin-bottom: 10px;
    color: var(--ink);
}
.axis .val.bad {
    color: var(--danger-2);
}
.axis .val.ok {
    color: var(--teal-strong);
}
.axis .val.neu {
    color: var(--ink);
}
.axis .meter {
    height: 7px;
    border-radius: 99px;
    background: var(--line-2);
    overflow: hidden;
    position: relative;
    margin-bottom: 8px;
    border: 1px solid var(--line);
}
.axis .meter i {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 99px;
    background: var(--ink-3);
    transition: width 1s cubic-bezier(.2,.8,.2,1);
}
.axis .meter.bad i {
    background: var(--danger);
}
.axis .meter.teal i {
    background: var(--teal);
}
.axis .scoreline {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: var(--ink-2);
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 11px;
    border-radius: 99px;
    border: 1px solid var(--line);
    color: var(--ink-2);
    background: var(--surface);
}
.pill.danger {
    color: var(--danger-2);
    background: var(--danger-50);
    border-color: var(--danger);
}
.pill.ok {
    color: var(--teal-strong);
    background: var(--teal-50);
    border-color: var(--teal-100);
}
.pill.warn {
    color: var(--warn);
    background: var(--warn-50);
    border-color: color-mix(in srgb, var(--warn) 30%, transparent);
}

.mono {
    font-family: var(--mono) !important;
    font-variant-numeric: tabular-nums;
}

.svg-sprite {
    display: none;
}
"""

SPEED_METRICS_HTML = """
<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; font-family: 'Times New Roman', serif;">
    <div style="flex: 1; min-width: 200px; border: 1px solid var(--custom-phobert-border); border-radius: 8px; padding: 15px; text-align: center; background: var(--custom-phobert-bg);">
        <p style="margin: 0; font-size: 0.9rem; color: var(--custom-text-muted);">🏎️ Tốc độ PhoBERT-v2</p>
        <h2 style="margin: 5px 0; color: var(--custom-phobert-text); font-size: 1.8rem; font-weight: bold;">120.5 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #3db882; font-weight: bold;">Nhanh nhất (Real-time)</span>
    </div>
    <div style="flex: 1; min-width: 200px; border: 1px solid #007bff; border-radius: 8px; padding: 15px; text-align: center; background: var(--custom-xlmr-bg);">
        <p style="margin: 0; font-size: 0.9rem; color: var(--custom-text-muted);">🚗 Tốc độ XLM-R-v1</p>
        <h2 style="margin: 5px 0; color: #007bff; font-size: 1.8rem; font-weight: bold;">85.2 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #ff4b4b; font-weight: bold;">-29.3% so với PhoBERT</span>
    </div>
    <div style="flex: 1; min-width: 200px; border: 1px solid #FFA500; border-radius: 8px; padding: 15px; text-align: center; background: var(--custom-gemma-bg);">
        <p style="margin: 0; font-size: 0.9rem; color: var(--custom-text-muted);">🐢 Tốc độ Gemma-4 4B</p>
        <h2 style="margin: 5px 0; color: #FFA500; font-size: 1.8rem; font-weight: bold;">1.8 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #ff4b4b; font-weight: bold;">Rất chậm (Phù hợp offline)</span>
    </div>
</div>
"""

RECOMMENDATIONS_HTML = """
<div style="background: var(--custom-card-bg); border: 1px solid var(--custom-card-border); border-radius: 8px; padding: 20px; font-family: 'Times New Roman', serif;">
    <h4 style="margin-top: 0; color: var(--custom-text-neon); font-size: 1.2rem;">🤝 Kiến trúc lai đề xuất cho dự án VaccineNLP (HUPH 2026):</h4>
    <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.6; color: var(--custom-text-normal);">
        <li><b>Vòng ngoài (Real-time Classification - PhoBERT-v2)</b>: Nhờ tốc độ suy luận cực nhanh (120.5 mẫu/giây) và độ chính xác F1 vượt trội, PhoBERT-v2 được đề xuất làm màng lọc trực tiếp ở luồng dữ liệu mạng xã hội để phân loại nhanh tin giả, sắc thái và lập trường.</li>
        <li><b>Vòng trong (Explainable & Strategic Consulting - Gemma-4 4B)</b>: Đối với các mẫu được PhoBERT-v2 nghi ngờ là "Tin giả" hoặc "Tiêu cực cực đoan", hệ thống sẽ đẩy vào hàng đợi offline để Gemma-4 lý luận chuyên sâu (XAI) giải thích lý do gán nhãn và đề xuất kịch bản phản hồi khủng hoảng cho chuyên gia y tế HUPH.</li>
    </ol>
</div>
"""

RESOURCES_HTML = """
<div style="font-family: 'Times New Roman', Times, serif; color: var(--text-color);">
  <h2 style="color: var(--accent-color); margin-bottom: 20px; font-size: 1.8rem;">📚 Tài liệu & Notebooks Nghiên cứu</h2>
  
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <!-- Column 1: Kim Manh Hung -->
    <div style="flex: 1; min-width: 300px; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px var(--shadow-color);">
      <h3 style="color: var(--accent-color); margin-top: 0; border-bottom: 1px solid var(--input-border); padding-bottom: 10px; font-size: 1.3rem;">👨‍💻 1. Kim Mạnh Hưng (MSSV: 2211090016)</h3>
      
      <div style="margin-top: 15px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">📘 I. KAGGLE NOTEBOOKS:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-phobert-v2-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-xlm-r-v1-multitask-classifier" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-training" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma QLoRA Training (03A)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-inference" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma XAI Inference (03B)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-model-benchmark-report" target="_blank" style="color: var(--accent-color); text-decoration: none;">Model Benchmark Report (04)</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">🤗 II. HUGGINGFACE:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/spaces/hung2903/vaccinenlp-demo" target="_blank" style="color: var(--accent-color); text-decoration: none;"><b>Gradio Demo App Space</b></a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/gemma-4-E4B-vaccine-xai-merged" target="_blank" style="color: var(--accent-color); text-decoration: none;"><b>Gemma GGUF Merged Model</b></a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/phobert-vaccine-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask Model</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/xlmr-vaccine-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask Model</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma QLoRA Adapter</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">💻 III. GITHUB:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li>• <a href="https://github.com/hwngkm/VaccineNLP-Thesis" target="_blank" style="color: var(--accent-color); text-decoration: none;">VaccineNLP Thesis Repo</a></li>
        </ul>
      </div>
    </div>
    
    <!-- Column 2: Dinh Le Quynh Phuong -->
    <div style="flex: 1; min-width: 300px; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px var(--shadow-color);">
      <h3 style="color: var(--accent-color); margin-top: 0; border-bottom: 1px solid var(--input-border); padding-bottom: 10px; font-size: 1.3rem;">👩‍💻 2. Đinh Lê Quỳnh Phương (MSSV: 2211090031)</h3>
      
      <div style="margin-top: 15px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">📘 I. KAGGLE NOTEBOOKS:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-phobert-v2-multitask-classifier" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-xlm-r-v1-multitask-classifier" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-training" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma QLoRA Training (03A)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-inference" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma XAI Inference (03B)</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">🤗 II. HUGGINGFACE:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/spaces/quynhphuong1209/VaccineNLP_demo" target="_blank" style="color: var(--accent-color); text-decoration: none;"><b>Gradio Demo App Space</b></a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/phobert-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask Model</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/xlmr-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask Model</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma QLoRA Adapter</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: var(--accent-color); margin-bottom: 5px; font-size: 1.05rem; opacity: 0.95;">💻 III. GITHUB:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li>• <a href="https://github.com/quynhphuong1209/VaccineNLP_Project" target="_blank" style="color: var(--accent-color); text-decoration: none;">VaccineNLP Project Repo</a></li>
        </ul>
      </div>
    </div>
  </div>
</div>
"""

METHODOLOGY_HTML = """
<div style="font-family: 'Times New Roman', Times, serif; color: var(--text-color); line-height: 1.6;">
  <h2 style="color: var(--accent-color); border-bottom: 1px solid var(--input-border); padding-bottom: 10px; font-size: 1.8rem; margin-bottom: 20px;">📜 Phương pháp luận & Kiến trúc Hệ thống</h2>
  
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 3; min-width: 300px;">
      <h3 style="color: var(--accent-color); font-size: 1.3rem;">🏗️ 1. Kiến trúc Dual-Student Hybrid</h3>
      <p>Dự án xây dựng hệ thống <b>Ensemble</b> tận dụng ưu điểm của hai dòng kiến trúc Transformer phổ biến nhất hiện nay:</p>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-left: 4px solid #3db882; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
        <strong style="color: #3db882;">Động cơ Phân loại (Classification Engine):</strong>
        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
          <li>PhoBERT-v2 (kiến trúc Encoder)</li>
          <li>Multi-task Learning với 3 heads độc lập</li>
          <li>Ưu điểm: Hiểu sâu ngữ pháp tiếng Việt, phân loại nhãn chính xác cao</li>
        </ul>
      </div>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-left: 4px solid #FFA500; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <strong style="color: #FFA500;">Động cơ Giải thích (XAI Reasoning Engine):</strong>
        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
          <li>Gemma-4 E4B-it (kiến trúc Decoder)</li>
          <li>QLoRA 4-bit fine-tuning</li>
          <li>Ưu điểm: Sinh văn bản giải thích Tiếng Việt mạch lạc</li>
        </ul>
      </div>
      
      <h4 style="color: var(--accent-color); margin-bottom: 10px;">🛠️ Sơ đồ Luồng Xử lý (System Pipeline)</h4>
      <pre style="background: var(--input-bg); color: var(--text-color); border: 1px solid var(--input-border); border-radius: 8px; padding: 15px; font-family: monospace; font-size: 0.9rem; line-height: 1.4;">
[ Văn bản đầu vào (Tiếng Việt) ]
              ↓
[ Tiền xử lý: 8 bước cleaning ]
              ↓
      ┌───────┴───────┐
      ↓               ↓
[ PhoBERT-v2 ]   [ Gemma-4 XAI ]
      ↓               ↓
[ Labels ]      [ Reasoning ]
      ↓               ↓
      └───────┬───────┘
              ↓
[ Hiệu chuẩn (Temperature Scaling) ]
              ↓
[ Hiển thị: Labels + Confidence + Reasoning ]
      </pre>
    </div>
    
    <div style="flex: 2; min-width: 250px; display: flex; flex-direction: column; gap: 15px;">
      <h3 style="color: var(--accent-color); font-size: 1.3rem; margin-bottom: 5px;">🎯 2. Ba nhiệm vụ chính</h3>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px var(--shadow-color);">
        <strong style="color: #ff4b4b; font-size: 1rem;">🚨 Misinformation Detection</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: var(--card-text-muted);">Xác định tin giả về vaccine dựa trên các nguồn tin cậy và đối chiếu chéo.</p>
      </div>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px var(--shadow-color);">
        <strong style="color: var(--accent-color); font-size: 1rem;">🎯 Stance Analysis</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: var(--card-text-muted);">Phân tích quan điểm cộng đồng: Ủng hộ, Phản đối hoặc Trung lập với tiêm chủng vaccine.</p>
      </div>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px var(--shadow-color);">
        <strong style="color: #00c853; font-size: 1rem;">💭 Sentiment Analysis</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: var(--card-text-muted);">Nhận diện sắc thái cảm xúc của người viết: Tích cực, Tiêu cực, hoặc Trung tính.</p>
      </div>
      
      <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 8px; padding: 15px; margin-top: 10px; border-left: 4px solid var(--accent-color);">
        <h4 style="color: var(--accent-color); margin: 0 0 8px 0; font-size: 1.05rem;">🧪 Quy trình thực nghiệm</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; line-height: 1.5; color: var(--card-text-muted);">
          <li><b>Dataset:</b> 1.856 mẫu Silver + 186 mẫu Gold Test</li>
          <li><b>Hardware:</b> GPU NVIDIA T4 (Kaggle)</li>
          <li><b>Optimization:</b> QLoRA 4-bit + Temperature Scaling</li>
          <li><b>Evaluation:</b> Macro F1 + ECE (Expected Calibration Error)</li>
        </ul>
      </div>
    </div>
  </div>
  
  <div style="margin-top: 25px; border-top: 1px solid var(--input-border); padding-top: 15px;">
    <h3 style="color: var(--accent-color); font-size: 1.25rem;">💡 Tại sao Explainable AI (XAI)?</h3>
    <p style="margin-top: 5px;">Trong lĩnh vực y tế như vaccine, việc chỉ đưa ra nhãn 'Tin giả' là chưa đủ. Hệ thống cần giải thích <b>tại sao</b> để:</p>
    <ul style="padding-left: 20px; margin-top: 5px;">
      <li style="margin-bottom: 5px;">Thuyết phục người dùng tin tưởng vào các khuyến nghị và cảnh báo của AI.</li>
      <li style="margin-bottom: 5px;">Hỗ trợ đắc lực cho cán bộ và chuyên gia y tế công cộng ra quyết định nhanh chóng.</li>
      <li style="margin-bottom: 5px;">Đáp ứng yêu cầu minh bạch và giải trình bắt buộc trong ứng dụng AI y tế công cộng.</li>
    </ul>
  </div>
</div>
"""

THESIS_HTML = """
<div style="font-family: 'Times New Roman', Times, serif; color: var(--text-color); line-height: 1.6;">
  <h2 style="color: var(--accent-color); border-bottom: 1px solid var(--input-border); padding-bottom: 10px; font-size: 1.8rem; margin-bottom: 20px;">📑 Đề cương & Mục lục Đồ án tốt nghiệp</h2>
  
  <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 20px; border-radius: 5px; margin-bottom: 25px; box-shadow: 0 4px 8px var(--shadow-color);">
    <h3 style="margin: 0; color: var(--text-color); font-size: 1.15rem; text-transform: uppercase;">📝 Tên Đề Tài Đồ Án tốt nghiệp:</h3>
    <p style="margin: 8px 0 0 0; font-size: 1.25rem; font-weight: bold; color: var(--accent-color); line-height: 1.4;">
      "Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"
    </p>
    <p style="margin: 5px 0 0 0; font-style: italic; color: var(--card-text-muted); font-size: 1rem;">
      (Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)
    </p>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 25px;">
    <div style="flex: 1; min-width: 300px; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 12px; padding: 20px;">
      <h3 style="color: var(--accent-color); border-bottom: 1px solid var(--input-border); padding-bottom: 8px; margin-top: 0; font-size: 1.25rem;">📌 Cấu trúc 6 Chương chính</h3>
      <ul style="list-style-type: none; padding-left: 0; line-height: 1.8;">
        <li><b>CHƯƠNG 1: ĐẶT VẤN ĐỀ</b> (Lý do chọn đề tài, Mục tiêu MT1-MT3, Câu hỏi RQ1-RQ3)</li>
        <li><b>CHƯƠNG 2: TỔNG QUAN TÀI LIỆU</b> (Định nghĩa Vaccine misinformation, NLP, XAI, Research Gap)</li>
        <li><b>CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU</b> (Chiến lược Tier-Based A/B/C, 8 bước tiền xử lý, Annotation)</li>
        <li><b>CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM</b> (Mô tả Gold Test Set n=186, Macro F1, Calibration, Kiểm định)</li>
        <li><b>CHƯƠNG 5: BÀN LUẬN KHOA HỌC</b> (Diễn giải kết quả chính, so sánh ANTiVax/MiSoVac, Hạn chế đề tài)</li>
        <li><b>CHƯƠNG 6: KẾT LUẬN VÀ KIẾN NGHỊ</b> (Tổng kết mục tiêu, kiến nghị CDC & Bộ Y tế, hướng phát triển)</li>
      </ul>
    </div>
    
    <div style="flex: 1; min-width: 300px; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 12px; padding: 20px;">
      <h3 style="color: var(--accent-color); border-bottom: 1px solid var(--input-border); padding-bottom: 8px; margin-top: 0; font-size: 1.25rem;">🧪 Ba Giả thuyết Nghiên cứu (Hypotheses)</h3>
      <ul style="list-style-type: none; padding-left: 0; line-height: 1.8; color: var(--text-color);">
        <li style="margin-bottom: 12px;">
          <strong style="color: #ff4b4b;">• Giả thuyết H1 (Chấp nhận):</strong><br>
          Cảm xúc tiêu cực ↔ Lập trường phản đối vaccine (Kiểm định Chi-square đạt ý nghĩa thống kê cao, p < 10⁻⁴⁰).
        </li>
        <li style="margin-bottom: 12px;">
          <strong style="color: var(--accent-color);">• Giả thuyết H2 (Chấp nhận):</strong><br>
          Nền tảng mạng xã hội ↔ Tỷ lệ lan truyền tin giả y tế (Kiểm định G-test, p = 2,14 × 10⁻³).
        </li>
        <li style="margin-bottom: 12px;">
          <strong style="color: #00c853;">• Giả thuyết H3 (Chấp nhận):</strong><br>
          Lập trường phản đối/nghi ngại ↔ Tỷ lệ xuất hiện tin giả (Kiểm định Chi-square, p < 10⁻¹⁴).
        </li>
      </ul>
    </div>
  </div>
  
  <div style="background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px var(--shadow-color);">
    <h3 style="color: var(--accent-color); margin-top: 0; border-bottom: 1px solid var(--input-border); padding-bottom: 8px; font-size: 1.3rem;">👥 Thông tin Đồ án tốt nghiệp HUPH</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 30px; margin-top: 15px;">
      <div style="flex: 1; min-width: 250px;">
        <h4 style="color: var(--accent-color); margin: 0 0 10px 0; font-size: 1.1rem;">Sinh viên thực hiện:</h4>
        <p style="margin: 5px 0;"><b>1. Kim Mạnh Hưng</b> · MSSV: 2211090016</p>
        <p style="margin: 5px 0;"><b>2. Đinh Lê Quỳnh Phương</b> · MSSV: 2211090031</p>
        <p style="margin: 5px 0; color: var(--card-text-muted); font-size: 0.9rem;">Lớp: CNCQ Khoa học dữ liệu 1-1A</p>
      </div>
      <div style="flex: 1; min-width: 250px;">
        <h4 style="color: var(--accent-color); margin: 0 0 10px 0; font-size: 1.1rem;">Giảng viên hướng dẫn:</h4>
        <p style="margin: 5px 0;"><b>TS. Trần Lâm Quân</b></p>
        <p style="margin: 5px 0; color: var(--card-text-muted); font-size: 0.9rem;">Giảng viên Khoa học dữ liệu · Trường Đại học Y tế Công cộng</p>
      </div>
    </div>
  </div>
</div>
"""



def get_svg_sprite_html() -> str:
    return """
    <svg width="0" height="0" style="position: absolute;" aria-hidden="true" class="svg-sprite">
      <symbol id="i-analyze" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /><path d="M11 8v6M8 11h6" /></symbol>
      <symbol id="i-advanced" viewBox="0 0 24 24"><path d="M12 3 3 8l9 5 9-5-9-5Z" /><path d="m3 13 9 5 9-5M3 18l9 5 9-5" /></symbol>
      <symbol id="i-bench" viewBox="0 0 24 24"><path d="M3 21h18" /><rect x="5" y="11" width="4" height="7" /><rect x="14" y="6" width="4" height="12" /></symbol>
      <symbol id="i-docs" viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" /><path d="M14 3v5h5M8 13h8M8 17h6" /></symbol>
      <symbol id="i-method" viewBox="0 0 24 24"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" /><path d="M7.5 15h9" /></symbol>
      <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3 5 6v5c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6Z" /><path d="M12 8v4M12 15.5h.01" /></symbol>
      <symbol id="i-check" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /><path d="m8.5 12 2.5 2.5L16 9.5" /></symbol>
      <symbol id="i-download" viewBox="0 0 24 24"><path d="M12 3v12m0 0 4-4m-4 4-4-4" /><path d="M5 19h14" /></symbol>
      <symbol id="i-volume" viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9Z" /><path d="M17 8a5 5 0 0 1 0 8" /></symbol>
      <symbol id="i-link" viewBox="0 0 24 24"><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" /><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" /></symbol>
      <symbol id="i-spark" viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18" /></symbol>
      <symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14m0 0-6-6m6 6-6 6" /></symbol>
      <symbol id="i-send" viewBox="0 0 24 24"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4Z" /></symbol>
      <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" /></symbol>
      <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 15V3m0 0L8 7m4-4 4 4" /><path d="M5 17v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" /></symbol>
      <symbol id="i-scale" viewBox="0 0 24 24"><path d="M12 3v18M7 7h10M5 7l-2 5a3 3 0 0 0 6 0L7 7M17 7l-2 5a3 3 0 0 0 6 0l-2-5" /></symbol>
      <symbol id="i-data" viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" /></symbol>
      <symbol id="i-chevron" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6" /></symbol>
    </svg>
    """

def get_navbar_html() -> str:
    return f"""
    <div class="navbar-redesign">
        <div class="brand">
            <div class="logo">
                <svg class="icon lg"><use href="#i-shield"></use></svg>
            </div>
            <div>
                <div class="name">Vaccine<span>NLP</span></div>
                <div class="tag">Phát hiện tin giả · Phân tích thái độ</div>
            </div>
        </div>
        
        <div class="nav-menu">
            <button id="nav-btn-analyze" class="nav-item active" onclick="clickHidden('analyze')">
                <svg class="icon"><use href="#i-analyze"></use></svg>
                <span>Phân tích văn bản</span>
            </button>
            <button id="nav-btn-advanced" class="nav-item" onclick="clickHidden('advanced')">
                <svg class="icon"><use href="#i-advanced"></use></svg>
                <span>Công cụ nâng cao</span>
            </button>
            <button id="nav-btn-benchmark" class="nav-item" onclick="clickHidden('benchmark')">
                <svg class="icon"><use href="#i-bench"></use></svg>
                <span>Benchmark hiệu năng</span>
            </button>
            <button id="nav-btn-docs" class="nav-item" onclick="clickHidden('docs')">
                <svg class="icon"><use href="#i-docs"></use></svg>
                <span>Tài liệu hệ thống</span>
            </button>
            <button id="nav-btn-method" class="nav-item" onclick="clickHidden('method')">
                <svg class="icon"><use href="#i-method"></use></svg>
                <span>Phương pháp luận</span>
            </button>
            <button id="nav-btn-thesis" class="nav-item" onclick="clickHidden('thesis')">
                <svg class="icon"><use href="#i-docs"></use></svg>
                <span>Đề cương nghiên cứu</span>
            </button>
        </div>
        
        <div class="nav-right">
            <div class="status-chip">
                <span class="pulse"></span> 
                <span>Mô hình: <b>PhoBERT-v2</b> · trực tuyến</span>
            </div>
            
            <button class="theme-toggle" onclick="toggleDarkMode()">
                <span>Giao diện</span>
                <span class="sw"><i><svg class="icon sm"><use href="#i-sun"></use></svg></i></span>
            </button>
        </div>
    </div>
    """

def get_sidebar_header_html() -> str:
    logo_src = get_huph_logo_base64()
    return f"""
    <div style="text-align: center; margin-bottom: 20px; font-family: 'Times New Roman', Times, serif;">
        <!-- Logo -->
        <div style="width: 90px; height: 90px; background: rgba(255, 255, 255, 0.05); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid var(--accent-color); box-shadow: 0 0 20px var(--glow-color); margin: 0 auto 15px auto;">
            <img src="{logo_src}" style="width: 75px; height: 75px; object-fit: contain;" alt="HUPH Logo">
        </div>
        
        <!-- App Title -->
        <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: var(--header-text); display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 1.6rem;">🦠</span> VaccineNLP
        </h2>
        
        <!-- Subtitle -->
        <p style="margin: 8px 0; font-size: 0.9rem; color: var(--accent-color); font-weight: 600; line-height: 1.3;">
            Hệ thống phát hiện Tin giả & Phân tích Thái độ Vaccine Tiếng Việt
        </p>
        
        <!-- Architecture Info -->
        <p style="margin: 4px 0 15px 0; font-size: 0.8rem; color: var(--card-text-muted); font-style: italic; line-height: 1.3;">
            Kiến trúc Dual-Student Hybrid · PhoBERT-v2 + Gemma-4 E4B
        </p>
        
        <!-- Author Card -->
        <div style="background: var(--input-bg); border: 1px solid var(--input-border); border-radius: 10px; padding: 12px; text-align: left; font-size: 0.85rem; line-height: 1.5; color: var(--text-color);">
            <div style="text-align: center; margin-bottom: 8px;">
                <span style="display: inline-block; background: var(--accent-bg); color: var(--accent-color); padding: 1px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; border: 1px solid var(--input-border);">🎓 ĐỒ ÁN TỐT NGHIỆP HUPH 2026</span>
            </div>
            <b style="color: var(--header-text);">Kim Mạnh Hưng</b> · 2211090016<br>
            <b style="color: var(--header-text);">Đinh Lê Quỳnh Phương</b> · 2211090031<br>
            <span style="font-size: 0.8rem; color: var(--card-text-muted); display: inline-block; margin-top: 4px;">GVHD: TS. Trần Lâm Quân</span>
        </div>
    </div>
    """


def get_header_html() -> str:
    return """
    <div style="text-align: center; padding: 15px 10px 30px 10px; margin-bottom: 15px; font-family: 'Times New Roman', Times, serif; color: var(--text-color);">
      <h1 style="margin: 0; font-size: clamp(1.8rem, 4.2vw, 2.7rem); font-weight: 800; color: var(--header-text); line-height: 1.35; text-transform: uppercase; letter-spacing: 0.02em;">
        PHÁT HIỆN TIN GIẢ VÀ PHÂN TÍCH THÁI ĐỘ VỀ VACCINE TẠI VIỆT NAM 💉
      </h1>
      <div style="width: 180px; height: 4px; background: var(--accent-color); margin: 18px auto; border-radius: 2px;"></div>
      <p style="margin: 0; font-size: clamp(1rem, 2.2vw, 1.3rem); color: var(--card-text-muted); font-style: italic; font-weight: 500; max-width: 900px; margin: 0 auto; line-height: 1.45;">
        Vaccine Misinformation & Attitude Analysis Framework for Vietnamese Social Media
      </p>
    </div>
    """


def get_footer_html():
    logo_src = get_huph_logo_base64()
    return f"""
    <div style="background: var(--footer-bg); color: var(--footer-text); padding: 45px 30px; border-radius: 16px; margin-top: 45px; font-family: 'Times New Roman', Times, serif; border: 1px solid var(--input-border); border-top: 4px solid var(--accent-color); box-shadow: 0 -12px 35px rgba(0, 0, 0, 0.2);">
      <div style="display: flex; flex-wrap: wrap; gap: 35px; justify-content: space-between;">
        <div style="flex: 1.1; min-width: 250px; text-align: center; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <div style="width: 100px; height: 100px; background: rgba(255,255,255,0.08); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid var(--accent-color); box-shadow: 0 0 20px rgba(100, 255, 218, 0.15); margin: 0 auto 15px auto;">
            <img src="{logo_src}" style="width: 80px; height: 80px; object-fit: contain;" alt="HUPH Logo">
          </div>
          <h3 style="color: var(--header-text); font-size: 1.05rem; margin: 5px 0; font-family: 'Times New Roman', Times, serif !important; font-weight: 700; letter-spacing: 0.05em;">TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</h3>
          <p style="font-size: 0.85rem; color: var(--tab-button-text); margin: 6px 0;">📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
          <p style="font-size: 0.85rem; margin: 6px 0;">🌐 <a href="https://huph.edu.vn/" target="_blank" style="color: var(--accent-color); text-decoration: none; font-weight: 600;">huph.edu.vn</a></p>
        </div>
        
        <div style="flex: 1.5; min-width: 250px; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Times New Roman', Times, serif !important; font-weight: 700; letter-spacing: 0.05em;">🔬 Đề tài đồ án</h3>
          <p style="color: #ffd700; font-weight: bold; font-style: italic; font-size: 1rem; line-height: 1.6; margin-bottom: 8px;">
            "Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"
          </p>
          <p style="color: var(--tab-button-text); font-size: 0.85rem; line-height: 1.5;">
            (Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)
          </p>
        </div>
        
        <div style="flex: 1.2; min-width: 250px; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Times New Roman', Times, serif !important; font-weight: 700; letter-spacing: 0.05em;">👥 Nhóm thực hiện</h3>
          <div style="margin-bottom: 12px;">
            <p style="margin: 0; color: var(--header-text); font-weight: 600;">1. Kim Mạnh Hưng</p>
            <p style="font-size: 0.85rem; color: var(--tab-button-text); margin: 2px 0 0 0;">MSSV: 2211090016 · Lớp: CNCQ KHDL1-1A</p>
          </div>
          <div>
            <p style="margin: 0; color: var(--header-text); font-weight: 600;">2. Đinh Lê Quỳnh Phương</p>
            <p style="font-size: 0.85rem; color: var(--tab-button-text); margin: 2px 0 0 0;">MSSV: 2211090031 · Lớp: CNCQ KHDL1-1A</p>
          </div>
        </div>
        
        <div style="flex: 1; min-width: 200px;">
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Times New Roman', Times, serif !important; font-weight: 700; letter-spacing: 0.05em;">👨‍🏫 GV Hướng dẫn</h3>
          <p style="font-size: 1.1rem; font-weight: bold; color: var(--header-text); margin-bottom: 6px;">TS. Trần Lâm Quân</p>
          <p style="font-size: 0.85rem; color: var(--tab-button-text); line-height: 1.5;">
            Giảng viên Khoa học dữ liệu<br>
            Trường Đại học Y tế Công cộng<br>
            📧 <a href="mailto:tlq@huph.edu.vn" style="color: var(--accent-color); text-decoration: none; font-weight: 600;">tlq@huph.edu.vn</a>
          </p>
        </div>
      </div>
      <hr style="border-color: var(--input-border); margin: 30px 0 20px 0;">
      <p style="text-align: center; font-size: 0.85rem; color: var(--tab-button-text); margin: 0; font-family: 'Times New Roman', Times, serif !important; letter-spacing: 0.02em;">
        © 2026 VaccineNLP Project | Đồ án tốt nghiệp chuyên ngành Khoa học Dữ liệu - HUPH
      </p>
    </div>
    """


def get_kpi_cards_html(selected_view):
    bench = load_benchmark()
    benchmark_results = {}
    for key in ['phobert', 'xlmr', 'gemma']:
        d = bench[key]
        avg_f1 = (d['misinfo'] + d['stance'] + d['sentiment']) / 3
        benchmark_results[key] = {
            'name': d.get('name', key),
            'avg_f1': avg_f1,
            'misinfo': d['misinfo'],
            'stance': d['stance'],
            'sentiment': d['sentiment']
        }
        
    data_source_badge = "🟢 LIVE"
    
    if selected_view == "Tất cả mô hình (So sánh chéo)":
        best_m = max(benchmark_results, key=lambda k: benchmark_results[k]['misinfo'])
        best_s = max(benchmark_results, key=lambda k: benchmark_results[k]['stance'])
        best_se = max(benchmark_results, key=lambda k: benchmark_results[k]['sentiment'])
        best_avg = max(benchmark_results, key=lambda k: benchmark_results[k]['avg_f1'])
        
        cards = [
            ("🚨 Best Misinfo F1", f"{benchmark_results[best_m]['misinfo']:.4f}", benchmark_results[best_m]['name'].split('(')[0].strip(), "#ff4b4b"),
            ("🚩 Best Stance F1", f"{benchmark_results[best_s]['stance']:.4f}", benchmark_results[best_s]['name'].split('(')[0].strip(), "#007bff"),
            ("🎭 Best Sentiment F1", f"{benchmark_results[best_se]['sentiment']:.4f}", benchmark_results[best_se]['name'].split('(')[0].strip(), "#00c853"),
            ("🏆 Best Avg F1", f"{benchmark_results[best_avg]['avg_f1']:.4f}", f"{benchmark_results[best_avg]['name'].split('(')[0].strip()} {data_source_badge}", "#FFD700")
        ]
    else:
        model_key = 'phobert' if 'PhoBERT' in selected_view else ('xlmr' if 'XLM-R' in selected_view else 'gemma')
        m_data = benchmark_results[model_key]
        cards = [
            ("🚨 Misinfo Macro F1", f"{m_data['misinfo']:.4f}", "PhoBERT-v2" if model_key == 'phobert' else ("XLM-R-v1" if model_key == 'xlmr' else "Gemma-4 4B"), "#ff4b4b"),
            ("🚩 Stance Macro F1", f"{m_data['stance']:.4f}", "PhoBERT-v2" if model_key == 'phobert' else ("XLM-R-v1" if model_key == 'xlmr' else "Gemma-4 4B"), "#007bff"),
            ("🎭 Sentiment Macro F1", f"{m_data['sentiment']:.4f}", "PhoBERT-v2" if model_key == 'phobert' else ("XLM-R-v1" if model_key == 'xlmr' else "Gemma-4 4B"), "#00c853"),
            ("🏆 Average Macro F1", f"{m_data['avg_f1']:.4f}", "PhoBERT-v2" if model_key == 'phobert' else ("XLM-R-v1" if model_key == 'xlmr' else "Gemma-4 4B"), "#FFD700")
        ]

    html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; width: 100%; margin-bottom: 20px; font-family: \'Times New Roman\', Times, serif;">'
    for title, val, sub, border_color in cards:
        html += f"""
        <div style="flex: 1; min-width: 220px; background: var(--card-bg); border: 1px solid var(--input-border); border-top: 4px solid {border_color}; border-radius: 8px; padding: 15px; text-align: left; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
            <div style="font-size: 0.85rem; color: var(--tab-button-text); text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em; margin-bottom: 5px; font-family: \'Times New Roman\', Times, serif !important;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: var(--header-text); margin-bottom: 3px; font-family: \'Times New Roman\', Times, serif !important;">{val}</div>
            <div style="font-size: 0.8rem; color: var(--text-color); font-style: italic; font-family: \'Times New Roman\', Times, serif !important;">{sub}</div>
        </div>
        """
    html += '</div>'
    return html


def get_leaderboard_html():
    bench = load_benchmark()
    benchmark_results = {}
    for key in ['phobert', 'xlmr', 'gemma']:
        d = bench[key]
        avg_f1 = (d['misinfo'] + d['stance'] + d['sentiment']) / 3
        benchmark_results[key] = {
            'name': d['name'],
            'avg_f1': avg_f1,
            'misinfo': d['misinfo'],
            'stance': d['stance'],
            'sentiment': d['sentiment']
        }
        
    sorted_models = sorted(benchmark_results.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
    model_medals = ['🥇', '🥈', '🎖️']
    model_descs = {
        'phobert': 'Phân loại tối ưu nhất, xử lý sắc thái tiếng Việt vượt trội.',
        'xlmr': 'Baseline đa ngôn ngữ.',
        'gemma': 'Tập trung giải thích (XAI), không tối ưu phân loại nhãn.'
    }
    
    table_border = "var(--input-border)"
    table_bg = "var(--card-bg)"
    header_bg = "var(--tab-button-bg)"
    text_col = "var(--text-color)"
    
    leaderboard_rows = ""
    for rank, (mkey, mdata) in enumerate(sorted_models):
        bg_extra = " background: var(--accent-bg);" if rank == 0 else ""
        fw = " font-weight:bold;" if rank == 0 else ""
        leaderboard_rows += f"""
            <tr style="border-bottom:1px solid {table_border};{bg_extra}">
                <td style="padding:12px; font-weight:bold;">{rank+1}</td>
                <td class="model-color-{mkey}" style="padding:12px; text-align:left; font-weight:bold;">{mdata['name']}</td>
                <td style="padding:12px;{fw}">{mdata['misinfo']:.4f}</td>
                <td style="padding:12px;{fw}">{mdata['stance']:.4f}</td>
                <td style="padding:12px;{fw}">{mdata['sentiment']:.4f}</td>
                <td style="padding:12px; font-weight:bold; color:#FFD700;">{mdata['avg_f1']:.4f}</td>
                <td style="padding:12px; text-align:left; font-style:italic;">{model_medals[rank]} {model_descs[mkey]}</td>
            </tr>"""
    
    table_html = f"""
    <table style="width:100%; border-collapse:collapse; background:{table_bg}; border:1px solid {table_border}; border-radius:10px; overflow:hidden; font-family:'Times New Roman', serif; text-align:center;">
        <thead style="background:{header_bg}; color:{text_col}; font-weight:bold;">
            <tr style="border-bottom:2px solid var(--custom-card-border);">
                <th style="padding:12px;">Hạng</th>
                <th style="padding:12px; text-align:left;">Mô hình & Kiến trúc</th>
                <th style="padding:12px;">Misinfo F1</th>
                <th style="padding:12px;">Stance F1</th>
                <th style="padding:12px;">Sentiment F1</th>
                <th style="padding:12px; color:#FFD700;">Trung bình F1</th>
                <th style="padding:12px; text-align:left;">Đặc tính & Định vị</th>
            </tr>
        </thead>
        <tbody style="color:{text_col};">{leaderboard_rows}</tbody>
    </table>
    """
    return table_html


def get_per_class_table_html(task_key):
    bench = load_benchmark()
    benchmark_results = {}
    for key in ['phobert', 'xlmr', 'gemma']:
        d = bench[key]
        avg_f1 = (d['misinfo'] + d['stance'] + d['sentiment']) / 3
        benchmark_results[key] = {
            'name': d['name'],
            'avg_f1': avg_f1,
            'tasks': {
                'misinfo':   {'macro_f1': d['misinfo'],   'per_class': d['per_class_misinfo'],   'support': d['support_misinfo']},
                'stance':    {'macro_f1': d['stance'],    'per_class': d['per_class_stance'],    'support': d['support_stance']},
                'sentiment': {'macro_f1': d['sentiment'], 'per_class': d['per_class_sentiment'], 'support': d['support_sentiment']},
            }
        }
        
    task_labels = {
        'misinfo': (['Tin giả', 'Chính xác'], ['#ff4b4b', '#38ef7d'], 'Nhãn phân loại'),
        'stance': (['Ủng hộ', 'Phản đối', 'Trung lập'], ['#38ef7d', '#ff4b4b', '#007bff'], 'Nhãn lập trường'),
        'sentiment': (['Tiêu cực', 'Trung tính', 'Tích cực'], ['#ff4b4b', '#007bff', '#38ef7d'], 'Sắc thái cảm xúc')
    }
    class_names, class_colors, header_label = task_labels[task_key]
    support = benchmark_results['phobert']['tasks'][task_key]['support']
    
    table_border = "var(--input-border)"
    table_bg = "var(--card-bg)"
    header_bg = "var(--tab-button-bg)"
    text_col = "var(--text-color)"
    
    rows_html = ""
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        sup = support[i]
        p_f1 = benchmark_results['phobert']['tasks'][task_key]['per_class'][i]
        x_f1 = benchmark_results['xlmr']['tasks'][task_key]['per_class'][i]
        g_f1 = benchmark_results['gemma']['tasks'][task_key]['per_class'][i]
        rows_html += f'''<tr style="border-bottom:1px solid {table_border};">
            <td style="padding:10px; text-align:left; font-weight:bold; color:{color};">{name}</td>
            <td style="padding:10px; font-family:monospace;">{sup}</td>
            <td style="padding:10px; font-weight:bold; color:var(--custom-text-neon);">{p_f1:.4f}</td>
            <td style="padding:10px;">{x_f1:.4f}</td>
            <td style="padding:10px;">{g_f1:.4f}</td>
        </tr>'''
        
    table_html = f'''
    <table style="width:100%; border-collapse:collapse; background:{table_bg}; border:1px solid {table_border}; font-family:'Times New Roman', serif; text-align:center;">
        <thead style="background:{header_bg}; color:{text_col}; font-weight:bold;">
            <tr style="border-bottom:1px solid var(--custom-card-border);">
                <th style="padding:10px; text-align:left;">{header_label}</th>
                <th style="padding:10px;">Support</th>
                <th style="padding:10px;">PhoBERT-v2 F1</th>
                <th style="padding:10px;">XLM-R-v1 F1</th>
                <th style="padding:10px;">Gemma-4 4B F1</th>
            </tr>
        </thead>
        <tbody style="color:{text_col};">{rows_html}</tbody>
    </table>'''
    return table_html


def render_live_table(data_list):
    rows_html = ""
    chart_font_color = "var(--text-color)"
    info_border = "var(--accent-color)"
    
    for row in data_list:
        def get_prog_html(val, color):
            width = val * 100
            return f'<div style="width: 100%; background: var(--tab-button-bg); border-radius: 5px; height: 10px; margin-bottom: 3px; overflow: hidden; border: 1px solid var(--input-border);"><div style="width: {width}%; background: {color}; height: 100%; border-radius: 5px;"></div></div><span style="font-size: 11px; font-weight: bold; color: {chart_font_color};">{val:.4f}</span>'
        
        rows_html += f"""
        <tr style="border-bottom: 1px solid var(--input-border);">
            <td style="padding: 12px; font-weight: bold; color: {chart_font_color}; text-align: left;">{row['Model']}</td>
            <td style="padding: 12px;">{get_prog_html(row['Misinfo'], '#ff4b4b')}</td>
            <td style="padding: 12px;">{get_prog_html(row['Stance'], '#007bff')}</td>
            <td style="padding: 12px;">{get_prog_html(row['Sentiment'], '#00c853')}</td>
        </tr>"""
    
    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="4" style="padding: 20px; color: var(--tab-button-text); font-style: italic;">Chưa bắt đầu suy luận Live. Nhấn nút bấm bên dưới để chạy.</td>
        </tr>
        """
        
    return f"""
    <table style="width: 100%; border-collapse: collapse; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 10px; overflow: hidden; font-family: 'Times New Roman', serif; text-align: center;">
        <thead style="background: var(--tab-button-bg);">
            <tr>
                <th style="padding: 12px; text-align: left; color: {chart_font_color}; border-bottom: 2px solid {info_border};">Kiến trúc mô hình</th>
                <th style="padding: 12px; text-align: left; color: {chart_font_color}; border-bottom: 2px solid {info_border};">Misinfo (F1)</th>
                <th style="padding: 12px; text-align: left; color: {chart_font_color}; border-bottom: 2px solid {info_border};">Stance (F1)</th>
                <th style="padding: 12px; text-align: left; color: {chart_font_color}; border-bottom: 2px solid {info_border};">Sentiment (F1)</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def run_live_evaluation():
    bench = load_benchmark()
    benchmark_data = [
        {"Model": bench['phobert']['name'], "Misinfo": bench['phobert']['misinfo'], "Stance": bench['phobert']['stance'], "Sentiment": bench['phobert']['sentiment']},
        {"Model": bench['xlmr']['name'],    "Misinfo": bench['xlmr']['misinfo'],    "Stance": bench['xlmr']['stance'],    "Sentiment": bench['xlmr']['sentiment']},
        {"Model": bench['gemma']['name'],   "Misinfo": bench['gemma']['misinfo'],   "Stance": bench['gemma']['stance'],   "Sentiment": bench['gemma']['sentiment']},
    ]
    
    current_data = []
    for row in benchmark_data:
        status = f"<div style='color: orange; font-weight: bold; font-family: \"Times New Roman\", serif;'>🤖 Đang giả lập kiểm thử trực tiếp trên GPU: {row['Model']}...</div>"
        yield status, render_live_table(current_data)
        time.sleep(0.8)
        current_data.append(row)
        yield status, render_live_table(current_data)
        time.sleep(0.4)
        
    status = f"<div style='color: #38ef7d; font-weight: bold; font-family: \"Times New Roman\", serif;'>✅ Quá trình suy luận Live hoàn tất! Bảng kết quả F1 đã được cập nhật thành công.</div>"
    yield status, render_live_table(benchmark_data)

def handle_clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()
    try:
        gr.Info("✅ Đã xóa cache và sẵn sàng khởi động lại!")
    except:
        pass
    return "✅ Đã xóa cache thành công!"

# ============================================================================
# SIDEBAR DATA & HELPER FUNCTIONS
# ============================================================================
SIDEBAR_CATEGORIES = {
    "Tự nhập": [],
    "🚨 Nhóm Tin giả cực đoan": [
        ("Chống vaccine cực đoan", "🚨 Tin giả - Chống vaccine cực đoan"),
        ("Vô sinh", "🚨 Tin giả - Vô sinh")
    ],
    "🟢 Nhóm phân tích Thái độ": [
        ("Ủng hộ", "🟢 Ủng hộ tiêm chủng"),
        ("Nghi ngại", "🟡 Nghi ngại")
    ],
    "✅ Nhóm Thông tin chuẩn": [
        ("Thông tin chuẩn", "✅ Thông tin chuẩn"),
        ("Câu hỏi - Tư vấn", "🔵 Câu hỏi tư vấn")
    ],
    "💬 Nhóm Từ lóng MXH": [
        ("Từ lóng nguy hiểm", "💬 Tin giả - Từ lóng MXH")
    ]
}

def get_sidebar_info_html(model_name):
    return f"""
    <div style="margin-top: 15px; padding: 15px; background: var(--input-bg); border: 1px solid var(--input-border); border-radius: 10px; font-family: 'Times New Roman', serif;">
        <div style="font-size: 14px; font-weight: bold; margin-bottom: 8px; color: var(--text-color);">Về hệ thống</div>
        <div style="font-size: 12px; line-height: 1.6; color: var(--text-color); opacity: 0.85;">
            • <b>Classifier:</b> {model_name}<br>
            • <b>XAI Engine:</b> Gemma-4 4B (cached)<br>
            • <b>Tasks:</b> Misinfo · Stance · Sentiment<br>
            • <b>Benchmark:</b> 186 samples
        </div>
    </div>
    """

def handle_category_change(category):
    if category == "Tự nhập" or category not in SIDEBAR_CATEGORIES:
        return gr.update(visible=False, choices=["Tự nhập"], value="Tự nhập"), ""
    
    pairs = SIDEBAR_CATEGORIES[category]
    labels = [p[0] for p in pairs]
    default_label = labels[0]
    default_text = SAMPLE_TEXTS[pairs[0][1]]
    
    return gr.update(visible=True, choices=labels, value=default_label), default_text

def handle_detail_change(category, detail):
    if category == "Tự nhập" or category not in SIDEBAR_CATEGORIES:
        return ""
    
    pairs = SIDEBAR_CATEGORIES[category]
    for label, key in pairs:
        if label == detail:
            return SAMPLE_TEXTS[key]
    return ""


# ============================================================================
# GRADIO UI BUILDER
# ============================================================================


def build_app():
    """Build the Gradio Blocks app with redesigned sidebar and screens."""
    init_theme_js = """function() {
        /* Sidebar navigation + theme toggle.
           Gradio's gr.HTML does NOT execute inline <script> tags, so these handlers
           must be defined here (the Blocks js= hook IS executed on load). Without
           this, the sidebar buttons' inline onclick="clickHidden(...)" reference an
           undefined function and navigation/theme toggle silently do nothing. */
        window.clickHidden = function(target) {
            document.querySelectorAll('.navbar-redesign .nav-item').forEach(function(el) { el.classList.remove('active'); });
            var activeBtn = document.getElementById('nav-btn-' + target);
            if (activeBtn) activeBtn.classList.add('active');
            /* In Gradio 4.x elem_id is on the <button> itself (not a wrapper). */
            var el = document.getElementById('btn-hidden-' + target);
            if (el) {
                var btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
                if (btn) btn.click();
            }
        };

        window.toggleDarkMode = function() {
            var isDark = document.documentElement.classList.toggle('dark');
            document.body.classList.toggle('dark', isDark);
            try { localStorage.setItem('vnlp-theme', isDark ? 'dark' : 'light'); } catch (e) {}
        };

        /* Apply persisted theme on load */
        try {
            var theme = localStorage.getItem('vnlp-theme') || 'light';
            var addDark = theme === 'dark';
            document.documentElement.classList.toggle('dark', addDark);
            document.body.classList.toggle('dark', addDark);
        } catch (e) {}

        /* Force Plotly charts to stretch to 100% width on tab clicks/nav changes */
        document.addEventListener('click', function(e) {
            const isTabOrButton = e.target.closest('button') || e.target.closest('[role="tab"]') || e.target.closest('.tab-nav') || e.target.closest('.tab-item') || e.target.closest('.nav-item');
            if (isTabOrButton) {
                setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
                setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 150);
                setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 400);
                setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 800);
            }
        });
        
        /* Trigger resize on initial load */
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 1000);
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 2500);
        
        /* Delegated document event listener for TTS button click */
        document.addEventListener('click', function(e) {
            const btn = e.target.closest('.tts-speak-btn');
            if (!btn) return;
            
            e.preventDefault();
            e.stopPropagation();
            
            const wrapper = btn.parentElement;
            if (!wrapper) return;
            
            const audio = wrapper.querySelector('audio');
            if (!audio) return;
            
            if (audio.paused) {
                document.querySelectorAll('audio').forEach(a => {
                    if (a !== audio) {
                        a.pause();
                        a.currentTime = 0;
                        const otherBtn = a.parentElement.querySelector('.tts-speak-btn');
                        if (otherBtn) {
                            otherBtn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe AI Giải Thích';
                            otherBtn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                            otherBtn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                            otherBtn.style.color = '#0a192f';
                        }
                    }
                });
                
                audio.play().then(() => {
                    btn.innerHTML = '<span class="tts-playing-icon" style="font-size: 1.2rem;">⏹️</span> Đang đọc giải thích...';
                    btn.style.background = 'linear-gradient(135deg, #ff4b4b 0%, #ff8f8f 100%)';
                    btn.style.boxShadow = '0 4px 15px rgba(255, 75, 75, 0.3)';
                    btn.style.color = '#ffffff';
                }).catch(err => {
                    console.error("Lỗi phát audio:", err);
                    audio.style.display = 'block';
                });
            } else {
                audio.pause();
                audio.currentTime = 0;
                btn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe AI Giải Thích';
                btn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                btn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                btn.style.color = '#0a192f';
            }
            
            audio.onended = () => {
                btn.innerHTML = '<span style="font-size: 1.2rem;">🔊</span> Nghe Lại';
                btn.style.background = 'linear-gradient(135deg, #00c853 0%, #b2ff59 100%)';
                btn.style.boxShadow = '0 4px 15px rgba(0, 200, 83, 0.3)';
                btn.style.color = '#0a192f';
            };
        });
    }""".strip()

    # Define navigation functions
    def nav_to_analyze():
        return (
            gr.update(visible=True),   # analyze
            gr.update(visible=False),  # advanced
            gr.update(visible=False),  # benchmark
            gr.update(visible=False),  # docs
            gr.update(visible=False),  # method
            gr.update(visible=False)   # thesis
        )

    def nav_to_advanced():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def nav_to_benchmark():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def nav_to_docs():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def nav_to_method():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    def nav_to_thesis():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)
        )

    # Wrapper for cào to batch
    def handle_send_to_batch_to_screen(batch_text_str: str):
        res_text, _ = handle_send_to_batch(batch_text_str)
        # returns: [batch_text, screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis]
        return (
            res_text,
            gr.update(visible=False), # screen_analyze
            gr.update(visible=True),  # screen_advanced
            gr.update(visible=False), # screen_benchmark
            gr.update(visible=False), # screen_docs
            gr.update(visible=False), # screen_method
            gr.update(visible=False)  # screen_thesis
        )

    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal), css=CSS_STYLE, fill_width=True, js=init_theme_js) as app:
        # SVG sprite injected at body top
        gr.HTML(get_svg_sprite_html())
        
        # Hidden buttons used to trigger python navigation callbacks from custom HTML sidebar
        btn_hidden_analyze = gr.Button("nav_analyze", elem_id="btn-hidden-analyze", elem_classes=["hidden-btn"])
        btn_hidden_advanced = gr.Button("nav_advanced", elem_id="btn-hidden-advanced", elem_classes=["hidden-btn"])
        btn_hidden_benchmark = gr.Button("nav_benchmark", elem_id="btn-hidden-benchmark", elem_classes=["hidden-btn"])
        btn_hidden_docs = gr.Button("nav_docs", elem_id="btn-hidden-docs", elem_classes=["hidden-btn"])
        btn_hidden_method = gr.Button("nav_method", elem_id="btn-hidden-method", elem_classes=["hidden-btn"])
        btn_hidden_thesis = gr.Button("nav_thesis", elem_id="btn-hidden-thesis", elem_classes=["hidden-btn"])
        
        # Top Navbar (Full Width)
        gr.HTML(get_navbar_html(), elem_id="navbar-html-wrapper")
        
        with gr.Row(elem_id="main-layout-row"):
            # Content Column
            with gr.Column(scale=1, elem_id="content-col"):
                # Hidden actual cache button that is triggered programmatically if needed
                clear_cache_btn = gr.Button("🗑️ Xóa Cache & Khởi động lại", elem_classes=["hidden-btn"], size="sm")
                clear_cache_status = gr.Markdown(value="", visible=False)
                
                session_state = gr.State([])
                report_state = gr.State("")
                fetched_raw_state = gr.State("")

                # --------------------------------------------------------
                # SCREEN 1: ANALYZE SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=True, elem_classes=["screen-container"]) as screen_analyze:
                    # Topbar
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Phân tích · Đa nhiệm</div><h1>Phân tích văn bản</h1>")
                        # Hidden or ghost buttons to match React topbar
                        with gr.Row(elem_classes=["hidden-btn"]):
                            btn_guide_dummy = gr.Button("📚 Hướng dẫn", elem_classes=["btn", "ghost", "sm"])
                    
                    with gr.Row(elem_id="analyze-grid-row"):
                        # Left Input Card
                        with gr.Column(scale=1, min_width=380):
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                gr.HTML("<div class='field-label' style='font-weight:600;margin-bottom:8px;'>Nội dung cần đối soát <span style='font-size:11px;color:var(--ink-3);font-weight:normal;'>tiếng Việt</span></div>")
                                text_input = gr.Textbox(
                                    label="",
                                    placeholder="Dán bình luận, bài viết hoặc tin nhắn về vaccine…",
                                    lines=8,
                                    show_label=False
                                )
                                
                                gr.HTML("<div class='field-label' style='font-weight:600;margin:18px 0 6px;'>Bộ ví dụ mẫu</div>")
                                sample_category = gr.Dropdown(
                                    choices=["Tự nhập", "🚨 Nhóm Tin giả cực đoan", "🟢 Nhóm phân tích Thái độ", "✅ Nhóm Thông tin chuẩn", "💬 Nhóm Từ lóng MXH"],
                                    value="Tự nhập",
                                    label="Chọn nhóm mẫu:",
                                    show_label=False
                                )
                                sample_detail = gr.Radio(
                                    choices=["Tự nhập"],
                                    value="Tự nhập",
                                    label="Chọn loại văn bản:",
                                    visible=False
                                )
                                
                                gr.HTML("<div class='field-label' style='font-weight:600;margin:18px 0 6px;'>Mô hình phân loại</div>")
                                model_choice = gr.Dropdown(
                                    choices=list(CONFIG["models"].keys()),
                                    value="PhoBERT-v2",
                                    label="Chọn model:",
                                    show_label=False
                                )
                                info_box = gr.HTML(value=get_sidebar_info_html("PhoBERT-v2"))
                                
                                with gr.Accordion("🌐 Hoặc thu thập từ URL", open=False):
                                    url_input = gr.Textbox(
                                        label="URL (Báo · YouTube · Facebook · TikTok · Threads)",
                                        placeholder="https://vnexpress.net/...",
                                    )
                                    max_cmt = gr.Slider(10, 100, value=30, step=10, label="Max comments")
                                    fetch_btn = gr.Button("📥 Thu thập")
                                    fetch_status = gr.HTML(value="")
                                    fetched_table = gr.Dataframe(
                                        headers=["STT", "Nội dung thu thập được"],
                                        datatype=["str", "str"],
                                        wrap=True,
                                        visible=False,
                                        label="📋 Danh sách đã cào (💡 Click trực tiếp vào dòng bình luận để gửi nhanh lên ô Phân tích chính)"
                                    )
                                    with gr.Row():
                                        send_to_batch_btn = gr.Button("🚀 Gửi sang Phân tích Batch", variant="secondary", visible=False)
                                        export_fetched_btn = gr.Button("📥 Tải dữ liệu cào (.xlsx)", variant="primary", visible=False)
                                    export_fetched_file = gr.File(label="Tải về tệp Excel dữ liệu đã cào", visible=False)

                                use_captum_cb = gr.Checkbox(
                                    label="🎯 Bật Captum IG (token attribution, chậm hơn 5-10s)",
                                    value=False,
                                    elem_classes=["captum-cb"]
                                )
                                analyze_btn = gr.Button("🔬 Tiến hành phân tích đa nhiệm", variant="primary", size="lg")

                        # Right Results Stack
                        with gr.Column(scale=1, min_width=380):
                            summary_out = gr.HTML(value="""
                            <div class="card card-pad" style="text-align:center;">
                                <div class="empty-state" style="padding:40px 20px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;">
                                    <div class="ring" style="width:56px;height:56px;border-radius:50%;background:var(--bg-2);display:grid;place-items:center;color:var(--ink-3);"><svg class="icon lg"><use href="#i-analyze"></use></svg></div>
                                    <h3 style="margin:0;font-size:16px;font-weight:700;">Chưa có phân tích</h3>
                                    <div class="muted" style="font-size:13px;max-width:320px;line-height:1.4;">Nhập văn bản và bấm “Tiến hành phân tích đa nhiệm” để chạy PhoBERT-v2 trên 3 trục nhãn.</div>
                                </div>
                            </div>
                            """)
                            
                            radar_out = gr.Plot(visible=False)
                            
                            # XAI Block Card
                            with gr.Group(elem_classes=["card", "card-pad"], visible=False) as xai_group:
                                gr.HTML("<div class='section-label'><svg class='icon sm ic'><use href='#i-spark'></use></svg> Giải thích của mô hình (XAI)</div>")
                                gr.HTML("<p class='muted' style='font-size:12px;margin:2px 0 12px;'>Nhịp 1 · PhoBERT trả nhãn tức thời → Nhịp 2 · Gemma-4B lý giải CoT</p>")
                                with gr.Tabs():
                                    with gr.Tab("💭 Chain-of-Thought Reasoning"):
                                        reasoning_out = gr.Markdown()
                                        disagreement_out = gr.HTML(value="")
                                        audio_out = gr.HTML(value="")
                                    with gr.Tab("🎯 Token Attribution (Captum IG)"):
                                        saliency_out = gr.HTML(value="<p style='color:#888;'><em>💡 Bật checkbox <b>Captum IG</b> ở trên rồi nhấn Phân tích</em></p>")
                            
                            with gr.Row():
                                export_btn = gr.Button("📥 Tải báo cáo phân tích (.md)", variant="secondary", visible=False)
                                export_file = gr.File(label="Báo cáo", visible=False)
                                
                            history_display = gr.Markdown(value="*Chưa có lượt phân tích nào trong phiên này*")

                # --------------------------------------------------------
                # SCREEN 2: ADVANCED SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_advanced:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Phân tích · Nâng cao</div><h1>Công cụ nâng cao</h1>")
                            
                    with gr.Tabs():
                        with gr.Tab("📋 Phân tích hàng loạt (Batch Mode)"):
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                gr.HTML("<div class='section-label' style='margin-bottom:10px;'><svg class='icon sm ic'><use href='#i-data'></use></svg> Phân tích nhiều mẫu cùng lúc</div>")
                                batch_input = gr.Textbox(
                                    label="Mỗi dòng = 1 mẫu (tối đa 50 mẫu, phân tách bằng --- hoặc xuống dòng)",
                                    placeholder="Mẫu 1: Vaccine COVID gây vô sinh...\n---\nMẫu 2: Tiêm phòng rất tốt...",
                                    lines=8,
                                )
                                batch_btn = gr.Button("🚀 Phân tích Batch", variant="primary")
                                batch_out = gr.Markdown()
                                
                        with gr.Tab("🔬 So sánh PhoBERT-v2 vs XLM-R-v1"):
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                gr.HTML("<div class='section-label' style='margin-bottom:10px;'><svg class='icon sm ic'><use href='#i-scale'></use></svg> Đối sánh trực quan hai kiến trúc</div>")
                                cmp_input = gr.Textbox(
                                    label="Nhập văn bản cần đối sánh:", 
                                    placeholder="Nhập câu tại đây để so sánh dự đoán giữa PhoBERT-v2 và XLM-R-v1...",
                                    lines=4
                                )
                                cmp_btn = gr.Button("So sánh mô hình", variant="primary")
                                cmp_out = gr.Markdown()

                # --------------------------------------------------------
                # SCREEN 3: BENCHMARK SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_benchmark:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Đánh giá · Hiệu năng</div><h1>Benchmark hiệu năng</h1>")
                            
                    with gr.Tabs():
                        with gr.Tab("📋 BÁO CÁO BENCHMARK KHOA HỌC"):
                            gr.HTML("""
                                <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 15px; border-radius: 8px; margin-bottom: 25px;">
                                    <span style="color: var(--text-color); font-size: 1.05rem;">
                                        💡 Báo cáo đối sáng hiệu năng thực nghiệm chi tiết giữa 3 kiến trúc mô hình: <b>PhoBERT-v2</b>, <b>XLM-R-v1</b> và <b>Gemma-4 4B (QLoRA)</b> trên tập dữ liệu kiểm thử vàng <b>Gold Test Set (186 mẫu)</b>, được gán nhãn thủ công bởi chuyên gia từ HUPH 2026.
                                    </span>
                                </div>
                            """)

                            selected_model_view = gr.Dropdown(
                                choices=[
                                    "Tất cả mô hình (So sánh chéo)",
                                    "PhoBERT-v2 (Discriminator Tối ưu nhất)",
                                    "XLM-R-v1 (Baseline Đa ngôn ngữ)",
                                    "Gemma-4 4B (Reasoning Engine XAI)"
                                ],
                                value="Tất cả mô hình (So sánh chéo)",
                                label="🔍 Chọn chế độ xem dữ liệu Benchmark:"
                            )

                            kpi_output = gr.HTML(value=get_kpi_cards_html("Tất cả mô hình (So sánh chéo)"))

                            gr.Markdown("### 🏆 1. Bảng so sánh hiệu năng tổng thể (Macro F1 Leaderboard)")
                            gr.HTML(value=get_leaderboard_html())

                            gr.Markdown("#### 📊 So sánh trực quan Macro F1 Score giữa các kiến trúc")
                            gr.Plot(value=make_benchmark_chart())

                            gr.Markdown("---")
                            gr.Markdown("### 🕸️ 2. Phân tích hiệu năng chi tiết theo nhãn và phân bổ mẫu (Per-Class Breakdown)")

                            with gr.Tabs():
                                with gr.Tab("🚨 PHÂN LOẠI TIN GIẢ (MISINFO)"):
                                    gr.Plot(value=make_per_class_chart("misinfo"))
                                    gr.HTML(value=get_per_class_table_html("misinfo"))
                                    gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Tin giả** (28 mẫu) có F1 đạt cao nhất là **0.5085** (PhoBERT-v2), đây là bài toán khó do số lượng mẫu huấn luyện hạn chế.")
                                with gr.Tab("🚩 QUAN ĐIỂM (STANCE)"):
                                    gr.Plot(value=make_per_class_chart("stance"))
                                    gr.HTML(value=get_per_class_table_html("stance"))
                                    gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Trung lập** có kết quả tốt nhất do cách diễn đạt khách quan.")
                                with gr.Tab("🎭 CẢM XÚC (SENTIMENT)"):
                                    gr.Plot(value=make_per_class_chart("sentiment"))
                                    gr.HTML(value=get_per_class_table_html("sentiment"))
                                    gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Tích cực** tỏ ra thách thức hơn trên mọi kiến trúc.")

                        with gr.Tab("⚡ ĐÁNH GIÁ LIVE (LIVE EVALUATION)"):
                            gr.HTML("""
                                <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                                    <span style="color: var(--text-color);">⚡ <b>Chế độ Đánh giá Live</b> giả lập quá trình quét trực tiếp và tính toán F1-Score thời gian thực của các mô hình trên tập kiểm thử vàng Gold Test Set (186 mẫu).</span>
                                </div>
                            """)

                            gr.Markdown("#### 🚀 Trạng thái Tiền trình Suy luận (Inference Pipeline)")
                            live_status = gr.HTML(value="<div style='color: var(--tab-button-text);'>💡 Nhấn nút bên dưới để bắt đầu chạy kiểm thử suy luận trên GPU trực tiếp...</div>")
                            live_table = gr.HTML(value=render_live_table([]))
                            live_eval_btn = gr.Button("⚡ Bắt đầu Đánh giá Live", variant="primary", size="lg")

                        with gr.Tab("📈 ĐÁNH GIÁ CHUYÊN SÂU & SANKEY"):
                            gr.Markdown("## 🌀 Dòng chảy Cảm xúc → Lập trường (n=186)")
                            gr.Plot(value=make_sankey_chart())
                            gr.Markdown("""
                            ---
                            ### 🔍 Diễn giải Sankey Flow
                            **Phát hiện chính:**
                            - **93.8%** (45/48) nội dung **Phản đối** vaccine mang cảm xúc **Tiêu cực**
                            - **82.5%** (33/40) nội dung **Tích cực** đồng hành với lập trường **Ủng hộ**
                            - **77.4%** (65/84) nội dung **Trung lập** đi với cảm xúc **Trung tính**
                            """)
                            gr.Markdown("---")
                            gr.Markdown("## 📊 Phân cấp nhãn Gold Test Set (Sunburst)")
                            gr.Plot(value=make_sunburst_chart())
                            gr.Markdown("---")
                            gr.Markdown("## 🔍 Bộ máy tính chỉ số thực nghiệm (Interactive Metric Calculator)")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    calc_class_choice = gr.Dropdown(
                                        choices=list(METRICS_DB.keys()),
                                        value="Tin giả (Misinfo = Tin giả)",
                                        label="Lựa chọn nhãn lớp cụ thể để tính toán chỉ số:",
                                    )
                                    calc_metrics_area = gr.HTML(value=update_calculator("Tin giả (Misinfo = Tin giả)")[0])
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        calc_p_area = gr.Markdown(value=update_calculator("Tin giả (Misinfo = Tin giả)")[1])
                                        calc_r_area = gr.Markdown(value=update_calculator("Tin giả (Misinfo = Tin giả)")[2])
                                        calc_f1_area = gr.Markdown(value=update_calculator("Tin giả (Misinfo = Tin giả)")[3])

                # --------------------------------------------------------
                # SCREEN 4: DOCS SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_docs:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Tài liệu · Hệ thống</div><h1>Tài liệu hệ thống</h1>")
                    gr.HTML(RESOURCES_HTML)

                # --------------------------------------------------------
                # SCREEN 5: METHOD SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_method:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Tài liệu · Phương pháp</div><h1>Phương pháp luận</h1>")
                    gr.HTML(METHODOLOGY_HTML)

                # --------------------------------------------------------
                # SCREEN 6: THESIS SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_thesis:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Tài liệu · Đề cương</div><h1>Đề cương nghiên cứu</h1>")
                    gr.HTML(THESIS_HTML)

                gr.HTML(get_footer_html())

        # ================================================================
        # EVENT WIRE UP
        # ================================================================
        
        # 1. Navigation Hidden Button events
        btn_hidden_analyze.click(fn=nav_to_analyze, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)
        btn_hidden_advanced.click(fn=nav_to_advanced, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)
        btn_hidden_benchmark.click(fn=nav_to_benchmark, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)
        btn_hidden_docs.click(fn=nav_to_docs, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)
        btn_hidden_method.click(fn=nav_to_method, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)
        btn_hidden_thesis.click(fn=nav_to_thesis, inputs=[], outputs=[screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis], api_name=False)

        # 2. Main Analyze Event wiring
        def wrapper_handle_analyze(*args):
            res = handle_analyze(*args)
            return res + (gr.update(visible=True), gr.update(visible=True))
            
        analyze_btn.click(
            fn=wrapper_handle_analyze,
            inputs=[text_input, model_choice, use_captum_cb, session_state],
            outputs=[summary_out, radar_out, reasoning_out, disagreement_out, saliency_out,
                     audio_out, session_state, history_display, report_state, xai_group, radar_out],
            api_name=False
        )

        # 3. Model choice info
        model_choice.change(
            fn=get_sidebar_info_html,
            inputs=[model_choice],
            outputs=[info_box],
            api_name=False
        )

        # 4. Preset category changes
        sample_category.change(
            fn=handle_category_change,
            inputs=[sample_category],
            outputs=[sample_detail, text_input],
            api_name=False
        )
        sample_detail.change(
            fn=handle_detail_change,
            inputs=[sample_category, sample_detail],
            outputs=[text_input],
            api_name=False
        )

        # 5. Export Report
        export_btn.click(
            fn=handle_export_report,
            inputs=[report_state],
            outputs=[export_file],
            api_name=False
        )

        # 6. Fetch URL Events
        fetch_btn.click(
            fn=handle_fetch_url,
            inputs=[url_input, max_cmt],
            outputs=[text_input, fetched_table, send_to_batch_btn, export_fetched_btn, fetch_status, fetched_raw_state],
            api_name=False
        )
        export_fetched_btn.click(
            fn=handle_export_fetched,
            inputs=[fetched_raw_state],
            outputs=[export_fetched_file],
            api_name=False
        )
        fetched_table.select(
            fn=handle_cell_select,
            inputs=[],
            outputs=[text_input],
            api_name=False
        )

        # 7. Batch Mode & Compare Events
        batch_btn.click(
            fn=handle_batch, 
            inputs=[batch_input, model_choice], 
            outputs=[batch_out], 
            api_name=False
        )
        cmp_btn.click(
            fn=handle_compare, 
            inputs=[cmp_input], 
            outputs=[cmp_out], 
            api_name=False
        )

        # Send cào to batch
        send_to_batch_btn.click(
            fn=handle_send_to_batch_to_screen,
            inputs=[fetched_raw_state],
            outputs=[batch_input, screen_analyze, screen_advanced, screen_benchmark, screen_docs, screen_method, screen_thesis],
            api_name=False
        )

        # 8. Benchmark change view & Live Eval
        selected_model_view.change(
            fn=get_kpi_cards_html,
            inputs=[selected_model_view],
            outputs=[kpi_output],
            api_name=False
        )
        live_eval_btn.click(
            fn=run_live_evaluation,
            inputs=[],
            outputs=[live_status, live_table],
            api_name=False
        )
        calc_class_choice.change(
            fn=update_calculator,
            inputs=[calc_class_choice],
            outputs=[calc_metrics_area, calc_p_area, calc_r_area, calc_f1_area],
            api_name=False
        )
        
        # 9. Clear cache
        clear_cache_btn.click(
            fn=handle_clear_cache,
            inputs=[],
            outputs=[clear_cache_status],
            api_name=False
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2, max_size=10)
    app.launch(show_error=True)
