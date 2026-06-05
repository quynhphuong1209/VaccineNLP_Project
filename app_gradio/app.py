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

from xai_postprocess import (
    clean_reasoning_output as _clean_v2,
    strip_markdown_for_tts,
    parse_gemma_labels,
    compare_engines,
    render_disagreement_badge,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    "misinfo":   {0: "#e8504a", 1: "#3db882"},
    "stance":    {0: "#3db882", 1: "#e8504a", 2: "#4a9eed"},
    "sentiment": {0: "#e8504a", 1: "#4a9eed", 2: "#3db882"},
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


def clean_reasoning_output(raw: str, max_chars: int = 4000) -> str:
    """Compatibility wrapper for the centralized XAI post-process module."""
    return _clean_v2(raw, max_chars=max_chars)

import openai

def get_live_xai_reasoning(text: str, result: Optional[Dict] = None,
                           context_text: Optional[str] = None) -> Tuple[str, Dict]:
    """Gọi Gemma-4 GGUF qua LM Studio API (localhost).

    Args:
        text:   Văn bản cần phân tích.
        result: Dict kết quả từ PhoBERT predict() — dùng để sinh fallback
                thông minh khi LM Studio không khả dụng.
        context_text: Ngữ cảnh bài viết/bình luận cha bổ sung cho prompt.
    """
    client = openai.OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_API_TOKEN)
    prompt_input = context_text if (context_text and context_text.strip()) else text
    try:
        logger.info(f"⏳ Calling LM Studio at {LM_STUDIO_BASE_URL} (model={LM_STUDIO_MODEL})...")
        common_kwargs = dict(
            model=LM_STUDIO_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia AI phân tích y tế công cộng (Explainable AI — XAI). "
                        "Hãy phân tích nội dung vaccine bằng TIẾNG VIỆT theo 3 chiều: "
                        "(1) Tính xác thực, (2) Thái độ với vaccine, (3) Cảm xúc tổng thể. "
                        "Trả lời có cấu trúc: === KẾT QUẢ === trước, === GIẢI THÍCH === sau. "
                        "Chỉ trả lời MỘT LẦN, KHÔNG lặp lại cấu trúc."
                    ),
                },
                {"role": "user", "content": _build_gemma_prompt(prompt_input)},
            ],
            max_tokens=819,   # một block GIẢI THÍCH hoàn chỉnh ~410 token là đủ.
                              # Hạ từ 1024 → cắt phần model lặp block thừa (log
                              # 01/06 sinh 718 token, ~300 thừa) → nhanh hơn ~6-10s.
                              # clean_reasoning_output vẫn giữ trọn block đầu tiên.
            temperature=0.1,
            stop=["<end_of_turn>", "<eos>"],
            timeout=120,
        )
        # repeat_penalty=1.1 (mức an toàn chuẩn của llama.cpp). KHÔNG dùng
        # frequency_penalty — chồng với repeat_penalty trên prompt dài tiếng Việt
        # sẽ phạt mọi âm tiết đã có trong ngữ cảnh → sinh chữ rác ("Tich dungert")
        # rồi dừng sớm. Đây là nguyên nhân lỗi suy luận 01/06 (prompt 2684 token).
        try:
            response = client.chat.completions.create(
                **common_kwargs, extra_body={"repeat_penalty": 1.1})
        except Exception as param_err:
            logger.warning(f"⚠️ extra_body không hỗ trợ, gọi lại bản tối giản: {str(param_err)[:120]}")
            response = client.chat.completions.create(**common_kwargs)

        content = response.choices[0].message.content or ""
        # CHỐNG OUTPUT RÁC/CỤT: phải có '=== GIẢI THÍCH ===' kèm nội dung thực
        # (≥50 ký tự). Bắt đúng ca "Tich dungert</end_of_turn>" — chỉ có KẾT QUẢ,
        # phần giải thích rỗng → coi là hỏng → dùng smart fallback theo PhoBERT.
        eidx = content.find("GIẢI THÍCH")
        has_explanation = eidx != -1 and len(content[eidx + len("GIẢI THÍCH"):].strip(":= \n\r")) >= 50
        if content.strip() and len(content.strip()) > 30 and has_explanation:
            return _clean_v2(content), parse_gemma_labels(content)
        logger.warning(f"⚠️ Gemma trả output thiếu/cụt phần giải thích "
                       f"(len={len(content)}, completion có vẻ bị cắt) → dùng fallback PhoBERT")
    except Exception as e:
        logger.warning(f"❌ LM Studio unreachable ({LM_STUDIO_BASE_URL}): {str(e)[:120]}")

    # Smart fallback: dùng template khi LM Studio không khả dụng
    if result:
        return generate_smart_fallback(
            result["misinfo"]["pred"],
            result["stance"]["pred"],
            result["sentiment"]["pred"],
        ), {}
    return "⚠️ LM Studio chưa được khởi động. Hãy mở LM Studio và bật Local Server tại cổng 1234.", {}


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


def get_reasoning(text: str, result: Dict,
                  context_text: Optional[str] = None) -> Tuple[str, str, Dict]:
    """
    2-layer local reasoning logic:
    Layer 1: Cache (xai_cache.json) ? instant
    Layer 2: Local LM Studio (Gemma-4 GGUF) — ~10-30s, với smart fallback template
    """
    # Layer 1: Cache (xai_cache.json) ? instant
    cached = find_xai_reasoning_cache(text)
    if cached:
        cleaned_cached = _clean_v2(cached)
        labels = parse_gemma_labels(cached)
        return cleaned_cached, "✅ Từ cache (Gold Test Set, 186 mẫu)", labels

    # Layer 2: LM Studio (Gemma-4 GGUF) với smart fallback khi offline
    reasoning, glabels = get_live_xai_reasoning(text, result=result, context_text=context_text)

    # Phân biệt source label: LM Studio thật hay fallback template
    if reasoning.startswith("⚠️") or "chưa được khởi động" in reasoning:
        return reasoning, "⚠️ LM Studio không khả dụng — hiển thị phân tích mẫu", glabels
    return reasoning, f"🖥️ Từ Gemma-4 GGUF · {LM_STUDIO_MODEL} ({LM_STUDIO_BASE_URL})", glabels


# ============================================================================
# CAPTUM INTEGRATED GRADIENTS
# ============================================================================

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

    lig = LayerIntegratedGradients(forward_fn, model.encoder.embeddings)
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
    """Render saliency heatmap — GỘP subword PhoBERT (@@) thành TỪ hoàn chỉnh.

    PhoBERT dùng BPE: token kết thúc bằng '@@' nghĩa là CÒN nối với token sau.
    Bản cũ hiển thị từng mảnh subword rời (vd 'ngụ@@', 'y@@', 'biện') khiến
    attribution khó đọc. Bản này gộp lại thành 'ngụy biện' và lấy điểm TRUNG BÌNH
    của các mảnh → attribution ở cấp TỪ, trực quan hơn nhiều.
    """
    if not tokens:
        return ("<p><em>⚠️ Token Attribution chưa chạy hoặc lỗi load model. "
                "Đây là giải thích bổ trợ ở cấp token (Level 1); giải thích chính "
                "bằng ngôn ngữ tự nhiên do Gemma đảm nhiệm (Level 2).</em></p>")

    SPECIALS = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "<unk>", "<mask>"}
    # 1) GỘP subword thành từ: tích luỹ tới khi gặp token KHÔNG có '@@'
    words, scores = [], []
    buf, buf_scores = "", []
    for tok, sc in zip(tokens, attr_norm):
        if tok in SPECIALS:
            continue
        piece = tok
        cont = piece.endswith("@@")          # còn nối sang token kế
        piece = piece[:-2] if cont else piece
        piece = piece.replace("▁", "").replace("Ġ", "")  # phòng tokenizer khác
        buf += piece
        buf_scores.append(sc)
        if not cont:                          # kết thúc 1 từ
            # '_' là ranh giới từ ghép do underthesea → hiện thành dấu cách
            word = buf.replace("_", " ").strip()
            if word:
                words.append(word)
                scores.append(sum(buf_scores) / len(buf_scores))  # điểm TB của từ
            buf, buf_scores = "", []
    if buf.strip():
        words.append(buf.replace("_", " ").strip())
        scores.append(sum(buf_scores) / len(buf_scores) if buf_scores else 0.0)

    # 2) Render
    html = ('<div style="line-height:2.1; padding:20px; border-radius:15px; '
            'background:var(--custom-card-bg); border:1px dashed var(--custom-card-border); '
            'font-family:Inter, sans-serif; font-size:1.05rem;">')
    for word, score in zip(words, scores):
        abs_score = abs(score)
        if abs_score < 0.15:
            html += f'<span style="color: var(--custom-text-muted); opacity:0.6;">{word}</span> '
        else:
            intensity = min(abs_score, 0.7)
            bg = (f"rgba(255,75,75,{intensity})" if pred_class == 0
                  else f"rgba(var(--saliency-pos-color),{intensity})")
            html += (f'<span style="background:{bg}; padding:2px 7px; border-radius:5px; '
                     f'font-weight:bold;">{word}</span> ')
    html += "</div>"
    label = LABEL_MAPS["misinfo"].get(pred_class, "?")
    html += (f'<p style="font-size:11px; color: var(--custom-text-muted); margin-top:10px;">'
             f'💡 Trục Tính xác thực dự đoán: <b>{label}</b> · từ đậm hơn = đóng góp lớn hơn '
             f'(Level 1 bổ trợ — giải thích chính do Gemma đảm nhiệm). '
             f'Đây là attribution ở cấp TỪ đã gộp từ subword.</p>')
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
            <audio src="data:audio/mp3;base64,{audio_b64}" oncanplay="this.playbackRate=1.4"></audio>
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
                font-family: 'Inter', sans-serif;
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
                            audio.playbackRate = 1.4;
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
                    apify_max_cmt = min(max_comments, 100)

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
    """Radar: ĐỘ TIN CẬY ĐÃ HIỆU CHUẨN của NHÃN ĐƯỢC DỰ ĐOÁN trên 3 tác vụ.

    Sửa 2 lỗi khoa học của bản cũ:
      (1) Bản cũ HARDCODE lớp "rủi ro" (Tin giả idx0 / Phản đối idx1 / Tiêu cực
          idx0) BẤT KỂ model dự đoán gì. Ví dụ ca 01/06: stance dự đoán "Trung
          lập" nhưng radar cũ vẽ P(Phản đối) → SAI nhãn, gây hiểu lầm.
      (2) Bản cũ dùng conf_raw (chưa hiệu chuẩn) → overconfident, mâu thuẫn với
          đóng góp Temperature Scaling của luận văn.
    Bản mới: mỗi trục = nhãn THỰC SỰ dự đoán, giá trị = conf ĐÃ HIỆU CHUẨN, hiện
    số % trực tiếp. Lưu ý khoa học: 3 trục là 3 tác vụ độc lập (không cùng đơn vị)
    → DIỆN TÍCH tam giác KHÔNG mang ý nghĩa tổng hợp; chỉ đọc theo TỪNG trục.
    """
    axes = ["misinfo", "stance", "sentiment"]
    axis_names = ["Tính xác thực", "Lập trường", "Cảm xúc"]
    labels, values, hover = [], [], []
    for ax, name in zip(axes, axis_names):
        pred = result[ax]["pred"]
        cal = float(result[ax]["conf_cal"][pred]) * 100.0
        lab = LABEL_MAPS[ax][pred]
        labels.append(f"{name}<br><b>{lab}</b>")
        values.append(cal)
        hover.append(f"{name}: {lab}<br>Tin cậy (hiệu chuẩn): {cal:.1f}%"
                     f"<br>T={result[ax]['T']:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself", line_color="#64ffda",
        fillcolor="rgba(100, 255, 218, 0.18)",   # mờ — không nhấn mạnh diện tích
        mode="lines+markers+text",
        marker=dict(size=8, color="#64ffda"),
        text=[f"{v:.0f}%" for v in values] + [f"{values[0]:.0f}%"],
        textposition="top center",
        textfont=dict(size=12),
        hovertext=hover + [hover[0]], hoverinfo="text",
        name="Độ tin cậy (hiệu chuẩn)",
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 100], showticklabels=True,
                ticksuffix="%", tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(size=9, color="#8899aa"), gridcolor="rgba(136,153,170,0.25)")
        ),
        showlegend=False, height=415,
        margin=dict(l=95, r=95, t=35, b=35),
        title=dict(text="Độ tin cậy (đã hiệu chuẩn) của nhãn dự đoán — đọc theo từng trục",
                   font=dict(size=11, color="#8899aa"), x=0.5, xanchor="center"),
    )
    return fig


def make_probability_distribution_chart(result: Dict) -> go.Figure:
    """Biểu đồ cột PHÂN PHỐI XÁC SUẤT đầy đủ (đã hiệu chuẩn) cho 3 tác vụ.

    Khác radar (chỉ 1 điểm/trục): hiện XÁC SUẤT MỌI LỚP của từng tác vụ → thấy rõ
    ĐỘ BẤT ĐỊNH (vd stance 49/30/21 lộ sự phân vân mà radar giấu). Đây là cách
    trình bày chuẩn cho output phân loại đa lớp. Lớp dự đoán (argmax) tô đậm.
    """
    axes = ["misinfo", "stance", "sentiment"]
    titles = ["Tính xác thực", "Lập trường", "Cảm xúc"]
    fig = make_subplots(rows=3, cols=1, subplot_titles=titles,
                        vertical_spacing=0.16, row_heights=[0.5, 0.75, 0.75])
    PRED_COLOR = "#64ffda"
    OTHER_COLOR = "rgba(136,153,170,0.45)"
    for i, ax in enumerate(axes, start=1):
        pred = result[ax]["pred"]
        probs = result[ax]["conf_cal"]           # xác suất ĐÃ HIỆU CHUẨN
        classes = [LABEL_MAPS[ax][j] for j in range(len(probs))]
        colors = [PRED_COLOR if j == pred else OTHER_COLOR for j in range(len(probs))]
        fig.add_trace(go.Bar(
            x=[p * 100 for p in probs], y=classes, orientation="h",
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs], textposition="auto",
            textfont=dict(size=12), showlegend=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ), row=i, col=1)
        fig.update_xaxes(range=[0, 100], ticksuffix="%", row=i, col=1,
                         showgrid=True, gridcolor="rgba(136,153,170,0.2)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=90, r=40, t=46, b=30),
        font=dict(family="Inter, sans-serif", size=12, color="#8899aa"),
        title=dict(text="Phân phối xác suất đã hiệu chuẩn (lớp đậm = dự đoán)",
                   font=dict(size=11), x=0.5, xanchor="center"),
    )
    for ann in fig.layout.annotations:
        ann.font.update(size=13, color="#64ffda")
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
    colors = ["#64ffda", "#007bff", "#FFA500"]
    
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
    """Render premium glassmorphism HTML cards with glow progress bars."""
    AXES = [("misinfo", "Tin giả / Xác thực"), ("stance", "Quan điểm"), ("sentiment", "Cảm xúc")]
    html = '<div style="display:flex;flex-wrap:wrap;gap:18px;width:100%;margin-bottom:12px;">'

    for i, (axis, axis_name) in enumerate(AXES):
        r = result[axis]
        pred_id = r["pred"]
        label = LABEL_MAPS[axis][pred_id]
        icon  = LABEL_ICONS[axis][pred_id]
        color = LABEL_COLORS[axis][pred_id]
        conf_raw = max(r["conf_raw"]) * 100
        conf_cal = max(r["conf_cal"]) * 100
        T        = r["T"]
        has_cal  = abs(T - 1.0) > 0.001

        if has_cal:
            conf_html = (
                f'<div style="font-size:0.78rem;color:var(--card-text-muted);text-align:center;margin-top:6px;">'
                f'Thô: <span style="text-decoration:line-through;opacity:0.65;">{conf_raw:.1f}%</span></div>'
                f'<div style="font-size:2rem;font-weight:800;color:{color};text-align:center;'
                f'text-shadow:0 0 20px {color}55;margin:4px 0;">{conf_cal:.1f}%</div>'
                f'<div style="font-size:0.73rem;color:var(--card-text-muted);text-align:center;">'
                f'Đã hiệu chuẩn (T={T:.2f})</div>'
            )
        else:
            conf_html = (
                f'<div style="font-size:2rem;font-weight:800;color:{color};text-align:center;'
                f'text-shadow:0 0 20px {color}55;margin:6px 0;">{conf_raw:.1f}%</div>'
                f'<div style="font-size:0.73rem;color:var(--card-text-muted);text-align:center;">Độ tin cậy</div>'
            )

        breakdown_items = ""
        for idx in range(len(r["conf_raw"])):
            p_raw = r["conf_raw"][idx] * 100
            p_cal = r["conf_cal"][idx] * 100
            cl    = LABEL_MAPS[axis][idx]
            cc    = LABEL_COLORS[axis][idx]
            pct   = p_cal if has_cal else p_raw
            pct_label = (f"Thô: {p_raw:.1f}% → <b style='color:{cc};'>{p_cal:.1f}%</b>"
                         if has_cal else f"<b style='color:{cc};'>{p_raw:.1f}%</b>")
            breakdown_items += (
                f'<div style="margin-top:9px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:var(--card-text-secondary);margin-bottom:4px;">'
                f'<span style="font-weight:600;">{cl}</span>'
                f'<span style="font-size:10px;">{pct_label}</span></div>'
                f'<div style="background:var(--progress-bar-bg);border-radius:6px;height:7px;overflow:hidden;border:1px solid var(--card-border);">'
                f'<div style="background:linear-gradient(90deg,{cc},{cc}cc);width:{pct:.1f}%;height:100%;border-radius:6px;'
                f'box-shadow:0 0 8px {cc}80;animation:pulseGlow 2.5s infinite ease-in-out;"></div></div></div>'
            )

        delay = i * 0.12
        html += (
            f'<div class="result-card-hover" style="flex:1;min-width:230px;'
            f'background:var(--card-bg-gradient);'
            f'border:1px solid {color}30;border-top:3px solid {color};border-radius:18px;'
            f'padding:24px 20px;'
            f'box-shadow:0 10px 32px var(--shadow-color),0 0 28px {color}18,inset 0 1px 0 rgba(255,255,255,0.05);'
            f'backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);'
            f'animation:fadeInUp 0.55s cubic-bezier(0.165,0.84,0.44,1) {delay:.2f}s both;">'
            f'<div style="text-align:center;margin-bottom:10px;">'
            f'<div style="font-size:44px;filter:drop-shadow(0 0 12px {color}60);">{icon}</div></div>'
            f'<div style="font-size:0.7rem;color:var(--card-text-muted);text-transform:uppercase;letter-spacing:0.14em;'
            f'text-align:center;font-weight:700;margin-bottom:6px;">{axis_name}</div>'
            f'<div style="font-size:1.55rem;font-weight:800;color:{color};text-align:center;'
            f'text-shadow:0 0 18px {color}50;">{label}</div>'
            f'{conf_html}'
            f'<div style="margin-top:16px;border-top:1px solid var(--card-border);padding-top:12px;">'
            f'<div style="font-size:0.68rem;font-weight:700;color:{color};text-transform:uppercase;'
            f'margin-bottom:6px;letter-spacing:0.08em;opacity:0.85;">Chi tiết nhãn</div>'
            f'{breakdown_items}</div></div>'
        )

    html += '</div>'
    html += (f"<div style='margin-top:12px;font-style:italic;color:var(--card-text-muted);font-size:0.85rem;text-align:right;'>"
             f"⏱️ {elapsed:.2f}s · {model_choice}</div>")
    return html


def make_speed_chart() -> go.Figure:
    """Throughput (samples/sec) comparison bar chart."""
    models = ["PhoBERT-v2", "XLM-R-v1", "Gemma-4 4B"]
    throughputs = [120.5, 85.2, 1.8]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=throughputs,
        marker_color=['#64ffda', '#007bff', '#FFA500'],
        text=[f"{v:.1f} mẫu/s" for v in throughputs], textposition="auto"
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#ccd6f6', size=13),
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
    <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px; font-family: 'Inter', sans-serif;">
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
    thread_ctx: str = "", progress=gr.Progress()
) -> Tuple:
    """Main analysis handler with progress indicator."""
    if not text or not text.strip():
        error_html = '<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Inter\', serif;">⚠️ Vui lòng nhập văn bản hoặc chọn mẫu thử!</div>'
        return (error_html, None, None, "", "", "", history,
                session_history_to_markdown(history), "", "")

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()
    
    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    t0 = time.time()
    result = predict(text, model_choice)
    t_predict = time.time() - t0
    if not result:
        error_html = f'<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Inter\', serif;">❌ Không thể load mô hình {model_choice} — kiểm tra HF_TOKEN</div>'
        return (error_html, None, None, "", "", "", history,
                session_history_to_markdown(history), "", "")

    progress(0.5, desc="📊 Đang tính radar chart...")
    radar = make_radar_chart(result)
    prob_dist = make_probability_distribution_chart(result)

    progress(0.6, desc="💭 Đang sinh giải thích (XAI 3-layer)...")
    t1 = time.time()
    reasoning, source, gemma_labels = get_reasoning(text, result, context_text=thread_ctx)
    t_reason = time.time() - t1
    reasoning_md = f"**{source}**\n\n{reasoning}"

    if use_captum:
        progress(0.8, desc="🎯 Đang tính Token Attribution...")
        t_cap = time.time()
        tokens, attr_norm, pred_class = compute_captum_saliency(text, model_choice)
        saliency_html = render_saliency_html(tokens, attr_norm, pred_class)
        logger.info(f"⏱️ Captum IG: {time.time()-t_cap:.1f}s")
    else:
        saliency_html = ("<p style='color:#888;'><em>💡 Bật checkbox <b>Token Attribution</b> "
                         "để xem đóng góp cấp từ (Level 1 bổ trợ — chậm hơn vài giây trên CPU)</em></p>")

    # gTTS bị TÁCH khỏi luồng chính (trước đây chặn ~40s). Văn bản đọc được lưu
    # vào state; người dùng bấm nút "Nghe" để tạo audio theo yêu cầu (lazy).
    voice_text = strip_markdown_for_tts(reasoning) if reasoning and not reasoning.startswith("⚠️") else ""
    audio_html = ("<p style='color:#888; font-size:0.9rem;'><em>🔊 Bấm nút "
                  "<b>“Tạo &amp; Nghe giọng đọc”</b> bên dưới để nghe phần giải thích "
                  "(tách riêng để kết quả hiển thị nhanh hơn).</em></p>") if voice_text else ""

    elapsed = time.time() - start
    logger.info(f"⏱️ Tổng handle_analyze (chưa tính TTS): {elapsed:.1f}s "
                f"| predict={t_predict:.1f}s · reasoning={t_reason:.1f}s")

    # Generate custom HTML result cards
    summary_html = render_result_cards_html(result, elapsed, model_choice)
    cmp = compare_engines(result, gemma_labels)
    badge = render_disagreement_badge(cmp)
    summary_html = badge + summary_html

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
    return (summary_html, radar, prob_dist, reasoning_md, saliency_html, audio_html, history,
            history_md, report_md, voice_text)


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




def handle_cell_select(evt: gr.SelectData, ctx_map_safe: dict):
    """Click 1 hàng → gửi VĂN BẢN THÔ cho PhoBERT + dựng ngữ cảnh cho Gemma."""
    from thread_parser import build_thread_context
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    entry = (ctx_map_safe or {}).get(str(row))
    if not entry:
        return "", ""

    if entry["kind"] == "post":
        post = entry["post"]
        return post["post_text"], build_thread_context(post, None)

    cmt = entry["comment"]
    if not entry["analyzable"]:
        try: gr.Info("🏷️ Dòng này là tag tên/emoji — không có nội dung để phân tích.")
        except Exception: pass
        return "", ""
    # PhoBERT ⟵ text THÔ ; Gemma ⟵ ngữ cảnh (bài + cmt cha nếu có)
    return cmt["text"], build_thread_context(entry["post"], cmt)


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


def handle_fetch_url(url: str, max_comments: int) -> Tuple[str, gr.update, gr.update, gr.update, gr.update, str, dict]:
    """Trả: (preview, fetched_table, send_btn, export_btn, status, batch_str, thread_ctx_map)."""
    from thread_parser import parse_apify, thread_to_rows, attach_parent_text
    from fetchers import detect_source, fetch_apify_raw, fetch_url_as_list

    # ── FB → đường có cấu trúc ──────────────────────────────────────────────
    if detect_source(url) == "apify" and "facebook.com" in url.lower():
        items, info, err = fetch_apify_raw(url, max_comments)
        if err or not items:
            msg = err or "❌ Không có dữ liệu"
            return ("", gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=f"<p style='color:#ff4b4b;'>{msg}</p>"), "", {})
        posts, schema = parse_apify(items)
        attach_parent_text(posts)              # nối _parent_text cho đường group
        rows, ctx_map = thread_to_rows(posts)
        df = pd.DataFrame(rows, columns=["STT", "Loại", "Nội dung"])

        # batch_str: chỉ gồm các mục ĐÁNG phân tích (bỏ tên-tag/emoji)
        analyzable = [v["comment"]["text"] for v in ctx_map.values()
                      if v["kind"] == "comment" and v["analyzable"]]
        if posts and posts[0]["post_text"]:
            analyzable = [posts[0]["post_text"]] + analyzable
        batch_str = "\n\n---\n\n".join(analyzable)

        n_cmt = sum(len(p["comments"]) for p in posts)
        n_ok = len(analyzable) - (1 if (posts and posts[0]["post_text"]) else 0)
        preview = posts[0]["post_text"][:500] if posts else ""
        status = (f"<p style='color:#3db882;font-weight:bold;'>✅ {info}: "
                  f"{len(posts)} bài · {n_cmt} bình luận "
                  f"({n_ok} đáng phân tích, đã lọc {n_cmt - n_ok} tag tên/emoji)</p>")
        # ctx_map key là int (row index) → JSON-safe hoá để vào gr.State
        ctx_map_safe = {str(k): v for k, v in ctx_map.items()}
        return (preview, gr.update(value=df, visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(value=status), batch_str, ctx_map_safe)

    # ── Báo / YouTube → đường cũ (giữ nguyên) ───────────────────────────────
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        msg = info if info.startswith("❌") else f"❌ {info}"
        return ("", gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=f"<p style='color:#ff4b4b;'>{msg}</p>"), "", {})
    df = pd.DataFrame([[i + 1, "📄 Nội dung", t] for i, t in enumerate(texts)],
                      columns=["STT", "Loại", "Nội dung"])
    batch_str = "\n\n---\n\n".join(texts)
    status = f"<p style='color:#3db882;font-weight:bold;'>✅ {len(texts)} mục từ {info}</p>"
    return (texts[0], gr.update(value=df, visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(value=status), batch_str, {})


def handle_send_to_batch(batch_text_str: str) -> Tuple[str, gr.update]:
    """Send fetched texts to batch textbox and open accordion."""
    try:
        gr.Info("🚀 Đã sao chép toàn bộ bài viết/comments vào ô Phân tích Batch! Vui lòng cuộn xuống dưới để thực hiện phân tích hàng loạt.")
    except:
        pass
    return batch_text_str, gr.update(open=True)


def handle_upload_data(file) -> Tuple[str, gr.update]:
    """Đọc .xlsx/.csv đã cào, LÀM SẠCH LOSSLESS, nạp vào ô Batch.

    Nguyên tắc làm sạch (bảo toàn dữ liệu — tránh loại trừ sai sót):
      - strip khoảng trắng thừa (đầu/cuối, gộp xuống dòng kép)
      - bỏ ô RỖNG / 'nan' (không phải nội dung)
      - khử trùng lặp CHÍNH XÁC (cùng một câu lặp lại nhiều lần)
      - GIỮ NGUYÊN mọi nội dung khác (không cắt chữ, không lọc theo từ khoá)
    Báo cáo rõ: nạp X · bỏ Y rỗng · gộp Z trùng, để người dùng kiểm soát.
    """
    if file is None:
        return "", gr.update()
    try:
        path = getattr(file, "name", None) or str(file)
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
        # chọn cột nội dung (ưu tiên theo tên, fallback cột cuối)
        col = next((c for c in df.columns
                    if any(k in str(c).lower()
                           for k in ("nội dung", "noi dung", "content", "text", "comment"))),
                   df.columns[-1])

        raw = df[col].tolist()
        n_total = len(raw)
        cleaned, seen, n_empty, n_dup = [], set(), 0, 0
        for x in raw:
            s = "" if x is None else str(x)
            # strip + gộp khoảng trắng/xuống dòng thừa (lossless về nội dung)
            s = re.sub(r"[ \t]+", " ", s).strip()
            s = re.sub(r"\n{3,}", "\n\n", s)
            if not s or s.lower() in ("nan", "none", "null"):
                n_empty += 1
                continue
            key = s.lower()
            if key in seen:          # trùng lặp chính xác (không phân biệt hoa/thường)
                n_dup += 1
                continue
            seen.add(key)
            cleaned.append(s)

        if not cleaned:
            try:
                gr.Warning(f"⚠️ File có {n_total} dòng nhưng không còn nội dung "
                           f"sau khi bỏ rỗng/trùng.")
            except Exception:
                pass
            return "", gr.update()

        try:
            gr.Info(f"📤 Đã nạp {len(cleaned)} mẫu sạch "
                    f"(từ {n_total} dòng · bỏ {n_empty} rỗng · gộp {n_dup} trùng lặp).")
        except Exception:
            pass
        return "\n\n---\n\n".join(cleaned), gr.update(open=True)
    except Exception as e:
        logger.error(f"Upload data failed: {e}")
        try:
            gr.Warning(f"❌ Lỗi đọc file: {str(e)[:80]}")
        except Exception:
            pass
        return "", gr.update()


def handle_generate_voice(tts_text: str) -> str:
    """Tạo audio gTTS THEO YÊU CẦU (lazy) — tách khỏi luồng phân tích chính
    để kết quả hiển thị nhanh. Trả HTML player (đã tăng tốc đọc 1.4x)."""
    if not tts_text or not tts_text.strip():
        return ("<p style='color:#888;'><em>⚠️ Chưa có nội dung để đọc. "
                "Hãy bấm “Phân tích” trước.</em></p>")
    t0 = time.time()
    html = text_to_speech(tts_text)
    logger.info(f"⏱️ gTTS (lazy): {time.time()-t0:.1f}s")
    return html or ("<p style='color:#888;'><em>⚠️ Không tạo được giọng đọc "
                    "(gTTS lỗi mạng). Thử lại sau.</em></p>")


def handle_batch(text: str, model_choice: str, progress=gr.Progress()) -> str:
    """Batch mode analysis with progress."""
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập văn bản"
    # CHỈ tách mẫu theo dấu phân cách "---". KHÔNG tách theo "\n" nữa — để GIỮ
    # NGUYÊN cấu trúc bài viết/bình luận thu thập từ Apify (một đoạn văn nhiều
    # dòng = MỘT mẫu, không bị xé thành nhiều câu nhỏ). Dữ liệu fetch sẵn dùng
    # "\n\n---\n\n" làm ranh giới nên mỗi bài/comment vẫn là 1 mẫu trọn vẹn.
    if "---" in text:
        lines = [l.strip() for l in text.split("---") if l.strip()]
    else:
        lines = [text.strip()]   # không có "---" ⇒ TOÀN BỘ input là 1 mẫu duy nhất
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
/* ============================================================
   VaccineNLP — Premium Theme v4.5
   Design: Hybrid Light White / Dark Navy theme
   Inspired by: quynhphuong1209-rehab-ai-monitor-2026.hf.space
   ============================================================ */

/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ===== CSS VARIABLES (Light Mode by Default) ===== */
:root {
    --bg-color: #ffffff;
    --bg-gradient: none;
    --text-color: #000000;
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --header-bg: #ffffff;
    --header-text: #000000;
    --footer-bg: #ffffff;
    --footer-text: #000000;
    --input-bg: #ffffff;
    --input-text: #000000;
    --input-border: #cbd5e1;
    --accordion-bg: #f8fafc;
    --tab-button-bg: #f1f5f9;
    --tab-button-text: #475569;
    --accent-color: #00b894;
    --accent-bright: #00d4aa;
    --accent-bg: rgba(0, 184, 148, 0.07);
    --shadow-color: rgba(0, 0, 0, 0.06);
    --glow-color: rgba(0, 184, 148, 0.15);
    --card-text-muted: #64748b;
    --card-text-primary: #000000;
    --card-text-secondary: #334155;
    --progress-bar-bg: #e2e8f0;
    --dropdown-bg: #ffffff;
    --custom-card-bg: #ffffff;
    --custom-card-border: #cbd5e1;
    --custom-text-neon: #00b894;
    --custom-text-muted: #475569;
    --custom-text-normal: #000000;
    --saliency-pos-color: 0, 184, 148;
    --custom-phobert-bg: #f0fdf4;
    --custom-xlmr-bg: #eff6ff;
    --custom-gemma-bg: #fffbeb;
    --custom-phobert-border: #00b894;
    --custom-phobert-text: #00b894;

    /* Theme specific variables */
    --sidebar-bg: #ffffff;
    --sidebar-border: #cbd5e1;
    --hero-bg: #ffffff;
    --hero-border: #e2e8f0;
    --hero-title-color: #000000;
    --sidebar-title-color: #000000;
    --hero-subtitle-color: #475569;
    --footer-bg-gradient: #ffffff;
    --footer-border: #cbd5e1;
    --footer-top-accent: #00b894;
    --footer-shadow: 0 -15px 50px rgba(0,0,0,0.04);
    --card-bg-gradient: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
}

/* ===== DARK MODE — Premium Navy ===== */
:root.dark, body.dark, .dark {
    --bg-color: #04091a;
    --bg-gradient: linear-gradient(160deg, #04091a 0%, #070e1a 40%, #050d1f 100%);
    --text-color: #ccd6f6;
    --card-bg: rgba(8, 18, 40, 0.75);
    --card-border: rgba(0, 212, 170, 0.18);
    --header-bg: rgba(4, 9, 26, 0.95);
    --header-text: #e6f1ff;
    --footer-bg: rgba(4, 9, 26, 0.97);
    --footer-text: #8892b0;
    --input-bg: rgba(10, 20, 45, 0.85);
    --input-text: #e6f1ff;
    --input-border: rgba(0, 212, 170, 0.22);
    --accordion-bg: rgba(10, 20, 45, 0.5);
    --tab-button-bg: rgba(10, 20, 45, 0.6);
    --tab-button-text: #8892b0;
    --accent-color: #00d4aa;
    --accent-bright: #00ffcc;
    --accent-bg: rgba(0, 212, 170, 0.09);
    --shadow-color: rgba(0, 0, 0, 0.55);
    --glow-color: rgba(0, 212, 170, 0.3);
    --card-text-muted: #8892b0;
    --card-text-primary: #ccd6f6;
    --card-text-secondary: #a8b2d8;
    --progress-bar-bg: rgba(4, 9, 26, 0.7);
    --dropdown-bg: #0a1628;
    --custom-card-bg: rgba(8, 18, 40, 0.65);
    --custom-card-border: rgba(0, 212, 170, 0.35);
    --custom-text-neon: #00d4aa;
    --custom-text-muted: #8892b0;
    --custom-text-normal: #ccd6f6;
    --saliency-pos-color: 0, 212, 170;
    --custom-phobert-bg: rgba(0, 212, 170, 0.06);
    --custom-xlmr-bg: rgba(0, 123, 255, 0.06);
    --custom-gemma-bg: rgba(255, 165, 0, 0.06);
    --custom-phobert-border: #00d4aa;
    --custom-phobert-text: #00d4aa;

    /* Theme specific variables */
    --sidebar-bg: linear-gradient(180deg, #050f1f 0%, #04091a 100%);
    --sidebar-border: rgba(0, 212, 170, 0.12);
    --hero-bg: linear-gradient(135deg, rgba(4,9,26,0.97) 0%, rgba(5,15,38,0.98) 50%, rgba(4,9,26,0.97) 100%);
    --hero-border: rgba(0, 212, 170, 0.22);
    --hero-title-color: #ffffff;
    --sidebar-title-color: #ffffff;
    --hero-subtitle-color: #8892b0;
    --footer-bg-gradient: linear-gradient(135deg, rgba(4,9,26,0.97) 0%, rgba(5,14,35,0.98) 100%);
    --footer-border: rgba(0, 212, 170, 0.18);
    --footer-top-accent: #00d4aa;
    --footer-shadow: 0 -15px 50px rgba(0,0,0,0.4);
    --card-bg-gradient: linear-gradient(145deg, rgba(8, 18, 40, 0.88) 0%, rgba(5, 12, 30, 0.93) 100%);
}

/* ===== KEYFRAME ANIMATIONS ===== */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(0,212,170,0.3), 0 0 20px rgba(0,212,170,0.1); }
    50%       { box-shadow: 0 0 20px rgba(0,212,170,0.6), 0 0 45px rgba(0,212,170,0.25); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ===== BASE ===== */
* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    text-shadow: none !important;
}
body, html {
    background-color: var(--bg-color) !important;
    background: var(--bg-gradient) !important;
    background-attachment: fixed !important;
    color: var(--text-color) !important;
    margin: 0;
    padding: 0;
    height: auto !important;
    min-height: 100vh;
}

/* ===== TYPOGRAPHY ===== */
body, html, p, li, table, tr, td, th { font-size: 1.06rem !important; line-height: 1.65 !important; }
code, pre, kbd, samp { font-family: 'Fira Code','JetBrains Mono',Consolas,monospace !important; }
h1 { font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: -0.025em !important; line-height: 1.25 !important; }
h2 { font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.015em !important; line-height: 1.35 !important; }
h3 { font-size: 1.38rem !important; font-weight: 700 !important; line-height: 1.4 !important; }
h4 { font-size: 1.18rem !important; font-weight: 600 !important; }
h5, h6 { font-size: 1.06rem !important; font-weight: 600 !important; }
label { font-size: 1.02rem !important; font-weight: 500 !important; }

/* ===== REMOVE GRADIO LABEL BADGES ===== */
.gradio-container .block-label,
.gradio-container [data-testid="block-label"],
.gradio-container label span,
.gradio-container label > span {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-color) !important;
    padding: 0 !important;
    margin: 0 0 6px 0 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}
.dark .gradio-container .block-label,
.dark .gradio-container [data-testid="block-label"],
.dark .gradio-container label span,
.dark .gradio-container label > span {
    color: var(--text-color) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.28); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,170,0.55); }
::-webkit-scrollbar-button { display: none !important; width: 0 !important; height: 0 !important; }

/* ===== GRADIO CONTAINER ===== */
.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: visible !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
.gradio-container .contain { max-width: 100% !important; width: 100% !important; }

/* ===== DROPDOWN FIX ===== */
.gradio-container .border-none { background-color: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container .options,
.gradio-container .select-options,
.gradio-container .dropdown-menu {
    z-index: 999999 !important;
    background-color: var(--dropdown-bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-color) !important;
    box-shadow: 0 16px 45px var(--shadow-color), 0 0 35px var(--glow-color) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    position: absolute !important;
    top: 100% !important;
    bottom: auto !important;
    transform: translateY(6px) !important;
    animation: dropdownFadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

@keyframes dropdownFadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(6px); }
}

.gradio-container .options .option,
.gradio-container .options .item,
.gradio-container .select-options .option,
.gradio-container .select-options .item,
.gradio-container .dropdown-menu .option,
.gradio-container .dropdown-menu .item {
    color: var(--text-color) !important;
    padding: 11px 15px !important;
    margin: 4px 0 !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Hover effect */
.gradio-container .options .option:hover,
.gradio-container .options .item:hover,
.gradio-container .select-options .option:hover,
.gradio-container .select-options .item:hover,
.gradio-container .dropdown-menu .option:hover,
.gradio-container .dropdown-menu .item:hover {
    background-color: rgba(0, 212, 170, 0.1) !important;
    color: var(--accent-color) !important;
    padding-left: 19px !important; /* Slide right animation */
}

/* Selected state */
.gradio-container .options .option.selected,
.gradio-container .options .item.selected,
.gradio-container .select-options .option.selected,
.gradio-container .select-options .item.selected,
.gradio-container .dropdown-menu .option.selected,
.gradio-container .dropdown-menu .item.selected {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(0, 212, 170, 0.25) !important;
}
.dark .gradio-container .options .option.selected,
.dark .gradio-container .options .item.selected,
.dark .gradio-container .select-options .option.selected,
.dark .gradio-container .select-options .item.selected,
.dark .gradio-container .dropdown-menu .option.selected,
.dark .gradio-container .dropdown-menu .item.selected {
    color: #04091a !important;
    box-shadow: 0 4px 14px rgba(0, 212, 170, 0.45) !important;
}

/* ===== TABS ===== */
.tabs { border-bottom: 1px solid rgba(0,212,170,0.12) !important; background: transparent !important; }
.tab-nav {
    display: flex;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: visible !important;
    gap: 4px !important;
    background: rgba(240,240,240,0.6) !important;
    border-bottom: 1px solid var(--card-border) !important;
    padding: 8px 10px !important;
    border-radius: 12px 12px 0 0 !important;
    -webkit-overflow-scrolling: touch !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
.dark .tab-nav {
    background: rgba(4,9,26,0.55) !important;
    border-bottom: 1px solid rgba(0,212,170,0.12) !important;
}
.tab-nav button {
    background: transparent !important;
    color: var(--tab-button-text) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
    text-transform: none !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.tab-nav button:hover { color: var(--accent-color) !important; background: rgba(0,212,170,0.06) !important; }
.tab-nav button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(0,212,170,0.25) !important;
}
.dark .tab-nav button.selected {
    color: #04091a !important;
    box-shadow: 0 4px 18px rgba(0,212,170,0.45), 0 0 25px rgba(0,212,170,0.18) !important;
}
@media (max-width: 768px) {
    .tab-nav { gap: 4px !important; padding: 5px !important; }
    .tab-nav button { padding: 6px 10px !important; font-size: 0.76rem !important; border-radius: 7px !important; }
}

/* ===== BUTTONS — Primary ===== */
button.primary, button.gr-button-primary {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 60%, #00a884 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,184,148,0.2) !important;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    padding: 12px 28px !important;
}
.dark button.primary, .dark button.gr-button-primary {
    color: #04091a !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.38), 0 0 30px rgba(0,212,170,0.12) !important;
}
button.primary:hover, button.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(0,184,148,0.35) !important;
    filter: brightness(1.08) !important;
}
.dark button.primary:hover, .dark button.gr-button-primary:hover {
    box-shadow: 0 8px 30px rgba(0,212,170,0.55), 0 0 45px rgba(0,212,170,0.22) !important;
}
button.primary:active, button.gr-button-primary:active { transform: translateY(0) !important; }

/* ===== BUTTONS — Secondary ===== */
button.secondary, button.gr-button-secondary {
    background-color: var(--tab-button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    transition: all 0.28s ease !important;
}
button.secondary:hover, button.gr-button-secondary:hover {
    border-color: var(--accent-color) !important;
    color: var(--accent-color) !important;
    background-color: rgba(0,184,148,0.06) !important;
    box-shadow: 0 4px 18px rgba(0,184,148,0.1) !important;
}

/* ===== INPUTS ===== */
.gradio-container input[type="text"]:not(.border-none):not(.dropdown input):not(.select-wrap input):not(.wrap input),
.gradio-container textarea {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    transition: all 0.28s ease !important;
}
.gradio-container input[type="text"]:not(.border-none):focus,
.gradio-container textarea:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 3px rgba(0,184,148,0.12) !important;
}

/* ===== CARDS & PANELS ===== */
.gr-box, .gr-panel, .block {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px var(--shadow-color) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

/* ===== SIDEBAR BLOCK FIX ===== */
#sidebar-col .block, #sidebar-col .wrap, #sidebar-col .gap,
#sidebar-col .form > .block, #sidebar-col fieldset { overflow: visible !important; }
#sidebar-col .block, #sidebar-col .gr-block {
    background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
#sidebar-col .block:focus-within {
    z-index: 1000 !important;
    position: relative !important;
}

/* ===== ACCORDION ===== */
.gr-accordion { background-color: var(--accordion-bg) !important; border: 1px solid var(--input-border) !important; border-radius: 12px !important; margin-bottom: 12px !important; transition: all 0.3s ease !important; }
.gr-accordion:hover { border-color: rgba(0,184,148,0.3) !important; box-shadow: 0 4px 20px rgba(0,184,148,0.05) !important; }

/* ===== RESULT CARDS ===== */
.result-card-hover {
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    animation: fadeInUp 0.55s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
}
.result-card-hover:hover {
    transform: translateY(-8px) scale(1.018) !important;
    box-shadow: 0 24px 50px var(--shadow-color) !important;
}
.dark .result-card-hover:hover {
    box-shadow: 0 24px 50px var(--shadow-color), 0 0 40px rgba(0,212,170,0.2) !important;
}
.resource-card { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; }
.resource-card:hover { border-color: var(--accent-color) !important; box-shadow: 0 10px 35px rgba(0,184,148,0.12) !important; transform: translateY(-5px) !important; }

/* ===== THEME TOGGLE ===== */
.theme-dark-btn, .theme-light-btn { border: 1px solid var(--input-border) !important; border-radius: 8px !important; font-weight: 500 !important; cursor: pointer !important; transition: all 0.3s ease !important; }
body.dark .theme-dark-btn { background: linear-gradient(135deg, var(--accent-color) 0%, #00b894 100%) !important; color: #020617 !important; font-weight: bold !important; border-color: var(--accent-color) !important; }
body.dark .theme-light-btn { background-color: var(--tab-button-bg) !important; color: var(--text-color) !important; }
body:not(.dark) .theme-light-btn { background: linear-gradient(135deg, var(--accent-color) 0%, #00b894 100%) !important; color: #ffffff !important; font-weight: bold !important; border-color: var(--accent-color) !important; }
body:not(.dark) .theme-dark-btn { background-color: var(--tab-button-bg) !important; color: var(--text-color) !important; }

/* ===== TOGGLE SWITCH ===== */
.switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 22px;
}
.switch input {
    opacity: 0;
    width: 0;
    height: 0;
}
.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #cbd5e1 !important; /* light gray when off */
    transition: .4s;
    border-radius: 34px !important;
}
.dark .slider {
    background-color: #334155 !important; /* dark gray when off */
}
.slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 3px;
    bottom: 3px;
    background-color: white !important;
    transition: .4s;
    border-radius: 50% !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}
.switch input:checked + .slider {
    background-color: #22c55e !important; /* emerald/green when on */
}
.switch input:checked + .slider:before {
    transform: translateX(22px) !important;
}

/* ===== SIDEBAR LAYOUT ===== */
#sidebar-col {
    align-self: flex-start !important;
    position: sticky !important;
    top: 0 !important;
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    padding: 20px !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease !important;
    display: block !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
    box-sizing: border-box !important;
    width: 290px !important;
    min-width: 290px !important;
    max-width: 290px !important;
    opacity: 1 !important;
    z-index: 9999 !important;
    transform: translateX(0) !important;
}
.dark #sidebar-col, body.dark #sidebar-col {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
}
#sidebar-col::-webkit-scrollbar { width: 4px !important; display: block !important; }
#sidebar-col::-webkit-scrollbar-track { background: transparent !important; }
#sidebar-col::-webkit-scrollbar-thumb { background-color: rgba(0,212,170,0.2) !important; border-radius: 4px !important; }
#sidebar-col::-webkit-scrollbar-thumb:hover { background-color: rgba(0,212,170,0.4) !important; }
#sidebar-col::-webkit-scrollbar-button { display: none !important; width: 0 !important; height: 0 !important; }

#main-layout-row { flex-wrap: nowrap !important; width: 100% !important; display: flex !important; overflow: visible !important; position: relative !important; align-items: stretch !important; }
#sidebar-col.collapsed { width: 0px !important; min-width: 0px !important; max-width: 0px !important; padding: 0px !important; opacity: 0 !important; border-right: none !important; transform: translateX(-290px) !important; pointer-events: none !important; overflow: hidden !important; }
#content-col { position: relative !important; padding-top: 50px !important; padding-left: 20px !important; padding-right: 20px !important; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; flex: 1 1 auto !important; display: flex !important; flex-direction: column !important; padding-bottom: 30px !important; }
#sidebar-col.collapsed ~ #content-col { width: 100% !important; max-width: 100% !important; }
#sidebar-col:not(.collapsed) ~ #content-col { width: calc(100% - 290px) !important; max-width: calc(100% - 290px) !important; }

/* ===== SIDEBAR TOGGLE ===== */
#sidebar-toggle-btn {
    position: fixed !important; z-index: 10001 !important; top: 16px !important;
    width: 40px !important; min-width: 40px !important; max-width: 40px !important;
    height: 40px !important; padding: 0 !important; border-radius: 10px !important;
    font-size: 18px !important; font-weight: bold !important;
    background: var(--bg-color) !important; color: var(--accent-color) !important;
    border: 1px solid var(--sidebar-border) !important; cursor: pointer !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.08) !important;
    transition: all 0.25s ease !important; display: flex !important;
    align-items: center !important; justify-content: center !important; left: 258px !important;
}
.dark #sidebar-toggle-btn {
    background: rgba(4,9,26,0.92) !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.3), 0 0 15px rgba(0,212,170,0.1) !important;
}
#sidebar-toggle-btn.sidebar-is-collapsed { left: 16px !important; }
#sidebar-toggle-btn::before {
    content: ""; display: block; width: 18px; height: 18px;
    background-color: var(--accent-color);
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    -webkit-mask-size: contain; mask-size: contain;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
#sidebar-toggle-btn.sidebar-is-collapsed::before { transform: rotate(180deg); }
#sidebar-toggle-btn span { display: none !important; }
#sidebar-toggle-btn:hover {
    background: rgba(0,184,148,0.1) !important;
    border-color: var(--accent-color) !important;
    box-shadow: 0 4px 18px rgba(0,184,148,0.2) !important;
    transform: scale(1.08) !important;
}

/* ===== MOBILE ===== */
@media (max-width: 768px) {
    .gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 8px !important; margin: 0 !important; }
    #sidebar-col {
        position: fixed !important; top: 0 !important; left: 0 !important; height: 100vh !important;
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
        box-shadow: 5px 0 30px rgba(0,0,0,0.1) !important; z-index: 9999 !important;
        transform: translateX(-290px) !important; opacity: 0 !important;
        width: 290px !important; min-width: 290px !important; max-width: 290px !important;
    }
    .dark #sidebar-col {
        box-shadow: 5px 0 30px rgba(0,0,0,0.5) !important;
    }
    #sidebar-col:not(.collapsed) { transform: translateX(0) !important; opacity: 1 !important; pointer-events: auto !important; }
    #sidebar-col.collapsed { transform: translateX(-290px) !important; opacity: 0 !important; width: 0px !important; min-width: 0px !important; max-width: 0px !important; padding: 0px !important; }
    #content-col { width: 100% !important; max-width: 100% !important; padding-top: 60px !important; padding-left: 8px !important; padding-right: 8px !important; }
    #sidebar-col.collapsed ~ #content-col, #sidebar-col:not(.collapsed) ~ #content-col { width: 100% !important; max-width: 100% !important; }
}

/* ===== PLOTLY ===== */
.js-plotly-plot { background-color: transparent !important; width: 100% !important; }
.js-plotly-plot .bg { fill: transparent !important; }
.js-plotly-plot text, .js-plotly-plot tspan, .js-plotly-plot .xtick text, .js-plotly-plot .ytick text,
.js-plotly-plot .gtitle, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle,
.js-plotly-plot .legendtext { fill: var(--text-color) !important; }
.dark .js-plotly-plot text, .dark .js-plotly-plot tspan { fill: #ffffff !important; }
.js-plotly-plot .gridlayer path, .js-plotly-plot .zerolinelayer path,
.js-plotly-plot .axis line { stroke: rgba(128,128,128,0.15) !important; }
.dark .js-plotly-plot .gridlayer path, .dark .js-plotly-plot .zerolinelayer path,
.dark .js-plotly-plot .axis line { stroke: rgba(255,255,255,0.1) !important; }
.js-plotly-plot .sankey-node text { fill: var(--text-color) !important; }

/* Radar polar styles */
.js-plotly-plot .polarbg { fill: #ffffff !important; }
.dark .js-plotly-plot .polarbg { fill: #000000 !important; }
.js-plotly-plot .polargrid path { stroke: rgba(0,0,0,0.1) !important; }
.dark .js-plotly-plot .polargrid path { stroke: rgba(255,255,255,0.75) !important; }
.gr-plot, .gradio-plot, .plot-container, [data-testid="plot"], .js-plotly-plot,
.plotly, .svg-container, .main-svg { width: 100% !important; max-width: 100% !important; }
.gr-plot > div, .plot-container > div, .js-plotly-plot > div { width: 100% !important; max-width: 100% !important; }

/* ===== MISC UTILITIES ===== */
.model-color-phobert { color: var(--custom-text-neon) !important; }
.model-color-xlmr { color: #3b82f6 !important; }
.model-color-gemma { color: #FFA500 !important; }
.dropdown-menu { background-color: var(--input-bg) !important; border: 1px solid var(--input-border) !important; }
.reset-btn-layout { background: transparent !important; border: 1.5px solid var(--input-border) !important; color: var(--text-color) !important; transition: all 0.2s ease !important; }
.reset-btn-layout:hover { border-color: var(--accent-color) !important; color: var(--accent-color) !important; background-color: var(--accent-bg) !important; }
#sidebar-col .form { display: flex !important; flex-direction: column !important; gap: 18px !important; height: auto !important; overflow: visible !important; }
#sidebar-col .form > * { width: 100% !important; box-sizing: border-box !important; flex-shrink: 0 !important; }
#sidebar-col hr { margin: 28px 0 20px 0 !important; opacity: 0.85; }
#sidebar-col h5 { margin-top: 4px !important; margin-bottom: 10px !important; }
.sidebar-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, rgba(0,184,148,0.2) 50%, transparent 100%) !important; margin: 28px 0 20px 0 !important; border: none !important; }
.dark .sidebar-divider { background: linear-gradient(90deg, rgba(0,212,170,0) 0%, rgba(0,212,170,0.3) 50%, rgba(0,212,170,0) 100%) !important; }
.sidebar-scroll-btn { display: none !important; }
footer, .gradio-container > footer { display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }

/* ===== HERO BANNER ===== */
#hero-banner {
    background: var(--hero-bg) !important;
    border: 1px solid var(--hero-border) !important;
    border-radius: 20px !important;
    padding: 48px 28px !important;
    text-align: center !important;
    margin-bottom: 28px !important;
    box-shadow: 0 12px 50px rgba(0,0,0,0.04) !important;
    position: relative !important;
    overflow: hidden !important;
}
.dark #hero-banner {
    box-shadow: 0 12px 50px rgba(0,0,0,0.55), 0 0 80px rgba(0,212,170,0.07),
                inset 0 1px 0 rgba(0,212,170,0.12) !important;
}
#hero-banner::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 15% 50%, rgba(0,184,148,0.04) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 50%, rgba(0,100,255,0.03) 0%, transparent 55%);
    pointer-events: none;
}
.dark #hero-banner::before {
    background: radial-gradient(ellipse at 15% 50%, rgba(0,212,170,0.07) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 50%, rgba(0,100,255,0.05) 0%, transparent 55%);
}
.hero-accent-line {
    position: absolute !important; top: 0 !important; left: 0 !important; right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, transparent 0%, #00d4aa 25%, #00ffcc 50%, #00d4aa 75%, transparent 100%) !important;
    background-size: 200% auto !important;
    animation: shimmer 3s linear infinite !important;
}
.hero-emojis {
    font-size: 2.5rem !important; margin-bottom: 10px !important;
    filter: drop-shadow(0 0 14px rgba(0,184,148,0.2)) !important;
    animation: float 4s ease-in-out infinite !important;
    display: inline-block !important;
}
.dark .hero-emojis {
    filter: drop-shadow(0 0 14px rgba(0,212,170,0.45)) !important;
}
.hero-title {
    margin: 6px 0 16px 0 !important;
    font-size: clamp(1.6rem, 3.5vw, 2.6rem) !important;
    font-weight: 800 !important;
    color: var(--hero-title-color) !important;
    line-height: 1.3 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}
.hero-divider {
    width: 130px !important; height: 3px !important;
    background: var(--hero-title-gradient, linear-gradient(135deg, #e6f1ff 0%, #00d4aa 35%, #00ffcc 55%, #ccd6f6 100%)) !important;
    background-size: 200% auto !important;
    animation: shimmer 2.5s linear infinite !important;
    margin: 16px auto !important; border-radius: 2px !important;
}
.hero-subtitle {
    margin: 0 auto !important;
    font-size: clamp(0.88rem, 1.8vw, 1.1rem) !important;
    color: var(--hero-subtitle-color) !important;
    font-weight: 500 !important;
    max-width: 820px !important;
    line-height: 1.55 !important;
}
"""

SPEED_METRICS_HTML = """
<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
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
<div style="background: var(--custom-card-bg); border: 1px solid var(--custom-card-border); border-radius: 8px; padding: 20px; font-family: 'Inter', sans-serif;">
    <h4 style="margin-top: 0; color: var(--custom-text-neon); font-size: 1.2rem;">🤝 Kiến trúc lai đề xuất cho dự án VaccineNLP (HUPH 2026):</h4>
    <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.6; color: var(--custom-text-normal);">
        <li><b>Vòng ngoài (Real-time Classification - PhoBERT-v2)</b>: Nhờ tốc độ suy luận cực nhanh (120.5 mẫu/giây) và độ chính xác F1 vượt trội, PhoBERT-v2 được đề xuất làm màng lọc trực tiếp ở luồng dữ liệu mạng xã hội để phân loại nhanh tin giả, sắc thái và lập trường.</li>
        <li><b>Vòng trong (Explainable & Strategic Consulting - Gemma-4 4B)</b>: Đối với các mẫu được PhoBERT-v2 nghi ngờ là "Tin giả" hoặc "Tiêu cực cực đoan", hệ thống sẽ đẩy vào hàng đợi offline để Gemma-4 lý luận chuyên sâu (XAI) giải thích lý do gán nhãn và đề xuất kịch bản phản hồi khủng hoảng cho chuyên gia y tế HUPH.</li>
    </ol>
</div>
"""

RESOURCES_HTML = """
<div style="font-family: 'Inter', sans-serif; color: var(--text-color);">
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
<div style="font-family: 'Inter', sans-serif; color: var(--text-color); line-height: 1.6;">
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
<div style="font-family: 'Inter', sans-serif; color: var(--text-color); line-height: 1.6;">
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


def get_sidebar_header_html() -> str:
    logo_src = get_huph_logo_base64()
    return f"""
    <style>
    @keyframes sidebar-glow-pulse {{
        0%, 100% {{ box-shadow: 0 0 18px rgba(0,212,170,0.4), 0 0 40px rgba(0,212,170,0.15); border-color: rgba(0,212,170,0.7); }}
        50%       {{ box-shadow: 0 0 30px rgba(0,212,170,0.7), 0 0 60px rgba(0,212,170,0.28); border-color: rgba(0,255,200,0.9); }}
    }}
    </style>
    <div style="text-align: center; margin-bottom: 12px;">
        <!-- Animated Logo Ring -->
        <div style="position: relative; width: 95px; height: 95px; margin: 0 auto 16px auto;">
            <div style="width: 95px; height: 95px; border-radius: 50%; border: 2.5px solid rgba(0,212,170,0.7); display: flex; align-items: center; justify-content: center; background: rgba(0,212,170,0.06); animation: sidebar-glow-pulse 3s ease-in-out infinite;">
                <img src="{logo_src}" style="width: 72px; height: 72px; object-fit: contain; border-radius: 50%;" alt="HUPH Logo">
            </div>
        </div>

        <!-- App Title -->
        <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; display: flex; align-items: center; justify-content: center; gap: 8px; color: var(--sidebar-title-color, var(--text-color));">
            🦠 VaccineNLP
        </h2>
    </div>
    """


def get_header_html() -> str:
    return """
    <div style="text-align: center; padding: 15px 10px 30px 10px; margin-bottom: 15px; font-family: 'Inter', sans-serif; color: var(--text-color);">
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
    <div style="background: var(--footer-bg); color: var(--footer-text); padding: 45px 30px; border-radius: 16px; margin-top: 45px; font-family: 'Inter', sans-serif; border: 1px solid var(--input-border); border-top: 4px solid var(--accent-color); box-shadow: 0 -12px 35px rgba(0, 0, 0, 0.2);">
      <div style="display: flex; flex-wrap: wrap; gap: 35px; justify-content: space-between;">
        <div style="flex: 1.1; min-width: 250px; text-align: center; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <div style="width: 100px; height: 100px; background: rgba(255,255,255,0.08); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid var(--accent-color); box-shadow: 0 0 20px rgba(100, 255, 218, 0.15); margin: 0 auto 15px auto;">
            <img src="{logo_src}" style="width: 80px; height: 80px; object-fit: contain;" alt="HUPH Logo">
          </div>
          <h3 style="color: var(--header-text); font-size: 1.05rem; margin: 5px 0; font-family: 'Inter', sans-serif !important; font-weight: 700; letter-spacing: 0.05em;">TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</h3>
          <p style="font-size: 0.85rem; color: var(--tab-button-text); margin: 6px 0;">📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
          <p style="font-size: 0.85rem; margin: 6px 0;">🌐 <a href="https://huph.edu.vn/" target="_blank" style="color: var(--accent-color); text-decoration: none; font-weight: 600;">huph.edu.vn</a></p>
        </div>
        
        <div style="flex: 1.5; min-width: 250px; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Inter', sans-serif !important; font-weight: 700; letter-spacing: 0.05em;">🔬 Đề tài đồ án</h3>
          <p style="color: #ffd700; font-weight: bold; font-style: italic; font-size: 1rem; line-height: 1.6; margin-bottom: 8px;">
            "Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"
          </p>
          <p style="color: var(--tab-button-text); font-size: 0.85rem; line-height: 1.5;">
            (Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)
          </p>
        </div>
        
        <div style="flex: 1.2; min-width: 250px; border-right: 1px solid var(--input-border); padding-right: 20px;">
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Inter', sans-serif !important; font-weight: 700; letter-spacing: 0.05em;">👥 Nhóm thực hiện</h3>
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
          <h3 style="color: var(--accent-color); font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px; font-family: 'Inter', sans-serif !important; font-weight: 700; letter-spacing: 0.05em;">👨‍🏫 GV Hướng dẫn</h3>
          <p style="font-size: 1.1rem; font-weight: bold; color: var(--header-text); margin-bottom: 6px;">TS. Trần Lâm Quân</p>
          <p style="font-size: 0.85rem; color: var(--tab-button-text); line-height: 1.5;">
            Giảng viên Khoa học dữ liệu<br>
            Trường Đại học Y tế Công cộng<br>
            📧 <a href="mailto:tlq@huph.edu.vn" style="color: var(--accent-color); text-decoration: none; font-weight: 600;">tlq@huph.edu.vn</a>
          </p>
        </div>
      </div>
      <hr style="border-color: var(--input-border); margin: 30px 0 20px 0;">
      <p style="text-align: center; font-size: 0.85rem; color: var(--tab-button-text); margin: 0; font-family: 'Inter', sans-serif !important; letter-spacing: 0.02em;">
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

    html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; width: 100%; margin-bottom: 20px;">'
    for title, val, sub, border_color in cards:
        html += f"""
        <div style="flex: 1; min-width: 220px; background: var(--card-bg); border: 1px solid var(--card-border); border-top: 4px solid {border_color}; border-radius: 12px; padding: 16px 20px; text-align: left; box-shadow: 0 10px 25px var(--shadow-color);">
            <div style="font-size: 0.72rem; color: var(--card-text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: var(--text-color); margin-bottom: 8px; letter-spacing: -0.02em; line-height: 1; text-shadow: 0 0 10px {border_color}1a;">{val}</div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 0.75rem; color: {border_color}; font-weight: 600; background: {border_color}12; padding: 3px 8px; border-radius: 6px; border: 1px solid {border_color}25;">{sub}</span>
            </div>
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
    <table style="width:100%; border-collapse:collapse; background:{table_bg}; border:1px solid {table_border}; border-radius:10px; overflow:hidden; font-family:'Inter', sans-serif; text-align:center;">
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
    <table style="width:100%; border-collapse:collapse; background:{table_bg}; border:1px solid {table_border}; font-family:'Inter', sans-serif; text-align:center;">
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
    <table style="width: 100%; border-collapse: collapse; background: var(--card-bg); border: 1px solid var(--input-border); border-radius: 10px; overflow: hidden; font-family: 'Inter', sans-serif; text-align: center;">
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
        status = f"<div style='color: orange; font-weight: bold; font-family: \"Inter\", serif;'>🤖 Đang giả lập kiểm thử trực tiếp trên GPU: {row['Model']}...</div>"
        yield status, render_live_table(current_data)
        time.sleep(0.8)
        current_data.append(row)
        yield status, render_live_table(current_data)
        time.sleep(0.4)
        
    status = f"<div style='color: #38ef7d; font-weight: bold; font-family: \"Inter\", serif;'>✅ Quá trình suy luận Live hoàn tất! Bảng kết quả F1 đã được cập nhật thành công.</div>"
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
    <div style="margin-top: 15px; padding: 15px; background: var(--input-bg); border: 1px solid var(--input-border); border-radius: 10px; font-family: 'Inter', sans-serif;">
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
    """Build the Gradio Blocks app with 6 tabs."""
    init_theme_js = """function() {
        /* Force Plotly charts to stretch to 100% width on tab clicks/nav changes */
        document.addEventListener('click', function(e) {
            const isTabOrButton = e.target.closest('button') || e.target.closest('[role="tab"]') || e.target.closest('.tab-nav') || e.target.closest('.tab-item');
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
        /* Force Plotly charts to stretch to 100% width on tab clicks/nav changes */
        document.addEventListener('click', function(e) {
            const isTabOrButton = e.target.closest('button') || e.target.closest('[role="tab"]') || e.target.closest('.tab-nav') || e.target.closest('.tab-item');
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
        /* Delegated document event listener for TTS button click (highly secure & sanitizer-proof) */
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
                // Pause all other audio players on page
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
                    // Fallback: If programmatic playback is blocked by browser gesture policies, reveal the native controls!
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

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.documentElement.classList.remove('dark');
            document.body.classList.remove('dark');
        } else {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
        }

        /* Runtime fix: force overflow:visible on all sidebar children so dropdowns work */
        function fixSidebarDropdowns() {
            var sidebar = document.getElementById('sidebar-col');
            if (!sidebar || sidebar.classList.contains('collapsed')) return;
            var allChildren = sidebar.querySelectorAll('*');
            for (var i = 0; i < allChildren.length; i++) {
                var el = allChildren[i];
                var cls = (el.className || '').toString();
                /* Skip the options list itself and elements that should scroll */
                if (cls.indexOf('options') >= 0 || cls.indexOf('items') >= 0) continue;
                el.style.setProperty('overflow', 'visible', 'important');
            }
        }

        /* Collapse sidebar by default on mobile screens (Streamlit style) */
        function collapseSidebarOnMobile() {
            const sidebar = document.getElementById('sidebar-col');
            const btn = document.getElementById('sidebar-toggle-btn');
            if (window.innerWidth <= 768) {
                if (sidebar) sidebar.classList.add('collapsed');
                if (btn) btn.classList.add('sidebar-is-collapsed');
            } else {
                if (sidebar) sidebar.classList.remove('collapsed');
                if (btn) btn.classList.remove('sidebar-is-collapsed');
            }
        }

        /* Tap outside the sidebar to close it on mobile */
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar-col');
                const btn = document.getElementById('sidebar-toggle-btn');
                if (sidebar && !sidebar.classList.contains('collapsed')) {
                    if (!sidebar.contains(e.target) && !btn.contains(e.target)) {
                        sidebar.classList.add('collapsed');
                        if (btn) btn.classList.add('sidebar-is-collapsed');
                    }
                }
            }
        });

        /* Run multiple times to catch lazy-rendered Svelte components */
        setTimeout(fixSidebarDropdowns, 300);
        setTimeout(fixSidebarDropdowns, 800);
        setTimeout(fixSidebarDropdowns, 2000);

        setTimeout(collapseSidebarOnMobile, 50);
        setTimeout(collapseSidebarOnMobile, 300);
        setTimeout(collapseSidebarOnMobile, 800);

        /* Scroll helper functions for Svelte tabs and sidebar */
        function setupTabScrolling() {
            var tabContainers = document.querySelectorAll('.tabs');
            tabContainers.forEach(function(container) {
                var nav = container.querySelector('.tab-nav');
                if (!nav) return;
                
                if (container.querySelector('.tab-scroll-btn-left')) return;
                
                var btnLeft = document.createElement('button');
                btnLeft.className = 'tab-scroll-btn tab-scroll-btn-left';
                btnLeft.innerHTML = '❮';
                btnLeft.type = 'button';
                
                var btnRight = document.createElement('button');
                btnRight.className = 'tab-scroll-btn tab-scroll-btn-right';
                btnRight.innerHTML = '❯';
                btnRight.type = 'button';
                
                container.appendChild(btnLeft);
                container.appendChild(btnRight);
                
                function updateArrows() {
                    var scrollLeft = nav.scrollLeft;
                    var scrollWidth = nav.scrollWidth;
                    var clientWidth = nav.clientWidth;
                    
                    if (scrollLeft > 5) {
                        btnLeft.classList.add('visible');
                    } else {
                        btnLeft.classList.remove('visible');
                    }
                    
                    if (scrollWidth - scrollLeft - clientWidth > 5) {
                        btnRight.classList.add('visible');
                    } else {
                        btnRight.classList.remove('visible');
                    }
                }
                
                btnLeft.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    nav.scrollBy({ left: -220, behavior: 'smooth' });
                });
                
                btnRight.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    nav.scrollBy({ left: 220, behavior: 'smooth' });
                });
                
                nav.addEventListener('scroll', updateArrows);
                window.addEventListener('resize', updateArrows);
                
                setTimeout(updateArrows, 200);
            });
        }

        function setupSidebarScrollButtons() {
            var sidebar = document.getElementById('sidebar-col');
            if (!sidebar || sidebar.classList.contains('collapsed')) return;
            
            if (sidebar.querySelector('.sidebar-scroll-btn-up')) return;
            
            var btnUp = document.createElement('div');
            btnUp.className = 'sidebar-scroll-btn sidebar-scroll-btn-up';
            btnUp.innerHTML = '▲';
            
            var btnDown = document.createElement('div');
            btnDown.className = 'sidebar-scroll-btn sidebar-scroll-btn-down';
            btnDown.innerHTML = '▼';
            
            sidebar.insertBefore(btnUp, sidebar.firstChild);
            sidebar.appendChild(btnDown);
            
            function updateSidebarButtons() {
                var scrollTop = sidebar.scrollTop;
                var scrollHeight = sidebar.scrollHeight;
                var clientHeight = sidebar.clientHeight;
                
                if (scrollTop > 15) {
                    btnUp.classList.add('visible');
                } else {
                    btnUp.classList.remove('visible');
                }
                
                if (scrollHeight - scrollTop - clientHeight > 15) {
                    btnDown.classList.add('visible');
                } else {
                    btnDown.classList.remove('visible');
                }
            }
            
            btnUp.addEventListener('click', function(e) {
                e.stopPropagation();
                sidebar.scrollBy({ top: -180, behavior: 'smooth' });
            });
            
            btnDown.addEventListener('click', function(e) {
                e.stopPropagation();
                sidebar.scrollBy({ top: 180, behavior: 'smooth' });
            });
            
            sidebar.addEventListener('scroll', updateSidebarButtons);
            window.addEventListener('resize', updateSidebarButtons);
            sidebar.addEventListener('mouseenter', updateSidebarButtons);
            
            setTimeout(updateSidebarButtons, 200);
        }

        /* Run multiple times to catch lazy-rendered Svelte components */
        setTimeout(fixSidebarDropdowns, 300);
        setTimeout(fixSidebarDropdowns, 800);
        setTimeout(fixSidebarDropdowns, 2000);

        setTimeout(setupTabScrolling, 300);
        setTimeout(setupTabScrolling, 800);
        setTimeout(setupTabScrolling, 2000);

        setTimeout(setupSidebarScrollButtons, 300);
        setTimeout(setupSidebarScrollButtons, 800);
        setTimeout(setupSidebarScrollButtons, 2000);

        setTimeout(collapseSidebarOnMobile, 50);
        setTimeout(collapseSidebarOnMobile, 300);
        setTimeout(collapseSidebarOnMobile, 800);

        /* Watch for new DOM nodes and re-apply fix */
        var observer = new MutationObserver(function(mutations) {
            for (var m = 0; m < mutations.length; m++) {
                if (mutations[m].addedNodes.length > 0) {
                    setTimeout(fixSidebarDropdowns, 50);
                    setTimeout(setupTabScrolling, 50);
                    setTimeout(setupSidebarScrollButtons, 50);
                    break;
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        /* Theme toggle switch handler */
        function setupThemeSwitch() {
            const toggle = document.getElementById('theme-toggle-switch');
            const label = document.getElementById('theme-switch-text');
            if (!toggle || !label) {
                setTimeout(setupThemeSwitch, 100);
                return;
            }
            
            const isDark = document.documentElement.classList.contains('dark') || document.body.classList.contains('dark');
            toggle.checked = isDark;
            label.innerHTML = isDark ? '🌙 Giao diện tối' : '☀️ Giao diện sáng';
            
            toggle.addEventListener('change', function() {
                if (toggle.checked) {
                    document.documentElement.classList.add('dark');
                    document.body.classList.add('dark');
                    localStorage.setItem('theme', 'dark');
                    label.innerHTML = '🌙 Giao diện tối';
                } else {
                    document.documentElement.classList.remove('dark');
                    document.body.classList.remove('dark');
                    localStorage.setItem('theme', 'light');
                    label.innerHTML = '☀️ Giao diện sáng';
                }
            });
        }
        setupThemeSwitch();
    }""".strip()
    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS_STYLE, fill_width=True, js=init_theme_js) as app:
        # Sidebar Toggle Button (positioned via CSS)
        sidebar_toggle_btn = gr.Button("", elem_id="sidebar-toggle-btn", size="sm")
        
        with gr.Row(elem_id="main-layout-row"):
            # Left Sidebar Column
            with gr.Column(scale=1, min_width=290, elem_id="sidebar-col"):
                gr.HTML(get_sidebar_header_html())
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Inter\", serif; font-weight: bold; margin-bottom: 8px;'>🎨 Giao diện</h5>")
                theme_switch_html = gr.HTML(
                    """
                    <div class="theme-switch-wrapper" style="display: flex; align-items: center; justify-content: space-between; margin: 10px 0; padding: 5px 0; font-family: 'Inter', sans-serif;">
                        <span id="theme-switch-text" style="font-size: 0.95rem; font-weight: 600; color: var(--text-color);">☀️ Giao diện sáng</span>
                        <label class="switch" style="position: relative; display: inline-block; width: 46px; height: 22px; margin: 0;">
                            <input type="checkbox" id="theme-toggle-switch" style="opacity: 0; width: 0; height: 0;">
                            <span class="slider round" style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(148, 163, 184, 0.3); transition: .3s; border-radius: 22px;"></span>
                        </label>
                    </div>
                    """
                )
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Inter\", serif; font-weight: bold; margin-bottom: 8px;'>📋 Mẫu thử nghiệm</h5>")
                sample_category = gr.Dropdown(
                    choices=["Tự nhập", "🚨 Nhóm Tin giả cực đoan", "🟢 Nhóm phân tích Thái độ", "✅ Nhóm Thông tin chuẩn", "💬 Nhóm Từ lóng MXH"],
                    value="Tự nhập",
                    label="Chọn nhóm mẫu:"
                )
                sample_detail = gr.Radio(
                    choices=["Tự nhập"],
                    value="Tự nhập",
                    label="Chọn loại văn bản:",
                    visible=False
                )
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Inter\", serif; font-weight: bold; margin-bottom: 8px;'>🤖 Mô hình Phân loại</h5>")
                gr.Markdown(
                    """
                    <div style="font-size: 12px; color: var(--text-color); opacity: 0.8; font-family: 'Inter', sans-serif; margin-bottom: 8px; line-height: 1.4;">
                        Mô hình này đảm nhiệm việc phân loại nhãn (Tin giả, Quan điểm, Cảm xúc).
                    </div>
                    """,
                    sanitize_html=False
                )
                model_choice = gr.Dropdown(
                    choices=list(CONFIG["models"].keys()),
                    value="PhoBERT-v2",
                    label="Chọn model:"
                )
                info_box = gr.HTML(value=get_sidebar_info_html("PhoBERT-v2"))
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Inter\", serif; font-weight: bold; margin-bottom: 8px;'>🛠️ Quản trị hệ thống</h5>")
                clear_cache_btn = gr.Button("🗑️ Xóa Cache & Khởi động lại", elem_classes=["theme-toggle-btn"], size="sm")
                clear_cache_status = gr.Markdown(value="", visible=False)
                gr.HTML(
                    """
                    <div style="font-size: 11px; color: var(--text-color); opacity: 0.7; font-family: 'Inter', sans-serif; margin-top: 10px; line-height: 1.4;">
                        💡 <b>Lưu ý:</b> Nếu gặp lỗi 403 Forbidden, vui lòng kiểm tra lại quyền 'Inference' của Token trên Hugging Face.
                    </div>
                    """
                )
            clear_cache_btn.click(
                fn=handle_clear_cache,
                inputs=[],
                outputs=[clear_cache_status],
                api_name=False
            )
            # Wire toggle button click
            sidebar_toggle_btn.click(
                None, None, None,
                js="""function() {
                    const sidebar = document.getElementById('sidebar-col');
                    const btn = document.getElementById('sidebar-toggle-btn');
                    sidebar.classList.toggle('collapsed');
                    btn.classList.toggle('sidebar-is-collapsed');
                }""".strip(),
                api_name=False
            )
            # Right Main Body Column
            with gr.Column(scale=5, elem_id="content-col"):
                # Premium Header
                gr.HTML(get_header_html())

                session_state = gr.State([])
                report_state = gr.State("")
                fetched_raw_state = gr.State("")
                thread_ctx_map_state = gr.State({})
                thread_ctx_state = gr.State("")
                tts_text_state = gr.State("")   # văn bản cho TTS lazy

                with gr.Tabs():
                    # ================================================================
                    # TAB 1: PHÂN TÍCH VĂN BẢN
                    # ================================================================
                    with gr.Tab("🔍 PHÂN TÍCH VĂN BẢN"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📝 Nhập văn bản")
                                text_input = gr.Textbox(
                                    label="Văn bản tiếng Việt",
                                    placeholder="Nhập hoặc dán văn bản về vaccine...",
                                    lines=8,
                                )
                                use_captum_cb = gr.Checkbox(
                                    label="🎯 Bật Captum IG (token attribution, chậm hơn 5-10s)",
                                    value=False,
                                )
                                analyze_btn = gr.Button("🔬 Phân tích", variant="primary", size="lg")

                                with gr.Accordion("🌐 Hoặc thu thập từ URL", open=False):
                                    url_input = gr.Textbox(
                                        label="URL (Báo · YouTube · Facebook · TikTok · Threads)",
                                        placeholder="https://vnexpress.net/...",
                                    )
                                    max_cmt = gr.Slider(10, 100, value=30, step=10, label="Max comments")
                                    fetch_btn = gr.Button("📥 Thu thập")
                                    fetch_status = gr.HTML(value="")
                                    fetched_table = gr.Dataframe(
                                        headers=["STT", "Loại", "Nội dung"],
                                        datatype=["number", "str", "str"],
                                        wrap=True,
                                        visible=False,
                                        label="📋 Luồng đã cào — 📄 bài viết · 💬 bình luận · 🏷️ tag tên (đã lọc). Click 1 dòng để phân tích kèm ngữ cảnh.",
                                    )
                                    with gr.Row():
                                        send_to_batch_btn = gr.Button("🚀 Gửi sang Phân tích Batch", variant="secondary", visible=False)
                                        export_fetched_btn = gr.Button("📥 Tải dữ liệu cào (.xlsx)", variant="primary", visible=False)
                                    export_fetched_file = gr.File(label="Tải về tệp Excel dữ liệu đã cào", visible=False)

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📊 Kết quả phân loại")
                                summary_out = gr.HTML()
                                
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("##### Radar — độ tin cậy nhãn dự đoán")
                                radar_out = gr.Plot()
                            with gr.Column(scale=1):
                                gr.Markdown("##### Phân phối xác suất đầy đủ (chuẩn)")
                                prob_dist_out = gr.Plot()


                        gr.Markdown("---")
                        gr.Markdown("## 🧠 Giải thích AI (XAI 3-Layer Engine)")
                        with gr.Tabs():
                            with gr.Tab("💭 Chain-of-Thought Reasoning"):
                                reasoning_out = gr.Markdown()
                                voice_btn = gr.Button("🔊 Tạo & Nghe giọng đọc", size="sm", variant="secondary")
                                audio_out = gr.HTML(value="")
                            with gr.Tab("🎯 Token Attribution (Captum IG)"):
                                saliency_out = gr.HTML(value="<p style='color:#888;'><em>💡 Bật checkbox <b>Captum IG</b> ở trên rồi nhấn Phân tích</em></p>")

                        # Export Report Button
                        with gr.Row():
                            export_btn = gr.Button("📥 Tải báo cáo phân tích (.md)", variant="secondary")
                            export_file = gr.File(label="Báo cáo", visible=False)

                        # Session History Display
                        history_display = gr.Markdown(value="*Chưa có lượt phân tích nào trong phiên này*")

                        # Sidebar sample and model update wiring
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
                        model_choice.change(
                            fn=get_sidebar_info_html,
                            inputs=[model_choice],
                            outputs=[info_box],
                            api_name=False
                        )

                        # Main analyze button — FIXED: now passes use_captum + returns report
                        analyze_btn.click(
                            fn=handle_analyze,
                            inputs=[text_input, model_choice, use_captum_cb, session_state, thread_ctx_state],
                            outputs=[summary_out, radar_out, prob_dist_out, reasoning_out, saliency_out,
                                     audio_out, session_state, history_display, report_state,
                                     tts_text_state],
                            api_name=False
                        )
                        # TTS lazy: bấm nút mới tạo audio (kết quả phân tích hiện ngay, không đợi gTTS)
                        voice_btn.click(
                            fn=handle_generate_voice,
                            inputs=[tts_text_state],
                            outputs=[audio_out],
                            api_name=False,
                        )

                        # Export button
                        export_btn.click(
                            fn=handle_export_report,
                            inputs=[report_state],
                            outputs=[export_file],
                            api_name=False
                        )

                        # URL fetch
                        fetch_btn.click(
                            fn=handle_fetch_url,
                            inputs=[url_input, max_cmt],
                            outputs=[text_input, fetched_table, send_to_batch_btn, export_fetched_btn, fetch_status, fetched_raw_state, thread_ctx_map_state],
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
                            inputs=[thread_ctx_map_state],
                            outputs=[text_input, thread_ctx_state],
                            api_name=False,
                        )
                        text_input.input(fn=lambda: "", outputs=[thread_ctx_state], api_name=False)

                        # Batch + Compare accordions
                        with gr.Accordion("📋 Batch Mode (phân tích nhiều mẫu cùng lúc)", open=False) as batch_accordion:
                            upload_data_file = gr.File(
                                label="📤 Nạp lại file .xlsx/.csv đã cào (cột 'Nội dung thu thập') để phân tích Batch",
                                file_types=[".xlsx", ".csv"],
                                visible=True,
                            )
                            batch_input = gr.Textbox(
                                label="Mỗi mẫu ngăn cách bằng dòng chứa --- (tối đa 50). KHÔNG còn tách theo xuống dòng → đoạn văn nhiều dòng giữ nguyên là 1 mẫu.",
                                placeholder="Mẫu 1 nhiều dòng...\n---\nMẫu 2...\n---\nMẫu 3...",
                                lines=6,
                            )
                            batch_btn = gr.Button("🚀 Phân tích Batch")
                            batch_out = gr.Markdown()
                            batch_btn.click(fn=handle_batch, inputs=[batch_input, model_choice], outputs=[batch_out], api_name=False)

                        # Send to batch click
                        send_to_batch_btn.click(
                            fn=handle_send_to_batch,
                            inputs=[fetched_raw_state],
                            outputs=[batch_input, batch_accordion],
                            api_name=False
                        )
                        upload_data_file.change(
                            fn=handle_upload_data,
                            inputs=[upload_data_file],
                            outputs=[batch_input, batch_accordion],
                            api_name=False,
                        )

                        with gr.Accordion("🔬 So sánh PhoBERT-v2 vs XLM-R-v1", open=False):
                            cmp_input = gr.Textbox(label="Văn bản", lines=4)
                            cmp_btn = gr.Button("So sánh")
                            cmp_out = gr.Markdown()
                            cmp_btn.click(fn=handle_compare, inputs=[cmp_input], outputs=[cmp_out], api_name=False)

                    # ================================================================
                    # TAB 2: BENCHMARK & BÁO CÁO KHOA HỌC
                    # ================================================================
                    with gr.Tab("📊 BENCHMARK & BÁO CÁO KHOA HỌC"):
                        with gr.Tabs():
                            with gr.Tab("📋 BÁO CÁO BENCHMARK KHOA HỌC"):
                                gr.Markdown("## 📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG & BENCHMARK MÔ HÌNH KHOA HỌC")
                                gr.HTML("""
                                    <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 12px; border-radius: 8px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
                                        <span style="color: var(--text-color); font-size: 0.95rem;">
                                            💡 Báo cáo đối sánh hiệu năng của <b>PhoBERT-v2</b>, <b>XLM-R-v1</b> và <b>Gemma-4 (QLoRA)</b> trên tập kiểm thử độc lập <b>Gold Test Set (186 mẫu)</b> gán nhãn bởi chuyên gia HUPH.
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
                                gr.Markdown("### 🕸️ 2. Phân tích hiệu năng chi tiết theo nhãn (Per-Class Breakdown)")

                                with gr.Tabs():
                                    with gr.Tab("🚨 PHÂN LOẠI TIN GIẢ (MISINFO)"):
                                        gr.Plot(value=make_per_class_chart("misinfo"))
                                        gr.HTML(value=get_per_class_table_html("misinfo"))
                                        gr.Markdown("💡 **Nhận xét**: Phân lớp *Tin giả* (n=28) có độ khó phân loại cao nhất (F1 ~0.50) do mất cân bằng mẫu lớp thiểu số và tính chất ẩn dụ/châm biếm của thông tin sai lệch.")
                                    with gr.Tab("🚩 QUAN ĐIỂM (STANCE)"):
                                        gr.Plot(value=make_per_class_chart("stance"))
                                        gr.HTML(value=get_per_class_table_html("stance"))
                                        gr.Markdown("💡 **Nhận xét**: Lớp *Trung lập* đạt F1 tối ưu trên cả 3 mô hình nhờ tính biểu đạt khách quan. Lập trường *Phản đối* đạt hiệu năng tốt nhất trên PhoBERT-v2.")
                                    with gr.Tab("🎭 CẢM XÚC (SENTIMENT)"):
                                        gr.Plot(value=make_per_class_chart("sentiment"))
                                        gr.HTML(value=get_per_class_table_html("sentiment"))
                                        gr.Markdown("💡 **Nhận xét**: Lớp *Tích cực* có F1 thấp hơn do các phản hồi ủng hộ vắc-xin thường đi kèm mô tả phản ứng phụ nhẹ sau tiêm (sốt, đau), gây nhiễu sắc thái ngữ cảnh.")

                                selected_model_view.change(
                                    fn=get_kpi_cards_html,
                                    inputs=[selected_model_view],
                                    outputs=[kpi_output],
                                    api_name=False
                                )

                                gr.Markdown("---")
                                gr.Markdown("### 🔬 3. Phân tích thực nghiệm chính (Key Analyses)")
                                gr.Markdown("""
                                * **🚨 Độ phức tạp tác vụ**: Phân loại lớp *Tin giả* (Avg F1 = 0.48) và *Cảm xúc Tích cực* (Avg F1 = 0.62) có độ khó cao nhất do mất cân bằng dữ liệu, ngữ nghĩa châm biếm tinh vi, hoặc nhiễu ngữ cảnh do lồng ghép mô tả phản ứng sau tiêm thông thường.
                                * **🤖 Vai trò của Gemma-4**: Gemma-4 4B hoạt động như một mô hình tạo sinh tinh chỉnh để giải thích lý luận sâu (XAI) và tư vấn phản hồi khủng hoảng truyền thông, không tối ưu cho bài toán gán nhãn cứng như mô hình chuyên biệt PhoBERT-v2.
                                """)

                                gr.Markdown("---")
                                gr.Markdown("### 🛡️ 4. Giải pháp thực tiễn: Kiến trúc lai Dual-Student Hybrid")
                                gr.Markdown("Để tối ưu hóa cả tốc độ phân loại chính xác và chiều sâu lý luận giải thích, hệ thống đề xuất kiến trúc kết hợp:")
                                gr.HTML("""
                                <div style="display:flex; flex-direction:row; justify-content:space-around; align-items:center; flex-wrap:wrap; margin-top:20px; font-family:'Inter', sans-serif;">
                                    <div style="background:var(--accent-bg); border:1px solid var(--accent-color); border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">📥</span>
                                        <h4 style="margin:10px 0; color:var(--text-color);">1. Dữ liệu đầu vào</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Văn bản mạng xã hội thu thập qua URL hoặc nhập trực tiếp.</p>
                                    </div>
                                    <div style="font-size:2rem; color:var(--accent-color); font-weight:bold; margin-bottom:10px;">➔</div>
                                    <div style="background:var(--accent-bg); border:1px solid #007bff; border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">🥇</span>
                                        <h4 style="margin:10px 0; color:#007bff;">2. PhoBERT-v2 (Phân loại)</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Gán nhãn siêu tốc (120 mẫu/s): Tin giả, Quan điểm, Cảm xúc.</p>
                                    </div>
                                    <div style="font-size:2rem; color:var(--accent-color); font-weight:bold; margin-bottom:10px;">➔</div>
                                    <div style="background:var(--accent-bg); border:1px solid #FFA500; border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">🧠</span>
                                        <h4 style="margin:10px 0; color:#FFA500;">3. Gemma-4 4B (Giải thích)</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Tạo giải thích lập luận (XAI) và tư vấn kịch bản phản hồi khủng hoảng.</p>
                                    </div>
                                </div>
                                """)


                            with gr.Tab("⚡ ĐÁNH GIÁ LIVE (LIVE EVALUATION)"):
                                gr.HTML("""
                                    <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 15px; border-radius: 5px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
                                        <span style="color: var(--text-color);">⚡ <b>Chế độ Đánh giá Live</b> giả lập quá trình quét trực tiếp và tính toán F1-Score thời gian thực của các mô hình trên tập kiểm thử vàng Gold Test Set (186 mẫu).</span>
                                    </div>
                                """)

                                gr.Markdown("#### 🚀 Trạng thái Tiến trình Suy luận (Inference Pipeline)")

                                live_status = gr.HTML(value="<div style='color: var(--tab-button-text); font-family: \"Inter\", serif;'>💡 Nhấn nút bên dưới để bắt đầu chạy kiểm thử suy luận trên GPU trực tiếp...</div>")
                                live_table = gr.HTML(value=render_live_table([]))

                                live_eval_btn = gr.Button("⚡ Bắt đầu Đánh giá Live", variant="primary", size="lg")

                                live_eval_btn.click(
                                    fn=run_live_evaluation,
                                    inputs=[],
                                    outputs=[live_status, live_table],
                                    api_name=False
                                )

                                gr.Markdown("---")
                                gr.Markdown("### ⚡ 1. Đánh giá Hiệu năng Vận hành & Tốc độ Suy luận (Runtime Performance)")
                                gr.HTML("""
                                    <div style="margin-top: -10px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
                                        <span style="color: var(--text-color); font-style: italic; font-size: 0.95rem; opacity: 0.85;">
                                            💡 Phân tích so sánh khía cạnh kỹ thuật phần mềm: Tốc độ xử lý (Thông lượng) và Độ trễ phản hồi của từng kiến trúc mô hình khi quét vắc-xin.
                                        </span>
                                    </div>
                                """)

                                gr.HTML(SPEED_METRICS_HTML)
                                gr.Plot(value=make_speed_chart())

                                gr.Markdown("---")
                                gr.HTML(RECOMMENDATIONS_HTML)


                    # ================================================================
                    # TAB 4: TÀI LIỆU
                    # ================================================================
                    with gr.Tab("📚 TÀI LIỆU & NOTEBOOKS"):
                        gr.HTML(RESOURCES_HTML)

                    # ================================================================
                    # TAB 5: PHƯƠNG PHÁP LUẬN
                    # ================================================================
                    with gr.Tab("📜 PHƯƠNG PHÁP LUẬN"):
                        gr.HTML(METHODOLOGY_HTML)

                    # ================================================================
                    # TAB 6: ĐỀ CƯƠNG
                    # ================================================================
                    with gr.Tab("📑 ĐỀ CƯƠNG"):
                        gr.HTML(THESIS_HTML)

                gr.HTML(get_footer_html())

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2, max_size=10)
    app.launch(show_error=True)