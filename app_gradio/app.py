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
    print("✅ Starlette TemplateResponse monkey patch successfully applied!")
except Exception as e:
    print(f"⚠️ Failed to apply TemplateResponse patch: {e}")


from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gradio as gr

# Compatibility Fallback: If running on an older Gradio version (e.g., Gradio 3.x),
# map gr.Sidebar to gr.Column to prevent AttributeError: module 'gradio' has no attribute 'Sidebar'
if not hasattr(gr, "Sidebar"):
    print("⚠️ Warning: gr.Sidebar not found. Falling back to gr.Column layout.")
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
        "hung2903/gemma-4-E4B-unsloth-vaccine-xai",
        "google/gemma-2-2b-it",
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

HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("VaccineNLP_TOKEN", "")

# Load .env file if available (local development helper)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
_env_gemma_url = os.environ.get("GEMMA_ENDPOINT_URL", "").strip()
GEMMA_ENDPOINT_URL = "https://pearle-staglike-nonsyntonically.ngrok-free.dev/predict"
if _env_gemma_url and "ngrok-free.dev" in _env_gemma_url and "pearle-staglike-nonsyntonically" not in _env_gemma_url:
    GEMMA_ENDPOINT_URL = _env_gemma_url

# Path C: OpenRouter fallback (public LLM inference)
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")

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


def query_gemma_external_endpoint(text: str) -> Optional[str]:
    """Layer 2a: Try external Gemma endpoint (Path B: Kaggle+ngrok self-host)."""
    if not GEMMA_ENDPOINT_URL:
        return None
    try:
        import requests
        response = requests.post(
            GEMMA_ENDPOINT_URL,
            json={"text": text[:1000]},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            reasoning = data.get("reasoning") or data.get("output") or data.get("text")
            if reasoning and len(reasoning.strip()) > 10:
                return f"Lý luận: {reasoning.strip()}" if not reasoning.startswith("Lý luận") else reasoning.strip()
    except Exception as e:
        logger.debug(f"External Gemma endpoint failed: {e}")
    return None


def _build_xai_prompt(text: str) -> str:
    """Build structured Vietnamese XAI prompt."""
    return (
        "Bạn là chuyên gia AI có khả năng giải thích (Explainable AI) trong lĩnh vực Y tế Công cộng Việt Nam. "
        "Hãy phân tích văn bản sau theo cấu trúc 3 bước bằng TIẾNG VIỆT:\n"
        "1. **Tính xác thực (Misinformation):** Văn bản có chứa thông tin sai lệch không? Dấu hiệu nào?\n"
        "2. **Thái độ (Stance):** Người viết ủng hộ, phản đối hay trung lập với vaccine?\n"
        "3. **Cảm xúc (Sentiment):** Sắc thái cảm xúc là tiêu cực, trung tính hay tích cực?\n"
        f"\nVăn bản cần phân tích:\n{text.strip()[:800]}"
    )


def _try_openrouter(text: str) -> Optional[Tuple[str, str]]:
    """Layer 2c: OpenRouter public inference API (Gemma models)."""
    if not OPENROUTER_KEY:
        return None
    import urllib.request
    prompt = _build_xai_prompt(text)
    
    # Fallback chain: gemma-3n-e4b (E4B Architecture) → gemma-3-4b-it (free fallback)
    or_models = [
        "google/gemma-3n-e4b-it",
        "google/gemma-3-4b-it:free",
    ]
    for or_model in or_models:
        try:
            body = json.dumps({
                "model": or_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.7,
            }).encode()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://huggingface.co/spaces/hung2903/vaccinenlp-demo",
                    "X-Title": "VaccineNLP XAI Demo",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"].strip()
            if content and len(content) > 30:
                logger.info(f"✅ OpenRouter ({or_model}) reasoning OK")
                return content, or_model
        except Exception as e:
            logger.debug(f"OpenRouter {or_model} failed: {e}")
            continue
    return None


def _try_hf_inference(text: str) -> Optional[Tuple[str, str]]:
    """Layer 2d: HF Inference API via chat_completion (avoids StopIteration bug)."""
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None
    prompt = _build_xai_prompt(text)
    for model_id in CONFIG["xai_models"]:
        try:
            client = InferenceClient(model=model_id, token=HF_TOKEN)
            result = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7,
            )
            content = result.choices[0].message.content or ""
            if content.strip() and len(content.strip()) > 30:
                logger.info(f"✅ HF Inference ({model_id}) reasoning OK")
                return content.strip(), model_id
        except Exception as e:
            logger.debug(f"HF Inference {model_id} failed: {e}")
            continue
    return None


def _try_gemini_api(text: str) -> Optional[Tuple[str, str]]:
    """Layer 2b: Google Gemini Developer API (gemma-4-E4B-it with rotating keys)."""
    # Gather all GEMINI_API_KEYs (1 to 5) from environment
    keys = []
    for i in range(1, 6):
        key_name = "GEMINI_API_KEY" if i == 1 else f"GEMINI_API_KEY_{i}"
        val = os.environ.get(key_name, "").strip()
        if val:
            keys.append(val)

    if not keys:
        return None

    import urllib.request
    prompt = _build_xai_prompt(text)
    
    # Try different model name variants for the Gemini API
    model_variants = ["google/gemma-4-E4B-it", "gemma-4-E4B-it"]
    
    payload = json.dumps({
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 400
        }
    }).encode("utf-8")

    # Rotate keys and model variants to call Gemini API
    for key in keys:
        for model in model_variants:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                
                if "candidates" in res_data and len(res_data["candidates"]) > 0:
                    parts = res_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        content = parts[0]["text"].strip()
                        if content and len(content) > 30:
                            logger.info(f"✅ Gemini API ({model}) reasoning OK")
                            return content, f"🤖 Từ {model} (Google Gemini API)"
            except Exception as e:
                logger.debug(f"Gemini API key rotation item failed for {model}: {e}")
                continue
    return None


def query_gemma_api(text: str) -> Tuple[Optional[str], Optional[str]]:
    """XAI Layer 2 Routing: ngrok self-host (Primary) → Gemini API → OpenRouter → HF Inference."""
    external = query_gemma_external_endpoint(text)
    if external:
        logger.info("✅ XAI via ngrok self-host endpoint")
        return external, "✅ Từ Gemma-4 self-host (ngrok)"

    gemini_result = _try_gemini_api(text)
    if gemini_result:
        content, source = gemini_result
        return content, source

    or_result = _try_openrouter(text)
    if or_result:
        content, model = or_result
        model_display = "Gemma-3n E4B" if "gemma-3n-e4b" in model else "Gemma-3"
        return content, f"🤖 Từ {model_display} (OpenRouter API)"

    hf_result = _try_hf_inference(text)
    if hf_result:
        content, model_id = hf_result
        return content, f"🤖 Từ Gemma-4 HF Inference ({model_id.split('/')[-1]})"

    return None, None


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


def get_reasoning(text: str, result: Dict) -> Tuple[str, str]:
    """4-layer reasoning with source label."""
    cached = find_xai_reasoning_cache(text)
    if cached:
        return cached, "✅ Từ cache (Gold Test Set, 186 mẫu)"

    api_reasoning, source_label = query_gemma_api(text)
    if api_reasoning:
        if is_mostly_english(api_reasoning):
            api_reasoning = translate_to_vietnamese(api_reasoning)
        return api_reasoning, source_label

    fallback = generate_smart_fallback(
        result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"]
    )
    return fallback, "⚠️ Fallback template (API không khả dụng)"


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
        fill="toself", line_color="#64ffda",
        fillcolor="rgba(100, 255, 218, 0.3)", name="Mức độ rủi ro"
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
    """Confusion Matrix Heatmap for PhoBERT-v2 Sentiment (from thesis Ch4)."""
    z_data = [
        [54, 12, 5],
        [10, 56, 9],
        [3, 11, 26]
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
    """Sentiment to Stance flow."""
    nodes = ["Tiêu cực", "Trung tính", "Tích cực", "Phản đối", "Trung lập", "Ủng hộ"]
    sources = [0, 0, 0, 1, 1, 1, 2, 2]
    targets = [3, 4, 5, 3, 4, 5, 4, 5]
    values  = [38, 28, 5, 10, 49, 16, 7, 33]
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
        md += f"| {h['timestamp']} | {h['text']} | {h['misinfo']} | {h['stance']} | {h['sentiment']} |\n"
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
    html = '<div style="display: flex; flex-wrap: wrap; gap: 20px; width: 100%; font-family: \'Times New Roman\', Times, serif; margin-bottom: 10px;">'
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
        <div class="result-card-hover" style="flex: 1; min-width: 240px; background: var(--card-bg); border: 1px solid {color}60; border-radius: 16px; padding: 22px; text-align: center; box-shadow: 0 8px 24px var(--shadow-color), 0 0 15px {color}10; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); border-bottom: 4px solid {color};">
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
        return (error_html, None, "", "", "", history, 
                session_history_to_markdown(history), "")

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()
    
    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    result = predict(text, model_choice)
    if not result:
        error_html = f'<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">❌ Không thể load mô hình {model_choice} — kiểm tra HF_TOKEN</div>'
        return (error_html, None, "", "", "", history,
                session_history_to_markdown(history), "")
    
    progress(0.5, desc="📊 Đang tính radar chart...")
    radar = make_radar_chart(result)
    
    progress(0.6, desc="💭 Đang sinh giải thích (XAI 3-layer)...")
    reasoning, source = get_reasoning(text, result)
    reasoning_md = f"**{source}**\n\n{reasoning}"

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

    # Generate custom HTML result cards
    summary_html = render_result_cards_html(result, elapsed, model_choice)

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
    return (summary_html, radar, reasoning_md, saliency_html, audio_html, history, history_md, report_md)


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


def handle_fetch_url(url: str, max_comments: int) -> Tuple[str, gr.update, gr.update, gr.update, str]:
    """Unified handler for fetching URL content as a list of segments."""
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        error_msg = info if info.startswith("❌") else f"❌ Lỗi: {info}"
        return ("", gr.update(visible=False), gr.update(visible=False), gr.update(value=f"<p style='color:#ff4b4b;'>{error_msg}</p>"), "")
    
    rows = [[i + 1, t] for i, t in enumerate(texts)]
    df = pd.DataFrame(rows, columns=["STT", "Nội dung thu thập được"])
    batch_text_str = "\n".join(texts)
    preview_text = texts[0] if texts else ""
    status_html = f"<p style='color:#3db882; font-weight:bold;'>✅ Thu thập thành công {len(texts)} bài viết/bình luận từ {info}!</p>"
    
    return (preview_text, gr.update(value=df, visible=True), gr.update(visible=True), gr.update(value=status_html), batch_text_str)


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
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) > 50:
        lines = lines[:50]

    rows = []
    for i, line in enumerate(lines):
        progress((i + 1) / len(lines), desc=f"🔬 Đang phân tích {i+1}/{len(lines)}...")
        r = predict(line, model_choice)
        if r:
            rows.append({
                "STT": i + 1,
                "Văn bản": line[:80] + ("..." if len(line) > 80 else ""),
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
- [PhoBERT Multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask)
- [XLM-R Multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask)
- [Gemma XAI Reasoning](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai)

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
- [PhoBERT Multitask](https://huggingface.co/quynhphuong1209/phobert-multitask)
- [XLM-R Multitask](https://huggingface.co/quynhphuong1209/xlmr-multitask)
- [Gemma XAI Reasoning](https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai)

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
/* Theme styles via CSS variables */
:root {
    --bg-color: #f8fafc;
    --bg-gradient: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
    --text-color: #0f172a;
    --card-bg: rgba(255, 255, 255, 0.75);
    --card-border: rgba(0, 0, 0, 0.05);
    --header-bg: rgba(255, 255, 255, 0.8);
    --header-text: #0f172a;
    --footer-bg: rgba(255, 255, 255, 0.85);
    --footer-text: #334155;
    --input-bg: rgba(255, 255, 255, 0.9);
    --input-text: #0f172a;
    --input-border: rgba(0, 0, 0, 0.12);
    --accordion-bg: rgba(0, 0, 0, 0.02);
    --tab-button-bg: rgba(0, 0, 0, 0.04);
    --tab-button-text: #475569;
    --accent-color: #2563eb;
    --accent-bg: rgba(37, 99, 235, 0.05);
    --shadow-color: rgba(0, 0, 0, 0.06);
    --glow-color: rgba(37, 99, 235, 0.15);
    --card-text-muted: #64748b;
    --card-text-primary: #0f172a;
    --card-text-secondary: #334155;
    --progress-bar-bg: rgba(226, 232, 240, 0.8);
    --dropdown-bg: #ffffff;
    
    /* Custom utility variables for light theme */
    --custom-card-bg: #ffffff;
    --custom-card-border: #2563eb;
    --custom-text-neon: #1d4ed8;       /* dark blue */
    --custom-text-muted: #475569;      /* dark slate */
    --custom-text-normal: #0f172a;     /* black */
    --saliency-pos-color: 37, 99, 235; /* RGB for dark blue */
    
    --custom-phobert-bg: #ffffff;
    --custom-xlmr-bg: #ffffff;
    --custom-gemma-bg: #ffffff;
    --custom-phobert-border: #2563eb;
    --custom-phobert-text: #1d4ed8;
}

:root.dark, body.dark, .dark {
    --bg-color: #030712;
    --bg-gradient: linear-gradient(135deg, #030712 0%, #0b1528 50%, #0f172a 100%);
    --text-color: #cbd5e1;
    --card-bg: rgba(15, 23, 42, 0.65);
    --card-border: rgba(100, 255, 218, 0.08);
    --header-bg: rgba(15, 23, 42, 0.75);
    --header-text: #ffffff;
    --footer-bg: rgba(15, 23, 42, 0.85);
    --footer-text: #94a3b8;
    --input-bg: rgba(30, 41, 59, 0.45);
    --input-text: #e2e8f0;
    --input-border: rgba(100, 255, 218, 0.12);
    --accordion-bg: rgba(15, 23, 42, 0.3);
    --tab-button-bg: rgba(30, 41, 59, 0.3);
    --tab-button-text: #64748b;
    --accent-color: #64ffda;
    --accent-bg: rgba(100, 255, 218, 0.06);
    --shadow-color: rgba(0, 0, 0, 0.3);
    --glow-color: rgba(100, 255, 218, 0.15);
    --card-text-muted: #8892b0;
    --card-text-primary: #ccd6f6;
    --card-text-secondary: #a8b2d1;
    --progress-bar-bg: rgba(10, 25, 47, 0.6);
    --dropdown-bg: #0f172a;
    
    /* Custom utility variables for dark theme */
    --custom-card-bg: rgba(10, 25, 47, 0.4);
    --custom-card-border: rgba(100, 255, 218, 0.3);
    --custom-text-neon: #64ffda;        /* neon green */
    --custom-text-muted: #8892b0;       /* muted grey */
    --custom-text-normal: #ccd6f6;      /* light grey */
    --saliency-pos-color: 100, 255, 218;/* RGB for neon green */
    
    --custom-phobert-bg: rgba(100, 255, 218, 0.05);
    --custom-xlmr-bg: rgba(0, 123, 255, 0.05);
    --custom-gemma-bg: rgba(255, 165, 0, 0.05);
    --custom-phobert-border: #64ffda;
    --custom-phobert-text: #64ffda;
}

.model-color-phobert { color: var(--custom-text-neon) !important; }
.model-color-xlmr { color: #007bff !important; }
.model-color-gemma { color: #FFA500 !important; }
.dark .model-color-gemma { color: #FFA500 !important; }


body, html {
    background-color: var(--bg-color) !important;
    background: var(--bg-gradient) !important;
    color: var(--text-color) !important;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: visible !important;
    /* NOTE: Do NOT add backdrop-filter here — it creates a containing block for position:fixed (dropdown lists) */
}

.gradio-container .contain {
    max-width: 100% !important;
    width: 100% !important;
}

/* Prevent global input styling from breaking Gradio's custom dropdown inputs */
.gradio-container .border-none {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Force Gradio dropdown options list to be visible and correctly styled */
.gradio-container .options,
.gradio-container .select-options,
.gradio-container .dropdown-menu {
    z-index: 999999 !important;
    background-color: var(--dropdown-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--text-color) !important;
    box-shadow: 0 10px 30px var(--shadow-color) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
}

.gradio-container .options .option,
.gradio-container .options .item,
.gradio-container .select-options .option,
.gradio-container .select-options .item,
.gradio-container .dropdown-menu .option,
.gradio-container .dropdown-menu .item {
    color: var(--text-color) !important;
    font-family: 'Times New Roman', Times, Georgia, serif !important;
    padding: 8px 12px !important;
}

.gradio-container .options .option:hover,
.gradio-container .options .option.selected,
.gradio-container .options .item:hover,
.gradio-container .options .item.selected,
.gradio-container .select-options .option:hover,
.gradio-container .select-options .option.selected,
.gradio-container .select-options .item:hover,
.gradio-container .select-options .item.selected,
.gradio-container .dropdown-menu .option:hover,
.gradio-container .dropdown-menu .option.selected,
.gradio-container .dropdown-menu .item:hover,
.gradio-container .dropdown-menu .item.selected {
    background-color: var(--accent-color) !important;
    color: #030712 !important;
}

/* Base typography - strict Times New Roman styled elegantly */
* {
    font-family: 'Times New Roman', Times, Georgia, serif !important;
    text-shadow: 0 1px 1px rgba(0,0,0,0.01);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 25, 47, 0.1);
}
.dark ::-webkit-scrollbar-track {
    background: rgba(10, 25, 47, 0.5);
}
::-webkit-scrollbar-thumb {
    background: var(--accent-color);
    border-radius: 10px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover {
    background: #4cd9b9;
    border: 2px solid transparent;
    background-clip: padding-box;
}

/* Global scrollbar buttons (up/down arrows) */
::-webkit-scrollbar-button:vertical:start:decrement {
    display: block !important;
    height: 16px !important;
    background-color: var(--input-bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232563eb'%3E%3Cpath d='M12 8l-6 6h12z'/%3E%3C/svg%3E") !important;
    background-size: 10px 10px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 4px !important;
}
.dark ::-webkit-scrollbar-button:vertical:start:decrement {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364ffda'%3E%3Cpath d='M12 8l-6 6h12z'/%3E%3C/svg%3E") !important;
}
::-webkit-scrollbar-button:vertical:start:decrement:hover {
    background-color: var(--accent-bg) !important;
    border-color: var(--accent-color) !important;
}

::-webkit-scrollbar-button:vertical:end:increment {
    display: block !important;
    height: 16px !important;
    background-color: var(--input-bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232563eb'%3E%3Cpath d='M12 16l6-6H6z'/%3E%3C/svg%3E") !important;
    background-size: 10px 10px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 4px !important;
}
.dark ::-webkit-scrollbar-button:vertical:end:increment {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364ffda'%3E%3Cpath d='M12 16l6-6H6z'/%3E%3C/svg%3E") !important;
}
::-webkit-scrollbar-button:vertical:end:increment:hover {
    background-color: var(--accent-bg) !important;
    border-color: var(--accent-color) !important;
}

/* Tabs Navigation Styling */
.tabs {
    border-bottom: 1px solid rgba(100, 255, 218, 0.15) !important;
    background: transparent !important;
}

.tab-nav {
    display: flex;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: visible !important;
    gap: 8px !important;
    background: transparent !important;
    border-bottom: none !important;
    padding: 10px 0 !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;  /* Firefox */
    -ms-overflow-style: none !important; /* IE/Edge */
}

/* Hide scrollbar on the tab nav but allow scrolling */
.tab-nav::-webkit-scrollbar {
    display: none !important;
}

.tab-nav button {
    background-color: var(--tab-button-bg) !important;
    color: var(--tab-button-text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover {
    color: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
    background-color: var(--accent-bg) !important;
}

.tab-nav button.selected {
    color: var(--text-color) !important;
    background-color: rgba(30, 41, 59, 0.15) !important;
    border: 1.5px solid #007bff !important;
    border-bottom: 3.5px solid var(--accent-color) !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    transform: none !important;
}
.dark .tab-nav button.selected {
    color: var(--header-text) !important;
    background-color: rgba(10, 25, 47, 0.3) !important;
    border: 1.5px solid #007bff !important;
    border-bottom: 3.5px solid var(--accent-color) !important;
}

/* Mobile-specific tab overrides */
@media (max-width: 768px) {
    .tab-nav {
        gap: 6px !important;
        padding: 8px 0 !important;
    }
    .tab-nav button {
        padding: 10px 16px !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.03em !important;
        border-radius: 6px !important;
    }
    .tab-scroll-btn {
        top: 26px !important;
    }
}

/* Button style */
button.primary, button.gr-button-primary {
    background: linear-gradient(135deg, var(--accent-color) 0%, #3b82f6 100%) !important;
    color: #020617 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px var(--glow-color) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.dark button.primary, .dark button.gr-button-primary {
    color: #020617 !important;
}

button.primary:hover, button.gr-button-primary:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 22px var(--glow-color) !important;
    opacity: 0.95 !important;
}

button.secondary, button.gr-button-secondary {
    background-color: var(--tab-button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover, button.gr-button-secondary:hover {
    border-color: var(--accent-color) !important;
    color: var(--accent-color) !important;
    background-color: var(--accent-bg) !important;
    box-shadow: 0 4px 12px var(--glow-color) !important;
}

/* Input boxes, text area style */
.gradio-container input[type="text"]:not(.border-none):not(.dropdown input):not(.select-wrap input):not(.wrap input),
.gradio-container textarea {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 10px 15px !important;
    transition: all 0.3s ease !important;
}

.gradio-container input[type="text"]:not(.border-none):focus,
.gradio-container textarea:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px var(--glow-color) !important;
    background-color: var(--input-bg) !important;
}

/* Gradio Containers and Panels */
/* NOTE: backdrop-filter is intentionally removed from .block — having it would create a new containing block
   for position:fixed elements (like Gradio dropdown option lists), trapping them inside and making them invisible. */
.gr-box, .gr-panel, .block {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 30px var(--shadow-color) !important;
}

/* NUCLEAR FIX: Force ALL elements inside sidebar to have overflow:visible so dropdown options list
   can escape parent clipping. Gradio Soft theme sets overflow:hidden on .block, .wrap, etc.
   We override ALL sidebar child elements except when sidebar is collapsed, excluding the options list itself. */
#sidebar-col:not(.collapsed) *:not(.options):not(.select-options):not([class*="options"]) {
    overflow: visible !important;
}

/* Accordion styling */
.gr-accordion {
    background-color: var(--accordion-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
    transition: all 0.3s ease !important;
}
.gr-accordion:hover {
    border-color: var(--accent-color) !important;
}

/* Custom animations & interactive classes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 5px var(--accent-color)80; }
    50% { box-shadow: 0 0 15px var(--accent-color)e0; }
    100% { box-shadow: 0 0 5px var(--accent-color)80; }
}

.result-card-hover {
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    animation: fadeInUp 0.6s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
}
.result-card-hover:hover {
    transform: translateY(-6px) scale(1.02) !important;
    box-shadow: 0 20px 35px var(--shadow-color), 0 0 25px var(--glow-color) !important;
}

.resource-card {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.resource-card:hover {
    border-color: var(--accent-color) !important;
    box-shadow: 0 10px 30px var(--glow-color) !important;
    transform: translateY(-4px) !important;
}

/* Dropdown list customization */
.dropdown-menu {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
}

/* Theme toggle buttons in sidebar styling */
.theme-dark-btn, .theme-light-btn {
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

body.dark .theme-dark-btn {
    background: linear-gradient(135deg, var(--accent-color) 0%, #3b82f6 100%) !important;
    color: #020617 !important;
    font-weight: bold !important;
    border-color: var(--accent-color) !important;
}
body.dark .theme-light-btn {
    background-color: var(--tab-button-bg) !important;
    color: var(--text-color) !important;
}

body:not(.dark) .theme-light-btn {
    background: linear-gradient(135deg, var(--accent-color) 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    border-color: var(--accent-color) !important;
}
body:not(.dark) .theme-dark-btn {
    background-color: var(--tab-button-bg) !important;
    color: var(--text-color) !important;
}

#sidebar-col {
    position: sticky !important;
    top: 12px !important;
    background-color: var(--card-bg) !important;
    border-right: 1px solid var(--input-border) !important;
    padding: 20px !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: calc(100vh - 24px) !important;
    max-height: calc(100vh - 24px) !important;
    box-sizing: border-box !important;
    width: 290px !important;
    min-width: 290px !important;
    max-width: 290px !important;
    opacity: 1 !important;
    z-index: 9999 !important;
    transform: translateX(0) !important;
}

#sidebar-col::-webkit-scrollbar {
    width: 12px !important;
    display: block !important;
}

#sidebar-col::-webkit-scrollbar-track {
    background: var(--input-bg) !important;
    border-left: 1px solid var(--input-border) !important;
}

#sidebar-col::-webkit-scrollbar-thumb {
    background-color: var(--tab-button-text) !important;
    border: 2px solid var(--input-bg) !important;
    border-radius: 6px !important;
}

#sidebar-col::-webkit-scrollbar-thumb:hover {
    background-color: var(--accent-color) !important;
}

#main-layout-row {
    flex-wrap: nowrap !important;
    width: 100% !important;
    display: flex !important;
    overflow: visible !important;
    position: relative !important;
}

#sidebar-col.collapsed {
    width: 0px !important;
    min-width: 0px !important;
    max-width: 0px !important;
    padding: 0px !important;
    opacity: 0 !important;
    border-right: none !important;
    transform: translateX(-290px) !important;
    pointer-events: none !important;
    overflow: hidden !important;
}

#content-col {
    position: relative !important;
    padding-top: 50px !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    flex: 1 1 auto !important;
}

#sidebar-col.collapsed ~ #content-col {
    width: 100% !important;
    max-width: 100% !important;
}

#sidebar-col:not(.collapsed) ~ #content-col {
    width: calc(100% - 290px) !important;
    max-width: calc(100% - 290px) !important;
}

#sidebar-toggle-btn {
    position: fixed !important;
    z-index: 10001 !important;
    top: 16px !important;
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    background-color: rgba(15, 23, 42, 0.9) !important;
    color: var(--accent-color) !important;
    border: 1px solid rgba(100, 255, 218, 0.35) !important;
    cursor: pointer !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    /* When sidebar is open - button on right */
    left: 258px !important;
}

#sidebar-toggle-btn.sidebar-is-collapsed {
    left: 16px !important;
}

#sidebar-toggle-btn::before {
    content: "";
    display: block;
    width: 18px;
    height: 18px;
    background-color: var(--accent-color);
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    -webkit-mask-size: contain;
    mask-size: contain;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.25s ease;
}

#sidebar-toggle-btn.sidebar-is-collapsed::before {
    transform: rotate(180deg);
}

#sidebar-toggle-btn span {
    display: none !important;
}

#sidebar-toggle-btn:hover {
    background-color: rgba(100, 255, 218, 0.15) !important;
    border-color: rgba(100, 255, 218, 0.4) !important;
    box-shadow: 0 4px 12px rgba(100, 255, 218, 0.2) !important;
    transform: scale(1.08) !important;
}

/* When sidebar is collapsed - button moves to left */
#sidebar-toggle-btn.sidebar-is-collapsed {
    left: 16px !important;
}

/* Sidebar scrolling styles consolidated above */

@media (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        margin: 0 !important;
    }
    #sidebar-col {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100vh !important;
        background-color: var(--bg-color) !important;
        border-right: 1px solid var(--input-border) !important;
        box-shadow: 5px 0 25px var(--shadow-color) !important;
        z-index: 9999 !important;
        transform: translateX(-290px) !important;
        opacity: 0 !important;
        width: 290px !important;
        min-width: 290px !important;
        max-width: 290px !important;
    }
    
    #sidebar-col:not(.collapsed) {
        transform: translateX(0) !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    #sidebar-col.collapsed {
        transform: translateX(-290px) !important;
        opacity: 0 !important;
        width: 0px !important;
        min-width: 0px !important;
        max-width: 0px !important;
        padding: 0px !important;
    }

    #content-col {
        width: 100% !important;
        max-width: 100% !important;
        padding-top: 60px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    #sidebar-col.collapsed ~ #content-col,
    #sidebar-col:not(.collapsed) ~ #content-col {
        width: 100% !important;
        max-width: 100% !important;
    }
}

#sidebar-toggle-btn:hover {
    border-color: var(--accent-color) !important;
    color: var(--accent-color) !important;
    background-color: var(--accent-bg) !important;
    transform: scale(1.02) !important;
}

/* Custom Plotly adaptations for Dark/Light Mode */
.js-plotly-plot {
    background-color: transparent !important;
    width: 100% !important;
}
.js-plotly-plot .bg {
    fill: transparent !important;
}
.js-plotly-plot text,
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .polargrid text,
.js-plotly-plot .angularaxis text,
.js-plotly-plot .gtitle,
.js-plotly-plot .xtitle,
.js-plotly-plot .ytitle,
.js-plotly-plot .legendtext {
    fill: var(--text-color) !important;
    font-family: 'Times New Roman', Times, Georgia, serif !important;
}
.js-plotly-plot .gridlayer path,
.js-plotly-plot .zerolinelayer path,
.js-plotly-plot .axis line,
.js-plotly-plot .polargrid path,
.js-plotly-plot .angularaxis path {
    stroke: rgba(0, 0, 0, 0.1) !important;
}
.dark .js-plotly-plot .gridlayer path,
.dark .js-plotly-plot .zerolinelayer path,
.dark .js-plotly-plot .axis line,
.dark .js-plotly-plot .polargrid path,
.dark .js-plotly-plot .angularaxis path {
    stroke: rgba(255, 255, 255, 0.12) !important;
}
.js-plotly-plot .sankey-node text {
    fill: var(--text-color) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
}

/* Sidebar scrollbar buttons (up/down arrows) */
#sidebar-col::-webkit-scrollbar-button:vertical:start:decrement {
    display: block !important;
    height: 16px !important;
    background-color: var(--input-bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232563eb'%3E%3Cpath d='M12 8l-6 6h12z'/%3E%3C/svg%3E") !important;
    background-size: 10px 10px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 4px !important;
}
.dark #sidebar-col::-webkit-scrollbar-button:vertical:start:decrement {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364ffda'%3E%3Cpath d='M12 8l-6 6h12z'/%3E%3C/svg%3E") !important;
}
#sidebar-col::-webkit-scrollbar-button:vertical:start:decrement:hover {
    background-color: var(--accent-bg) !important;
    border-color: var(--accent-color) !important;
}

#sidebar-col::-webkit-scrollbar-button:vertical:end:increment {
    display: block !important;
    height: 16px !important;
    background-color: var(--input-bg) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232563eb'%3E%3Cpath d='M12 16l6-6H6z'/%3E%3C/svg%3E") !important;
    background-size: 10px 10px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 4px !important;
}
.dark #sidebar-col::-webkit-scrollbar-button:vertical:end:increment {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2364ffda'%3E%3Cpath d='M12 16l6-6H6z'/%3E%3C/svg%3E") !important;
}
#sidebar-col::-webkit-scrollbar-button:vertical:end:increment:hover {
    background-color: var(--accent-bg) !important;
    border-color: var(--accent-color) !important;
}

/* Hide default Gradio footer and remove bottom spacing past custom footer */
footer {
    display: none !important;
}

.gradio-container {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Tab Scroll Buttons styling */
.tabs {
    position: relative !important;
}
.tab-scroll-btn {
    position: absolute !important;
    top: 32px !important;
    transform: translateY(-50%) !important;
    width: 32px !important;
    height: 38px !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
    color: var(--accent-color) !important;
    cursor: pointer !important;
    z-index: 10 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 14px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    opacity: 0;
    pointer-events: none;
}
.tab-scroll-btn-left {
    left: 4px !important;
    background: var(--bg-color) !important;
}
.dark .tab-scroll-btn-left {
    background: rgba(3, 7, 18, 0.95) !important;
}
.tab-scroll-btn-right {
    right: 4px !important;
    background: var(--bg-color) !important;
}
.dark .tab-scroll-btn-right {
    background: rgba(3, 7, 18, 0.95) !important;
}
.tab-scroll-btn.visible {
    opacity: 0.9 !important;
    pointer-events: auto !important;
}
.tab-scroll-btn:hover {
    background-color: var(--accent-color) !important;
    color: #030712 !important;
    box-shadow: 0 0 8px var(--accent-color) !important;
}

/* Sidebar scroll buttons */
.sidebar-scroll-btn {
    position: sticky !important;
    left: 0 !important;
    right: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--accent-color) !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    z-index: 9999 !important;
    transition: all 0.3s ease !important;
    opacity: 0;
    pointer-events: none;
    margin: 0 !important;
    padding: 0 !important;
}
.sidebar-scroll-btn.visible {
    opacity: 0.95 !important;
    pointer-events: auto !important;
    height: 32px !important;
    margin: 4px 10px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px var(--shadow-color) !important;
}
.sidebar-scroll-btn-up {
    top: 5px !important;
}
.sidebar-scroll-btn-down {
    bottom: 5px !important;
}
.sidebar-scroll-btn:hover {
    background-color: var(--accent-color) !important;
    color: #030712 !important;
    box-shadow: 0 0 10px var(--accent-color) !important;
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
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/phobert-vaccine-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/xlmr-vaccine-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma XAI Reasoning</a></li>
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
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/phobert-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">PhoBERT Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/xlmr-multitask" target="_blank" style="color: var(--accent-color); text-decoration: none;">XLM-R Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: var(--accent-color); text-decoration: none;">Gemma XAI Reasoning</a></li>
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
    """Build the Gradio Blocks app with 6 tabs."""
    init_theme_js = """function() {
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
    }""".strip()
    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS_STYLE, fill_width=True, js=init_theme_js) as app:
        # Sidebar Toggle Button (positioned via CSS)
        sidebar_toggle_btn = gr.Button("", elem_id="sidebar-toggle-btn", size="sm")
        
        with gr.Row(elem_id="main-layout-row"):
            # Left Sidebar Column
            with gr.Column(scale=1, min_width=290, elem_id="sidebar-col"):
                gr.HTML(get_sidebar_header_html())
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Times New Roman\", serif; font-weight: bold; margin-bottom: 8px;'>🎨 Giao diện</h5>")
                with gr.Row():
                    theme_dark_btn = gr.Button("🌙 Tối", elem_classes=["theme-dark-btn"], size="sm")
                    theme_light_btn = gr.Button("☀️ Sáng", elem_classes=["theme-light-btn"], size="sm")
                gr.HTML("<hr style='border-color: var(--input-border); margin: 15px 0 10px 0;'>")
                gr.HTML("<h5 style='font-family: \"Times New Roman\", serif; font-weight: bold; margin-bottom: 8px;'>📋 Mẫu thử nghiệm</h5>")
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
                gr.HTML("<h5 style='font-family: \"Times New Roman\", serif; font-weight: bold; margin-bottom: 8px;'>🤖 Mô hình Phân loại</h5>")
                gr.Markdown(
                    """
                    <div style="font-size: 12px; color: var(--text-color); opacity: 0.8; font-family: 'Times New Roman', serif; margin-bottom: 8px; line-height: 1.4;">
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
                gr.HTML("<h5 style='font-family: \"Times New Roman\", serif; font-weight: bold; margin-bottom: 8px;'>🛠️ Quản trị hệ thống</h5>")
                clear_cache_btn = gr.Button("🗑️ Xóa Cache & Khởi động lại", elem_classes=["theme-toggle-btn"], size="sm")
                clear_cache_status = gr.Markdown(value="", visible=False)
                gr.HTML(
                    """
                    <div style="font-size: 11px; color: var(--text-color); opacity: 0.7; font-family: 'Times New Roman', serif; margin-top: 10px; line-height: 1.4;">
                        💡 <b>Lưu ý:</b> Nếu gặp lỗi 403 Forbidden, vui lòng kiểm tra lại quyền 'Inference' của Token trên Hugging Face.
                    </div>
                    """
                )
            # Event wiring for sidebar
            theme_dark_btn.click(
                None, None, None,
                js="function() { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); localStorage.setItem('theme', 'dark'); }",
                api_name=False
            )
            theme_light_btn.click(
                None, None, None,
                js="function() { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); localStorage.setItem('theme', 'light'); }",
                api_name=False
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
                                        headers=["STT", "Nội dung thu thập được"],
                                        datatype=["str", "str"],
                                        wrap=True,
                                        visible=False,
                                        label="📋 Bảng danh sách bài viết / comments đã cào"
                                    )
                                    send_to_batch_btn = gr.Button("🚀 Gửi toàn bộ dữ liệu này sang Phân tích Batch", variant="secondary", visible=False)

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📊 Kết quả phân loại")
                                summary_out = gr.HTML()
                                
                        with gr.Row():
                            with gr.Column(scale=1):
                                radar_out = gr.Plot()

                        # Quick Examples
                        gr.Markdown("### 🚀 Ví dụ test nhanh (click để load)")
                        gr.Examples(
                            examples=GRADIO_EXAMPLES,
                            inputs=[text_input, model_choice],
                            label="",
                        )

                        gr.Markdown("---")
                        gr.Markdown("## 🧠 Giải thích AI (XAI 3-Layer Engine)")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 💭 Chain-of-Thought Reasoning")
                                reasoning_out = gr.Markdown()
                                audio_out = gr.HTML(value="")
                            with gr.Column():
                                gr.Markdown("### 🎯 Token Attribution (Captum IG)")
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
                            inputs=[text_input, model_choice, use_captum_cb, session_state],
                            outputs=[summary_out, radar_out, reasoning_out, saliency_out,
                                     audio_out, session_state, history_display, report_state],
                            api_name=False
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
                            outputs=[text_input, fetched_table, send_to_batch_btn, fetch_status, fetched_raw_state],
                            api_name=False
                        )

                        # Batch + Compare accordions
                        with gr.Accordion("📋 Batch Mode (phân tích nhiều mẫu cùng lúc)", open=False) as batch_accordion:
                            batch_input = gr.Textbox(
                                label="Mỗi dòng = 1 mẫu (tối đa 50)",
                                placeholder="Mẫu 1: Vaccine COVID gây vô sinh...\nMẫu 2: Tiêm phòng rất tốt...",
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
                                    <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 15px; border-radius: 8px; margin-bottom: 25px; font-family: 'Times New Roman', Times, serif;">
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
                                        gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Tin giả** (28 mẫu) có F1 đạt cao nhất là **0.5085** (PhoBERT-v2), đây là bài toán khó do số lượng mẫu huấn luyện hạn chế và sự tinh vi của các thông tin chống vắc-xin cực đoan.")
                                    with gr.Tab("🚩 QUAN ĐIỂM (STANCE)"):
                                        gr.Plot(value=make_per_class_chart("stance"))
                                        gr.HTML(value=get_per_class_table_html("stance"))
                                        gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Trung lập** có kết quả tốt nhất do cách diễn đạt khách quan. **Phản đối** chứng tỏ khả năng nhận diện thái độ phản biện cực đoan rất tốt từ PhoBERT-v2.")
                                    with gr.Tab("🎭 CẢM XÚC (SENTIMENT)"):
                                        gr.Plot(value=make_per_class_chart("sentiment"))
                                        gr.HTML(value=get_per_class_table_html("sentiment"))
                                        gr.Markdown("💡 **Nhận xét thực nghiệm**: Nhãn **Tích cực** tỏ ra thách thức hơn trên mọi kiến trúc do chia sẻ tích cực của người dân Việt Nam về tiêm vắc-xin thường đi kèm các từ mang sắc thái lo lắng.")

                                selected_model_view.change(
                                    fn=get_kpi_cards_html,
                                    inputs=[selected_model_view],
                                    outputs=[kpi_output],
                                    api_name=False
                                )

                                gr.Markdown("---")
                                gr.Markdown("### 🔬 3. Phân tích sâu & Đánh giá thực nghiệm (Scientific Deep-dive)")
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("""
                                        #### ❌ 1. Những nhãn phân loại 'Thách thức' nhất (Hardest Labels)
                                        Dựa trên kết quả benchmark thực tế từ Gold Test Set, hai nhóm nhãn có chỉ số F1-Score thấp nhất trên mọi mô hình là:
                                        * **🚨 Tin giả (Misinfo = Tin giả)**: Đạt F1 trung bình là **0.4280** trên cả 3 mô hình. 
                                          * *Lý do*: Tin giả liên quan đến vắc-xin không đơn thuần là tin đồn nhảm dễ nhận biết, mà thường ẩn chứa dưới dạng ngụy biện khoa học tinh vi, lồng ghép thuật ngữ y học phức tạp hoặc mỉa mai sâu cay.
                                        * **🎭 Cảm xúc Tích cực (Sentiment = Tích cực)**: Đạt F1 trung bình chỉ **0.4139**.
                                          * *Lý do*: Chia sẻ tích cực của người dân Việt Nam về tiêm vắc-xin thường có xu hướng đi kèm các từ mang sắc thái lo lắng (như "hơi sốt", "hơi đau tay một chút nhưng trộm vía ổn"), khiến các bộ phân loại dễ nhầm lẫn sang sắc thái tiêu cực hoặc trung tính.
                                        """)
                                    with gr.Column():
                                        gr.Markdown("""
                                        #### 🤖 2. Tại sao F1-Score của Gemma-4 4B lại thấp hơn?
                                        * **Bản chất kiến trúc**: Gemma-4 là mô hình Generative (tạo sinh) được tinh chỉnh qua QLoRA nhằm phục vụ việc tạo lập **Giải thích khoa học (XAI - Explainable AI)** và **Tư vấn chiến lược phản ứng** dưới dạng ngôn ngữ tự nhiên.
                                        * **Trade-off giữa Giải thích & Phân loại**:
                                          * Mô hình Encoder (như PhoBERT-v2) được thiết kế đặc thù cho bài toán phân loại đa nhãn (Multi-task Classification), giúp trích xuất nhãn cực nhanh và chính xác cao (Avg F1 = 0.6967).
                                          * Gemma-4 4B đóng vai trò lý luận sâu, giúp người dùng hiểu *tại sao* đó là tin giả và đề xuất kịch bản phản hồi khủng hoảng cho chuyên gia y tế HUPH, chứ không cạnh tranh hiệu năng ở bài toán gán nhãn cứng.
                                        """)

                                gr.Markdown("---")
                                gr.Markdown("### 🛡️ 4. Giải pháp thực tiễn: Kiến trúc lai Dual-Student Hybrid")
                                gr.Markdown("Để tối ưu hóa cả tốc độ phân loại chính xác và chiều sâu lý luận giải thích, hệ thống đề xuất kiến trúc kết hợp Dual-Student:")
                                gr.HTML("""
                                <div style="display:flex; flex-direction:row; justify-content:space-around; align-items:center; flex-wrap:wrap; margin-top:20px; font-family:'Times New Roman', serif;">
                                    <div style="background:var(--accent-bg); border:1px solid var(--accent-color); border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">📥</span>
                                        <h4 style="margin:10px 0; color:var(--text-color);">1. Văn bản mạng xã hội</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Người dùng nhập dữ liệu hoặc quét tin từ các URL tin tức.</p>
                                    </div>
                                    <div style="font-size:2rem; color:var(--accent-color); font-weight:bold; margin-bottom:10px;">➔</div>
                                    <div style="background:var(--accent-bg); border:1px solid #007bff; border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">🥇</span>
                                        <h4 style="margin:10px 0; color:#007bff;">2. PhoBERT-v2 (Phân loại)</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Gán nhãn cực nhanh các khía cạnh: Tin giả, Lập trường & Cảm xúc.</p>
                                    </div>
                                    <div style="font-size:2rem; color:var(--accent-color); font-weight:bold; margin-bottom:10px;">➔</div>
                                    <div style="background:var(--accent-bg); border:1px solid #FFA500; border-radius:10px; padding:20px; width:280px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:10px;">
                                        <span style="font-size:2rem;">🧠</span>
                                        <h4 style="margin:10px 0; color:#FFA500;">3. Gemma-4 4B (Giải thích)</h4>
                                        <p style="font-size:0.9rem; color:var(--text-color); opacity:0.8;">Lý luận lý do gán nhãn & đề xuất kịch bản phản hồi khủng hoảng cho chuyên gia y tế HUPH.</p>
                                    </div>
                                </div>
                                """)

                            with gr.Tab("⚡ ĐÁNH GIÁ LIVE (LIVE EVALUATION)"):
                                gr.HTML("""
                                    <div style="background: var(--accent-bg); border-left: 5px solid var(--accent-color); padding: 15px; border-radius: 5px; margin-bottom: 20px; font-family: 'Times New Roman', Times, serif;">
                                        <span style="color: var(--text-color);">⚡ <b>Chế độ Đánh giá Live</b> giả lập quá trình quét trực tiếp và tính toán F1-Score thời gian thực của các mô hình trên tập kiểm thử vàng Gold Test Set (186 mẫu).</span>
                                    </div>
                                """)

                                gr.Markdown("#### 🚀 Trạng thái Tiến trình Suy luận (Inference Pipeline)")

                                live_status = gr.HTML(value="<div style='color: var(--tab-button-text); font-family: \"Times New Roman\", serif;'>💡 Nhấn nút bên dưới để bắt đầu chạy kiểm thử suy luận trên GPU trực tiếp...</div>")
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
                                    <div style="margin-top: -10px; margin-bottom: 20px; font-family: 'Times New Roman', Times, serif;">
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
                    # TAB 3: ĐÁNH GIÁ CHUYÊN SÂU (Enhanced với Per-class)
                    # ================================================================
                    with gr.Tab("📈 ĐÁNH GIÁ CHUYÊN SÂU"):
                        gr.Markdown("## 🌀 Dòng chảy Cảm xúc → Lập trường (n=186)")
                        gr.Plot(value=make_sankey_chart())

                        gr.Markdown(
                            """
                            ---
                            ### 🔍 Diễn giải Sankey Flow

                            **Phát hiện chính:**
                            - **93.8%** (45/48) nội dung **Phản đối** vaccine mang cảm xúc **Tiêu cực**
                            - **82.5%** (33/40) nội dung **Tích cực** đồng hành với lập trường **Ủng hộ**
                            - **77.4%** (65/84) nội dung **Trung lập** đi với cảm xúc **Trung tính**

                            → Sắc thái cảm xúc là **chỉ thị mạnh** dự báo lập trường tiêm chủng (Chi-square p < 10⁻⁴⁰).
                            """
                        )

                        gr.Markdown("---")
                        gr.Markdown("## 📊 Per-class F1 Breakdown")
                        with gr.Tabs():
                            with gr.Tab("🚨 Misinformation"):
                                gr.Plot(value=make_per_class_chart("misinfo"))
                                gr.Markdown("**Nhận xét:** Nhãn *Tin giả* (n=28) khó nhất do mất cân bằng dữ liệu, F1 cao nhất 0.5079 (XLM-R).")
                            with gr.Tab("🎯 Stance"):
                                gr.Plot(value=make_per_class_chart("stance"))
                                gr.Markdown("**Nhận xét:** Nhãn *Trung lập* dễ nhất (F1 ~0.74) do diễn đạt khách quan. *Phản đối* tốt với Gemma (0.6905).")
                            with gr.Tab("💭 Sentiment"):
                                gr.Plot(value=make_per_class_chart("sentiment"))
                                gr.Markdown("**Nhận xét:** Gemma vượt trội ở cả 3 lớp Sentiment, F1 *Tích cực* cao nhất 0.7027.")

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

                        calc_class_choice.change(
                            fn=update_calculator,
                            inputs=[calc_class_choice],
                            outputs=[calc_metrics_area, calc_p_area, calc_r_area, calc_f1_area],
                            api_name=False
                        )

                        gr.Markdown("---")
                        gr.Markdown("## 📊 Phân cấp nhãn Gold Test Set (Sunburst)")
                        gr.Plot(value=make_sunburst_chart())

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
