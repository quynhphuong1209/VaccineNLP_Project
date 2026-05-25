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
        "hung2903/gemma-4-E4B-unsloth-vaccine-xai",
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
APIFY_TOKENS = [os.environ.get(f"APIFY_TOKEN_{i}", "") for i in range(1, 6)]
APIFY_TOKENS = [t for t in APIFY_TOKENS if t]

# Path B: External Gemma endpoint (Kaggle+ngrok)
GEMMA_ENDPOINT_URL = os.environ.get("GEMMA_ENDPOINT_URL", "").strip()

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
    "🟡 Nghi ngại": "Cún mình chỉ tiêm mũi ở viện về nhà là ko tiêm gì nữa. Bây giờ 2 tuổi rồi. Ai hỏi t vẫn nói tiêm đủ.",
    "✅ Thông tin chuẩn": "Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng để phòng các bệnh truyền nhiễm nguy hiểm.",
    "🔵 Câu hỏi tư vấn": "Trâm Trần ví dụ như Ko có tiêm 6in1 hay 5in1, mà tiêm từng mũi từng bệnh phải không ạ?",
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
        # CRITICAL v2.1: low_cpu_mem_usage=False to prevent meta tensor errors.
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
        # Load weights (strict=False handles head_ vs heads. naming variants)
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}...")
        
        # CRITICAL FIX v2.1: Materialize any remaining meta tensors to CPU.
        meta_count = 0
        for name, param in list(model.named_parameters()):
            if param.is_meta:
                meta_count += 1
                with torch.no_grad():
                    parent = model
                    parts = name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    new_tensor = torch.zeros(param.shape, dtype=torch.float32, device="cpu")
                    if "weight" in parts[-1] and len(param.shape) == 2:
                        nn.init.xavier_uniform_(new_tensor)
                    new_param = nn.Parameter(new_tensor, requires_grad=False)
                    setattr(parent, parts[-1], new_param)
        
        if meta_count > 0:
            logger.warning(f"Materialized {meta_count} meta tensor(s) — checkpoint may be incomplete")
        
        # Force entire model to CPU as final safety check
        model = model.to("cpu")
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


def _try_openrouter(text: str) -> Optional[str]:
    """Layer 2b: OpenRouter public inference API (google/gemma-3-4b-it or llama fallback)."""
    if not OPENROUTER_KEY:
        return None
    import urllib.request
    prompt = _build_xai_prompt(text)
    payload = json.dumps({
        "model": "google/gemma-3-4b-it:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.7,
    }).encode()
    # Fallback chain: gemma-3-4b → llama-3.2-3b
    or_models = [
        "google/gemma-3-4b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
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
                return content
        except Exception as e:
            logger.debug(f"OpenRouter {or_model} failed: {e}")
            continue
    return None


def _try_hf_inference(text: str) -> Optional[str]:
    """Layer 2c: HF Inference API via chat_completion (avoids StopIteration bug)."""
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
            # Use chat_completion (OpenAI-compatible) instead of text_generation
            # to avoid StopIteration error from empty stream
            result = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7,
            )
            content = result.choices[0].message.content or ""
            if content.strip() and len(content.strip()) > 30:
                logger.info(f"✅ HF Inference ({model_id}) reasoning OK")
                return content.strip()
        except Exception as e:
            logger.debug(f"HF Inference {model_id} failed: {e}")
            continue
    return None


def query_gemma_api(text: str) -> Optional[str]:
    """XAI Layer 2: ngrok → OpenRouter → HF Inference (ordered by reliability)."""
    # Layer 2a: Dedicated self-hosted endpoint (Kaggle+ngrok)
    external = query_gemma_external_endpoint(text)
    if external:
        logger.info("✅ XAI via ngrok self-host endpoint")
        return external

    # Layer 2b: OpenRouter (free public inference — most reliable for demo)
    or_result = _try_openrouter(text)
    if or_result:
        return or_result

    # Layer 2c: HF Inference API (merged Gemma model)
    hf_result = _try_hf_inference(text)
    if hf_result:
        return hf_result

    return None


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
    """4-layer reasoning: Cache → ngrok → OpenRouter → HF Inference → Fallback."""
    # Layer 1: XAI cache (instant, 266 entries)
    cached = find_xai_reasoning_cache(text)
    if cached:
        return cached, "✅ Từ XAI Cache (Gold Test Set)"

    # Layers 2a-2c: Live inference
    api_reasoning = query_gemma_api(text)
    if api_reasoning:
        # Translate if response came back in English
        if is_mostly_english(api_reasoning):
            api_reasoning = translate_to_vietnamese(api_reasoning)
        # Determine source label
        if GEMMA_ENDPOINT_URL and api_reasoning:
            source_label = "✅ Từ Gemma-4 self-host (ngrok)"
        elif OPENROUTER_KEY:
            source_label = "🤖 Từ Gemma-3 (OpenRouter API)"
        else:
            source_label = "🤖 Từ Gemma-4 HF Inference API"
        return api_reasoning, source_label

    # Layer 4: Template fallback
    fallback = generate_smart_fallback(
        result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"]
    )
    return fallback, "⚠️ Fallback template (Tất cả API không khả dụng)"


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

    html = '<div style="line-height:1.9; padding:20px; border-radius:15px; background:#f8f9fa; border:1px dashed rgba(100,255,218,0.4); font-family:Times New Roman, serif;">'
    for tok, score in zip(tokens, attr_norm):
        if tok in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "<unk>"]:
            continue
        tok_clean = tok.replace("▁", " ").replace("@@", "").replace("Ġ", " ")
        abs_score = abs(score)
        if abs_score < 0.15:
            html += f'<span style="color:#666; opacity:0.55;">{tok_clean}</span> '
        else:
            intensity = min(abs_score, 0.7)
            if pred_class == 0:
                bg = f"rgba(255,75,75,{intensity})"
            else:
                bg = f"rgba(100,255,218,{intensity})"
            html += f'<span style="background:{bg}; padding:2px 6px; border-radius:4px; font-weight:bold;">{tok_clean}</span> '
    html += "</div>"
    label = LABEL_MAPS["misinfo"].get(pred_class, "?")
    html += f'<p style="font-size:11px; color:#888; margin-top:10px;">💡 Dự đoán: <b>{label}</b> · Token có màu đậm hơn = đóng góp lớn hơn vào quyết định model</p>'
    return html


# ============================================================================
# AI VOICE (gTTS) - Thread-safe with unique temp file
# ============================================================================

def text_to_speech(text: str) -> Optional[str]:
    """Generate audio with unique filename to avoid race condition."""
    if not text:
        return None
    try:
        from gtts import gTTS
        # Hash text để tạo unique filename
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        # Dùng tempfile để đảm bảo writable + unique
        tmp_file = tempfile.NamedTemporaryFile(
            prefix=f"tts_{text_hash}_", suffix=".mp3", delete=False, dir="/tmp"
        )
        tmp_file.close()
        
        tts = gTTS(text=text[:500], lang="vi")
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        logger.warning(f"gTTS failed: {e}")
        return None


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


def fetch_url(url: str, max_comments: int = 30) -> Tuple[str, str]:
    """Main fetcher dispatcher."""
    if not url or not url.strip():
        return "", "Vui lòng nhập URL"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "", "❌ URL không hợp lệ"

    kind = detect_source(url)
    if kind == "news":
        content = fetch_news(url)
        info = "📰 Báo điện tử (Tier 1, ~2s)"
    elif kind == "youtube":
        content = fetch_youtube(url, max_comments)
        info = "🎬 YouTube (Tier 2, ~10s)"
    else:
        content = fetch_apify(url)
        info = "📱 Mạng xã hội (Tier 3, ~60s)"

    if content.startswith("❌"):
        return "", content
    return content, f"**Nguồn:** {info}"


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
        barmode="group", height=400,
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
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
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
        barmode="group", height=380,
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
    fig.update_layout(height=420, margin=dict(l=15, r=15, t=15, b=15))
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
    # Check both same-level and parent-level to support both local workspace and HF Space
    for p in [Path(__file__).parent / "huph_logo.png", Path(__file__).parent.parent / "huph_logo.png"]:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            except Exception:
                pass
    return "https://huph.edu.vn/uploads/logo/logo-huph.png"


def render_result_cards_html(result: Dict, elapsed: float, model_choice: str) -> str:
    """Render beautiful HTML cards with progress bars for multi-task predictions."""
    html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; width: 100%; font-family: \'Times New Roman\', Times, serif; margin-bottom: 10px;">'
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
            <div style="font-size: 0.85rem; color: #8892b0; margin-top: 5px;">
                Thô: <span style="text-decoration: line-through;">{conf_raw:.1f}%</span>
            </div>
            <div style="font-size: 1.05rem; color: {color}; font-weight: bold; margin-top: 2px;">
                Đã hiệu chuẩn (T={T:.2f}): {conf_cal:.1f}%
            </div>
            """
        else:
            conf_html = f"""
            <div style="font-size: 0.95rem; color: #8892b0; margin-top: 5px;">
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
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #a8b2d1;">
                        <span>{class_label}</span>
                        <span style="font-size: 10px; color: #8892b0;">Thô: {pct_raw:.1f}% → <strong style="color: {class_color};">{pct_cal:.1f}%</strong></span>
                    </div>
                    <div style="background: #112240; border-radius: 5px; height: 6px; margin-top: 2px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="background: {class_color}; width: {pct_cal}%; height: 100%; border-radius: 5px; box-shadow: 0 0 5px {class_color}80;"></div>
                    </div>
                </div>
                """
            else:
                breakdown_items += f"""
                <div style="margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #a8b2d1;">
                        <span>{class_label}</span>
                        <span style="color: {class_color}; font-weight: bold;">{pct_raw:.1f}%</span>
                    </div>
                    <div style="background: #112240; border-radius: 5px; height: 6px; margin-top: 2px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="background: {class_color}; width: {pct_raw}%; height: 100%; border-radius: 5px; box-shadow: 0 0 5px {class_color}80;"></div>
                    </div>
                </div>
                """

        html += f"""
        <div style="flex: 1; min-width: 240px; background: rgba(17, 34, 64, 0.6); border: 1px solid {color}80; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3); transition: all 0.3s ease; backdrop-filter: blur(10px);">
            <div style="font-size: 40px; margin-bottom: 5px;">{icon}</div>
            <div style="font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px;">{axis_name}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-bottom: 8px;">{label}</div>
            {conf_html}
            
            <div style="margin-top: 15px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 10px; text-align: left;">
                <div style="font-size: 0.8rem; font-weight: bold; color: #64ffda; text-transform: uppercase; margin-bottom: 5px;">Chi tiết nhãn:</div>
                {breakdown_items}
            </div>
        </div>
        """
    html += '</div>'
    html += f"<div style='margin-top: 15px; font-style: italic; color: #8892b0; font-family: \"Times New Roman\", serif; font-size: 0.9rem; text-align: right;'>⏱️ Thời gian xử lý: {elapsed:.2f}s · Mô hình: {model_choice}</div>"
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
        height=320,
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
        height=380
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
        <div style="flex: 1; min-width: 130px; border: 1px solid #64ffda; border-radius: 8px; padding: 10px; text-align: center; background: rgba(100,255,218,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: #8892b0;">Support (Tổng mẫu)</p>
            <h3 style="margin: 5px 0; color: #64ffda; font-size: 1.5rem;">{support}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #3db882; border-radius: 8px; padding: 10px; text-align: center; background: rgba(61,184,130,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: #8892b0;">True Positives (TP)</p>
            <h3 style="margin: 5px 0; color: #3db882; font-size: 1.5rem;">{tp}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #ff4b4b; border-radius: 8px; padding: 10px; text-align: center; background: rgba(255,75,75,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: #8892b0;">False Positives (FP)</p>
            <h3 style="margin: 5px 0; color: #ff4b4b; font-size: 1.5rem;">{fp}</h3>
        </div>
        <div style="flex: 1; min-width: 130px; border: 1px solid #FFA500; border-radius: 8px; padding: 10px; text-align: center; background: rgba(255,165,0,0.03);">
            <p style="margin: 0; font-size: 0.85rem; color: #8892b0;">False Negatives (FN)</p>
            <h3 style="margin: 5px 0; color: #FFA500; font-size: 1.5rem;">{fn}</h3>
        </div>
    </div>
    <div style="font-style: italic; color: #ccd6f6; font-size: 0.95rem; margin-bottom: 20px;">
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
        return (error_html, None, "", "", None, history, 
                session_history_to_markdown(history), "")

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()
    
    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    result = predict(text, model_choice)
    if not result:
        error_html = f'<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">❌ Không thể load mô hình {model_choice} — kiểm tra HF_TOKEN</div>'
        return (error_html, None, "", "", None, history,
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
    audio_path = text_to_speech(voice_text) if voice_text else None

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
    return (summary_html, radar, reasoning_md, saliency_html, audio_path, history, history_md, report_md)


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


def handle_fetch(url: str, max_comments: int) -> Tuple[str, str]:
    """Multi-source URL fetcher."""
    content, info = fetch_url(url, max_comments)
    return content, info


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
/* VaccineNLP — Original Design (Times New Roman, High Contrast Dark) */

body, html, .gradio-container {
    background-color: #0a1628 !important;
    color: #ccd6f6 !important;
    font-family: 'Times New Roman', Times, Georgia, serif !important;
}

/* Global font — Times New Roman throughout */
*, p, span, div, li, td, th, label, input, textarea, select, button {
    font-family: 'Times New Roman', Times, Georgia, serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Times New Roman', Times, Georgia, serif !important;
    color: #ffffff !important;
    font-weight: bold !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(10, 22, 40, 0.8) !important; }
::-webkit-scrollbar-thumb { background: rgba(0, 123, 255, 0.5) !important; border-radius: 4px !important; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 123, 255, 0.9) !important; }

/* Tabs */
.tab-nav button {
    background-color: transparent !important;
    color: #8892b0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    font-family: 'Times New Roman', Times, serif !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover {
    color: #ffffff !important;
    background-color: rgba(0, 123, 255, 0.08) !important;
}
.tab-nav button.selected {
    color: #64ffda !important;
    border-bottom: 3px solid #64ffda !important;
    font-weight: 700 !important;
    background-color: rgba(100, 255, 218, 0.05) !important;
}

/* Primary button */
button.primary, button.gr-button-primary {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4) !important;
    transition: all 0.2s ease !important;
    font-family: 'Times New Roman', Times, serif !important;
}
button.primary:hover, button.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0, 123, 255, 0.6) !important;
    background: linear-gradient(135deg, #1a8aff 0%, #007bff 100%) !important;
}

/* Secondary button */
button.secondary, button.gr-button-secondary {
    background-color: rgba(17, 34, 64, 0.8) !important;
    color: #ccd6f6 !important;
    border: 1px solid rgba(100, 255, 218, 0.3) !important;
    border-radius: 8px !important;
    font-family: 'Times New Roman', Times, serif !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover, button.gr-button-secondary:hover {
    border-color: #64ffda !important;
    color: #64ffda !important;
    background-color: rgba(100, 255, 218, 0.08) !important;
}

/* Inputs */
input, textarea, select {
    background-color: rgba(10, 22, 40, 0.9) !important;
    color: #e8f0fe !important;
    border: 1px solid rgba(100, 255, 218, 0.25) !important;
    border-radius: 8px !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #64ffda !important;
    box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.2) !important;
    background-color: rgba(10, 22, 40, 1) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder {
    color: #8892b0 !important;
    opacity: 1 !important;
}

/* Accordion */
.gr-accordion {
    background-color: rgba(10, 22, 40, 0.6) !important;
    border: 1px solid rgba(100, 255, 218, 0.15) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}

/* Markdown text */
.prose, .markdown-body, .gr-markdown {
    color: #ccd6f6 !important;
    font-family: 'Times New Roman', Times, serif !important;
    line-height: 1.7 !important;
    font-size: 1rem !important;
}
.prose a, .markdown-body a, .gr-markdown a {
    color: #64ffda !important;
}
.prose strong, .markdown-body strong {
    color: #ffffff !important;
}

/* Labels */
label, .gr-label {
    color: #ccd6f6 !important;
    font-weight: 600 !important;
    font-family: 'Times New Roman', Times, serif !important;
}
"""

SPEED_METRICS_HTML = """
<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; font-family: 'Times New Roman', serif;">
    <div style="flex: 1; min-width: 200px; border: 1px solid #64ffda; border-radius: 8px; padding: 15px; text-align: center; background: rgba(100,255,218,0.05);">
        <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">🏎️ Tốc độ PhoBERT-v2</p>
        <h2 style="margin: 5px 0; color: #64ffda; font-size: 1.8rem; font-weight: bold;">120.5 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #3db882; font-weight: bold;">Nhanh nhất (Real-time)</span>
    </div>
    <div style="flex: 1; min-width: 200px; border: 1px solid #007bff; border-radius: 8px; padding: 15px; text-align: center; background: rgba(0,123,255,0.05);">
        <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">🚗 Tốc độ XLM-R-v1</p>
        <h2 style="margin: 5px 0; color: #007bff; font-size: 1.8rem; font-weight: bold;">85.2 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #ff4b4b; font-weight: bold;">-29.3% so với PhoBERT</span>
    </div>
    <div style="flex: 1; min-width: 200px; border: 1px solid #FFA500; border-radius: 8px; padding: 15px; text-align: center; background: rgba(255,165,0,0.05);">
        <p style="margin: 0; font-size: 0.9rem; color: #8892b0;">🐢 Tốc độ Gemma-4 4B</p>
        <h2 style="margin: 5px 0; color: #FFA500; font-size: 1.8rem; font-weight: bold;">1.8 mẫu/s</h2>
        <span style="font-size: 0.8rem; color: #ff4b4b; font-weight: bold;">Rần chậm (Phù hợp offline)</span>
    </div>
</div>
"""

RECOMMENDATIONS_HTML = """
<div style="background: rgba(100, 255, 218, 0.03); border: 1px solid rgba(100, 255, 218, 0.2); border-radius: 8px; padding: 20px; font-family: 'Times New Roman', serif;">
    <h4 style="margin-top: 0; color: #64ffda; font-size: 1.2rem;">🤝 Kiến trúc lai đề xuất cho dự án VaccineNLP (HUPH 2026):</h4>
    <ol style="margin-bottom: 0; padding-left: 20px; line-height: 1.6; color: #ccd6f6;">
        <li><b>Vòng ngoài (Real-time Classification - PhoBERT-v2)</b>: Nhờ tốc độ suy luận cực nhanh (120.5 mẫu/giây) và độ chính xác F1 vượt trội, PhoBERT-v2 được đề xuất làm màng lọc trực tiếp ở luồng dữ liệu mạng xã hội để phân loại nhanh tin giả, sắc thái và lập trường.</li>
        <li><b>Vòng trong (Explainable & Strategic Consulting - Gemma-4 4B)</b>: Đối với các mẫu được PhoBERT-v2 nghi ngờ là "Tin giả" hoặc "Tiêu cực cực đoan", hệ thống sẽ đẩy vào hàng đợi offline để Gemma-4 lý luận chuyên sâu (XAI) giải thích lý do gán nhãn và đề xuất kịch bản phản hồi khủng hoảng cho chuyên gia y tế HUPH.</li>
    </ol>
</div>
"""

RESOURCES_HTML = """
<div style="font-family: 'Times New Roman', Times, serif; color: #ccd6f6;">
  <h2 style="color: #64ffda; margin-bottom: 20px; font-size: 1.8rem;">📚 Tài liệu & Notebooks Nghiên cứu</h2>
  
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <!-- Column 1: Kim Manh Hung -->
    <div style="flex: 1; min-width: 300px; background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.2); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
      <h3 style="color: #64ffda; margin-top: 0; border-bottom: 1px solid rgba(100,255,218,0.3); padding-bottom: 10px; font-size: 1.3rem;">👨‍💻 1. Kim Mạnh Hưng (MSSV: 2211090016)</h3>
      
      <div style="margin-top: 15px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">📘 I. KAGGLE NOTEBOOKS:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-phobert-v2-multitask" target="_blank" style="color: #64ffda; text-decoration: none;">PhoBERT Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-xlm-r-v1-multitask-classifier" target="_blank" style="color: #64ffda; text-decoration: none;">XLM-R Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-training" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma QLoRA Training (03A)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-gemma-4-inference" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma XAI Inference (03B)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/kimmnhhng/vaccinenlp-model-benchmark-report" target="_blank" style="color: #64ffda; text-decoration: none;">Model Benchmark Report (04)</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">🤗 II. HUGGINGFACE:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/phobert-vaccine-multitask" target="_blank" style="color: #64ffda; text-decoration: none;">PhoBERT Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/xlmr-vaccine-multitask" target="_blank" style="color: #64ffda; text-decoration: none;">XLM-R Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma XAI Reasoning</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">💻 III. GITHUB:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li>• <a href="https://github.com/hwngkm/VaccineNLP-Thesis" target="_blank" style="color: #64ffda; text-decoration: none;">VaccineNLP Thesis Repo</a></li>
        </ul>
      </div>
    </div>
    
    <!-- Column 2: Dinh Le Quynh Phuong -->
    <div style="flex: 1; min-width: 300px; background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.2); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
      <h3 style="color: #64ffda; margin-top: 0; border-bottom: 1px solid rgba(100,255,218,0.3); padding-bottom: 10px; font-size: 1.3rem;">👩‍💻 2. Đinh Lê Quỳnh Phương (MSSV: 2211090031)</h3>
      
      <div style="margin-top: 15px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">📘 I. KAGGLE NOTEBOOKS:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-phobert-v2-multitask-classifier" target="_blank" style="color: #64ffda; text-decoration: none;">PhoBERT Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-xlm-r-v1-multitask-classifier" target="_blank" style="color: #64ffda; text-decoration: none;">XLM-R Multitask Classifier</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-training" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma QLoRA Training (03A)</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://www.kaggle.com/code/inhlqunhphng/vaccinenlp-gemma-4-inference" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma XAI Inference (03B)</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">🤗 II. HUGGINGFACE:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/phobert-multitask" target="_blank" style="color: #64ffda; text-decoration: none;">PhoBERT Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/xlmr-multitask" target="_blank" style="color: #64ffda; text-decoration: none;">XLM-R Multitask</a></li>
          <li style="margin-bottom: 8px;">• <a href="https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai" target="_blank" style="color: #64ffda; text-decoration: none;">Gemma XAI Reasoning</a></li>
        </ul>
      </div>
      
      <div style="margin-top: 20px;">
        <h4 style="color: #007bff; margin-bottom: 5px; font-size: 1.05rem;">💻 III. GITHUB:</h4>
        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
          <li>• <a href="https://github.com/quynhphuong1209/VaccineNLP_Project" target="_blank" style="color: #64ffda; text-decoration: none;">VaccineNLP Project Repo</a></li>
        </ul>
      </div>
    </div>
  </div>
</div>
"""

METHODOLOGY_HTML = """
<div style="font-family: 'Times New Roman', Times, serif; color: #ccd6f6; line-height: 1.6;">
  <h2 style="color: #64ffda; border-bottom: 1px solid rgba(100,255,218,0.2); padding-bottom: 10px; font-size: 1.8rem; margin-bottom: 20px;">📜 Phương pháp luận & Kiến trúc Hệ thống</h2>
  
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 3; min-width: 300px;">
      <h3 style="color: #007bff; font-size: 1.3rem;">🏗️ 1. Kiến trúc Dual-Student Hybrid</h3>
      <p>Dự án xây dựng hệ thống <b>Ensemble</b> tận dụng ưu điểm của hai dòng kiến trúc Transformer phổ biến nhất hiện nay:</p>
      
      <div style="background: rgba(17,34,64,0.3); border-left: 4px solid #3db882; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
        <strong style="color: #3db882;">Động cơ Phân loại (Classification Engine):</strong>
        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
          <li>PhoBERT-v2 (kiến trúc Encoder)</li>
          <li>Multi-task Learning với 3 heads độc lập</li>
          <li>Ưu điểm: Hiểu sâu ngữ pháp tiếng Việt, phân loại nhãn chính xác cao</li>
        </ul>
      </div>
      
      <div style="background: rgba(17,34,64,0.3); border-left: 4px solid #FFA500; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <strong style="color: #FFA500;">Động cơ Giải thích (XAI Reasoning Engine):</strong>
        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
          <li>Gemma-4 E4B-it (kiến trúc Decoder)</li>
          <li>QLoRA 4-bit fine-tuning</li>
          <li>Ưu điểm: Sinh văn bản giải thích Tiếng Việt mạch lạc</li>
        </ul>
      </div>
      
      <h4 style="color: #64ffda; margin-bottom: 10px;">🛠️ Sơ đồ Luồng Xử lý (System Pipeline)</h4>
      <pre style="background: #112240; color: #64ffda; border: 1px solid rgba(100,255,218,0.2); border-radius: 8px; padding: 15px; font-family: monospace; font-size: 0.9rem; line-height: 1.4;">
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
      <h3 style="color: #007bff; font-size: 1.3rem; margin-bottom: 5px;">🎯 2. Ba nhiệm vụ chính</h3>
      
      <div style="background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.1); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
        <strong style="color: #ff4b4b; font-size: 1rem;">🚨 Misinformation Detection</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #8892b0;">Xác định tin giả về vaccine dựa trên các nguồn tin cậy và đối chiếu chéo.</p>
      </div>
      
      <div style="background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.1); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
        <strong style="color: #007bff; font-size: 1rem;">🎯 Stance Analysis</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #8892b0;">Phân tích quan điểm cộng đồng: Ủng hộ, Phản đối hoặc Trung lập với tiêm chủng vaccine.</p>
      </div>
      
      <div style="background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.1); border-radius: 8px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
        <strong style="color: #00c853; font-size: 1rem;">💭 Sentiment Analysis</strong>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #8892b0;">Nhận diện sắc thái cảm xúc của người viết: Tích cực, Tiêu cực, hoặc Trung tính.</p>
      </div>
      
      <div style="background: rgba(17,34,64,0.4); border: 1px solid rgba(0,123,255,0.2); border-radius: 8px; padding: 15px; margin-top: 10px;">
        <h4 style="color: #007bff; margin: 0 0 8px 0; font-size: 1.05rem;">🧪 Quy trình thực nghiệm</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; line-height: 1.5; color: #8892b0;">
          <li><b>Dataset:</b> 1.856 mẫu Silver + 186 mẫu Gold Test</li>
          <li><b>Hardware:</b> GPU NVIDIA T4 (Kaggle)</li>
          <li><b>Optimization:</b> QLoRA 4-bit + Temperature Scaling</li>
          <li><b>Evaluation:</b> Macro F1 + ECE (Expected Calibration Error)</li>
        </ul>
      </div>
    </div>
  </div>
  
  <div style="margin-top: 25px; border-top: 1px solid rgba(100,255,218,0.2); padding-top: 15px;">
    <h3 style="color: #64ffda; font-size: 1.25rem;">💡 Tại sao Explainable AI (XAI)?</h3>
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
<div style="font-family: 'Times New Roman', Times, serif; color: #ccd6f6; line-height: 1.6;">
  <h2 style="color: #64ffda; border-bottom: 1px solid rgba(100,255,218,0.2); padding-bottom: 10px; font-size: 1.8rem; margin-bottom: 20px;">📑 Đề cương & Mục lục Đồ án tốt nghiệp</h2>
  
  <div style="background: rgba(0, 123, 255, 0.05); border-left: 5px solid #007bff; padding: 20px; border-radius: 5px; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
    <h3 style="margin: 0; color: #ffffff; font-size: 1.15rem; text-transform: uppercase;">📝 Tên Đề Tài Đồ Án tốt nghiệp:</h3>
    <p style="margin: 8px 0 0 0; font-size: 1.25rem; font-weight: bold; color: #64ffda; line-height: 1.4;">
      "Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"
    </p>
    <p style="margin: 5px 0 0 0; font-style: italic; color: #8892b0; font-size: 1rem;">
      (Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)
    </p>
  </div>

  <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 25px;">
    <div style="flex: 1; min-width: 300px; background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.1); border-radius: 12px; padding: 20px;">
      <h3 style="color: #007bff; border-bottom: 1px solid rgba(0,123,255,0.2); padding-bottom: 8px; margin-top: 0; font-size: 1.25rem;">📌 Cấu trúc 6 Chương chính</h3>
      <ul style="list-style-type: none; padding-left: 0; line-height: 1.8;">
        <li><b>CHƯƠNG 1: ĐẶT VẤN ĐỀ</b> (Lý do chọn đề tài, Mục tiêu MT1-MT3, Câu hỏi RQ1-RQ3)</li>
        <li><b>CHƯƠNG 2: TỔNG QUAN TÀI LIỆU</b> (Định nghĩa Vaccine misinformation, NLP, XAI, Research Gap)</li>
        <li><b>CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU</b> (Chiến lược Tier-Based A/B/C, 8 bước tiền xử lý, Annotation)</li>
        <li><b>CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM</b> (Mô tả Gold Test Set n=186, Macro F1, Calibration, Kiểm định)</li>
        <li><b>CHƯƠNG 5: BÀN LUẬN KHOA HỌC</b> (Diễn giải kết quả chính, so sánh ANTiVax/MiSoVac, Hạn chế đề tài)</li>
        <li><b>CHƯƠNG 6: KẾT LUẬN VÀ KIẾN NGHỊ</b> (Tổng kết mục tiêu, kiến nghị CDC & Bộ Y tế, hướng phát triển)</li>
      </ul>
    </div>
    
    <div style="flex: 1; min-width: 300px; background: rgba(17,34,64,0.4); border: 1px solid rgba(100,255,218,0.1); border-radius: 12px; padding: 20px;">
      <h3 style="color: #007bff; border-bottom: 1px solid rgba(0,123,255,0.2); padding-bottom: 8px; margin-top: 0; font-size: 1.25rem;">🧪 Ba Giả thuyết Nghiên cứu (Hypotheses)</h3>
      <ul style="list-style-type: none; padding-left: 0; line-height: 1.8; color: #ccd6f6;">
        <li style="margin-bottom: 12px;">
          <strong style="color: #ff4b4b;">• Giả thuyết H1 (Chấp nhận):</strong><br>
          Cảm xúc tiêu cực ↔ Lập trường phản đối vaccine (Kiểm định Chi-square đạt ý nghĩa thống kê cao, p < 10⁻⁴⁰).
        </li>
        <li style="margin-bottom: 12px;">
          <strong style="color: #007bff;">• Giả thuyết H2 (Chấp nhận):</strong><br>
          Nền tảng mạng xã hội ↔ Tỷ lệ lan truyền tin giả y tế (Kiểm định G-test, p = 2,14 × 10⁻³).
        </li>
        <li style="margin-bottom: 12px;">
          <strong style="color: #00c853;">• Giả thuyết H3 (Chấp nhận):</strong><br>
          Lập trường phản đối/nghi ngại ↔ Tỷ lệ xuất hiện tin giả (Kiểm định Chi-square, p < 10⁻¹⁴).
        </li>
      </ul>
    </div>
  </div>
  
  <div style="background: rgba(17,34,64,0.5); border: 1px solid rgba(100,255,218,0.2); border-radius: 12px; padding: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
    <h3 style="color: #64ffda; margin-top: 0; border-bottom: 1px solid rgba(100,255,218,0.3); padding-bottom: 8px; font-size: 1.3rem;">👥 Thông tin Đồ án tốt nghiệp HUPH</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 30px; margin-top: 15px;">
      <div style="flex: 1; min-width: 250px;">
        <h4 style="color: #007bff; margin: 0 0 10px 0; font-size: 1.1rem;">Sinh viên thực hiện:</h4>
        <p style="margin: 5px 0;"><b>1. Kim Mạnh Hưng</b> · MSSV: 2211090016</p>
        <p style="margin: 5px 0;"><b>2. Đinh Lê Quỳnh Phương</b> · MSSV: 2211090031</p>
        <p style="margin: 5px 0; color: #8892b0; font-size: 0.9rem;">Lớp: CNCQ Khoa học dữ liệu 1-1A</p>
      </div>
      <div style="flex: 1; min-width: 250px;">
        <h4 style="color: #007bff; margin: 0 0 10px 0; font-size: 1.1rem;">Giảng viên hướng dẫn:</h4>
        <p style="margin: 5px 0;"><b>TS. Trần Lâm Quân</b></p>
        <p style="margin: 5px 0; color: #8892b0; font-size: 0.9rem;">Giảng viên Khoa học dữ liệu · Trường Đại học Y tế Công cộng</p>
      </div>
    </div>
  </div>
</div>
"""


def get_header_html():
    logo_src = get_huph_logo_base64()
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px; color: white; font-family: 'Times New Roman', serif; box-shadow: 0 8px 20px rgba(0,0,0,0.2);">
      <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
        <img src="{logo_src}" style="width: 70px; height: 70px; object-fit: contain; filter: drop-shadow(0 0 8px rgba(255,255,255,0.5));" alt="HUPH Logo">
        <div style="flex: 2; min-width: 300px;">
          <h1 style="margin: 0; font-size: 2rem;">🦠 VaccineNLP</h1>
          <p style="margin: 5px 0; font-size: 1.1rem; opacity: 0.95;">
            Phát hiện Tin giả & Phân tích Thái độ Vaccine tiếng Việt
          </p>
          <p style="margin: 5px 0; font-size: 0.95rem; opacity: 0.85; font-style: italic;">
            Kiến trúc Dual-Student Hybrid · PhoBERT-v2 + Gemma-4 E4B
          </p>
        </div>
        <div style="flex: 1; min-width: 250px; border-left: 2px solid rgba(255,255,255,0.3); padding-left: 20px;">
          <div style="font-size: 0.85rem; opacity: 0.9; line-height: 1.6;">
            <b>🎓 Đồ án tốt nghiệp HUPH 2026</b><br>
            Kim Mạnh Hưng · 2211090016<br>
            Đinh Lê Quỳnh Phương · 2211090031<br>
            <span style="font-size: 0.8rem; opacity: 0.8;">GVHD: TS. Trần Lâm Quân</span>
          </div>
        </div>
      </div>
    </div>
    """


def get_footer_html():
    logo_src = get_huph_logo_base64()
    return f"""
    <div style="background: linear-gradient(135deg, #0a192f 0%, #112240 100%); color: #a8b2d1; padding: 40px 20px; border-radius: 15px; margin-top: 40px; font-family: 'Times New Roman', serif; border-top: 4px solid #007bff; box-shadow: 0 -10px 25px rgba(0, 123, 255, 0.15);">
      <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: space-around;">
        <div style="flex: 1.2; min-width: 250px; text-align: center; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 15px;">
          <img src="{logo_src}" style="width: 90px; height: 90px; object-fit: contain; margin-bottom: 10px; filter: drop-shadow(0 0 8px rgba(0,123,255,0.2));" alt="HUPH Logo">
          <h3 style="color: #ffffff; font-size: 1rem; margin: 5px 0;">TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</h3>
          <p style="font-size: 0.85rem; color: #8892b0; margin: 5px 0;">📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
          <p style="font-size: 0.85rem; margin: 5px 0;">🌐 <a href="https://huph.edu.vn/" target="_blank" style="color: #64ffda; text-decoration: none;">huph.edu.vn</a></p>
        </div>
        <div style="flex: 1.5; min-width: 250px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 15px;">
          <h3 style="color: #007bff; font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px;">🔬 Đề tài đồ án</h3>
          <p style="color: #ffd700; font-weight: bold; font-style: italic; font-size: 0.95rem; line-height: 1.5;">
            "Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"
          </p>
          <p style="color: #8892b0; font-size: 0.85rem; margin-top: 8px;">
            (Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)
          </p>
        </div>
        <div style="flex: 1.2; min-width: 250px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 15px;">
          <h3 style="color: #007bff; font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px;">👥 Nhóm thực hiện</h3>
          <p style="margin: 5px 0;"><b>1. Kim Mạnh Hưng</b></p>
          <p style="font-size: 0.85rem; color: #8892b0; margin: 0 0 10px 0;">MSSV: 2211090016 · Lớp: CNCQ KHDL1-1A</p>
          <p style="margin: 5px 0;"><b>2. Đinh Lê Quỳnh Phương</b></p>
          <p style="font-size: 0.85rem; color: #8892b0; margin: 0 0 10px 0;">MSSV: 2211090031 · Lớp: CNCQ KHDL1-1A</p>
        </div>
        <div style="flex: 1; min-width: 200px;">
          <h3 style="color: #007bff; font-size: 1.1rem; text-transform: uppercase; margin-bottom: 15px;">👨‍🏫 GV Hướng dẫn</h3>
          <p style="font-size: 1.05rem; font-weight: bold; color: #ffffff;">TS. Trần Lâm Quân</p>
          <p style="font-size: 0.85rem; color: #8892b0; margin-top: 5px; line-height: 1.4;">
            Giảng viên Khoa học dữ liệu<br>
            Trường Đại học Y tế Công cộng<br>
            📧 <a href="mailto:tlq@huph.edu.vn" style="color: #64ffda; text-decoration: none;">tlq@huph.edu.vn</a>
          </p>
        </div>
      </div>
      <hr style="border-color: rgba(255,255,255,0.1); margin: 25px 0 15px 0;">
      <p style="text-align: center; font-size: 0.85rem; color: #8892b0; margin: 0;">
        © 2026 VaccineNLP Project | Đồ án tốt nghiệp chuyên ngành Khoa học Dữ liệu - HUPH
      </p>
    </div>
    """

# ============================================================================
# GRADIO UI BUILDER
# ============================================================================

def build_app():
    """Build the Gradio Blocks app with 6 tabs."""
    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS_STYLE) as app:
        # Premium Header
        gr.HTML(get_header_html())

        session_state = gr.State([])
        report_state = gr.State("")

        with gr.Tabs():
            # ================================================================
            # TAB 1: PHÂN TÍCH VĂN BẢN
            # ================================================================
            with gr.Tab("🔍 PHÂN TÍCH VĂN BẢN"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Nhập văn bản")
                        model_choice = gr.Dropdown(
                            choices=list(CONFIG["models"].keys()),
                            value="PhoBERT-v2",
                            label="Mô hình phân loại",
                        )
                        sample_choice = gr.Dropdown(
                            choices=["Tự nhập"] + list(SAMPLE_TEXTS.keys()),
                            value="Tự nhập",
                            label="Hoặc chọn mẫu thử",
                        )
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
                            fetch_status = gr.Markdown()

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Kết quả phân loại")
                        summary_out = gr.HTML()
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
                        audio_out = gr.Audio(label="🔊 AI Voice (gTTS)", type="filepath")
                    with gr.Column():
                        gr.Markdown("### 🎯 Token Attribution (Captum IG)")
                        saliency_out = gr.HTML(value="<p style='color:#888;'><em>💡 Bật checkbox <b>Captum IG</b> ở trên rồi nhấn Phân tích</em></p>")

                # Export Report Button
                with gr.Row():
                    export_btn = gr.Button("📥 Tải báo cáo phân tích (.md)", variant="secondary")
                    export_file = gr.File(label="Báo cáo", visible=False)

                # Session History Display
                history_display = gr.Markdown(value="*Chưa có lượt phân tích nào trong phiên này*")

                # Sample selection updates text
                def update_text_from_sample(sample_key):
                    if sample_key == "Tự nhập":
                        return ""
                    return SAMPLE_TEXTS.get(sample_key, "")

                sample_choice.change(
                    fn=update_text_from_sample,
                    inputs=[sample_choice],
                    outputs=[text_input],
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
                    fn=handle_fetch,
                    inputs=[url_input, max_cmt],
                    outputs=[text_input, fetch_status],
                    api_name=False
                )

                # Batch + Compare accordions
                with gr.Accordion("📋 Batch Mode (phân tích nhiều mẫu cùng lúc)", open=False):
                    batch_input = gr.Textbox(
                        label="Mỗi dòng = 1 mẫu (tối đa 50)",
                        placeholder="Mẫu 1: Vaccine COVID gây vô sinh...\nMẫu 2: Tiêm phòng rất tốt...",
                        lines=6,
                    )
                    batch_btn = gr.Button("🚀 Phân tích Batch")
                    batch_out = gr.Markdown()
                    batch_btn.click(fn=handle_batch, inputs=[batch_input, model_choice], outputs=[batch_out], api_name=False)

                with gr.Accordion("🔬 So sánh PhoBERT-v2 vs XLM-R-v1", open=False):
                    cmp_input = gr.Textbox(label="Văn bản", lines=4)
                    cmp_btn = gr.Button("So sánh")
                    cmp_out = gr.Markdown()
                    cmp_btn.click(fn=handle_compare, inputs=[cmp_input], outputs=[cmp_out], api_name=False)

            # ================================================================
            # TAB 2: BENCHMARK (Enhanced với Confusion Matrix)
            # ================================================================
            with gr.Tab("📊 BENCHMARK & BÁO CÁO"):
                gr.Markdown(render_benchmark_md())
                gr.Plot(value=make_benchmark_chart())

                gr.Markdown("---")
                gr.Markdown("## 🔥 Confusion Matrix — PhoBERT-v2 Sentiment")
                gr.Plot(value=make_confusion_matrix_chart())

                gr.Markdown("---")
                gr.Markdown("## 🏎️ Hiệu năng Vận hành & Tốc độ Suy luận (Throughput)")
                gr.HTML(SPEED_METRICS_HTML)
                gr.Plot(value=make_speed_chart())

                gr.Markdown("---")
                gr.HTML(RECOMMENDATIONS_HTML)

                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### 💡 Phân tích Kết quả

                    **PhoBERT-v2** đạt Macro F1 trung bình cao nhất (0.6967), khẳng định vai trò Classification Engine.

                    **XLM-R-v1** đạt F1 Misinformation cao nhất (0.7038) nhờ ưu thế đa ngôn ngữ — phù hợp khi tin giả vaccine VN có nguồn dịch từ tiếng Anh.

                    **Gemma-4 4B** đạt F1 Sentiment cao nhất (0.7700), vượt PhoBERT 4.3 điểm phần trăm — minh chứng cho thiết kế Dual-Student Hybrid với Decoder model bổ sung cho tác vụ sắc thái cảm xúc.
                    """
                )

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
