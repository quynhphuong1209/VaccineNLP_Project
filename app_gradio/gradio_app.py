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
APIFY_TOKENS = [os.environ.get(f"APIFY_TOKEN_{i}", "") for i in range(1, 6)]
APIFY_TOKENS = [t for t in APIFY_TOKENS if t]

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
        self.encoder = AutoModel.from_pretrained(
            model_name, token=token, trust_remote_code=True, low_cpu_mem_usage=True
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


def query_gemma_api(text: str) -> Optional[str]:
    """Layer 2: HF Inference API with multi-model fallback."""
    if not HF_TOKEN:
        return None

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None

    short_text = text.strip()[:1000]

    for model_id in CONFIG["xai_models"]:
        try:
            if "gemma-4-E4B" in model_id:
                prompt = (
                    f"Bạn là Trí tuệ Nhân tạo có khả năng giải thích (Explainable AI) trong lĩnh vực Y tế Công cộng. "
                    f"Hãy phân tích văn bản sau đây về chủ đề vắc-xin, đưa ra lý luận chi tiết HOÀN TOÀN bằng tiếng Việt "
                    f"về tính xác thực, thái độ và cảm xúc. Tuyệt đối không dùng tiếng Anh.\n\nVăn bản: {short_text}"
                )
                formatted = f"<|turn>user\n{prompt}\n<|turn>model\nLý luận: "
                stop_seqs = ["<|turn>", "<end_of_turn>"]
            else:
                prompt = f"Hãy phân tích nội dung sau về vắc-xin và giải thích tại sao bằng tiếng Việt: '{short_text}'"
                formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                stop_seqs = ["<end_of_turn>"]

            client = InferenceClient(model=model_id, token=HF_TOKEN)
            response = client.text_generation(
                formatted, max_new_tokens=350, temperature=0.7,
                repetition_penalty=1.2, stop_sequences=stop_seqs,
            )
            if response and len(response.strip()) > 10:
                clean = response.replace("<end_of_turn>", "").replace("<|turn>", "").strip()
                if "gemma-4-E4B" in model_id and not clean.startswith("Lý luận"):
                    clean = "Lý luận: " + clean
                elif "gemma-4-E4B" not in model_id:
                    clean = f"{clean}\n\n*(Giải thích từ Gemma-Engine fallback)*"
                return clean
        except Exception as e:
            logger.debug(f"Model {model_id} failed: {e}")
            continue
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
    """3-layer reasoning with source label."""
    cached = find_xai_reasoning_cache(text)
    if cached:
        return cached, "✅ Từ cache (Gold Test Set, 186 mẫu)"

    api_reasoning = query_gemma_api(text)
    if api_reasoning:
        if is_mostly_english(api_reasoning):
            api_reasoning = translate_to_vietnamese(api_reasoning)
        return api_reasoning, "✅ Từ Gemma-4 HF Inference API"

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
# HANDLERS (Enhanced with progress + report export)
# ============================================================================

def handle_analyze(
    text: str, model_choice: str, use_captum: bool, history: List,
    progress=gr.Progress()
) -> Tuple:
    """Main analysis handler with progress indicator."""
    if not text or not text.strip():
        return ("⚠️ Vui lòng nhập văn bản", None, "", "", None, history, 
                session_history_to_markdown(history), "")

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()
    
    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    result = predict(text, model_choice)
    if not result:
        return ("❌ Không thể load model — kiểm tra HF_TOKEN", None, "", "", None, history,
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

    # Summary với details
    summary_md = ""
    for axis, axis_name in [("misinfo", "Tin giả/Chính xác"), ("stance", "Quan điểm"), ("sentiment", "Cảm xúc")]:
        r = result[axis]
        label = LABEL_MAPS[axis][r["pred"]]
        icon = LABEL_ICONS[axis][r["pred"]]
        color = LABEL_COLORS[axis][r["pred"]]
        conf_raw = max(r["conf_raw"]) * 100
        conf_cal = max(r["conf_cal"]) * 100
        T = r["T"]
        has_cal = abs(T - 1.0) > 0.001
        if has_cal:
            conf_text = f"Thô: ~~{conf_raw:.1f}%~~ → **Hiệu chuẩn (T={T:.2f}): {conf_cal:.1f}%**"
        else:
            conf_text = f"Độ tin cậy: **{conf_raw:.1f}%**"
        summary_md += f"### {icon} {axis_name}: <span style='color:{color}'>**{label}**</span>\n\n{conf_text}\n\n"
    summary_md += f"\n*⏱️ Thời gian: {elapsed:.2f}s · Mô hình: {model_choice}*"

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
    return (summary_md, radar, reasoning_md, saliency_html, audio_path, history, history_md, report_md)


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

- **Dataset:** 1.856 mẫu Silver + 186 mẫu Gold Test
- **Hardware:** GPU NVIDIA T4 (Kaggle)
- **Optimization:** QLoRA 4-bit + Temperature Scaling
- **Evaluation:** Macro F1 + ECE (Expected Calibration Error)

### 💡 Tại sao Explainable AI (XAI)?

Trong lĩnh vực y tế như vaccine, việc chỉ đưa ra nhãn 'Tin giả' là chưa đủ. Hệ thống cần giải thích **tại sao** để:
1. Thuyết phục người dùng tin tưởng AI
2. Hỗ trợ chuyên gia y tế ra quyết định
3. Đáp ứng yêu cầu minh bạch trong y tế công cộng
"""

THESIS_MD = """## 📑 Đề cương & Mục lục Đồ án

### 📝 Tên Đề Tài

**"Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam"**

*(Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)*

### 📌 Cấu trúc 6 Chương

**CHƯƠNG 1: ĐẶT VẤN ĐỀ**
- 1.1 Lý do chọn đề tài
- 1.2 Mục tiêu nghiên cứu (MT1-MT3)
- 1.3 Câu hỏi nghiên cứu (RQ1-RQ3)
- 1.4 Giả thuyết H1, H2, H3

**CHƯƠNG 2: TỔNG QUAN TÀI LIỆU**
- 2.1 Vaccine misinformation: định nghĩa và phân loại
- 2.2 NLP trong public health surveillance
- 2.3 Transformer architectures, LLM, XAI
- 2.4 Khoảng trống nghiên cứu

**CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU**
- 3.2 Thu thập dữ liệu Tier-Based (A/B/C)
- 3.3 Tiền xử lý 8 bước tiếng Việt
- 3.4 LLM-assisted Annotation + HITL
- 3.5 Kiến trúc Dual-Student Hybrid
- 3.6 Phương pháp thống kê (Chi-square, G-test)

**CHƯƠNG 4: KẾT QUẢ**
- 4.1 Mô tả Gold Test Set (n=186)
- 4.2 Macro F1 + Per-class + Calibration
- 4.3 Đánh giá XAI (Gemma + Captum)
- 4.4 Kiểm định H1, H2, H3

**CHƯƠNG 5: BÀN LUẬN**
- 5.1 Diễn giải kết quả chính
- 5.2 So sánh với nghiên cứu trước (ANTiVax, MiSoVac)
- 5.3 Hạn chế của nghiên cứu
- 5.4 Ứng dụng y tế công cộng

**CHƯƠNG 6: KẾT LUẬN VÀ KIẾN NGHỊ**
- 6.1 Tổng kết MT1, MT2, MT3
- 6.2 Kiến nghị cho Bộ Y tế và CDC
- 6.3 Hướng phát triển mở rộng

### 🧪 Ba Giả thuyết Nghiên cứu (Cập nhật 21/05/2026)

- **H1:** Cảm xúc tiêu cực ↔ Lập trường phản đối (Chi-square, p < 10⁻⁴⁰)
- **H2:** Nền tảng nguồn tin ↔ Tỷ lệ tin giả (G-test, p = 2,14 × 10⁻³)
- **H3:** Lập trường ↔ Tỷ lệ tin giả (Chi-square, p < 10⁻¹⁴)

### 👥 Thông tin Nhóm

**Sinh viên thực hiện:**
- Kim Mạnh Hưng · MSSV: 2211090016
- Đinh Lê Quỳnh Phương · MSSV: 2211090031

**Giảng viên hướng dẫn:**
- TS. Trần Lâm Quân

**Cơ sở đào tạo:**
- Trường Đại học Y tế Công cộng (HUPH)
- Lớp: CNCQ KHDL1-1A
- Năm: 2026
"""

HEADER_HTML = """
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px; color: white; font-family: 'Times New Roman', serif; box-shadow: 0 8px 20px rgba(0,0,0,0.2);">
  <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
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

FOOTER_HTML = """
<div style="background: linear-gradient(135deg, #0a192f 0%, #112240 100%); color: #a8b2d1; padding: 30px; border-radius: 10px; margin-top: 30px; font-family: 'Times New Roman', serif;">
  <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: space-around;">
    <div style="flex: 1; min-width: 200px;">
      <h3 style="color: #007bff;">🏫 TRƯỜNG ĐẠI HỌC Y TẾ CÔNG CỘNG</h3>
      <p>📍 Số 1A, Đức Thắng, Bắc Từ Liêm, Hà Nội</p>
      <p>🌐 <a href="https://huph.edu.vn/" style="color: #64ffda;">huph.edu.vn</a></p>
    </div>
    <div style="flex: 1; min-width: 250px;">
      <h3 style="color: #007bff;">👥 NHÓM THỰC HIỆN</h3>
      <p><b>Kim Mạnh Hưng</b> · 2211090016</p>
      <p><b>Đinh Lê Quỳnh Phương</b> · 2211090031</p>
      <p>Lớp: CNCQ KHDL1-1A</p>
    </div>
    <div style="flex: 1; min-width: 200px;">
      <h3 style="color: #007bff;">👨‍🏫 GVHD</h3>
      <p><b>TS. Trần Lâm Quân</b></p>
      <p>📧 tlq@huph.edu.vn</p>
    </div>
  </div>
  <hr style="border-color: rgba(255,255,255,0.1); margin: 20px 0;">
  <p style="text-align: center; opacity: 0.7;">© 2026 VaccineNLP Project · Đồ án tốt nghiệp HUPH · v2.0</p>
</div>
"""


# ============================================================================
# GRADIO UI BUILDER
# ============================================================================

def build_app():
    """Build the Gradio Blocks app with 6 tabs."""
    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue="indigo")) as app:
        # Premium Header
        gr.HTML(HEADER_HTML)

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
                        summary_out = gr.Markdown()
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
                )

                # Main analyze button — FIXED: now passes use_captum + returns report
                analyze_btn.click(
                    fn=handle_analyze,
                    inputs=[text_input, model_choice, use_captum_cb, session_state],
                    outputs=[summary_out, radar_out, reasoning_out, saliency_out,
                             audio_out, session_state, history_display, report_state],
                )

                # Export button
                export_btn.click(
                    fn=handle_export_report,
                    inputs=[report_state],
                    outputs=[export_file],
                )

                # URL fetch
                fetch_btn.click(
                    fn=handle_fetch,
                    inputs=[url_input, max_cmt],
                    outputs=[text_input, fetch_status],
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
                    batch_btn.click(fn=handle_batch, inputs=[batch_input, model_choice], outputs=[batch_out])

                with gr.Accordion("🔬 So sánh PhoBERT-v2 vs XLM-R-v1", open=False):
                    cmp_input = gr.Textbox(label="Văn bản", lines=4)
                    cmp_btn = gr.Button("So sánh")
                    cmp_out = gr.Markdown()
                    cmp_btn.click(fn=handle_compare, inputs=[cmp_input], outputs=[cmp_out])

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

            # ================================================================
            # TAB 4: TÀI LIỆU
            # ================================================================
            with gr.Tab("📚 TÀI LIỆU & NOTEBOOKS"):
                gr.Markdown(RESOURCES_MD)

            # ================================================================
            # TAB 5: PHƯƠNG PHÁP LUẬN
            # ================================================================
            with gr.Tab("📜 PHƯƠNG PHÁP LUẬN"):
                gr.Markdown(METHODOLOGY_MD)

            # ================================================================
            # TAB 6: ĐỀ CƯƠNG
            # ================================================================
            with gr.Tab("📑 ĐỀ CƯƠNG"):
                gr.Markdown(THESIS_MD)

        gr.HTML(FOOTER_HTML)

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2, max_size=10)
    app.launch(show_error=True)
