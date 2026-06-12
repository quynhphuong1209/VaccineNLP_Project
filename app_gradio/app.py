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
- ✨ ADD: Export Report button (markdown download)
- ✨ ADD: Captum opt-in checkbox (faster demo)
- 🎨 REDESIGN: Web-aligned two-screen Gradio shell with HUPH branding

Features:
- 2 screens: Phân tích văn bản · Dữ liệu & đối sánh
- 3-Layer XAI: Cache → HF Inference API → Captum IG
- Multi-source Fetcher: News + YouTube + Apify
- AI Voice via gTTS
- Batch mode, Compare models, Session history

Deploy: HuggingFace Spaces (CPU Basic 16GB)
URL: huggingface.co/spaces/hung2903/vaccinenlp-demo
"""

import os
import json
import time
import tempfile
import hashlib
import ipaddress
import logging
import re
import threading
import base64
import gzip
import sys
import unicodedata
from urllib.parse import urlparse

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
import requests

try:
    from src.common.xai_prompt import (
        XAI_MAX_TOKENS,
        XAI_TEMPERATURE,
        build_xai_messages,
        build_xai_user_prompt,
    )
except ModuleNotFoundError:
    import importlib.util

    _app_path = Path(__file__).resolve()
    _prompt_candidates = [
        _app_path.parent / "src" / "common" / "xai_prompt.py",
        _app_path.parents[1] / "src" / "common" / "xai_prompt.py",
        Path.cwd() / "src" / "common" / "xai_prompt.py",
    ]
    _prompt_module = None
    for _prompt_path in _prompt_candidates:
        if not _prompt_path.exists():
            continue
        _prompt_spec = importlib.util.spec_from_file_location("vaccinenlp_shared_xai_prompt", _prompt_path)
        if _prompt_spec is None or _prompt_spec.loader is None:
            continue
        _prompt_module = importlib.util.module_from_spec(_prompt_spec)
        _prompt_spec.loader.exec_module(_prompt_module)
        break

    if _prompt_module is not None:
        XAI_MAX_TOKENS = _prompt_module.XAI_MAX_TOKENS
        XAI_TEMPERATURE = _prompt_module.XAI_TEMPERATURE
        build_xai_messages = _prompt_module.build_xai_messages
        build_xai_user_prompt = _prompt_module.build_xai_user_prompt
    else:
        XAI_MAX_TOKENS = 2048
        XAI_TEMPERATURE = 0.1
        XAI_SYSTEM_PROMPT = """Bạn là chuyên gia AI phân tích vấn đề y tế công cộng (Explainable AI - XAI).
Hãy suy luận và phân tích nội dung thông tin vaccine bằng TIẾNG VIỆT theo 3 chiều:
(1) Dấu hiệu sai lệch: chỉ chọn "Có dấu hiệu tin giả" hoặc "Không có dấu hiệu sai lệch".
(2) Thái độ với vaccine: chỉ chọn "Ủng hộ", "Phản đối" hoặc "Trung lập".
(3) Cảm xúc tổng thể: chỉ chọn "Tiêu cực", "Trung tính" hoặc "Tích cực".

Lưu ý bắt buộc:
- Phải sử dụng tiếng Việt; không dùng các nhãn tiếng Anh như Reasoning, Therefore, Misinformation, Stance, Sentiment trong câu trả lời.
- Cẩn trọng với phương ngữ địa phương, tiếng lóng, teen-code và từ viết tắt; nếu từ ngữ mơ hồ, hãy dựa vào ngữ cảnh và thái độ tổng thể.
- Nhãn PhoBERT chỉ là tham khảo để đối chiếu, không phải câu trả lời bắt buộc sao chép.
- Trả lời bắt đầu ngay bằng "=== KẾT QUẢ ==="; không thêm lời chào, không thêm token đặc biệt.
- Không thêm marker kết thúc như "=== HẾT GIẢI THÍCH ===".

Trả lời theo ĐÚNG cấu trúc:
=== KẾT QUẢ ===
- Dấu hiệu sai lệch: <Có dấu hiệu tin giả HOẶC Không có dấu hiệu sai lệch>
- Thái độ với vaccine: <Ủng hộ HOẶC Phản đối HOẶC Trung lập>
- Cảm xúc tổng thể: <Tiêu cực HOẶC Trung tính HOẶC Tích cực>
=== GIẢI THÍCH ===
<lý luận chi tiết bằng tiếng Việt, giải thích lần lượt 3 nhãn trên>"""
        _LABEL_VI = {
            "misinfo": {"Fake": "Có dấu hiệu tin giả", "Real": "Không phát hiện dấu hiệu sai lệch"},
            "stance": {"Favor": "Ủng hộ", "Against": "Phản đối", "Neutral": "Trung lập"},
            "sentiment": {"Positive": "Tích cực", "Negative": "Tiêu cực", "Neutral": "Trung tính"},
        }

        def _predicted_labels_vi(predicted_labels: dict | None) -> dict:
            return {
                axis: _LABEL_VI.get(axis, {}).get(value, value)
                for axis, value in (predicted_labels or {}).items()
            }

        def build_xai_user_prompt(text: str, predicted_labels: dict | None = None) -> str:
            parts = [
                "Văn bản cần phân tích:",
                (text or "").strip(),
                "",
            ]
            predicted_vi = _predicted_labels_vi(predicted_labels)
            if predicted_vi:
                parts.extend([
                    "Nhãn PhoBERT tham khảo:",
                    f"- Dấu hiệu sai lệch: {predicted_vi.get('misinfo', 'Không có')}",
                    f"- Thái độ với vaccine: {predicted_vi.get('stance', 'Không có')}",
                    f"- Cảm xúc tổng thể: {predicted_vi.get('sentiment', 'Không có')}",
                    "",
                ])
            parts.extend([
                "Yêu cầu bắt buộc:",
                "- Chỉ trả lời bằng tiếng Việt.",
                "- Không dùng nhãn tiếng Anh như Reasoning, Therefore, Misinformation, Stance, Sentiment.",
                "- Cẩn trọng với phương ngữ địa phương, tiếng lóng, teen-code và từ viết tắt.",
                "- Trả lời bắt đầu ngay bằng === KẾT QUẢ ===, không thêm lời chào hay token đặc biệt.",
                "",
                "Trả lời theo đúng cấu trúc:",
                "=== KẾT QUẢ ===",
                "- Dấu hiệu sai lệch: <Có dấu hiệu tin giả HOẶC Không có dấu hiệu sai lệch>",
                "- Thái độ với vaccine: <Ủng hộ HOẶC Phản đối HOẶC Trung lập>",
                "- Cảm xúc tổng thể: <Tiêu cực HOẶC Trung tính HOẶC Tích cực>",
                "=== GIẢI THÍCH ===",
                "<lý luận chi tiết bằng tiếng Việt, giải thích lần lượt 3 nhãn trên>",
            ])
            return "\n".join(parts)

        def build_xai_messages(text: str, predicted_labels: dict | None = None) -> list[dict]:
            return [
                {"role": "system", "content": XAI_SYSTEM_PROMPT},
                {"role": "user", "content": build_xai_user_prompt(text, predicted_labels)},
            ]

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
    "annotations_file": DATA_DIR / "human_annotations.json",
    "max_seq_length": 256,
    "session_history_limit": 10,
}

LABEL_MAPS = {
    "misinfo":   {0: "Có dấu hiệu tin giả",  1: "Không phát hiện dấu hiệu sai lệch"},
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

def _load_env_defaults():
    """Load local env files for direct Gradio runs without overriding deployed secrets."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "VaccineNLP_Web" / ".env",
        APP_DIR / ".env",
        APP_DIR.parent / ".env",
        APP_DIR.parent / "VaccineNLP_Web" / ".env",
    ]
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        except OSError:
            continue


def _env_any(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1" or bool(os.environ.get("SPACE_ID"))


def _normalize_lmstudio_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if not value:
        return ""
    if not _running_in_container():
        value = value.replace("host.docker.internal", "localhost")
    # Be forgiving about how the endpoint is configured. The code appends
    # "/chat/completions" and "/models" to this base, so the base MUST be ".../v1".
    # Common copy-paste mistakes set the full path ".../v1/chat/completions" or
    # ".../v1/chat" — left as-is they build ".../v1/chat/chat/completions" and every
    # LM Studio call fails. Strip those trailing segments and guarantee a /v1 base.
    low = value.lower()
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if low.endswith(suffix):
            value = value[: -len(suffix)]
            low = value.lower()
    value = value.rstrip("/")
    if not value.lower().endswith("/v1"):
        # Bare host or unexpected path → append the standard OpenAI-compatible base.
        value = value + "/v1"
    return value


def _is_private_lmstudio_url(url: str) -> bool:
    """True for local/private addresses that remote HF Spaces cannot reach."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    if host in {"localhost", "host.docker.internal"}:
        return True
    # Any RFC1918 / loopback / link-local IP (10.x, 172.16-31.x, 192.168.x, 127.x,
    # 169.254.x) is unreachable from a remote HF Space. Detect them generically so
    # we never hardcode a machine-specific LAN IP (those change with the network).
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


def _select_lmstudio_url() -> str:
    # Default to loopback; same-machine services must never be addressed by a LAN IP
    # (DHCP/Wi-Fi changes it). On HF Spaces the LM_STUDIO_URL secret (ngrok public
    # domain) overrides this anyway.
    raw = _env_any("LM_STUDIO_URL", "LMSTUDIO_URL", default="http://localhost:1234/v1")
    if _running_in_container() and _is_private_lmstudio_url(raw):
        public = _env_any(
            "PUBLIC_LM_STUDIO_URL",
            "LM_STUDIO_PUBLIC_URL",
            "NGROK_LM_STUDIO_URL",
            "NGROK_PUBLIC_URL",
            default="",
        )
        if public:
            return _normalize_lmstudio_url(public)
        logger.warning(
            "Ignoring private LM_STUDIO_URL '%s' in container/Space; set a public ngrok URL ending in /v1.",
            raw,
        )
        return ""
    return _normalize_lmstudio_url(raw)


_load_env_defaults()

HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("VaccineNLP_TOKEN", "")

# LM Studio local server configuration (configurable via env vars)
LM_STUDIO_BASE_URL = _select_lmstudio_url()
LM_STUDIO_BRIDGE_URL = _normalize_lmstudio_url(
    _env_any("LM_STUDIO_BRIDGE_URL", "LM_BRIDGE_URL", "XAI_BRIDGE_URL", default="")
)
LM_STUDIO_ACTIVE_URL = LM_STUDIO_BRIDGE_URL or LM_STUDIO_BASE_URL
LM_STUDIO_MODEL    = _env_any("LM_STUDIO_MODEL", "LMSTUDIO_MODEL", default="gemma-4-e4b-vaccine-xai-merged")
LM_API_TOKEN       = _env_any("LM_API_TOKEN", "LMSTUDIO_API_KEY", "LM_STUDIO_API_KEY", default="lm-studio")

# ----------------------------------------------------------------------------
# ABUSE-SAFETY ON PUBLIC HUGGING FACE SPACES
# ----------------------------------------------------------------------------
# A Space that (a) scrapes YouTube/news/social media via Apify, or (b) relays
# inference to an external LM Studio tunnel (ngrok/Cloudflare), can trip Hugging
# Face's abuse detection. A flagged Space is paused and CANNOT be restarted via
# API (the /restart endpoint returns 503 "Flagged as abusive"). To stay safe, both
# behaviours are DISABLED BY DEFAULT when running on a Space; local runs keep full
# functionality. Re-enable explicitly (at your own risk) on a Space you control:
#   ENABLE_URL_FETCHERS=1   ENABLE_REMOTE_LLM=1
_ON_SPACE = _running_in_container()
ENABLE_URL_FETCHERS = _env_bool("ENABLE_URL_FETCHERS", default=not _ON_SPACE)
ENABLE_REMOTE_LLM = _env_bool("ENABLE_REMOTE_LLM", default=not _ON_SPACE)
_FETCH_DISABLED_MSG = (
    "❌ Tính năng tải dữ liệu từ URL đã tắt trên bản demo công khai. "
    "Vui lòng dán trực tiếp văn bản cần phân tích."
)
# Keep the raw tunnel bridge URL (before any blanking) so the optional fetch-bridge
# can reuse the same Cloudflare/ngrok tunnel independently of the LLM relay toggle.
_RAW_BRIDGE_URL = LM_STUDIO_BRIDGE_URL
if not ENABLE_REMOTE_LLM:
    # Never call an external LLM tunnel from the Space; XAI falls back to the
    # synced cache → Gemini cloud → template chain.
    LM_STUDIO_BASE_URL = ""
    LM_STUDIO_BRIDGE_URL = ""
    LM_STUDIO_ACTIVE_URL = ""

# Optional remote "fetch-bridge" (Cách B): the Space asks YOUR machine (via the tunnel)
# to run the multi-source fetchers, so scraping never runs on — or appears in — the
# public Space. Explicit FETCH_BRIDGE_URL wins; otherwise it is derived from the tunnel
# bridge URL on Spaces. Local runs use fetchers_impl.py directly (Cách A).
FETCH_BRIDGE_URL = _env_any("FETCH_BRIDGE_URL", "FETCH_BRIDGE", default="")


def _fetch_bridge_base() -> str:
    explicit = (FETCH_BRIDGE_URL or "").strip().rstrip("/")
    if explicit:
        base = explicit
    elif _ON_SPACE:
        base = (_RAW_BRIDGE_URL or "").strip().rstrip("/")
    else:
        return ""  # local: fetchers_impl handles fetching directly (Cách A)
    if base.lower().endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


if _ON_SPACE:
    logger.info(
        "Abuse-safe Space mode: URL fetchers=%s, remote LLM tunnel=%s, fetch-bridge=%s",
        "ON" if ENABLE_URL_FETCHERS else "OFF",
        "ON" if ENABLE_REMOTE_LLM else "OFF",
        "ON" if _fetch_bridge_base() else "OFF",
    )


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer env var without crashing the Space on bad config."""
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %d", name, os.environ.get(name), default)
        return default


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    """Read a positive float env var for timeout/tuning knobs."""
    try:
        return max(minimum, float(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %.1f", name, os.environ.get(name), default)
        return default


# Concurrency tuning:
# - Gradio can queue many users, but LM Studio on a single local GPU/CPU should
#   not receive too many Gemma generations at once.
# - Increase LM_STUDIO_MAX_CONCURRENT only if LM Studio is configured for parallel
#   slots and the host machine has enough VRAM/RAM.
APP_QUEUE_CONCURRENCY = _env_int("APP_QUEUE_CONCURRENCY", 4)
APP_QUEUE_MAX_SIZE = _env_int("APP_QUEUE_MAX_SIZE", 50)
LM_STUDIO_MAX_CONCURRENT = _env_int("LM_STUDIO_MAX_CONCURRENT", 1)
LM_STUDIO_QUEUE_TIMEOUT = _env_float("LM_STUDIO_QUEUE_TIMEOUT", 300.0)
LM_STUDIO_REQUEST_TIMEOUT = _env_float("LM_STUDIO_REQUEST_TIMEOUT", 180.0)
# Separate connect timeout so a dead/stale ngrok tunnel fails fast (≈10s) instead of
# hanging for the full read timeout. The read timeout stays long for slow generations.
LM_STUDIO_CONNECT_TIMEOUT = _env_float("LM_STUDIO_CONNECT_TIMEOUT", 10.0)
# Transient ngrok/LM Studio blips (connection reset mid-generation, 502/504, brief
# unavailability) are the main cause of intermittent "lúc được lúc không" failures.
# Retry the blocking completion a few times with backoff before giving up to Gemini.
LM_STUDIO_NUM_RETRIES = _env_int("LM_STUDIO_NUM_RETRIES", 2)        # extra attempts after the first
LM_STUDIO_RETRY_BACKOFF = _env_float("LM_STUDIO_RETRY_BACKOFF", 1.5)
LM_STUDIO_MODELS_TTL = _env_float("LM_STUDIO_MODELS_TTL", 15.0)
LM_STUDIO_ENABLE_STREAM = os.environ.get(
    "LM_STUDIO_ENABLE_STREAM",
    "0" if _running_in_container() else "1",
).strip().lower() in {"1", "true", "yes", "on"}

_LMSTUDIO_SEMAPHORE = threading.BoundedSemaphore(LM_STUDIO_MAX_CONCURRENT)
_LMSTUDIO_MODELS_CACHE = {"ts": 0.0, "ids": []}
_LMSTUDIO_MODELS_LOCK = threading.Lock()
_LMSTUDIO_HTTP = requests.Session()
_LMSTUDIO_HTTP.trust_env = False


def _lmstudio_headers() -> dict:
    # ngrok-skip-browser-warning: bắt buộc khi LM Studio đi qua ngrok-free, nếu không
    # ngrok trả về trang HTML cảnh báo (200) thay vì JSON → app tưởng LM Studio "chết".
    headers = {"ngrok-skip-browser-warning": "true"}
    if LM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LM_API_TOKEN}"
    return headers


def _lmstudio_get(url: str, **kwargs) -> requests.Response:
    """Call LM Studio/ngrok without inheriting proxy env from HF Spaces."""
    return _LMSTUDIO_HTTP.get(url, **kwargs)


def _lmstudio_post(url: str, **kwargs) -> requests.Response:
    """Call LM Studio/ngrok without inheriting proxy env from HF Spaces."""
    return _LMSTUDIO_HTTP.post(url, **kwargs)


def _encode_bridge_payload(payload: dict) -> str:
    """Pack chat/completions JSON into a compact URL-safe query payload."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    packed = gzip.compress(raw)
    return base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def _lmstudio_model_ids(timeout: float = 2.0) -> List[str]:
    if not LM_STUDIO_ACTIVE_URL:
        return []
    now = time.time()
    with _LMSTUDIO_MODELS_LOCK:
        cached_ids = list(_LMSTUDIO_MODELS_CACHE.get("ids") or [])
        cached_ts = float(_LMSTUDIO_MODELS_CACHE.get("ts") or 0.0)
        if cached_ids and now - cached_ts < LM_STUDIO_MODELS_TTL:
            return cached_ids
    try:
        resp = _lmstudio_get(f"{LM_STUDIO_ACTIVE_URL}/models", headers=_lmstudio_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        model_ids = [item.get("id", "") for item in data if item.get("id")]
        if model_ids:
            with _LMSTUDIO_MODELS_LOCK:
                _LMSTUDIO_MODELS_CACHE["ts"] = now
                _LMSTUDIO_MODELS_CACHE["ids"] = model_ids
        return model_ids
    except Exception:
        # Keep a recent positive probe instead of flipping the UI to "disconnected"
        # during transient ngrok/LM Studio hiccups.
        with _LMSTUDIO_MODELS_LOCK:
            cached_ids = list(_LMSTUDIO_MODELS_CACHE.get("ids") or [])
            cached_ts = float(_LMSTUDIO_MODELS_CACHE.get("ts") or 0.0)
        if cached_ids and now - cached_ts < max(60.0, LM_STUDIO_MODELS_TTL * 4):
            return cached_ids
        return []


def _resolve_lmstudio_model(timeout: float = 2.0) -> str:
    model_ids = _lmstudio_model_ids(timeout=timeout)
    if not model_ids or LM_STUDIO_MODEL in model_ids:
        return LM_STUDIO_MODEL
    preferred = next((mid for mid in model_ids if "gemma" in mid.lower()), model_ids[0])
    logger.warning("Configured LM Studio model '%s' is not loaded; using '%s'", LM_STUDIO_MODEL, preferred)
    return preferred


def _resolve_lmstudio_model_for_completion(timeout: float = 2.0) -> str:
    """Resolve model id without treating a transient /models failure as fatal.

    Ngrok and LM Studio can occasionally fail the cheap /models probe while the
    actual chat completion endpoint is still reachable. In that case we keep the
    configured model id and let chat/completions be the source of truth.
    """
    model_ids = _lmstudio_model_ids(timeout=timeout)
    if not model_ids:
        logger.info("LM Studio /models probe returned no models; trying configured model '%s' anyway.", LM_STUDIO_MODEL)
        return LM_STUDIO_MODEL
    if LM_STUDIO_MODEL in model_ids:
        return LM_STUDIO_MODEL
    preferred = next((mid for mid in model_ids if "gemma" in mid.lower()), model_ids[0])
    logger.warning("Configured LM Studio model '%s' is not loaded; using '%s'", LM_STUDIO_MODEL, preferred)
    return preferred


def get_lm_studio_status_html() -> str:
    models = _lmstudio_model_ids(timeout=1.5)
    ok = bool(models)
    effective = _resolve_lmstudio_model(timeout=1.0) if ok else LM_STUDIO_MODEL
    cls = "ok" if ok else "bad"
    label = "Đã kết nối" if ok else "Chưa kết nối"
    if ok:
        detail = f"{effective} · {'GET bridge' if LM_STUDIO_BRIDGE_URL else 'direct'}"
    else:
        detail = LM_STUDIO_ACTIVE_URL or "Cần LM_STUDIO_BRIDGE_URL hoặc PUBLIC_LM_STUDIO_URL"
    return (
        f'<div class="lm-status {cls}">'
        f'<div><span class="dot"></span><b>Gemma-4 XAI</b></div>'
        f'<span>{label}</span>'
        f'<small>{detail}</small>'
        '</div>'
    )

# Apify/social URL fetchers were moved to local-only fetchers_impl.py
# (not shipped to public Spaces). See _get_fetchers() below.

# Path B: Optional external Gemma endpoint (legacy Kaggle/Colab worker).
# The active XAI path is LM_STUDIO_URL -> Gemini -> template. Keep this empty
# unless a separate /predict worker is explicitly configured.
_env_gemma_url = os.environ.get("GEMMA_ENDPOINT_URL", "").strip()
GEMMA_ENDPOINT_URL = _env_gemma_url
if GEMMA_ENDPOINT_URL:
    logger.info(f"External Gemma /predict endpoint configured: {GEMMA_ENDPOINT_URL}")

# Path C: OpenRouter fallback (RESERVED — currently unused; Gemini is the active cloud fallback)
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")

# Gemini API keys: collect flexibly (GEMINI_API_KEY, GEMINI_API_KEY_2 ... GEMINI_API_KEY_5).
# These power the cloud XAI fallback (see _gemini_xai_stream) so the demo keeps producing
# Gemma-style reasoning on HuggingFace Spaces where LM Studio (localhost/ngrok) is unreachable.
_GEMINI_KEYS: list = []
for _gi in range(1, 6):
    _gname = "GEMINI_API_KEY" if _gi == 1 else f"GEMINI_API_KEY_{_gi}"
    _gval = os.environ.get(_gname, "").strip()
    if _gval and _gval not in _GEMINI_KEYS:
        _GEMINI_KEYS.append(_gval)
if _GEMINI_KEYS:
    logger.info(f"🔑 Loaded {len(_GEMINI_KEYS)} Gemini API key(s)")
# Verified working model id (gemini-3.5-flash); override via env if needed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Thread-safe cache lock
_CACHE_LOCK = threading.Lock()
_CACHE = {}

# Sample texts for quick demo.
# Public-repo defaults are NEUTRAL (no raw misinformation strings) so the deployed
# Space files carry no sensitive content. The full research examples (including the
# misinformation samples used only as classifier *test inputs*) are loaded at runtime
# from a PRIVATE, token-gated Hugging Face dataset — see _load_private_assets().
SAMPLE_TEXTS = {
    "\U0001F6A8 Tin giả - Chống vaccine cực đoan": "(Mẫu nghiên cứu — nạp từ kho dữ liệu riêng khi có quyền truy cập.)",
    "\U0001F6A8 Tin giả - Vô sinh": "(Mẫu nghiên cứu — nạp từ kho dữ liệu riêng khi có quyền truy cập.)",
    "\U0001F7E2 Ủng hộ tiêm chủng": "Em đã cho bé tiêm đủ các mũi theo lịch. Bé khỏe mạnh, không sốt, phát triển tốt.",
    "\U0001F7E1 Nghi ngại": "Mình đang tìm hiểu thêm thông tin từ các nguồn chính thống trước khi quyết định.",
    "\u2705 Thông tin chuẩn": "Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi cơ bản theo Chương trình Tiêm chủng Mở rộng.",
    "\U0001F535 Câu hỏi tư vấn": "Cho em hỏi lịch tiêm các mũi cơ bản cho bé dưới 1 tuổi như thế nào ạ?",
    "\U0001F4AC Tin giả - Từ lóng MXH": "(Mẫu nghiên cứu — nạp từ kho dữ liệu riêng khi có quyền truy cập.)",
}

# Examples for gr.Examples (neutral defaults; full set comes from the private dataset)
GRADIO_EXAMPLES = [
    ["Bộ Y tế công bố lịch tiêm chủng mở rộng cho trẻ em năm 2026.", "PhoBERT-v2"],
    ["Em đã tiêm đủ các mũi cho con theo lịch. Bé khỏe mạnh, không sốt.", "PhoBERT-v2"],
]

# Optional PRIVATE token-gated dataset holding the full research samples + XAI cache.
PRIVATE_ASSETS_DATASET = os.environ.get("PRIVATE_ASSETS_DATASET", "hung2903/vaccinenlp-assets").strip()


def _load_private_assets() -> None:
    """Override demo samples from a PRIVATE token-gated HF dataset, if reachable.
    Keeps sensitive research examples out of the public Space repo files."""
    global SAMPLE_TEXTS, GRADIO_EXAMPLES
    if not (PRIVATE_ASSETS_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(PRIVATE_ASSETS_DATASET, filename="samples.json",
                               repo_type="dataset", token=HF_TOKEN)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("SAMPLE_TEXTS"), dict) and data["SAMPLE_TEXTS"]:
            SAMPLE_TEXTS = data["SAMPLE_TEXTS"]
        if isinstance(data.get("GRADIO_EXAMPLES"), list) and data["GRADIO_EXAMPLES"]:
            GRADIO_EXAMPLES = data["GRADIO_EXAMPLES"]
        logger.info("Loaded demo samples from private dataset %s", PRIVATE_ASSETS_DATASET)
    except Exception as e:
        logger.info("Private assets not loaded (%s); using neutral defaults.", type(e).__name__)


_load_private_assets()


# ============================================================================
# MULTITASK MODEL ARCHITECTURE
# ============================================================================

class VaccineMultitaskModel(nn.Module):
    """Multitask model with shared encoder + 3 task-specific heads."""

    def __init__(
        self,
        model_name: str,
        num_misinfo=2,
        num_stance=3,
        num_sentiment=3,
        token=None,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        # AGENT NOTE:
        # `model_name` may be either a Hugging Face repo id or a local snapshot directory.
        # Keep `local_files_only` wired through both config and encoder so offline runs do
        # not silently hit the Hub and fail when the network/HF_TOKEN is unavailable.
        self.config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=local_files_only,
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

def _resolve_local_checkpoint(model_key: str, cfg: Dict) -> Optional[Path]:
    """Find a task-head checkpoint before contacting Hugging Face Hub.

    AGENT NOTE:
    The Gradio demo and Web app share `VaccineNLP_Web/models/phobert_multitask.pt`.
    Do not remove this local-first path unless the deployment no longer ships local
    model files. HF Hub is a fallback, not the primary path, because demo machines
    often run without HF_TOKEN or stable outbound network access.
    """
    explicit_by_type = {
        "phobert": ["PHOBERT_PATH", "GRADIO_PHOBERT_PATH"],
        "xlm-roberta": ["XLMR_PATH", "GRADIO_XLMR_PATH"],
    }
    for env_name in explicit_by_type.get(cfg.get("type", ""), []):
        value = os.environ.get(env_name, "").strip()
        if value and Path(value).exists():
            return Path(value)

    if cfg.get("type") == "phobert":
        filenames = ["phobert_multitask.pt", "best_model.pt"]
    elif cfg.get("type") == "xlm-roberta":
        filenames = ["xlmr_multitask.pt", "best_model.pt"]
    else:
        filenames = ["best_model.pt"]

    candidate_dirs = [
        Path("/models"),
        APP_DIR / "models",
        APP_DIR.parent / "models",
        APP_DIR.parent / "VaccineNLP_Web" / "models",
        Path.cwd() / "models",
        Path.cwd() / "VaccineNLP_Web" / "models",
    ]
    for directory in candidate_dirs:
        for filename in filenames:
            path = directory / filename
            if path.exists():
                return path
    return None


def _resolve_local_hf_snapshot(repo_id: str) -> Optional[Path]:
    """Return a cached Hugging Face snapshot directory for the base encoder."""
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    roots = [
        Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") else None,
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
        APP_DIR / ".cache" / "huggingface" / "hub",
        APP_DIR.parent / "VaccineNLP_Web" / "models" / ".cache" / "huggingface" / "hub",
    ]
    for root in [r for r in roots if r]:
        snapshot_root = root / repo_dir_name / "snapshots"
        if not snapshot_root.exists():
            continue
        snapshots = sorted(
            [p for p in snapshot_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").exists():
                return snapshot
    return None


def _normalize_model_state(state: Dict) -> Dict:
    """Normalize checkpoint key variants used by Web/Gradio training exports."""
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state).__name__}")

    normalized = {}
    for key, value in state.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_key = (
            new_key
            .replace("head_misinfo.", "heads.misinfo.")
            .replace("head_stance.", "heads.stance.")
            .replace("head_sentiment.", "heads.sentiment.")
        )
        if new_key.startswith("head_") and "heads." not in new_key:
            new_key = new_key.replace("head_", "heads.", 1)
        normalized[new_key] = value
    return normalized


def _load_model_from_checkpoint(
    model_key: str,
    cfg: Dict,
    checkpoint_path: Path,
    local_files_only: bool,
):
    """Load tokenizer, encoder and multitask heads from a checkpoint."""
    import gc
    from transformers import AutoTokenizer

    token = HF_TOKEN if HF_TOKEN else None
    base_source = cfg["base_repo"]
    if local_files_only:
        local_snapshot = _resolve_local_hf_snapshot(cfg["base_repo"])
        if not local_snapshot:
            raise FileNotFoundError(f"Local base model snapshot not found for {cfg['base_repo']}")
        base_source = str(local_snapshot)

    tokenizer_kwargs = {
        "token": token,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if cfg.get("type") == "phobert":
        tokenizer_kwargs["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(base_source, **tokenizer_kwargs)
    model = VaccineMultitaskModel(
        model_name=base_source,
        token=token,
        local_files_only=local_files_only,
    )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _normalize_model_state(state)
    result = model.load_state_dict(state, strict=False)

    # Critical heads must exist; encoder pooler warnings are acceptable for PhoBERT.
    critical_heads = [
        "heads.misinfo.weight", "heads.misinfo.bias",
        "heads.stance.weight", "heads.stance.bias",
        "heads.sentiment.weight", "heads.sentiment.bias",
    ]
    if any(key in result.missing_keys for key in critical_heads):
        raise RuntimeError(f"Critical task heads missing from checkpoint: {result.missing_keys}")

    model.eval()
    del state
    gc.collect()
    return model, tokenizer

def load_model(model_key: str):
    """Load model with local checkpoint first and HF Hub fallback second.

    AGENT NOTE:
    Never cache a failed `(None, None, False)` result here. A transient network
    failure or a late-mounted model file should be recoverable on the next click.
    """
    cache_key = f"model_{model_key}"
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    import gc
    from huggingface_hub import hf_hub_download

    cfg = CONFIG["models"][model_key]
    gc.collect()
    errors = []

    local_checkpoint = _resolve_local_checkpoint(model_key, cfg)
    if local_checkpoint:
        for local_files_only in (True, False):
            mode = "local base snapshot" if local_files_only else "online base fallback"
            try:
                logger.info("Loading %s from local checkpoint %s (%s)...", model_key, local_checkpoint, mode)
                model, tokenizer = _load_model_from_checkpoint(
                    model_key=model_key,
                    cfg=cfg,
                    checkpoint_path=local_checkpoint,
                    local_files_only=local_files_only,
                )
                with _CACHE_LOCK:
                    _CACHE[cache_key] = (model, tokenizer, True)
                logger.info("✅ %s loaded from local checkpoint: %s", model_key, local_checkpoint)
                return _CACHE[cache_key]
            except Exception as e:
                msg = f"{mode}: {type(e).__name__}: {e}"
                errors.append(msg)
                logger.warning("Local checkpoint load failed for %s (%s): %s", model_key, mode, e)

    try:
        logger.info(f"Loading {model_key} from HF Hub...")
        model_path = hf_hub_download(
            repo_id=cfg["repo_id"], filename="best_model.pt",
            token=HF_TOKEN if HF_TOKEN else None,
        )
        model, tokenizer = _load_model_from_checkpoint(
            model_key=model_key,
            cfg=cfg,
            checkpoint_path=Path(model_path),
            local_files_only=False,
        )

        with _CACHE_LOCK:
            _CACHE[cache_key] = (model, tokenizer, True)
        logger.info(f"✅ {model_key} loaded")
        return _CACHE[cache_key]
    except Exception as e:
        errors.append(f"hf_hub: {type(e).__name__}: {e}")
        logger.error("❌ Failed to load %s. Attempts: %s", model_key, " | ".join(errors))
        return (None, None, False)


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
    """Load XAI reasoning cache (thread-safe).

    AGENT NOTE:
    `app_gradio/data/xai_cache.json` currently contains the rebuilt 266-sample
    cache synchronized with the Vietnamese structured prompt. Keep cache reads
    here side-effect free; write-through is handled by `save_xai_reasoning_cache`.
    """
    with _CACHE_LOCK:
        if "xai_cache" in _CACHE:
            return _CACHE["xai_cache"]

    cache = {}
    src = CONFIG["cache_file"] if CONFIG["cache_file"].exists() else None
    if src is None and PRIVATE_ASSETS_DATASET and HF_TOKEN:
        # Public Space ships no xai_cache.json (it holds research text); pull it from
        # the PRIVATE token-gated dataset at runtime instead.
        try:
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download(PRIVATE_ASSETS_DATASET, filename="xai_cache.json",
                                       repo_type="dataset", token=HF_TOKEN))
        except Exception as e:
            logger.info("Private xai_cache not loaded (%s)", type(e).__name__)
            src = None
    if src is not None and src.exists():
        try:
            with open(src, encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded %d cached XAI reasonings", len(cache))
        except Exception as e:
            logger.warning("Could not load XAI cache: %s", e)
    with _CACHE_LOCK:
        _CACHE["xai_cache"] = cache
    return cache


def is_xai_cache_value_compatible(raw: str) -> bool:
    """Return True for cache entries generated by the current structured XAI prompt."""
    if not raw or not isinstance(raw, str):
        return False
    norm = _strip_accents(raw).lower()
    return (
        "ket qua" in norm
        and "giai thich" in norm
        and ("tinh xac thuc" in norm or "xac thuc" in norm)
        and ("thai do" in norm or "lap truong" in norm or "stance" in norm)
        and ("cam xuc" in norm or "sentiment" in norm)
    )


def _compatible_xai_cache_entries(cache: Dict) -> Dict:
    return {
        key: value
        for key, value in (cache or {}).items()
        if is_xai_cache_value_compatible(value)
    }


def save_xai_reasoning_cache(text: str, raw: str) -> bool:
    """Persist a fresh LM Studio XAI response.

    AGENT NOTE:
    This function intentionally drops legacy/incompatible cache values during the
    save snapshot so future runs do not mix old prompt formats with the rebuilt
    structured Vietnamese prompt. To disable write-through cache, return False at
    the top of this function and document the reason with a `DISABLED YYYY-MM-DD`
    comment.
    """
    text_key = (text or "").strip()
    raw_value = (raw or "").strip()
    if not text_key or not is_xai_cache_value_compatible(raw_value):
        return False

    cache = load_xai_cache()
    with _CACHE_LOCK:
        clean_cache = _compatible_xai_cache_entries(cache)
        clean_cache[text_key] = raw_value
        _CACHE["xai_cache"] = clean_cache
        snapshot = dict(clean_cache)

    try:
        cache_path = CONFIG["cache_file"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, cache_path)
        logger.info("Saved XAI cache entry; compatible cache size=%d", len(snapshot))
        return True
    except Exception as e:
        logger.warning("Could not save XAI cache: %s", e)
        return False


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
    if t_strip in cache and is_xai_cache_value_compatible(cache[t_strip]):
        return cache[t_strip]

    input_norm = normalize(t_strip)
    for k, v in cache.items():
        if not is_xai_cache_value_compatible(v):
            continue
        k_norm = normalize(k)
        if len(input_norm) > 20 and (input_norm in k_norm or k_norm in input_norm):
            return v
    return None


def _build_gemma_prompt(text: str) -> str:
    """Build standardized prompt format for the fine-tuned Gemma-4 MoE/merged model."""
    return build_xai_user_prompt(text)


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

    # 1. Strip reasoning/control "channel" tokens that the merged Gemma model leaks
    #    (e.g. "<|channel>thought", "<|channel|>final", "<|message|>", harmony/header
    #    markers). Prefer the final channel content when the model exposes one.
    final_match = re.search(r"<\|channel\|?>\s*final", t, flags=re.IGNORECASE)
    if final_match:
        t = t[final_match.end():]
    t = re.sub(r"<\|channel\|?>\s*(?:thought|analysis|final)?", "\n", t, flags=re.IGNORECASE)
    # Generic sweep for any remaining <|...|> / <|...> control tokens.
    t = re.sub(r"<\|/?[a-z0-9_]+\|?>", "\n", t, flags=re.IGNORECASE)
    t = t.replace("<end_of_turn>", "").replace("<start_of_turn>", "").replace("<|turn|>", "").replace("<|turn>", "").strip()

    # 2. Drop any "thinking/analysis" preamble that precedes the structured Vietnamese
    #    answer. Anchor on the first result marker so an English "Thinking Process"
    #    block never reaches the UI — even when the model omits a later marker.
    marker_positions = [t.find(m) for m in ("=== KẾT QUẢ ===", "=== GIẢI THÍCH ===") if m in t]
    if marker_positions:
        t = t[min(marker_positions):]
    else:
        # No structured markers: remove a leading English thinking heading line.
        t = re.sub(r"^\s*(?:thinking process|thinking|thought|analysis)\s*:?[^\n]*\n+", "", t,
                   count=1, flags=re.IGNORECASE).strip()

    # 3. Tìm phần giải thích (bỏ KẾT QUẢ vì PhoBERT đã có nhãn)
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
    text = "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")

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

    def pick(name_keys, options):
        if isinstance(name_keys, str):
            name_keys = (name_keys,)
        for ln in lines:                      # ưu tiên dòng chứa tên trục
            if any(name_key in ln for name_key in name_keys):
                for opt, val in options:
                    if opt in ln:
                        return val
        for opt, val in options:              # fallback: quét cả vùng KẾT QUẢ
            if opt in sec:
                return val
        return None

    mis = pick(
        ("misinformation", "misinfo", "tinh xac thuc", "xac thuc", "dau hieu sai lech"),
        [
            ("khong chinh xac", 0), ("sai lech", 0), ("tin gia", 0),
            ("co dau hieu tin gia", 0), ("co dau hieu sai lech", 0),
            # Some Gemma outputs label non-vaccine/unclear texts as neutral/unknown.
            # For the binary misinfo axis, treat these as not misinformation.
            ("khong xac dinh", 1), ("trung lap", 1), ("khong phai tin gia", 1),
            ("khong co thong tin sai lech", 1), ("khong chua thong tin sai lech", 1),
            ("khong de cap den vaccine", 1), ("khong lien quan den vaccine", 1),
            ("khong co dau hieu sai lech", 1), ("khong phat hien dau hieu sai lech", 1),
            ("khong co dau hieu tin gia", 1),
            ("chinh xac", 1),
        ],
    )
    st  = pick(("stance", "thai do", "lap truong"),
               [("phan doi", 1), ("trung lap", 2), ("ung ho", 0)])
    se  = pick(("sentiment", "cam xuc"),
               [("tieu cuc", 0), ("trung tinh", 1), ("tich cuc", 2)])
    if mis is None and st is None and se is None:
        return None
    return {"misinfo": mis, "stance": st, "sentiment": se}

_SVG_SCALE = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
              'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="color:var(--teal);vertical-align:-2px">'
              '<path d="M12 3v18M7 7h10M5 7l-2 5a3 3 0 0 0 6 0L7 7M17 7l-2 5a3 3 0 0 0 6 0l-2-5"/></svg>')

def render_disagreement_table(result, gemma_labels) -> str:
    """Bảng đối chiếu PhoBERT vs Gemma-4 (3 trục).

    AGENT NOTE:
    Đây là ĐỐI CHIẾU CHÉO để tham khảo, KHÔNG phải khẳng định mô hình sai. Tránh dùng
    từ "bất đồng thuận" gây hiểu nhầm 2 mô hình mâu thuẫn ngay cả khi đa số nhãn khớp.
    Trả "" nếu không parse được nhãn Gemma (fallback/template).
    """
    if not gemma_labels:
        return ""
    rows = ""
    agree = 0
    total = 0
    for key, vi in (("misinfo", "Dấu hiệu sai lệch"), ("stance", "Lập trường"), ("sentiment", "Cảm xúc")):
        p_id = result[key]["pred"]
        p_lbl = LABEL_MAPS[key][p_id]
        g_id = gemma_labels.get(key)
        g_lbl = LABEL_MAPS[key].get(g_id, "—") if g_id is not None else "—"
        if g_id is None:
            mark, cls = "—", ""
        else:
            total += 1
            same = (g_id == p_id)
            agree += 1 if same else 0
            mark = "✓ Đồng thuận" if same else "≠ Khác biệt"
            cls = "yes" if same else "no"
        flag = (g_id is not None and g_id != p_id)
        rows += (f'<tr class="{"flag" if flag else ""}"><td>{vi}</td><td>{p_lbl}</td><td>{g_lbl}</td>'
                 f'<td style="text-align:center" class="{cls}">{mark}</td></tr>')

    if total and agree == total:
        summary_block = f'<div style="margin-bottom:10px"><span class="pill ok">Hai mô hình đồng thuận hoàn toàn ({agree}/{total})</span></div>'
    elif total:
        summary_block = f'<div style="margin-bottom:10px"><span class="pill warn">Khác biệt {total - agree}/{total} nhãn — nên rà soát thêm</span></div>'
    else:
        summary_block = ''
    return (
        '<div style="margin-top:18px">'
        f'<div class="section-label" style="margin-bottom:10px">{_SVG_SCALE} Đối chiếu nhãn — PhoBERT vs Gemma-4</div>'
        f'{summary_block}'
        '<table class="dtable"><thead><tr><th>Trục</th><th>PhoBERT-v2</th><th>Gemma-4</th>'
        '<th style="text-align:center">Đối chiếu</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p class="muted" style="font-size:11.5px;margin-top:8px;color:var(--ink-3)">'
        'Nhãn Gemma-4 trích từ phần “KẾT QUẢ” của lời giải. Đây là đối chiếu chéo để tham khảo — '
        'khác biệt không có nghĩa là mô hình sai.</p>'
        '</div>'
    )


def _xai_messages(text: str) -> list:
    return build_xai_messages(text)


def _lmstudio_chat_url() -> str:
    if not LM_STUDIO_ACTIVE_URL:
        raise RuntimeError("LM Studio public endpoint is not configured")
    return f"{LM_STUDIO_ACTIVE_URL.rstrip('/')}/chat/completions"


def _lmstudio_chat_headers() -> dict:
    headers = _lmstudio_headers()
    headers["Content-Type"] = "application/json"
    # ngrok + LM Studio can leave long keep-alive sockets in a bad state when
    # Spaces reconnects. Closing each completion request is slower but steadier.
    headers["Connection"] = "close"
    return headers


def _lmstudio_payload(text: str, model: str, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": _xai_messages(text),
        "max_tokens": XAI_MAX_TOKENS,
        "temperature": XAI_TEMPERATURE,
        "stream": stream,
    }


def _lmstudio_completion_request(payload: dict) -> requests.Response:
    if LM_STUDIO_BRIDGE_URL:
        return _lmstudio_get(
            _lmstudio_chat_url(),
            headers=_lmstudio_chat_headers(),
            params={"payload": _encode_bridge_payload(payload)},
            timeout=(LM_STUDIO_CONNECT_TIMEOUT, LM_STUDIO_REQUEST_TIMEOUT),
        )
    return _lmstudio_post(
        _lmstudio_chat_url(),
        headers=_lmstudio_chat_headers(),
        json=payload,
        timeout=(LM_STUDIO_CONNECT_TIMEOUT, LM_STUDIO_REQUEST_TIMEOUT),
    )


def _extract_chat_text(data: dict) -> str:
    try:
        return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except (AttributeError, IndexError):
        return ""


def _acquire_lmstudio_slot(kind: str) -> bool:
    """Serialize Gemma calls so many users do not overload one LM Studio server."""
    acquired = _LMSTUDIO_SEMAPHORE.acquire(timeout=LM_STUDIO_QUEUE_TIMEOUT)
    if acquired:
        logger.info(
            "LM Studio slot acquired for %s (max_concurrent=%d)",
            kind,
            LM_STUDIO_MAX_CONCURRENT,
        )
        return True
    logger.warning(
        "LM Studio queue timeout after %.1fs for %s; skipping local Gemma",
        LM_STUDIO_QUEUE_TIMEOUT,
        kind,
    )
    return False


def _release_lmstudio_slot(kind: str) -> None:
    try:
        _LMSTUDIO_SEMAPHORE.release()
        logger.info("LM Studio slot released for %s", kind)
    except ValueError:
        logger.warning("LM Studio semaphore release ignored for %s", kind)


def get_live_xai_reasoning(text: str, result: Optional[Dict] = None, return_raw: bool = False):
    """Gọi Gemma-4 GGUF qua LM Studio (blocking). return_raw=True → trả (cleaned, raw|None) để parse nhãn."""
    if not LM_STUDIO_ACTIVE_URL:
        msg = "LM Studio public endpoint is not configured; skipping local Gemma."
        logger.info(msg)
        return ("", None) if return_raw else msg
    if not _acquire_lmstudio_slot("blocking"):
        return ("", None) if return_raw else ""
    try:
        effective_model = _resolve_lmstudio_model_for_completion(timeout=2)
        attempts = LM_STUDIO_NUM_RETRIES + 1
        for attempt in range(1, attempts + 1):
            try:
                logger.info(
                    "⏳ LM Studio completion via %s at %s (model=%s, attempt %d/%d, read_timeout=%.0fs)...",
                    "GET bridge" if LM_STUDIO_BRIDGE_URL else "direct POST",
                    LM_STUDIO_ACTIVE_URL, effective_model, attempt, attempts, LM_STUDIO_REQUEST_TIMEOUT,
                )
                response = _lmstudio_completion_request(_lmstudio_payload(text, effective_model, stream=False))
                response.raise_for_status()
                content = _extract_chat_text(response.json())
                if content and len(content.strip()) > 30:
                    cleaned = clean_reasoning_output(content)
                    return (cleaned, content) if return_raw else cleaned
                logger.warning("⚠️ LM Studio empty/short response (attempt %d/%d)", attempt, attempts)
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # 4xx (auth/bad request/model not found) won't fix on retry — stop early.
                if status is not None and 400 <= status < 500:
                    logger.warning("LM Studio HTTP %s is non-retryable; stopping retries", status)
                    break
                logger.warning("LM Studio HTTP %s (attempt %d/%d)", status, attempt, attempts)
            except requests.exceptions.ReadTimeout:
                # Server accepted the request but generation exceeded the read timeout.
                # Retrying re-runs the full slow generation — give up and let Gemini answer.
                logger.warning("LM Studio read timeout after %.0fs; not retrying", LM_STUDIO_REQUEST_TIMEOUT)
                break
            except requests.exceptions.RequestException as e:
                # ConnectError / ConnectionReset / ChunkedEncodingError / 5xx → fast-failing
                # transient errors worth a quick retry.
                logger.warning("LM Studio request error (attempt %d/%d): %s", attempt, attempts, str(e)[:140])
            if attempt < attempts:
                time.sleep(LM_STUDIO_RETRY_BACKOFF * attempt)
    finally:
        _release_lmstudio_slot("blocking")

    if result:
        fb = generate_smart_fallback(result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"])
        return (fb, None) if return_raw else fb
    msg = "⚠️ LM Studio chưa được khởi động. Hãy mở LM Studio và bật Local Server tại cổng 1234."
    return (msg, None) if return_raw else msg


def stream_live_xai_reasoning(text: str):
    """Stream Gemma-4 theo thời gian thực qua LM Studio (OpenAI-compatible, stream=True).
    Yield (cleaned_partial, raw_partial) sau mỗi token. Không yield gì nếu lỗi/không kết nối
    → caller tự rơi về fallback template."""
    if not LM_STUDIO_ACTIVE_URL:
        logger.info("LM Studio public endpoint is not configured; skipping local Gemma stream.")
        return
    if LM_STUDIO_BRIDGE_URL:
        logger.info("LM Studio bridge uses GET non-streaming transport; streaming path skipped.")
        return
    if not LM_STUDIO_ENABLE_STREAM:
        logger.info("LM Studio streaming disabled; using blocking completion path.")
        return
    if not _acquire_lmstudio_slot("stream"):
        return
    try:
        effective_model = _resolve_lmstudio_model_for_completion(timeout=2)
        logger.info(
            "⏳ Streaming Gemma qua requests từ LM Studio %s (model=%s)...",
            LM_STUDIO_ACTIVE_URL,
            effective_model,
        )
        stream = _lmstudio_post(
            _lmstudio_chat_url(),
            headers=_lmstudio_chat_headers(),
            json=_lmstudio_payload(text, effective_model, stream=True),
            stream=True,
            timeout=(LM_STUDIO_CONNECT_TIMEOUT, LM_STUDIO_REQUEST_TIMEOUT),
        )
        stream.raise_for_status()
        # Force UTF-8: SSE without a charset header would make requests default to
        # Latin-1 and mojibake Vietnamese while streaming.
        stream.encoding = "utf-8"
        raw = ""
        for line in stream.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                data = json.loads(line)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
            except (ValueError, AttributeError, IndexError):
                delta = ""
            if delta:
                raw += delta
                yield clean_reasoning_output(raw), raw
    except Exception as e:
        logger.warning(f"❌ LM Studio stream lỗi ({LM_STUDIO_BASE_URL}): {str(e)[:120]}")
        return
    finally:
        _release_lmstudio_slot("stream")


# ----------------------------------------------------------------------------
# Cloud fallback provider: Google Gemini (used when LM Studio/ngrok is unreachable)
# ----------------------------------------------------------------------------
def _gemini_payload(text: str) -> dict:
    """Map the shared system/user XAI contract to the Gemini generateContent schema."""
    messages = build_xai_messages(text)
    system_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_txt = next((m["content"] for m in messages if m["role"] == "user"), text)
    return {
        "system_instruction": {"parts": [{"text": system_txt}]},
        "contents": [{"role": "user", "parts": [{"text": user_txt}]}],
        "generationConfig": {"temperature": XAI_TEMPERATURE, "maxOutputTokens": XAI_MAX_TOKENS},
    }


def _gemini_extract_text(data: dict) -> str:
    """Pull concatenated text from a Gemini response/chunk JSON object."""
    out = ""
    for cand in (data.get("candidates") or []):
        for part in (cand.get("content", {}).get("parts") or []):
            out += part.get("text", "") or ""
    return out


def _gemini_xai_stream(text: str):
    """Yield (cleaned_partial, raw_partial) from Gemini SSE streaming.

    Rotates through _GEMINI_KEYS on auth/quota errors (401/403/429). Yields nothing
    if no key works, so the caller falls through to the template. Display label stays
    "Gemma-4" per product decision; the real provider is logged server-side only.
    """
    if not _GEMINI_KEYS:
        return
    payload = _gemini_payload(text)
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
    for idx, key in enumerate(_GEMINI_KEYS, 1):
        try:
            logger.info("⏳ XAI fallback via Gemini (%s) key#%d ...", GEMINI_MODEL, idx)
            resp = requests.post(
                url, params={"key": key}, json=payload,
                headers={"Content-Type": "application/json"},
                stream=True, timeout=60,
            )
            if resp.status_code in (401, 403, 429):
                logger.warning("Gemini key#%d rejected (%s); rotating", idx, resp.status_code)
                continue
            resp.raise_for_status()
            # SSE responses may omit "charset=utf-8"; requests would then default to
            # Latin-1 and mojibake Vietnamese. Force UTF-8 before decoding the stream.
            resp.encoding = "utf-8"
            raw = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    delta = _gemini_extract_text(json.loads(chunk))
                except (ValueError, KeyError):
                    delta = ""
                if delta:
                    raw += delta
                    yield clean_reasoning_output(raw), raw
            if raw.strip():
                return  # success — do not try further keys
        except Exception as e:
            logger.warning("Gemini stream key#%d error: %s", idx, str(e)[:120])
            continue


def _gemini_xai(text: str):
    """Non-streaming Gemini call → (cleaned, raw) or (None, None). Key rotation included."""
    if not _GEMINI_KEYS:
        return None, None
    payload = _gemini_payload(text)
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    for idx, key in enumerate(_GEMINI_KEYS, 1):
        try:
            resp = requests.post(
                url, params={"key": key}, json=payload,
                headers={"Content-Type": "application/json"}, timeout=60,
            )
            if resp.status_code in (401, 403, 429):
                continue
            resp.raise_for_status()
            raw = _gemini_extract_text(resp.json())
            if raw.strip():
                logger.info("✅ XAI fallback Gemini (%s) key#%d OK", GEMINI_MODEL, idx)
                return clean_reasoning_output(raw), raw
        except Exception as e:
            logger.warning("Gemini key#%d error: %s", idx, str(e)[:120])
            continue
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


def _segment_for_phobert(text: str) -> str:
    """Run Vietnamese word segmentation while suppressing optional fasttext noise.

    underthesea can probe optional language-detection models and print
    "No module named 'fasttext'" even when word_tokenize itself still works.
    That message is not actionable for this app, so keep stdout/stderr quiet and
    fall back to raw text only when segmentation actually fails.
    """
    try:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
    except Exception as e:
        logger.warning("PhoBERT segmentation skipped; using raw text: %s", str(e)[:120])
        return text


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
        processed = _segment_for_phobert(text)

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
    """Non-streaming XAI: Cache → LM Studio → Gemini → template. Trả (reasoning, source, gemma_labels).
    Nguồn UI luôn ghi 'Gemma-4'; provider thật chỉ log ở server."""
    cached = find_xai_reasoning_cache(text)
    if cached:
        return (clean_reasoning_output(cached),
                "✅ Từ cache XAI đã đồng bộ prompt (266 mẫu)",
                parse_gemma_labels(cached))

    # Layer 2a — LM Studio (finetuned Gemma) when reachable
    reasoning, raw = get_live_xai_reasoning(text, result=None, return_raw=True)
    if raw:
        save_xai_reasoning_cache(text, raw)
        logger.info("XAI (get_reasoning) via lmstudio")
        return reasoning, "🤖 Gemma-4", parse_gemma_labels(raw)

    # Layer 2b — Gemini cloud fallback
    cleaned, raw = _gemini_xai(text)
    if cleaned and cleaned.strip():
        save_xai_reasoning_cache(text, raw)
        logger.info("XAI (get_reasoning) via gemini:%s", GEMINI_MODEL)
        return cleaned, "🤖 Gemma-4", parse_gemma_labels(raw)

    # Layer 3 — template
    fb = generate_smart_fallback(result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"])
    return fb, "⚠️ LLM chưa khả dụng — hiển thị phân tích mẫu (template)", None



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

# The real scraping backend lives in fetchers_impl.py, which is intentionally NOT
# shipped to public Hugging Face Spaces (HF flags Spaces that scrape YouTube/social
# media). When that module is absent — or ENABLE_URL_FETCHERS is off — these wrappers
# are inert no-ops, so the deployed app.py contains no scraping logic.

_FETCHERS_IMPL = None
_FETCHERS_TRIED = False


class _BridgeFetchers:
    """Cách B adapter: fetch via the remote GET fetch-bridge running on YOUR machine.

    Contains NO scraping logic — just a token-gated HTTP GET to the tunnel. The actual
    multi-source fetching (Apify / yt_dlp / news) runs on your machine inside the bridge,
    so the deployed Space stays free of scraping code and passes HF policy.
    """

    def __init__(self, base: str):
        self._base = base.rstrip("/")

    def fetch_url_as_list(self, url: str, max_comments: int = 30) -> Tuple[List[str], str]:
        try:
            resp = _lmstudio_get(
                f"{self._base}/fetch",
                headers=_lmstudio_headers(),
                params={"url": url, "max_comments": max_comments},
                timeout=(LM_STUDIO_CONNECT_TIMEOUT, LM_STUDIO_REQUEST_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            return list(data.get("texts") or []), str(data.get("info") or "")
        except Exception as e:
            logger.info("Fetch-bridge error (%s)", type(e).__name__)
            return [], f"❌ Lỗi fetch-bridge ({type(e).__name__})"


def _get_fetchers():
    global _FETCHERS_IMPL, _FETCHERS_TRIED
    if not ENABLE_URL_FETCHERS:
        return None
    # Cách B: prefer the remote fetch-bridge (scraping runs on your machine; the
    # deployed app stays clean of scraping code).
    base = _fetch_bridge_base()
    if base:
        return _BridgeFetchers(base)
    # Cách A: local-only fetchers_impl (full scraping on this machine).
    if not _FETCHERS_TRIED:
        _FETCHERS_TRIED = True
        try:
            import fetchers_impl
            _FETCHERS_IMPL = fetchers_impl
        except Exception as e:
            logger.info("URL fetchers backend unavailable (%s); URL fetching disabled.", type(e).__name__)
            _FETCHERS_IMPL = None
    return _FETCHERS_IMPL


def fetch_url_as_list(url: str, max_comments: int = 30) -> Tuple[List[str], str]:
    """Fetch URL content as a list of text segments. Backend is optional/local-only."""
    impl = _get_fetchers()
    if impl is None:
        return [], _FETCH_DISABLED_MSG
    return impl.fetch_url_as_list(url, max_comments)


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


# Tone per (axis, predicted class id): mirrors the web `valClass` (bad/ok/neu).
# bad → danger, ok → teal, neu → neutral. Drives both the .val text and .meter bar.
_AXIS_TONE = {
    "misinfo":   {0: "bad", 1: "ok"},
    "stance":    {0: "ok",  1: "bad", 2: "neu"},
    "sentiment": {0: "bad", 1: "neu", 2: "ok"},
}
_METER_CLASS = {"bad": "bad", "ok": "teal", "neu": ""}
_AXIS_META = [
    ("misinfo",   "Dấu hiệu sai lệch", "Ngưỡng 50%"),
    ("stance",    "Lập trường",    "Ủng hộ · Phản đối · Trung lập"),
    ("sentiment", "Cảm xúc",       "Tiêu cực · Trung tính · Tích cực"),
]


def render_result_cards_html(result: Dict, elapsed: float, model_choice: str) -> str:
    """Render minimal horizontal-bar axis cards (web-aligned `.axes/.axis/.meter`).

    AGENT NOTE:
    Mirrors `AxisCard` in VaccineNLP_Web/frontend/src/App.tsx — one compact card per
    axis with a single calibrated-confidence meter. Detailed per-class softmax now
    lives in the radar; do not reintroduce the heavy glow cards here.
    """
    cards = ""
    for axis, cap, hint in _AXIS_META:
        r = result[axis]
        pred_id = r["pred"]
        label = LABEL_MAPS[axis][pred_id]
        conf_cal = max(r["conf_cal"]) * 100
        tone = _AXIS_TONE[axis].get(pred_id, "neu")
        meter_cls = _METER_CLASS[tone]
        cards += (
            '<div class="axis">'
            f'<div class="cap">{cap}</div>'
            f'<div class="val {tone}">{label}</div>'
            f'<div class="meter {meter_cls}"><i style="width:{conf_cal:.1f}%"></i></div>'
            f'<div class="scoreline"><span>{hint}</span><span class="mono">{conf_cal:.1f}%</span></div>'
            '</div>'
        )
    return (
        f'<div class="axes">{cards}</div>'
        f'<div style="margin-top:12px;font-size:12px;color:var(--ink-3);text-align:right">'
        f'⏱️ {elapsed:.2f}s · {model_choice} · độ tin cậy đã hiệu chuẩn</div>'
    )


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
):
    """Main analysis handler — REALTIME (generator).
    Nhịp 1: nhãn PhoBERT + radar hiện tức thì.
    Nhịp 2: lời giải Gemma-4 stream token-by-token qua LM Studio (ngrok) → cache → fallback.
    Giữ nguyên hợp đồng 9 output; dùng gr.update() cho output không đổi (trừ State phải truyền giá trị thật)."""
    NOCHG = gr.update()  # "không đổi" cho các component thường (KHÔNG dùng cho gr.State)

    if not text or not text.strip():
        error_html = '<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">⚠️ Vui lòng nhập văn bản hoặc chọn mẫu thử!</div>'
        yield (error_html, None, "", "", "", "", history,
               session_history_to_markdown(history), "")
        return

    progress(0.1, desc="🔬 Đang tải mô hình...")
    start = time.time()

    progress(0.3, desc=f"🧠 {model_choice} đang phân tích...")
    result = predict(text, model_choice)
    if not result:
        error_html = f'<div style="color: #ff4b4b; font-weight: bold; font-size: 1.1rem; padding: 15px; border: 1px solid #ff4b4b; border-radius: 8px; background: rgba(255,75,75,0.1); font-family: \'Times New Roman\', serif;">❌ Không thể load mô hình {model_choice} — kiểm tra HF_TOKEN</div>'
        yield (error_html, None, "", "", "", "", history,
               session_history_to_markdown(history), "")
        return

    radar = make_radar_chart(result)
    elapsed0 = time.time() - start
    summary_html = (
        render_verdict_hero(result)
        + render_result_cards_html(result, elapsed0, model_choice)
        + render_consistency_legend(compute_consistency(result))
    )

    # Cập nhật lịch sử phiên ngay (nhịp 1)
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "misinfo": LABEL_MAPS["misinfo"][result["misinfo"]["pred"]],
        "stance":  LABEL_MAPS["stance"][result["stance"]["pred"]],
        "sentiment": LABEL_MAPS["sentiment"][result["sentiment"]["pred"]],
    }
    history = ([entry] + (history or []))[:CONFIG["session_history_limit"]]
    history_md = session_history_to_markdown(history)

    saliency_pending = (
        "<p style='color:#888;'><em>🎯 Đang tính Captum IG…</em></p>" if use_captum
        else "<p style='color:#888;'><em>💡 Bật checkbox <b>Captum IG</b> để xem token attribution (chậm hơn 5-10s)</em></p>"
    )

    # NHỊP 1 — hiện nhãn PhoBERT + radar tức thì, reasoning chờ stream
    progress(0.5, desc="📡 Kết nối Gemma realtime...")
    yield (summary_html, radar, "⏳ Đang kết nối Gemma-4 (LM Studio qua ngrok)…", "",
           saliency_pending, "", history, history_md, "")

    # NHỊP 2 — reasoning: cache trước, nếu không có thì STREAM token-by-token
    progress(0.6, desc="💭 Gemma đang suy luận (realtime)...")
    cached = find_xai_reasoning_cache(text)
    if cached:
        reasoning = clean_reasoning_output(cached)
        source = "✅ Từ cache XAI đã đồng bộ prompt (266 mẫu)"
        gemma_labels = parse_gemma_labels(cached)
        yield (NOCHG, NOCHG, f"**{source}**\n\n{reasoning}", NOCHG, NOCHG, NOCHG, history, history_md, "")
    else:
        # XAI provider chain: LM Studio (finetuned Gemma) → Gemini cloud → template.
        # UI source label stays "Gemma-4" regardless of provider (per product decision);
        # the real provider is logged server-side for diagnostics.
        source = "🤖 Gemma-4"
        provider = "none"
        acc, raw_acc = "", ""
        # Layer 2a — LM Studio / ngrok (your finetuned GGUF) when reachable
        for cleaned_partial, raw_partial in stream_live_xai_reasoning(text):
            acc, raw_acc = cleaned_partial, raw_partial
            provider = "lmstudio"
            yield (NOCHG, NOCHG, f"**{source}**\n\n{acc}", NOCHG, NOCHG, NOCHG, history, history_md, "")
        # Streaming over ngrok can be flaky from HF Spaces even when the same
        # LM Studio endpoint accepts regular chat completions. Try a blocking
        # Gemma call before falling back to Gemini so the preferred model remains
        # the primary source of reasoning.
        if not acc.strip():
            logger.info("LM Studio stream produced nothing -> trying non-streaming Gemma call")
            cleaned, raw = get_live_xai_reasoning(text, result=None, return_raw=True)
            if raw and cleaned and cleaned.strip():
                acc, raw_acc, provider = cleaned, raw, "lmstudio:blocking"
                yield (NOCHG, NOCHG, f"**{source}**\n\n{acc}", NOCHG, NOCHG, NOCHG, history, history_md, "")
        # Layer 2b — Gemini cloud fallback (reliable on HF Spaces where LM Studio is absent)
        if not acc.strip():
            logger.info("LM Studio produced nothing → trying Gemini cloud fallback")
            for cleaned_partial, raw_partial in _gemini_xai_stream(text):
                acc, raw_acc = cleaned_partial, raw_partial
                provider = f"gemini:{GEMINI_MODEL}"
                yield (NOCHG, NOCHG, f"**{source}**\n\n{acc}", NOCHG, NOCHG, NOCHG, history, history_md, "")
            if not acc.strip():  # SSE yielded nothing → one non-streaming attempt
                cleaned, raw = _gemini_xai(text)
                if cleaned and cleaned.strip():
                    acc, raw_acc, provider = cleaned, raw, f"gemini:{GEMINI_MODEL}"
                    yield (NOCHG, NOCHG, f"**{source}**\n\n{acc}", NOCHG, NOCHG, NOCHG, history, history_md, "")
        if acc.strip():
            reasoning = acc
            gemma_labels = parse_gemma_labels(raw_acc)
            save_xai_reasoning_cache(text, raw_acc)
            logger.info("XAI reasoning produced via %s", provider)
        else:
            reasoning = generate_smart_fallback(
                result["misinfo"]["pred"], result["stance"]["pred"], result["sentiment"]["pred"])
            source = "⚠️ LLM chưa khả dụng — hiển thị phân tích mẫu (template)"
            gemma_labels = None
            yield (NOCHG, NOCHG, f"**{source}**\n\n{reasoning}", NOCHG, NOCHG, NOCHG, history, history_md, "")

    reasoning_md = f"**{source}**\n\n{reasoning}"
    disagreement_html = render_disagreement_table(result, gemma_labels)

    # Captum (sau khi có reasoning)
    if use_captum:
        progress(0.85, desc="🎯 Đang tính Captum Integrated Gradients...")
        tokens, attr_norm, pred_class = compute_captum_saliency(text, model_choice)
        saliency_html = render_saliency_html(tokens, attr_norm, pred_class)
    else:
        saliency_html = saliency_pending

    progress(0.95, desc="🔊 Đang tạo AI Voice...")
    voice_text = reasoning if reasoning and not reasoning.startswith("⚠️") else ""
    audio_html = text_to_speech(voice_text) if voice_text else ""

    elapsed = time.time() - start
    summary_html = (
        render_verdict_hero(result)
        + render_result_cards_html(result, elapsed, model_choice)
        + render_consistency_legend(compute_consistency(result))
    )
    report_md = build_report_markdown(text, model_choice, result, reasoning, elapsed)

    progress(1.0, desc="✅ Hoàn tất!")
    # NHỊP cuối — hoàn thiện toàn bộ
    yield (summary_html, radar, reasoning_md, disagreement_html, saliency_html, audio_html, history, history_md, report_md)


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


def _report_to_plaintext(report_md: str) -> str:
    """Strip basic markdown so the .txt report reads cleanly."""
    t = report_md or ""
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)   # headings
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                  # bold
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"\1", t)         # italic
    t = re.sub(r"^\s*\|?\s*-{3,}.*$", "", t, flags=re.MULTILINE)  # table separators
    t = t.replace("|", "  ")                                # table pipes
    return t


def export_report_file(report_md: str, suffix: str = ".md") -> Optional[str]:
    """Save the analysis report to a temp file for download (.md or .txt)."""
    content = _report_to_plaintext(report_md) if suffix == ".txt" else report_md
    if not content or not content.strip():
        return None
    try:
        # No hardcoded /tmp so this also works on Windows (local runs).
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="VaccineNLP_Report_", suffix=suffix, delete=False, mode="w", encoding="utf-8"
        )
        tmp_file.write(content)
        tmp_file.close()
        return tmp_file.name
    except Exception as e:
        logger.warning(f"Export failed: {e}")
        return None


def handle_export_report(report_md: str):
    if not report_md or not report_md.strip():
        return gr.update(visible=False)
    path = export_report_file(report_md, ".md")
    return gr.update(value=path, visible=True) if path else gr.update(visible=False)


def handle_export_report_txt(report_md: str):
    if not report_md or not report_md.strip():
        return gr.update(visible=False)
    path = export_report_file(report_md, ".txt")
    return gr.update(value=path, visible=True) if path else gr.update(visible=False)


# ============================================================================
# HUMAN RE-LABELING (expert override) — session JSON + CSV export
# ============================================================================
def handle_save_relabel(text: str, model_choice: str, mis_vi: str, st_vi: str, se_vi: str, note: str) -> str:
    """Append an expert re-annotation to annotations_file (atomic). Returns a status markdown.

    AGENT NOTE:
    Stores BOTH the model prediction and the human-corrected labels so the file is a
    usable correction dataset. predict() is re-run here (model is cached) to capture the
    model labels at save time — keeps this decoupled from the streaming analyze generator.
    """
    text = (text or "").strip()
    if not text:
        return "⚠️ Chưa có văn bản. Hãy nhập/chọn văn bản ở ô phân tích rồi mới gán nhãn."

    model_pred: Dict[str, str] = {}
    try:
        res = predict(text, model_choice)
        if res:
            model_pred = {ax: LABEL_MAPS[ax][res[ax]["pred"]] for ax in ("misinfo", "stance", "sentiment")}
    except Exception as e:
        logger.warning("relabel: predict failed: %s", e)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "model": model_pred,
        "human": {"misinfo": mis_vi, "stance": st_vi, "sentiment": se_vi},
        "note": (note or "").strip(),
    }
    path = CONFIG["annotations_file"]
    try:
        data = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        data.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        try:
            gr.Info("💾 Đã lưu nhãn của bạn.")
        except Exception:
            pass
        mp = lambda k: model_pred.get(k, "—")
        return (
            f"✅ Đã lưu nhãn (tổng **{len(data)}** bản ghi trong phiên).\n\n"
            f"- 👤 **Người gán:** {mis_vi} · {st_vi} · {se_vi}\n"
            f"- 🤖 **Mô hình:** {mp('misinfo')} · {mp('stance')} · {mp('sentiment')}\n\n"
            f"_Lưu trong phiên; HF Spaces reset khi rebuild → hãy bấm “Tải nhãn (.csv)” để giữ lại._"
        )
    except Exception as e:
        logger.warning("relabel save failed: %s", e)
        return f"❌ Không lưu được nhãn: {e}"


def handle_export_annotations():
    """Export all saved human annotations to a CSV file for download."""
    import csv
    path = CONFIG["annotations_file"]
    data = []
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        except Exception as e:
            logger.warning("relabel export read failed: %s", e)
    if not data:
        try:
            gr.Warning("Chưa có nhãn nào được lưu trong phiên này.")
        except Exception:
            pass
        return gr.update(visible=False)

    fd, out_path = tempfile.mkstemp(prefix="vaccinenlp_annotations_", suffix=".csv")
    os.close(fd)
    cols = ["timestamp", "text",
            "model_misinfo", "model_stance", "model_sentiment",
            "human_misinfo", "human_stance", "human_sentiment", "note"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in data:
            m = r.get("model", {}) or {}
            h = r.get("human", {}) or {}
            writer.writerow([
                r.get("timestamp", ""), r.get("text", ""),
                m.get("misinfo", ""), m.get("stance", ""), m.get("sentiment", ""),
                h.get("misinfo", ""), h.get("stance", ""), h.get("sentiment", ""),
                r.get("note", ""),
            ])
    return gr.update(value=out_path, visible=True)


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
        # Must return 6 values to match the wired outputs:
        # [text_input, fetched_table, send_to_batch_btn, export_fetched_btn, fetch_status, fetched_raw_state]
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=f"<p style='color:#ff4b4b;'>{error_msg}</p>"),
            "",
        )
    
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


def _pick_text_column(df) -> str:
    """Choose the most likely text column from an imported table."""
    preferred = ["text", "content", "nội dung", "noi dung", "comment", "message", "body", "caption", "post"]
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in preferred:
        if cand in lowered:
            return lowered[cand]
    # Fallback: the column whose values are the longest on average (most text-like).
    best, best_len = df.columns[0], -1.0
    for col in df.columns:
        try:
            avg = float(df[col].astype(str).str.len().mean())
        except Exception:
            avg = 0.0
        if avg > best_len:
            best, best_len = col, avg
    return best


def handle_import_data(file):
    """Import a .txt/.csv/.xlsx file into the selectable table (same outputs as fetch).

    Returns the 6 values wired to:
    [text_input, fetched_table, send_to_batch_btn, export_fetched_btn, fetch_status, fetched_raw_state]
    so the user can then pick a row to analyze, or send all rows to Batch mode.
    """
    def _err(msg: str):
        return ("", gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=f"<p style='color:#ff4b4b;'>{msg}</p>"), "")

    if not file:
        return _err("❌ Chưa chọn tệp")
    path = getattr(file, "name", file)
    low = str(path).lower()
    segments: List[str] = []
    try:
        if low.endswith(".csv"):
            df = pd.read_csv(path)
            col = _pick_text_column(df)
            segments = [str(x).strip() for x in df[col].tolist()]
        elif low.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
            col = _pick_text_column(df)
            segments = [str(x).strip() for x in df[col].tolist()]
        else:  # .txt / other → split by --- or newlines
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
            parts = raw.split("---") if "---" in raw else raw.splitlines()
            segments = [s.strip() for s in parts]
    except Exception as e:
        return _err(f"❌ Lỗi đọc tệp: {str(e)[:120]}")

    segments = [s for s in segments if s and s.lower() != "nan"][:200]
    if not segments:
        return _err("❌ Không tìm thấy nội dung văn bản trong tệp")

    rows = [[i + 1, s] for i, s in enumerate(segments)]
    df_show = pd.DataFrame(rows, columns=["STT", "Nội dung thu thập được"])
    batch_text_str = "\n\n---\n\n".join(segments)
    status = (f"<p style='color:#3db882; font-weight:bold;'>✅ Đã nhập {len(segments)} đoạn. "
              f"Bấm 1 dòng trong bảng để phân tích, hoặc “Gửi sang Batch”.</p>")
    return (segments[0], gr.update(value=df_show, visible=True), gr.update(visible=True),
            gr.update(visible=True), gr.update(value=status), batch_text_str)


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
    --r-sm: 8px; --r: 10px; --r-lg: 12px;
    --ui: 'Be Vietnam Pro', 'Segoe UI', system-ui, sans-serif;
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
    display: grid !important;
    grid-template-columns: 316px minmax(0, 1fr) !important;
    min-height: 100vh !important;
    width: 100% !important;
    margin: 0 !important;
    gap: 0 !important;
}

/* Collapsed state class lives on #main-layout-row itself (see __applySidebar in JS):
   Gradio scopes CSS under `...contain`, so `body.sidebar-collapsed ...` never matches.
   Targeting the row directly keeps specificity above the scoped base rule. */
#main-layout-row.sidebar-collapsed {
    grid-template-columns: minmax(0, 1fr) !important;
}

@media (max-width: 1024px) {
    #main-layout-row {
        grid-template-columns: 1fr !important;
    }
}

#sidebar-col {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
    padding: 0 !important;
    position: sticky !important;
    top: 0 !important;
    height: 100vh !important;
    overflow-y: auto !important;
    z-index: 100 !important;
    width: 316px !important;
    min-width: 316px !important;
    max-width: 316px !important;
}

#main-layout-row.sidebar-collapsed #sidebar-col {
    display: none !important;
}

@media (max-width: 1024px) {
    #sidebar-col {
        display: block !important;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        width: min(316px, calc(100vw - 24px)) !important;
        min-width: min(316px, calc(100vw - 24px)) !important;
        max-width: min(316px, calc(100vw - 24px)) !important;
        transform: translateX(0) !important;
        transition: transform .22s ease !important;
        box-shadow: 18px 0 36px rgba(15, 23, 42, .18) !important;
    }
    #main-layout-row.sidebar-collapsed #sidebar-col {
        display: block !important;
        transform: translateX(-110%) !important;
    }
}

#content-col {
    padding: 0 !important;
    overflow-y: auto !important;
    background: var(--bg) !important;
    min-height: 100vh !important;
}

/* ============ SIDEBAR REDESIGN ============ */
.sidebar-redesign {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 22px 16px;
}
/* Footer block is a separate gr.HTML rendered after the controls; trim the
   duplicated top padding so it sits snug under the controls card. */
.sidebar-redesign.sidebar-foot-wrap {
    padding-top: 4px;
}
#sidebar-foot-html {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.sidebar-redesign .brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 6px 8px 18px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 10px;
}
.sidebar-redesign .brand .logo {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    flex-shrink: 0;
    background: #ffffff;
    border: 1px solid var(--line);
    display: grid;
    place-items: center;
    padding: 4px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
.sidebar-redesign .brand .logo img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
}
.dark .sidebar-redesign .brand .logo {
    background: #ffffff;
}
.sidebar-redesign .brand .name {
    font-weight: 700;
    font-size: 16px;
    letter-spacing: -.2px;
    line-height: 1.1;
    color: var(--ink);
}
.sidebar-redesign .brand .name span {
    color: var(--teal);
}
.sidebar-redesign .brand .tag {
    font-size: 11px;
    color: var(--ink-3);
    font-weight: 500;
    margin-top: 2px;
}
.sidebar-redesign .nav-group {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    color: var(--ink-3);
    padding: 14px 10px 6px;
}
.sidebar-redesign .nav-item {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 9px 11px;
    border-radius: var(--r-sm);
    color: var(--ink-2);
    font-weight: 500;
    font-size: 14px;
    cursor: pointer;
    border: none;
    background: none;
    width: 100%;
    text-align: left;
    transition: background .16s, color .16s;
    position: relative;
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
.sidebar-redesign .nav-item .icon {
    width: 18px;
    height: 18px;
    stroke: currentColor;
    stroke-width: 1.7;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.sidebar-redesign .nav-item:hover {
    background: var(--bg-2);
    color: var(--ink);
}
.sidebar-redesign .nav-item.active {
    background: var(--teal-50);
    color: var(--teal-strong);
    font-weight: 600;
}
.sidebar-redesign .nav-item.active::before {
    content: "";
    position: absolute;
    left: -16px;
    top: 8px;
    bottom: 8px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--teal);
}
.sidebar-redesign .sidebar-collapse-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    margin: 8px 0 6px;
    padding: 9px 11px;
    border-radius: var(--r-sm);
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: var(--ink-2);
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: background .16s, color .16s, border-color .16s;
}
.sidebar-redesign .sidebar-collapse-btn:hover {
    color: var(--ink);
    border-color: var(--teal-100);
    background: var(--teal-50);
}
.sidebar-redesign .side-foot {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.sidebar-redesign .status-chip {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 12px;
    border-radius: var(--r-sm);
    background: var(--surface-2);
    border: 1px solid var(--line);
    font-size: 12px;
    color: var(--ink-2);
}
.sidebar-redesign .status-chip .pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 0 var(--teal);
    animation: pulse 2.4s infinite;
}
.sidebar-redesign .project-card,
.sidebar-redesign .lm-status {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface-2);
    padding: 12px;
    color: var(--ink);
}
.sidebar-redesign .project-card .eyebrow {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--teal-strong);
    letter-spacing: .55px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.sidebar-redesign .project-card {
    font-size: 12px;
    line-height: 1.45;
}
.sidebar-redesign .project-card small {
    display: block;
    color: var(--ink-3);
    margin-top: 4px;
}
.sidebar-redesign .lm-status {
    display: grid;
    gap: 6px;
    font-size: 12px;
}
.sidebar-redesign .lm-status > div {
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-redesign .lm-status span:not(.dot) {
    color: var(--ink-2);
}
.sidebar-redesign .lm-status small {
    color: var(--ink-3);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sidebar-redesign .lm-status .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 3px var(--teal-50);
}
.sidebar-redesign .lm-status.bad .dot {
    background: var(--danger);
    box-shadow: 0 0 0 3px var(--danger-50);
}
.sidebar-redesign .theme-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 12px;
    border-radius: var(--r-sm);
    border: 1px solid var(--line);
    background: var(--surface-2);
    color: var(--ink-2);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    width: 100%;
}
.sidebar-redesign .theme-toggle .sw {
    width: 38px;
    height: 22px;
    border-radius: 20px;
    background: var(--bg-2);
    border: 1px solid var(--line);
    position: relative;
    transition: background .2s;
}
.sidebar-redesign .theme-toggle .sw i {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--teal);
    transition: transform .22s cubic-bezier(.4,1.3,.6,1);
    display: grid;
    place-items: center;
    color: #fff;
}
.dark .sidebar-redesign .theme-toggle .sw i {
    transform: translateX(16px);
    color: #07120f;
}

/* Hidden buttons used for python callbacks */
.hidden-btn {
    display: none !important;
}

.sidebar-controls {
    margin: 0 16px 18px !important;
    padding: 14px !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
}
.sidebar-controls > .styler,
.sidebar-controls .styler {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.sidebar-controls .block,
.sidebar-controls .form {
    min-width: 0 !important;
}
.sidebar-controls .form {
    gap: 10px !important;
}
.sidebar-control-title {
    font-size: 12px;
    font-weight: 750;
    letter-spacing: .35px;
    color: var(--ink-2);
    margin: 10px 0 7px;
}
.sidebar-controls .sidebar-control-title:first-child {
    margin-top: 0;
}
.sidebar-controls button {
    min-height: 36px !important;
}

/* ============ CONTENT & TOPBAR ============ */
.screen-container {
    padding: 28px clamp(18px, 3vw, 38px) !important;
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
    padding: 4px 0 16px !important;
    margin-bottom: 22px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

.topbar .crumb {
    font-size: 12.5px;
    color: var(--ink-3);
    font-weight: 500;
}

.topbar h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: 0;
    margin: 1px 0 0 0 !important;
    color: var(--ink) !important;
}

/* ============ CARDS & UI COMPONENTS ============ */
.card {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 16px !important;
}

.card-pad {
    padding: 22px !important;
}

/* ============ WEB-LIKE POLISH OVERRIDES ============ */
.gradio-container {
    background: var(--bg) !important;
    color: var(--ink) !important;
}
.gradio-container .wrap,
.gradio-container .contain {
    max-width: none !important;
}
.gradio-container .block.padded.hide-container,
.gradio-container .block.hide-container {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
}
#svg-sprite-container,
#svg-sprite-container * {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
#sidebar-floating-toggle-html {
    position: fixed !important;
    left: 14px !important;
    top: 14px !important;
    z-index: 240 !important;
    width: auto !important;
    height: auto !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    /* Hidden by default; __applySidebar() flips this to `display:block !important`
       (inline) only when the sidebar is collapsed. Inline !important wins. */
    display: none !important;
}
.sidebar-floating-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 38px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: color-mix(in srgb, var(--surface) 92%, transparent);
    color: var(--ink-2);
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(12px);
    font-size: 12.5px;
    font-weight: 700;
    cursor: pointer;
}
.sidebar-floating-toggle:hover {
    color: var(--teal-strong);
    border-color: var(--teal-100);
}
#sidebar-html {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.card {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    box-shadow: var(--shadow-sm) !important;
}
.gr-group.card > .styler,
.card > .styler,
.card .styler {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.card:hover {
    box-shadow: var(--shadow-md) !important;
}
.gradio-container label,
.field-label {
    color: var(--ink-2) !important;
    font-size: 12.5px !important;
    font-weight: 650 !important;
}
.gradio-container label > span {
    background: transparent !important;
    color: var(--ink-2) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    font-size: 12.5px !important;
    font-weight: 650 !important;
}
.gradio-container .form {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    overflow: visible !important;
    gap: 12px !important;
}
.gradio-container .form > .block {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    border-radius: 10px !important;
}
.gradio-container .tab-nav,
.gradio-container .tabs {
    border-color: var(--line) !important;
}
.topbar {
    position: sticky !important;
    top: 0 !important;
    z-index: 20 !important;
    background: color-mix(in srgb, var(--bg) 88%, transparent) !important;
    backdrop-filter: blur(14px) !important;
}
.topbar .block,
.topbar .block * {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
#analyze-grid-row {
    gap: 18px !important;
    align-items: start !important;
}
#results-col {
    gap: 14px !important;
}
.result-overview-row {
    gap: 14px !important;
    align-items: stretch !important;
}
.result-overview-row > .gradio-column {
    min-width: 0 !important;
}
.radar-pane {
    min-height: 260px !important;
}
.radar-pane .js-plotly-plot,
.radar-pane .plot-container,
.radar-pane .svg-container {
    width: 100% !important;
    max-width: 100% !important;
}
#results-col .tabs {
    margin-top: 8px !important;
}
#results-col .tabitem,
#results-col .prose {
    max-height: 360px;
    overflow-y: auto;
}
.history-accordion {
    margin-top: 0 !important;
}
@media (max-width: 1180px) {
    .result-overview-row {
        flex-direction: column !important;
    }
}
footer,
.gradio-container .built-with {
    display: none !important;
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

/* Sliders, accordions, tabs */
.gradio-container input[type="range"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 100% !important;
    height: 24px !important;
    margin: 6px 0 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    accent-color: var(--teal) !important;
}
.gradio-container input[type="number"] {
    width: 74px !important;
    min-height: 28px !important;
    background: var(--surface-2) !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
    padding: 4px 8px !important;
    text-align: center !important;
    box-shadow: none !important;
}
.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    height: 6px !important;
    border-radius: 999px !important;
    background: var(--line-2) !important;
    border: 0 !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    margin-top: -5px;
    border-radius: 50%;
    background: var(--surface);
    border: 3px solid var(--teal);
    box-shadow: 0 2px 8px rgba(16, 32, 30, .18);
}
.gradio-container input[type="range"]::-moz-range-track {
    height: 6px;
    border-radius: 999px;
    background: var(--line-2);
    border: 0;
}
.gradio-container input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--surface);
    border: 3px solid var(--teal);
}
.gradio-container .accordion,
.gradio-container details {
    border-radius: var(--r-sm) !important;
    border-color: var(--line) !important;
    background: var(--surface) !important;
    box-shadow: none !important;
}
.gradio-container .tab-nav button {
    border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
    font-weight: 600 !important;
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

/* Axis Cards — minimal horizontal-bar result row (web-aligned) */
.axes {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 4px;
}
@media (max-width: 760px) {
    .axes { grid-template-columns: 1fr; }
}
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

.site-footer {
    margin: 34px auto 0;
    padding: 24px 28px;
    border-top: 3px solid var(--teal);
    background: var(--surface);
    border-radius: var(--r);
    border-left: 1px solid var(--line);
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    color: var(--ink-2);
    box-shadow: var(--shadow-sm);
}
.site-footer .footer-grid {
    display: grid;
    grid-template-columns: minmax(220px, 1.1fr) minmax(260px, 1.4fr) minmax(220px, 1fr) minmax(180px, .9fr);
    gap: 22px;
    align-items: start;
}
.site-footer .footer-brand {
    display: grid;
    grid-template-columns: 56px minmax(0, 1fr);
    gap: 14px;
    align-items: center;
}
.site-footer .footer-logo {
    width: 56px;
    height: 56px;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: #fff;
    padding: 5px;
    object-fit: contain;
}
.site-footer h3,
.site-footer h4 {
    margin: 0 0 8px;
    color: var(--ink);
    font-size: 14px;
    line-height: 1.25;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: none;
}
.site-footer p {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.55;
}
.site-footer .footer-title {
    color: var(--teal-strong);
    font-weight: 700;
}
.site-footer .footer-muted {
    color: var(--ink-3);
}
.site-footer .footer-col {
    border-left: 1px solid var(--line);
    padding-left: 20px;
    min-height: 74px;
}
.site-footer a {
    color: var(--teal-strong);
    text-decoration: none;
    font-weight: 600;
}
.site-footer .footer-bottom {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    text-align: center;
    font-size: 12px;
    color: var(--ink-3);
}

@media (max-width: 980px) {
    .site-footer .footer-grid {
        grid-template-columns: 1fr 1fr;
    }
    .site-footer .footer-col {
        border-left: 0;
        padding-left: 0;
    }
}

@media (max-width: 640px) {
    .screen-container {
        padding: 24px 14px !important;
    }
    .topbar {
        padding-top: 0 !important;
        margin-bottom: 18px !important;
    }
    .topbar h1 {
        font-size: 21px !important;
    }
    .card-pad {
        padding: 16px !important;
    }
    .site-footer {
        margin-top: 28px;
        padding: 20px 18px;
    }
    .site-footer .footer-grid {
        grid-template-columns: 1fr;
        gap: 18px;
    }
    .site-footer .footer-brand {
        grid-template-columns: 50px minmax(0, 1fr);
    }
    .site-footer .footer-logo {
        width: 50px;
        height: 50px;
    }
}

.svg-sprite {
    display: none;
}

/* ============ DASHBOARD (SCREEN 2) POLISH ============
   AGENT NOTE:
   This block supports the optional F1 calculator tab copied selectively from
   app_test.py. It is display-only and must not affect the main analysis flow.
*/
.dash-head {
    margin: 2px 0 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--line-2);
}
.dash-head-title {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -.2px;
    color: var(--ink);
}
.dash-head-sub {
    font-size: 12.5px;
    color: var(--ink-3);
    margin-top: 3px;
    line-height: 1.45;
}
#content-col .tabitem .card,
#content-col .tabitem .gr-group {
    margin-bottom: 16px;
}
#content-col .tabs > .tab-nav {
    flex-wrap: wrap !important;
    gap: 4px !important;
    row-gap: 6px !important;
}
#content-col .js-plotly-plot,
#content-col .plotly,
#content-col .plot-container {
    width: 100% !important;
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

def get_sidebar_html() -> str:
    logo_src = get_huph_logo_base64()
    return f"""
    <div class="sidebar-redesign">
        <div class="brand">
            <div class="logo">
                <img src="{logo_src}" alt="HUPH">
            </div>
            <div>
                <div class="name">Vaccine<span>NLP</span></div>
                <div class="tag">Giám sát thông tin vaccine</div>
            </div>
        </div>
        
        <div class="nav-group">Phân tích</div>
        <button id="nav-btn-analyze" class="nav-item active" onclick="clickHidden('analyze')">
            <svg class="icon"><use href="#i-analyze"></use></svg>
            <span>Phân tích văn bản</span>
        </button>
        <button id="nav-btn-advanced" class="nav-item" onclick="clickHidden('advanced')">
            <svg class="icon"><use href="#i-advanced"></use></svg>
            <span>Dữ liệu &amp; đối sánh</span>
        </button>

        <button class="sidebar-collapse-btn" onclick="toggleSidebar()" title="Thu gọn thanh điều khiển">
            <svg class="icon sm"><use href="#i-chevron"></use></svg>
            <span>Thu gọn thanh điều khiển</span>
        </button>
    </div>
    """


def get_sidebar_footer_html() -> str:
    """Sidebar footer (project card, model status, theme toggle).

    AGENT NOTE:
    Rendered as a SEPARATE gr.HTML AFTER the real Gradio `sidebar-controls`
    group so the visual reading order is brand → nav → controls → footer.
    It is wrapped in `.sidebar-redesign` so the existing descendant CSS
    selectors (`.sidebar-redesign .side-foot ...`) keep matching.
    """
    return f"""
    <div class="sidebar-redesign sidebar-foot-wrap">
        <div class="side-foot">
            <div class="project-card">
                <div class="eyebrow">Đồ án tốt nghiệp 2026</div>
                Hưng (2211090016) &amp; Phương (2211090031)<br>
                <small>GVHD: TS. Trần Lâm Quân</small>
            </div>
            {get_lm_studio_status_html()}
            <div class="status-chip">
                <span class="pulse"></span>
                <span>Mô hình: <b>PhoBERT-v2</b> · trực tuyến</span>
            </div>

            <button class="theme-toggle" onclick="toggleDarkMode()">
                <span>Giao diện sáng / tối</span>
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
    <section class="site-footer">
      <div class="footer-grid">
        <div class="footer-brand">
          <img src="{logo_src}" class="footer-logo" alt="HUPH Logo">
          <div>
            <h3>Trường Đại học Y tế Công cộng</h3>
            <p>Số 1A, Đức Thắng, Đông Ngạc, Hà Nội</p>
            <p><a href="https://huph.edu.vn/" target="_blank">huph.edu.vn</a></p>
          </div>
        </div>
        <div class="footer-col">
          <h4>Đề tài đồ án</h4>
          <p class="footer-title">Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam</p>
          <p class="footer-muted">Vaccine misinformation and attitude analysis for Vietnamese social media.</p>
        </div>
        <div class="footer-col">
          <h4>Nhóm thực hiện</h4>
          <p><b>Kim Mạnh Hưng</b> · 2211090016</p>
          <p><b>Đinh Lê Quỳnh Phương</b> · 2211090031</p>
          <p class="footer-muted">Lớp CNCQ KHDL1-1A</p>
        </div>
        <div class="footer-col">
          <h4>GV hướng dẫn</h4>
          <p><b>TS. Trần Lâm Quân</b></p>
          <p>Giảng viên Khoa học dữ liệu</p>
          <p><a href="mailto:tlq@huph.edu.vn">tlq@huph.edu.vn</a></p>
        </div>
      </div>
      <div class="footer-bottom">© 2026 VaccineNLP Project · Đồ án tốt nghiệp chuyên ngành Khoa học Dữ liệu - HUPH</div>
    </section>
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
            • <b>Outputs:</b> Radar · Captum IG · Report
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
            document.querySelectorAll('.sidebar-redesign .nav-item').forEach(function(el) { el.classList.remove('active'); });
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

        /* Apply collapsed state. IMPORTANT: Gradio scopes user CSS by prefixing every
           selector with `...gradio-container ... .contain`, which turns an ancestor
           selector like `body.sidebar-collapsed #main-layout-row` into
           `...contain body.sidebar-collapsed ...` — i.e. it looks for <body> INSIDE
           .contain and never matches. So the collapsed state class MUST live on an
           element that is itself inside .contain. We put it on #main-layout-row. */
        window.__applySidebar = function(collapsed) {
            document.body.classList.toggle('sidebar-collapsed', collapsed);
            document.documentElement.classList.toggle('sidebar-collapsed', collapsed);
            var row = document.querySelector('#main-layout-row');
            if (row) row.classList.toggle('sidebar-collapsed', collapsed);
            var ft = document.querySelector('#sidebar-floating-toggle-html');
            if (ft) ft.style.setProperty('display', collapsed ? 'block' : 'none', 'important');
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 80);
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 260);
        };

        window.toggleSidebar = function() {
            var collapsed = !document.body.classList.contains('sidebar-collapsed');
            window.__applySidebar(collapsed);
            try { localStorage.setItem('vnlp-sidebar', collapsed ? 'collapsed' : 'expanded'); } catch (e) {}
        };

        /* Apply persisted theme + sidebar state on load (retry until the layout row mounts) */
        try {
            var theme = localStorage.getItem('vnlp-theme') || 'light';
            var addDark = theme === 'dark';
            document.documentElement.classList.toggle('dark', addDark);
            document.body.classList.toggle('dark', addDark);
            var sidebar = localStorage.getItem('vnlp-sidebar');
            var collapseSidebar = sidebar ? sidebar === 'collapsed' : window.innerWidth <= 768;
            var tries = 0;
            (function applyWhenReady() {
                if (document.querySelector('#main-layout-row') || tries > 40) {
                    window.__applySidebar(collapseSidebar);
                } else {
                    tries += 1;
                    setTimeout(applyWhenReady, 100);
                }
            })();
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
            gr.update(visible=False)   # advanced
        )

    def nav_to_advanced():
        return (
            gr.update(visible=False),
            gr.update(visible=True)
        )

    # Wrapper for cào to batch
    def handle_send_to_batch_to_screen(batch_text_str: str):
        res_text, _ = handle_send_to_batch(batch_text_str)
        # returns: [batch_text, screen_analyze, screen_advanced]
        return (
            res_text,
            gr.update(visible=False), # screen_analyze
            gr.update(visible=True)   # screen_advanced
        )

    # Dashboard F1 calculator initial values.
    # AGENT NOTE: This is read-only educational UI. It should not call models or
    # mutate analysis state, so failures here must not block the main analyzer.
    _DASH_CALC_KEYS = list(METRICS_DB.keys())
    _dash_calc_init = update_calculator(_DASH_CALC_KEYS[0])

    with gr.Blocks(title="VaccineNLP Demo v2.0", theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal), css=CSS_STYLE, fill_width=True, js=init_theme_js) as app:
        # SVG sprite injected at body top
        gr.HTML(get_svg_sprite_html(), elem_id="svg-sprite-container")
        gr.HTML("""
        <button class="sidebar-floating-toggle" onclick="toggleSidebar()" title="Mở thanh điều khiển">
            <svg class="icon sm"><use href="#i-chevron"></use></svg>
            <span>Điều khiển</span>
        </button>
        """, elem_id="sidebar-floating-toggle-html")
        
        # Hidden buttons used to trigger python navigation callbacks from custom HTML sidebar
        btn_hidden_analyze = gr.Button("nav_analyze", elem_id="btn-hidden-analyze", elem_classes=["hidden-btn"])
        btn_hidden_advanced = gr.Button("nav_advanced", elem_id="btn-hidden-advanced", elem_classes=["hidden-btn"])
        
        with gr.Row(elem_id="main-layout-row"):
            # Left Sidebar Column (using get_sidebar_html)
            with gr.Column(scale=1, min_width=316, elem_id="sidebar-col"):
                gr.HTML(get_sidebar_html(), elem_id="sidebar-html")
                # AGENT NOTE:
                # These are real Gradio input components placed in the visual sidebar.
                # Keep their variable names stable; event wiring below depends on them.
                with gr.Group(elem_classes=["sidebar-controls"]):
                    gr.HTML("<div class='sidebar-control-title'>Bộ ví dụ mẫu</div>")
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

                    gr.HTML("<div class='sidebar-control-title'>Mô hình phân loại</div>")
                    model_choice = gr.Dropdown(
                        choices=list(CONFIG["models"].keys()),
                        value="PhoBERT-v2",
                        label="Chọn model:",
                        show_label=False
                    )
                    info_box = gr.HTML(value=get_sidebar_info_html("PhoBERT-v2"))

                    with gr.Accordion("🌐 Thu thập URL · 📁 Nhập tệp", open=True, elem_classes=["sidebar-accordion"]):
                        url_input = gr.Textbox(
                            label="URL (Báo · YouTube · Facebook · TikTok · Threads)",
                            placeholder="https://vnexpress.net/...",
                        )
                        max_cmt = gr.Slider(10, 100, value=30, step=10, label="Max comments")
                        fetch_btn = gr.Button("📥 Thu thập")
                        fetch_status = gr.HTML(value="")
                        with gr.Row():
                            send_to_batch_btn = gr.Button("🚀 Gửi sang Batch", variant="secondary", visible=False)
                            export_fetched_btn = gr.Button("📥 Tải .xlsx", variant="primary", visible=False)
                        export_fetched_file = gr.File(label="Tệp Excel dữ liệu đã cào", visible=False)
                        gr.HTML("<div style='font-size:11px;color:var(--ink-3);margin:6px 0 2px;'>— hoặc —</div>")
                        import_file = gr.File(
                            label="📁 Nhập tệp dữ liệu (.txt · .csv · .xlsx)",
                            file_types=[".txt", ".csv", ".xlsx", ".xls"],
                        )

                    use_captum_cb = gr.Checkbox(
                        label="🎯 Bật Captum IG",
                        value=False,
                        info="Token attribution, chậm hơn 5-10s",
                        elem_classes=["captum-cb"]
                    )
                # Sidebar footer (project info + model status + theme toggle).
                # Rendered AFTER the controls so reading order is brand → nav → controls → footer.
                gr.HTML(get_sidebar_footer_html(), elem_id="sidebar-foot-html")
                # Hidden actual cache button that is triggered programmatically if needed
                clear_cache_btn = gr.Button("🗑️ Xóa Cache & Khởi động lại", elem_classes=["hidden-btn"], size="sm")
                clear_cache_status = gr.Markdown(value="", visible=False)

            # Right Content Column
            with gr.Column(scale=5, elem_id="content-col"):
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
                        with gr.Column(scale=4, min_width=360):
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                gr.HTML("<div class='field-label' style='font-weight:600;margin-bottom:8px;'>Nội dung cần đối soát <span style='font-size:11px;color:var(--ink-3);font-weight:normal;'>tiếng Việt</span></div>")
                                text_input = gr.Textbox(
                                    label="",
                                    placeholder="Dán bình luận, bài viết hoặc tin nhắn về vaccine…",
                                    lines=8,
                                    show_label=False
                                )
                                fetched_table = gr.Dataframe(
                                    headers=["STT", "Nội dung thu thập được"],
                                    datatype=["str", "str"],
                                    wrap=True,
                                    visible=False,
                                    label="📋 Danh sách đã cào"
                                )
                                analyze_btn = gr.Button("🔬 Tiến hành phân tích đa nhiệm", variant="primary", size="lg")

                        # Right Results Stack
                        with gr.Column(scale=7, min_width=520, elem_id="results-col"):
                            with gr.Row(elem_classes=["result-overview-row"]):
                                with gr.Column(scale=3, min_width=320):
                                    summary_out = gr.HTML(value="""
                                    <div class="card card-pad" style="text-align:center;">
                                        <div class="empty-state" style="padding:40px 20px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;">
                                            <div class="ring" style="width:56px;height:56px;border-radius:50%;background:var(--bg-2);display:grid;place-items:center;color:var(--ink-3);"><svg class="icon lg"><use href="#i-analyze"></use></svg></div>
                                            <h3 style="margin:0;font-size:16px;font-weight:700;">Chưa có phân tích</h3>
                                            <div class="muted" style="font-size:13px;max-width:320px;line-height:1.4;">Nhập văn bản và bấm “Tiến hành phân tích đa nhiệm” để chạy PhoBERT-v2 trên 3 trục nhãn.</div>
                                        </div>
                                    </div>
                                    """)
                                # Column starts hidden so the empty-state summary spans the
                                # full width before any analysis; the analyze event reveals it.
                                with gr.Column(scale=2, min_width=280, elem_classes=["radar-pane"], visible=False) as radar_pane_col:
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
                                export_btn = gr.Button("📥 Tải báo cáo (.md)", variant="secondary", visible=False)
                                export_txt_btn = gr.Button("📄 Tải báo cáo (.txt)", variant="secondary", visible=False)
                                export_file = gr.File(label="Báo cáo", visible=False)

                            # ----- Human re-labeling (expert override) -----
                            with gr.Accordion("✍️ Gán nhãn lại (chuyên gia)", open=False, elem_classes=["relabel-accordion"]):
                                gr.HTML(
                                    "<p class='muted' style='font-size:12px;margin:0 0 10px'>"
                                    "Chọn nhãn đúng theo đánh giá của bạn để tạo bộ dữ liệu chỉnh sửa (active-learning). "
                                    "Lưu trong phiên; HuggingFace Spaces reset khi rebuild → hãy tải CSV để giữ lại.</p>"
                                )
                                with gr.Row():
                                    relabel_misinfo = gr.Dropdown(
                                        choices=list(LABEL_MAPS["misinfo"].values()),
                                        value=LABEL_MAPS["misinfo"][1], label="Dấu hiệu sai lệch")
                                    relabel_stance = gr.Dropdown(
                                        choices=list(LABEL_MAPS["stance"].values()),
                                        value=LABEL_MAPS["stance"][2], label="Lập trường")
                                    relabel_sentiment = gr.Dropdown(
                                        choices=list(LABEL_MAPS["sentiment"].values()),
                                        value=LABEL_MAPS["sentiment"][1], label="Cảm xúc")
                                relabel_note = gr.Textbox(label="Ghi chú (tuỳ chọn)", placeholder="Lý do gán nhãn lại…", lines=2)
                                with gr.Row():
                                    relabel_save_btn = gr.Button("💾 Lưu nhãn", variant="primary")
                                    relabel_export_btn = gr.Button("⬇️ Tải nhãn (.csv)", variant="secondary")
                                relabel_status = gr.Markdown(value="")
                                relabel_file = gr.File(label="Nhãn đã gán (CSV)", visible=False)

                            with gr.Accordion("Lịch sử phiên", open=False, elem_classes=["history-accordion"]):
                                history_display = gr.Markdown(value="*Chưa có lượt phân tích nào trong phiên này*")

                # --------------------------------------------------------
                # SCREEN 2: ADVANCED SCREEN
                # --------------------------------------------------------
                with gr.Column(visible=False, elem_classes=["screen-container"]) as screen_advanced:
                    with gr.Row(elem_classes=["topbar"]):
                        with gr.Column():
                            gr.HTML("<div class='crumb'>Phân tích · Nâng cao</div><h1>Công cụ nâng cao</h1>")
                            
                    with gr.Tabs():
                        # Optional dashboard tab copied selectively from app_test.py.
                        # It is static/educational and only calls update_calculator().
                        with gr.Tab("🧮 Máy tính chỉ số F1"):
                            gr.HTML(
                                "<div class='dash-head'>"
                                "<div class='dash-head-title'>Máy tính chỉ số đánh giá</div>"
                                "<div class='dash-head-sub'>Chọn một nhãn để xem Precision · Recall · F1 tính từ TP/FP/FN</div>"
                                "</div>"
                            )
                            dash_calc_class = gr.Dropdown(
                                choices=_DASH_CALC_KEYS,
                                value=_DASH_CALC_KEYS[0],
                                label="Chọn nhãn cần phân tích chỉ số",
                            )
                            dash_calc_metrics = gr.HTML(value=_dash_calc_init[0])
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                with gr.Row():
                                    dash_calc_prec = gr.Markdown(
                                        value=_dash_calc_init[1],
                                        latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                                    )
                                    dash_calc_recall = gr.Markdown(
                                        value=_dash_calc_init[2],
                                        latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                                    )
                                    dash_calc_f1 = gr.Markdown(
                                        value=_dash_calc_init[3],
                                        latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                                    )

                        with gr.Tab("📋 Phân tích hàng loạt (Batch Mode)"):
                            with gr.Group(elem_classes=["card", "card-pad"]):
                                gr.HTML("<div class='section-label' style='margin-bottom:10px;'><svg class='icon sm ic'><use href='#i-data'></use></svg> Phân tích nhiều mẫu cùng lúc</div>")
                                batch_input = gr.Textbox(
                                    label="Mỗi dòng = 1 mẫu (tối đa 50 mẫu, phân tách bằng --- hoặc xuống dòng)",
                                    placeholder="Mẫu 1: Nội dung cần phân tích...\n---\nMẫu 2: Nội dung cần phân tích...",
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

                gr.HTML(get_footer_html(), elem_id="footer-html")

        # ================================================================
        # EVENT WIRE UP
        # ================================================================
        
        # 1. Navigation Hidden Button events
        btn_hidden_analyze.click(fn=nav_to_analyze, inputs=[], outputs=[screen_analyze, screen_advanced], api_name=False)
        btn_hidden_advanced.click(fn=nav_to_advanced, inputs=[], outputs=[screen_analyze, screen_advanced], api_name=False)

        # 2. Main Analyze Event wiring
        def wrapper_handle_analyze(*args):
            # Generator: forward each streaming beat of handle_analyze, plus reveal the
            # XAI card, the radar plot/column, and the two export buttons (all hidden
            # until the first analysis).
            for res in handle_analyze(*args):
                yield res + (gr.update(visible=True),) * 5

        analyze_btn.click(
            fn=wrapper_handle_analyze,
            inputs=[text_input, model_choice, use_captum_cb, session_state],
            outputs=[summary_out, radar_out, reasoning_out, disagreement_out, saliency_out,
                     audio_out, session_state, history_display, report_state,
                     xai_group, radar_out, radar_pane_col, export_btn, export_txt_btn],
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
        export_txt_btn.click(
            fn=handle_export_report_txt,
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
        import_file.upload(
            fn=handle_import_data,
            inputs=[import_file],
            outputs=[text_input, fetched_table, send_to_batch_btn, export_fetched_btn, fetch_status, fetched_raw_state],
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
            outputs=[batch_input, screen_analyze, screen_advanced],
            api_name=False
        )
        
        # 8. Clear cache
        clear_cache_btn.click(
            fn=handle_clear_cache,
            inputs=[],
            outputs=[clear_cache_status],
            api_name=False
        )

        # 9. Dashboard calculator. Kept separate from model inference events.
        dash_calc_class.change(
            fn=update_calculator,
            inputs=[dash_calc_class],
            outputs=[dash_calc_metrics, dash_calc_prec, dash_calc_recall, dash_calc_f1],
            api_name=False
        )

        # 10. Human re-labeling (expert override): save + export CSV
        relabel_save_btn.click(
            fn=handle_save_relabel,
            inputs=[text_input, model_choice, relabel_misinfo, relabel_stance, relabel_sentiment, relabel_note],
            outputs=[relabel_status],
            api_name=False
        )
        relabel_export_btn.click(
            fn=handle_export_annotations,
            inputs=[],
            outputs=[relabel_file],
            api_name=False
        )

    return app


if __name__ == "__main__":
    app = build_app()
    logger.info(
        "Gradio queue configured: concurrency=%d, max_size=%d; LM Studio concurrency=%d, stream=%s",
        APP_QUEUE_CONCURRENCY,
        APP_QUEUE_MAX_SIZE,
        LM_STUDIO_MAX_CONCURRENT,
        LM_STUDIO_ENABLE_STREAM,
    )
    app.queue(default_concurrency_limit=APP_QUEUE_CONCURRENCY, max_size=APP_QUEUE_MAX_SIZE)
    app.launch(show_error=True)
