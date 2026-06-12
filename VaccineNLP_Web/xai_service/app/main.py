"""
xai_service/app/main.py — PHƯƠNG ÁN 4 (public): GGUF-CPU làm backend XAI on-demand, có streaming.

Công tắc backend qua env XAI_BACKEND:
  - "lmstudio"  (mặc định, dùng cho LOCAL dev có LM Studio + GPU)
  - "local"     (dùng cho PUBLIC: GGUF chạy CPU NGAY trong container, on-demand)

Tối ưu CPU đã áp (xem ghi chú nghiên cứu):
  - flash_attn=True (bật thủ công; tự rớt nếu bản build không nhận kwarg)
  - n_threads = số nhân; n_ctx nhỏ; warm-up lúc khởi động
  - LlamaRAMCache để tái dùng KV của SYSTEM_PROMPT cố định (bỏ prefill lặp)
  - _gen_lock: SERIALIZE mọi generation (CPU 1 lần/luồng-đầy tốt hơn song song tranh nhau)
  - stream trực tiếp từ GGUF (create_chat_completion stream=True) -> token chảy ra UI thật

LƯU Ý: SYSTEM_PROMPT yêu cầu sinh NHÃN (Kết quả) TRƯỚC, GIẢI THÍCH chi tiết SAU,
và bắt buộc suy luận bằng Tiếng Việt (cảnh giác tiếng lóng / phương ngữ / từ viết tắt
mới chưa được đào tạo). parse_gemma_output đã được làm robust cho CẢ HAI thứ tự
(nhãn-trước hoặc nhãn-sau) để tương thích ngược.
"""
import os, re, json, sys, threading, unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from src.common.xai_prompt import (
        LABEL_VI,
        XAI_MAX_TOKENS,
        XAI_STOP,
        XAI_SYSTEM_PROMPT,
        XAI_TEMPERATURE,
        build_xai_messages,
    )
except ModuleNotFoundError:
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.common.xai_prompt import (
        LABEL_VI,
        XAI_MAX_TOKENS,
        XAI_STOP,
        XAI_SYSTEM_PROMPT,
        XAI_TEMPERATURE,
        build_xai_messages,
    )

# ----------------------------- Cấu hình -----------------------------
def _load_env_defaults():
    """Load local .env files without overriding Docker/real environment."""
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "VaccineNLP_Web" / ".env",
        here.parents[2] / ".env",  # VaccineNLP_Web/.env
        here.parents[3] / ".env",  # repo root .env
    ]
    seen = set()
    for path in candidates:
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

def _running_in_container() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"

def _default_lmstudio_url() -> str:
    return "http://host.docker.internal:1234/v1" if _running_in_container() else "http://localhost:1234/v1"

def _normalize_lmstudio_url(url: str) -> str:
    value = (url or _default_lmstudio_url()).strip().rstrip("/")
    if not _running_in_container():
        value = value.replace("host.docker.internal", "localhost")
    return value

_load_env_defaults()

XAI_BACKEND    = _env_any("XAI_BACKEND", default="lmstudio").lower()   # "lmstudio" | "local"
LMSTUDIO_URL   = _normalize_lmstudio_url(_env_any("LMSTUDIO_URL", "LM_STUDIO_URL", default=_default_lmstudio_url()))
LMSTUDIO_MODEL = _env_any("LMSTUDIO_MODEL", "LM_STUDIO_MODEL", default="gemma-4-e4b-vaccine-xai-merged")
LM_API_TOKEN   = _env_any("LM_API_TOKEN", "LMSTUDIO_API_KEY", "LM_STUDIO_API_KEY", default="")
GGUF_PATH      = _env_any("GGUF_MODEL_PATH", default="/models/gemma-4-4b-Q4_K_M.gguf")
XAI_CACHE_PATH = _env_any(
    "XAI_CACHE_PATH",
    default=str(Path(__file__).resolve().parents[3] / "app_gradio" / "data" / "xai_cache.json"),
)
LLM_THREADS    = int(os.environ.get("LLM_THREADS", str(os.cpu_count() or 4)))
N_CTX          = int(os.environ.get("N_CTX", "4096"))   # đủ chỗ cho prompt + phần GIẢI THÍCH chi tiết
# Đo thực tế (LM Studio): prompt ~330 tok, completion ~220 tok, finish=stop (KHÔNG bị cắt).
# 2048 cho output để phần "Giải thích" dài vẫn còn dư; N_CTX=4096 vẫn chứa được input dài hơn.
MAX_TOKENS, TEMPERATURE, STOP = XAI_MAX_TOKENS, XAI_TEMPERATURE, XAI_STOP
# Nạp GGUF nếu chạy backend local, HOẶC giữ làm Safe Mode khi backend lmstudio
LOAD_LOCAL = (XAI_BACKEND == "local") or (os.environ.get("LOCAL_FALLBACK", "0") == "1")

# Gemini cloud fallback setup
_GEMINI_KEYS: list = []
for _gi in range(1, 6):
    _gname = "GEMINI_API_KEY" if _gi == 1 else f"GEMINI_API_KEY_{_gi}"
    _gval = os.environ.get(_gname, "").strip()
    if _gval and _gval not in _GEMINI_KEYS:
        _GEMINI_KEYS.append(_gval)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

ALLOWED = {
    "misinfo":   {"Fake", "Real"},
    "stance":    {"Favor", "Against", "Neutral"},
    "sentiment": {"Positive", "Negative", "Neutral"},
}
llm = {}
_gen_lock = threading.Lock()   # SERIALIZE generation trên CPU
_cache_lock = threading.Lock()
_xai_cache: dict[str, str] | None = None

def _get_headers() -> dict:
    headers = {"ngrok-skip-browser-warning": "true"}
    if LM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LM_API_TOKEN}"
    return headers

def _load_xai_cache() -> dict[str, str]:
    """Load rebuilt Gradio XAI cache for Web/XAI service reuse."""
    global _xai_cache
    with _cache_lock:
        if _xai_cache is not None:
            return _xai_cache

        cache_path = Path(XAI_CACHE_PATH)
        cache: dict[str, str] = {}
        try:
            if cache_path.exists():
                loaded = json.loads(cache_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    cache = {
                        str(key): str(value)
                        for key, value in loaded.items()
                        if isinstance(key, str) and isinstance(value, str)
                    }
                print(f"[OK] [xai_service] loaded XAI cache entries={len(cache)} path={cache_path}", flush=True)
            else:
                print(f"[INFO] [xai_service] XAI cache not found path={cache_path}", flush=True)
        except Exception as e:
            print(f"[WARN] [xai_service] could not load XAI cache ({type(e).__name__}: {e})", flush=True)
        _xai_cache = cache
        return _xai_cache

def _find_xai_cache(text: str) -> str | None:
    """Exact + conservative fuzzy match against rebuilt prompt-compatible cache."""
    def normalize(value: str) -> str:
        folded = _strip_accents(value or "").lower()
        return re.sub(r"[^a-z0-9]+", "", folded)

    text_strip = (text or "").strip()
    if not text_strip:
        return None

    cache = _load_xai_cache()
    exact = cache.get(text_strip)
    if exact:
        return exact

    input_norm = normalize(text_strip)
    if len(input_norm) <= 24:
        return None
    for key, value in cache.items():
        key_norm = normalize(key)
        if key_norm and (input_norm in key_norm or key_norm in input_norm):
            return value
    return None

def _lmstudio_model_ids(timeout: float = 2.0) -> list[str]:
    try:
        resp = requests.get(f"{LMSTUDIO_URL}/models", headers=_get_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [item.get("id", "") for item in data if item.get("id")]
    except Exception:
        return []

def _resolve_lmstudio_model(timeout: float = 2.0) -> str:
    model_ids = _lmstudio_model_ids(timeout=timeout)
    if not model_ids or LMSTUDIO_MODEL in model_ids:
        return LMSTUDIO_MODEL
    preferred = next((mid for mid in model_ids if "gemma" in mid.lower()), model_ids[0])
    print(f"[WARN] [xai_service] configured LMSTUDIO_MODEL='{LMSTUDIO_MODEL}' not loaded; using '{preferred}'", flush=True)
    return preferred

def _load_local_model():
    from llama_cpp import Llama
    base_kwargs = dict(model_path=GGUF_PATH, n_ctx=N_CTX, n_threads=LLM_THREADS,
                       n_threads_batch=LLM_THREADS, n_batch=768, n_gpu_layers=0,
                       use_mmap=True, verbose=False)
    # flash_attn có thể không tồn tại ở vài bản llama-cpp-python -> thử rồi rớt
    try:
        model = Llama(flash_attn=True, **base_kwargs)
    except TypeError:
        print("[WARN] [xai_service] llama-cpp-python does not accept flash_attn -> loading without flash_attn", flush=True)
        model = Llama(**base_kwargs)
    # Prompt-prefix cache: tái dùng KV của SYSTEM_PROMPT cố định
    try:
        from llama_cpp import LlamaRAMCache
        model.set_cache(LlamaRAMCache(capacity_bytes=256 * 1024 * 1024))
    except Exception as e:
        print(f"[INFO] [xai_service] could not enable LlamaRAMCache ({type(e).__name__}) - skipped", flush=True)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[INFO] [xai_service] XAI_BACKEND={XAI_BACKEND} | LM Studio={LMSTUDIO_URL} | n_threads={LLM_THREADS} n_ctx={N_CTX}", flush=True)
    _load_xai_cache()
    if LOAD_LOCAL:
        try:
            llm["model"] = _load_local_model()
            print(f"[OK] [xai_service] GGUF loaded ({GGUF_PATH})", flush=True)
            # WARM-UP: lần gọi đầu luôn chậm -> nuốt nó lúc khởi động
            try:
                with _gen_lock:
                    llm["model"].create_chat_completion(
                        messages=[{"role": "user", "content": "ok"}], max_tokens=1)
                print("[OK] [xai_service] warm-up done", flush=True)
            except Exception as e:
                print(f"[INFO] [xai_service] warm-up failed ({type(e).__name__}) - skipped", flush=True)
        except Exception as e:
            print(f"[WARN] [xai_service] could not load GGUF ({type(e).__name__}: {e})", flush=True)
    yield
    llm.clear()

app = FastAPI(title="VaccineNLP XAI Engine (PA4)", lifespan=lifespan)

@app.get("/")
def health():
    model_ids: list[str] = []
    if XAI_BACKEND == "lmstudio":
        model_ids = _lmstudio_model_ids(timeout=2)
    cache = _load_xai_cache()
    return {"service": "XAI Engine", "backend": XAI_BACKEND,
            "lm_studio_reachable": bool(model_ids), "local_gguf_loaded": "model" in llm,
            "lmstudio_url": LMSTUDIO_URL, "configured_model": LMSTUDIO_MODEL,
            "effective_model": _resolve_lmstudio_model(timeout=1) if model_ids else LMSTUDIO_MODEL,
            "available_models": model_ids[:8],
            "xai_cache_entries": len(cache),
            "xai_cache_path": XAI_CACHE_PATH}

class ExplainRequest(BaseModel):
    text: str
    predicted_labels: dict

SYSTEM_PROMPT = XAI_SYSTEM_PROMPT
_LABEL_VI = LABEL_VI

def _build_messages(text: str, predicted_labels: dict):
    return build_xai_messages(text, predicted_labels)

# Parse Gemma output: accept the notebook block format and legacy pipe format,
# while cleaning LM Studio special tokens before the frontend sees them.
_RESULT_MARKER_RE = re.compile(r"={2,}\s*(?:kết\s*quả|ket\s*qua|result)\s*={2,}", re.IGNORECASE)
_EXPLAIN_MARKER_RE = re.compile(r"={2,}\s*(?:giải\s*thích|giai\s*thich|explanation)\s*={2,}", re.IGNORECASE)

def _strip_accents(text: str) -> str:
    folded = unicodedata.normalize("NFD", text or "")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d").replace("Đ", "D")

def _norm_text(text: str) -> str:
    folded = _strip_accents(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()

def _clean_generation(raw: str) -> str:
    text = raw or ""
    for token in (
        "<turn|>", "<|turn|>", "<|turn>", "<end_of_turn>", "</start_of_turn>",
        "<eos>", "<|channel>thought", "<|channel|>thought", "<|channel>",
    ):
        text = text.replace(token, "")
    text = re.sub(r"<start_of_turn>\s*(?:user|model)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|?channel\|?>\s*(?:analysis|thought|final)?", "", text, flags=re.IGNORECASE)
    return text.strip()

def _axis_from_key(key: str) -> str | None:
    norm = _norm_text(key)
    if any(term in norm for term in ("tinh xac thuc", "misinformation", "misinfo", "xac thuc", "dau hieu sai lech")):
        return "misinfo"
    if any(term in norm for term in ("thai do", "lap truong", "stance")):
        return "stance"
    if any(term in norm for term in ("cam xuc", "cam thieu", "cam thich", "tong the", "sentiment")):
        return "sentiment"
    return None

def _strip_axis_prefix(value: str) -> str:
    cleaned = value.strip(" -*\t\r\n")
    bracket = re.search(r"\[([^\]]+)\]", cleaned)
    if bracket:
        cleaned = bracket.group(1)
    if ":" in cleaned:
        maybe_key, maybe_value = cleaned.split(":", 1)
        if _axis_from_key(maybe_key):
            cleaned = maybe_value
    return cleaned.strip(" -*[]\t\r\n")

def _map_label(axis: str, value: str) -> str | None:
    norm = _norm_text(_strip_axis_prefix(value))
    if not norm:
        return None

    if axis == "misinfo":
        if any(term in norm for term in ("khong tin gia", "not misinformation", "not fake", "real", "accurate", "correct", "thong tin dung")):
            return "Real"
        if any(term in norm for term in (
            "khong xac dinh", "trung lap", "khong phai tin gia",
            "khong co thong tin sai lech", "khong chua thong tin sai lech",
            "khong de cap den vaccine", "khong lien quan den vaccine",
            "khong co dau hieu sai lech", "khong phat hien dau hieu sai lech", "khong co dau hieu tin gia",
        )):
            return "Real"
        if "chinh xac" in norm and "khong chinh xac" not in norm:
            return "Real"
        if any(term in norm for term in (
            "tin gia", "sai lech", "khong chinh xac", "misinformation", "fake", "false", "inaccurate",
            "co dau hieu tin gia", "co dau hieu sai lech",
        )):
            return "Fake"
    elif axis == "stance":
        if any(term in norm for term in ("khong ung ho", "phan doi", "anti vaccine", "against", "oppose", "opposition")):
            return "Against"
        if any(term in norm for term in ("trung lap", "neutral")):
            return "Neutral"
        if any(term in norm for term in ("ung ho", "dong tinh", "tan thanh", "support", "favor", "pro vaccine")):
            return "Favor"
    elif axis == "sentiment":
        if any(term in norm for term in ("trung tinh", "neutral")):
            return "Neutral"
        if any(term in norm for term in ("tieu cuc", "tieu thich", "tieu thieu", "negative")):
            return "Negative"
        if any(term in norm for term in ("tich cuc", "positive", "lac quan")):
            return "Positive"
    return None

def _parse_axis_lines(section: str) -> dict:
    labels = {}
    for line in section.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        axis = _axis_from_key(key)
        if not axis:
            continue
        mapped = _map_label(axis, value)
        if mapped:
            labels[axis] = mapped
    return labels

def _parse_pipe_labels(text: str) -> dict:
    line = text.strip()
    line = re.sub(r"^(?:kết\s*quả|ket\s*qua|kết\s*luận|ket\s*luan|result)\s*:\s*", "", line, flags=re.IGNORECASE)
    parts = [_strip_axis_prefix(part) for part in line.split("|") if part.strip()]
    if len(parts) < 3:
        return {}
    labels = {
        "misinfo": _map_label("misinfo", parts[0]),
        "stance": _map_label("stance", parts[1]),
        "sentiment": _map_label("sentiment", parts[2]),
    }
    return {axis: value for axis, value in labels.items() if value}

def _clean_reasoning_text(text: str) -> str:
    cleaned = _clean_generation(text)
    cleaned = re.sub(r"^\s*(?:lý\s*luận|ly\s*luan|giải\s*thích|giai\s*thich|reasoning|analysis)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?is)\n?\s*(?:therefore|do đó|vi vậy|vì vậy)\s*:.*$", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*={2,}\s*(?:hết\s*)?(?:giải\s*thích|giai\s*thich|end\s*explanation)\s*={2,}\s*$", "", cleaned).strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def _valid_labels(labels: dict) -> bool:
    return all(labels.get(axis) in ALLOWED[axis] for axis in ALLOWED)

def parse_gemma_output(raw: str) -> dict:
    out = {"reasoning": "", "gemma_labels": {}, "raw_output": raw, "parse_ok": False}
    if not raw or not raw.strip():
        return out

    clean_raw = _clean_generation(raw)
    labels = {}
    reasoning_text = ""

    result_marker = _RESULT_MARKER_RE.search(clean_raw)
    if result_marker:
        explain_marker = _EXPLAIN_MARKER_RE.search(clean_raw, result_marker.end())
        result_section = clean_raw[result_marker.end(): explain_marker.start() if explain_marker else len(clean_raw)]
        labels = _parse_axis_lines(result_section)
        if not labels:
            labels = _parse_pipe_labels(result_section)
        if explain_marker:
            reasoning_text = clean_raw[explain_marker.end():].strip()

    if not labels:
        line_match = re.search(
            r"(?:Kết\s*quả|Ket\s*qua|Kết\s*luận|Ket\s*luan|Result)\s*:\s*([^\n\r]+)",
            clean_raw,
            flags=re.IGNORECASE,
        )
        if line_match:
            labels = _parse_pipe_labels(line_match.group(0))
            after = clean_raw[line_match.end():].strip()
            before = clean_raw[:line_match.start()].strip()
            reasoning_text = after or before

    if _valid_labels(labels):
        out.update({
            "reasoning": _clean_reasoning_text(reasoning_text),
            "gemma_labels": labels,
            "parse_ok": True,
        })
    return out

# ----------------------------- Backends -----------------------------
def _lmstudio_stream(messages):
    effective_model = _resolve_lmstudio_model(timeout=5)
    body = {"model": effective_model, "messages": messages, "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE, "stop": STOP, "stream": True}
    with requests.post(f"{LMSTUDIO_URL}/chat/completions", json=body, headers=_get_headers(),
                       stream=True, timeout=(10, 300)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta

def _local_stream(messages):
    """Stream token trực tiếp từ GGUF (CPU). Giữ _gen_lock suốt stream -> serialize."""
    if "model" not in llm:
        raise RuntimeError("GGUF local chưa nạp")
    with _gen_lock:
        for chunk in llm["model"].create_chat_completion(
                messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                stop=STOP, stream=True):
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta

def _gemini_payload(messages: list) -> dict:
    """Map the shared system/user XAI contract to the Gemini generateContent schema."""
    system_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_txt = next((m["content"] for m in messages if m["role"] == "user"), "")
    return {
        "system_instruction": {"parts": [{"text": system_txt}]},
        "contents": [{"role": "user", "parts": [{"text": user_txt}]}],
        "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS},
    }

def _gemini_extract_text(data: dict) -> str:
    """Pull concatenated text from a Gemini response/chunk JSON object."""
    out = ""
    for cand in (data.get("candidates") or []):
        for part in (cand.get("content", {}).get("parts") or []):
            out += part.get("text", "") or ""
    return out

def _gemini_stream(messages: list):
    """Yield delta tokens from Gemini SSE streaming.
    Rotates through _GEMINI_KEYS on auth/quota errors (401/403/429).
    """
    if not _GEMINI_KEYS:
        return
    payload = _gemini_payload(messages)
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
    for idx, key in enumerate(_GEMINI_KEYS, 1):
        try:
            print(f"[INFO] [xai_service] Trying Gemini API key#{idx}...", flush=True)
            resp = requests.post(
                url, params={"key": key}, json=payload,
                headers={"Content-Type": "application/json"},
                stream=True, timeout=60,
            )
            if resp.status_code in (401, 403, 429):
                print(f"[WARN] [xai_service] Gemini key#{idx} rejected with status {resp.status_code}; rotating...", flush=True)
                continue
            resp.raise_for_status()
            
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
                except (ValueError, KeyError, json.JSONDecodeError):
                    delta = ""
                if delta:
                    raw += delta
                    yield delta
            if raw.strip():
                return  # success — do not try further keys
        except Exception as e:
            print(f"[WARN] [xai_service] Gemini stream key#{idx} error: {e}", flush=True)
            continue

def _stream_backend(messages):
    """Chọn backend; nếu lmstudio lỗi thì thử local GGUF hoặc Gemini cloud fallback."""
    if XAI_BACKEND == "local":
        yield from _local_stream(messages)
        return
    try:
        yield from _lmstudio_stream(messages)
    except Exception as e:
        print(f"[WARN] [xai_service] LM Studio failed ({type(e).__name__}: {e})", flush=True)
        if "model" in llm:
            print("[INFO] [xai_service] Falling back to local GGUF...", flush=True)
            yield from _local_stream(messages)   # Safe Mode
        elif _GEMINI_KEYS:
            print("[INFO] [xai_service] Falling back to Gemini cloud...", flush=True)
            yield from _gemini_stream(messages)
        else:
            raise

def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ----------------------------- Endpoints -----------------------------
def _safe_evidence(text: str) -> list:
    try:
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parents[3])
        orig_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != repo_root and p not in ("", ".")]
        try:
            from app.evidence import retrieve_evidence
            return retrieve_evidence(text, k=3)
        finally:
            sys.path = orig_path
    except Exception as e:
        print(f"[WARN] [xai_service] retrieve_evidence failed ({type(e).__name__}: {e})", flush=True)
        return []

def _safe_anomaly(text: str) -> dict:
    try:
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parents[3])
        orig_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != repo_root and p not in ("", ".")]
        try:
            from app.anomaly import embedding_anomaly
            return embedding_anomaly(text)
        finally:
            sys.path = orig_path
    except Exception as e:
        print(f"[WARN] [xai_service] embedding_anomaly failed ({type(e).__name__}: {e})", flush=True)
        return {"is_anomalous": True, "max_similarity": 0.0, "anomaly_tau": 0.45}

@app.post("/api/explain")
def explain(req: ExplainRequest):
    """Non-stream (cho cache/background). Gom hết stream rồi parse."""
    cached = _find_xai_cache(req.text)
    if cached:
        out = parse_gemma_output(cached)
        out["mode"] = "cache"
        out["cache_hit"] = True
        out["evidence"] = _safe_evidence(req.text)
        out["anomaly"] = _safe_anomaly(req.text)
        return out

    messages = _build_messages(req.text, req.predicted_labels)
    try:
        raw = "".join(_stream_backend(messages))
    except Exception as e:
        return {"reasoning": "", "gemma_labels": {},
                "raw_output": f"XAI lỗi ({type(e).__name__}).", "parse_ok": False, "mode": "error",
                "evidence": _safe_evidence(req.text),
                "anomaly": _safe_anomaly(req.text)}
    out = parse_gemma_output(raw)
    out["mode"] = XAI_BACKEND
    out["cache_hit"] = False
    out["evidence"] = _safe_evidence(req.text)
    out["anomaly"] = _safe_anomaly(req.text)
    return out

@app.post("/api/explain-stream")
def explain_stream(req: ExplainRequest):
    """SSE: stream token sống; sự kiện 'final' chứa nhãn đã parse."""
    cached = _find_xai_cache(req.text)
    if cached:
        parsed = parse_gemma_output(cached)
        parsed["mode"] = "cache"
        parsed["cache_hit"] = True

        def gen_cached():
            reasoning = parsed.get("reasoning", "")
            chunk_size = 64
            for i in range(0, len(reasoning), chunk_size):
                yield _sse({"type": "token", "content": reasoning[i:i + chunk_size]})
            parsed["evidence"] = _safe_evidence(req.text)
            parsed["anomaly"] = _safe_anomaly(req.text)
            yield _sse({"type": "final", **parsed})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen_cached(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    messages = _build_messages(req.text, req.predicted_labels)

    def gen():
        acc = []
        try:
            for delta in _stream_backend(messages):
                acc.append(delta)
                yield _sse({"type": "token", "content": delta})
        except Exception as e:
            yield _sse({"type": "error", "message": f"XAI lỗi ({type(e).__name__})."})
            yield "data: [DONE]\n\n"
            return
        parsed = parse_gemma_output("".join(acc))
        parsed["mode"] = XAI_BACKEND
        parsed["cache_hit"] = False
        parsed["evidence"] = _safe_evidence(req.text)
        parsed["anomaly"] = _safe_anomaly(req.text)
        yield _sse({"type": "final", **parsed})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
