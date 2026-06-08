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
import os, re, json, threading
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ----------------------------- Cấu hình -----------------------------
XAI_BACKEND    = os.environ.get("XAI_BACKEND", "lmstudio").lower()   # "lmstudio" | "local"
LMSTUDIO_URL   = os.environ.get("LMSTUDIO_URL", "http://host.docker.internal:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "gemma-4-e4b-vaccine-xai-merged")
LM_API_TOKEN   = os.environ.get("LM_API_TOKEN", "")
GGUF_PATH      = os.environ.get("GGUF_MODEL_PATH", "/models/gemma-4-4b-Q4_K_M.gguf")
LLM_THREADS    = int(os.environ.get("LLM_THREADS", str(os.cpu_count() or 4)))
N_CTX          = int(os.environ.get("N_CTX", "3072"))   # đủ chỗ cho prompt + phần GIẢI THÍCH chi tiết
# Đo thực tế (LM Studio): prompt ~330 tok, completion ~220 tok, finish=stop (KHÔNG bị cắt).
# 1536 cho output để phần "Giải thích" dài vẫn còn dư; N_CTX=3072 vẫn chứa được input dài hơn.
MAX_TOKENS, TEMPERATURE, STOP = 1536, 0.2, ["\n\n\n"]
# Nạp GGUF nếu chạy backend local, HOẶC giữ làm Safe Mode khi backend lmstudio
LOAD_LOCAL = (XAI_BACKEND == "local") or (os.environ.get("LOCAL_FALLBACK", "0") == "1")

ALLOWED = {
    "misinfo":   {"Fake", "Real"},
    "stance":    {"Favor", "Against", "Neutral"},
    "sentiment": {"Positive", "Negative", "Neutral"},
}
llm = {}
_gen_lock = threading.Lock()   # SERIALIZE generation trên CPU

def _get_headers() -> dict:
    return {"Authorization": f"Bearer {LM_API_TOKEN}"} if LM_API_TOKEN else {}

def _load_local_model():
    from llama_cpp import Llama
    base_kwargs = dict(model_path=GGUF_PATH, n_ctx=N_CTX, n_threads=LLM_THREADS,
                       n_threads_batch=LLM_THREADS, n_batch=768, n_gpu_layers=0,
                       use_mmap=True, verbose=False)
    # flash_attn có thể không tồn tại ở vài bản llama-cpp-python -> thử rồi rớt
    try:
        model = Llama(flash_attn=True, **base_kwargs)
    except TypeError:
        print("⚠️ [xai_service] bản llama-cpp-python không nhận flash_attn -> nạp không flash_attn", flush=True)
        model = Llama(**base_kwargs)
    # Prompt-prefix cache: tái dùng KV của SYSTEM_PROMPT cố định
    try:
        from llama_cpp import LlamaRAMCache
        model.set_cache(LlamaRAMCache(capacity_bytes=256 * 1024 * 1024))
    except Exception as e:
        print(f"ℹ️ [xai_service] không bật được LlamaRAMCache ({type(e).__name__}) — bỏ qua", flush=True)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"ℹ️ [xai_service] XAI_BACKEND={XAI_BACKEND} | LM Studio={LMSTUDIO_URL} | n_threads={LLM_THREADS} n_ctx={N_CTX}", flush=True)
    if LOAD_LOCAL:
        try:
            llm["model"] = _load_local_model()
            print(f"✅ [xai_service] GGUF loaded ({GGUF_PATH})", flush=True)
            # WARM-UP: lần gọi đầu luôn chậm -> nuốt nó lúc khởi động
            try:
                with _gen_lock:
                    llm["model"].create_chat_completion(
                        messages=[{"role": "user", "content": "ok"}], max_tokens=1)
                print("✅ [xai_service] warm-up xong", flush=True)
            except Exception as e:
                print(f"ℹ️ [xai_service] warm-up lỗi ({type(e).__name__}) — bỏ qua", flush=True)
        except Exception as e:
            print(f"⚠️ [xai_service] KHÔNG nạp được GGUF ({type(e).__name__}: {e})", flush=True)
    yield
    llm.clear()

app = FastAPI(title="VaccineNLP XAI Engine (PA4)", lifespan=lifespan)

@app.get("/")
def health():
    lm_ok = False
    if XAI_BACKEND == "lmstudio":
        try:
            lm_ok = requests.get(f"{LMSTUDIO_URL}/models", headers=_get_headers(), timeout=2).ok
        except Exception:
            lm_ok = False
    return {"service": "XAI Engine", "backend": XAI_BACKEND,
            "lm_studio_reachable": lm_ok, "local_gguf_loaded": "model" in llm}

class ExplainRequest(BaseModel):
    text: str
    predicted_labels: dict

SYSTEM_PROMPT = (
    "You are a highly rigorous Public Health Data Analyst in Vietnam. Your task is to analyze social media comments regarding vaccines and annotate them strictly.\n\n"
    "CRITICAL LANGUAGE RULE — BẮT BUỘC: It is mandatory to reason ENTIRELY in Vietnamese (Tiếng Việt). "
    "Hãy đặc biệt thận trọng với tiếng lóng / teen-code (tiếng lóng), phương ngữ vùng miền (phương ngữ), "
    "và các từ viết tắt mới / phi chuẩn (từ viết tắt mới) mà bạn có thể CHƯA TỪNG gặp khi huấn luyện: "
    "TUYỆT ĐỐI không suy đoán nghĩa một cách mù quáng — hãy luận nghĩa từ ngữ cảnh xung quanh; nếu một từ vẫn mơ hồ, "
    "hãy dựa vào thái độ TỔNG THỂ của tác giả thay vì bám vào token lạ. "
    "Do not be distracted by philosophical arguments, political rhetoric, or manipulative half-truths. "
    "Focus entirely on identifying the author's TRUE attitude toward VACCINATION, which is typically revealed in the opening or closing sentences.\n\n"
    "Classify the comment based on these 3 criteria:\n"
    "1. Misinformation: [Chính xác] or [Tin giả]\n"
    "2. Stance: [Ủng hộ] or [Phản đối] or [Trung lập]\n"
    "3. Sentiment: [Tích cực] or [Tiêu cực] or [Trung tính]\n\n"
    "OUTPUT STRUCTURE — PHẢI theo đúng thứ tự này (sinh NHÃN TRƯỚC, GIẢI THÍCH SAU):\n"
    "- DÒNG ĐẦU TIÊN phải khớp CHÍNH XÁC định dạng: "
    "Kết quả: <Misinformation Label> | <Stance Label> | <Sentiment Label>\n"
    "- SAU dòng đó, viết 'Giải thích:' rồi trình bày suy luận từng bước bằng Tiếng Việt, "
    "biện giải chi tiết cho TỪNG nhãn trong 3 tiêu chí ở trên."
)

def _build_messages(text: str, predicted_labels: dict):
    user = f"Văn bản: {text}\nNhãn dự đoán: {predicted_labels}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

# ----------------------------- parse GIỮ NGUYÊN -----------------------------
def parse_gemma_output(raw: str) -> dict:
    out = {"reasoning": "", "gemma_labels": {}, "raw_output": raw, "parse_ok": False}
    if not raw or not raw.strip():
        return out
    label_mapping = {
        "Chính Xác": "Real", "Không Tin Giả": "Real", "Không Thông Tin Giả": "Real",
        "Tin Giả / Sai Lệch": "Fake", "Tin Giả": "Fake",
        "Ủng Hộ": "Favor", "Phản Đối": "Against", "Trung Lập": "Neutral",
        "Tích Cực": "Positive", "Tiêu Cực": "Negative", "Trung Tính": "Neutral",
    }
    thought = ""
    clean_raw = raw
    if "<channel|>" in raw:
        parts = raw.split("<channel|>")
        clean_raw = parts[-1].strip()
        if "<|channel>thought" in parts[0]:
            thought = parts[0].split("<|channel>thought")[-1].strip()
    clean_raw = clean_raw.replace('<turn|>', '').replace('</start_of_turn>', '').replace('<end_of_turn>', '').replace('<eos>', '').strip()
    m_res = re.search(r"(?:Kết quả|Kết luận|Result)\s*:\s*(.+)", clean_raw, flags=re.IGNORECASE)
    reasoning_text = clean_raw
    if m_res:
        # Robust cho CẢ HAI thứ tự: nhãn-TRƯỚC -> giải thích nằm SAU dòng "Kết quả";
        # nhãn-SAU (tương thích ngược) -> giải thích nằm TRƯỚC dòng "Kết quả".
        before, _, after = clean_raw.partition(m_res.group(0))
        reasoning_text = after.strip() or before.strip()
    reasoning_text = re.sub(r'^(?:Lý luận|Giải thích|Reasoning)\s*:\s*', '', reasoning_text, flags=re.IGNORECASE).strip()
    if m_res:
        parts = [p.strip("[] \t\r\n") for p in m_res.group(1).split("|") if p.strip()]
        if len(parts) >= 3:
            cand = {"misinfo": parts[0].title(), "stance": parts[1].title(), "sentiment": parts[2].title()}
            mapped_cand = {k: label_mapping.get(v, v) for k, v in cand.items()}
            if all(mapped_cand[k] in ALLOWED[k] for k in ALLOWED):
                out.update({"gemma_labels": mapped_cand, "parse_ok": True,
                            "reasoning": reasoning_text if reasoning_text else thought})
    return out

# ----------------------------- Backends -----------------------------
def _lmstudio_stream(messages):
    body = {"model": LMSTUDIO_MODEL, "messages": messages, "max_tokens": MAX_TOKENS,
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

def _stream_backend(messages):
    """Chọn backend; nếu lmstudio lỗi mà có GGUF thì rớt sang local."""
    if XAI_BACKEND == "local":
        yield from _local_stream(messages)
        return
    try:
        yield from _lmstudio_stream(messages)
    except Exception:
        if "model" in llm:
            yield from _local_stream(messages)   # Safe Mode
        else:
            raise

def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ----------------------------- Endpoints -----------------------------
@app.post("/api/explain")
def explain(req: ExplainRequest):
    """Non-stream (cho cache/background). Gom hết stream rồi parse."""
    messages = _build_messages(req.text, req.predicted_labels)
    try:
        raw = "".join(_stream_backend(messages))
    except Exception as e:
        return {"reasoning": "", "gemma_labels": {},
                "raw_output": f"XAI lỗi ({type(e).__name__}).", "parse_ok": False, "mode": "error"}
    out = parse_gemma_output(raw)
    out["mode"] = XAI_BACKEND
    return out

@app.post("/api/explain-stream")
def explain_stream(req: ExplainRequest):
    """SSE: stream token sống; sự kiện 'final' chứa nhãn đã parse."""
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
        yield _sse({"type": "final", **parsed})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
