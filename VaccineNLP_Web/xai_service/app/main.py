import os, re, json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

MODEL_PATH = os.environ.get("GGUF_MODEL_PATH", "/models/gemma-4-4b-Q4_K_M.gguf")
ALLOWED = {
    "misinfo":   {"Fake", "Real"},
    "stance":    {"Favor", "Against", "Neutral"},
    "sentiment": {"Positive", "Negative", "Neutral"},
}
llm = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        llm["model"] = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=int(os.environ.get("LLM_THREADS", "4")),
            n_gpu_layers=0,
            verbose=False,
        )
        print(f"✅ [xai_service] REAL Gemma GGUF loaded từ {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"⚠️ [xai_service] KHÔNG nạp được Gemma ({type(e).__name__}: {e}) -> /api/explain sẽ trả lỗi sạch", flush=True)
    yield
    llm.clear()

app = FastAPI(title="VaccineNLP XAI Engine", lifespan=lifespan)

@app.get("/")
def health():
    return {"service": "XAI Engine", "model_loaded": "model" in llm}

class ExplainRequest(BaseModel):
    text: str
    predicted_labels: dict

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích. Cho văn bản và 3 nhãn dự đoán, hãy giải thích NGẮN GỌN "
    "vì sao văn bản đồng thời mang 3 nhãn (misinfo, stance, sentiment), rồi kết luận. "
    "Định dạng BẮT BUỘC:\nLý luận: <giải thích>\nKết quả: <misinfo> | <stance> | <sentiment>"
)

def parse_gemma_output(raw: str) -> dict:
    out = {"reasoning": "", "gemma_labels": {}, "raw_output": raw, "parse_ok": False}
    if not raw or not raw.strip():
        return out
    try:
        j = json.loads(raw); labels = j.get("labels") or j.get("gemma_labels") or {}
        if isinstance(labels, dict):
            cand = {k: str(labels.get(k, "")).strip().title() for k in ALLOWED}
            if all(cand[k] in ALLOWED[k] for k in ALLOWED):
                return {"reasoning": str(j.get("reasoning","")).strip(), "gemma_labels": cand, "raw_output": raw, "parse_ok": True}
    except (json.JSONDecodeError, TypeError):
        pass
    m_res = re.search(r"(?:Kết quả|Kết luận|Result)\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    m_rea = re.search(r"(?:Lý luận|Giải thích|Reasoning)\s*:\s*(.+?)(?:\n\s*(?:Kết quả|Kết luận|Result)|$)", raw, flags=re.IGNORECASE|re.DOTALL)
    if m_res:
        parts = [p.strip() for p in re.split(r"[|/,]", m_res.group(1)) if p.strip()]
        if len(parts) >= 3:
            cand = {"misinfo": parts[0].title(), "stance": parts[1].title(), "sentiment": parts[2].title()}
            if all(cand[k] in ALLOWED[k] for k in ALLOWED):
                out.update({"gemma_labels": cand, "parse_ok": True,
                            "reasoning": (m_rea.group(1).strip() if m_rea else "")})
    return out

@app.post("/api/explain")
def explain(req: ExplainRequest):
    if "model" not in llm:
        return {"reasoning": "", "gemma_labels": {}, "raw_output": "XAI model chưa được nạp do lỗi file gguf.", "parse_ok": False}
        
    user = f"Văn bản: {req.text}\nNhãn dự đoán: {req.predicted_labels}"
    resp = llm["model"].create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        max_tokens=256,
        temperature=0.2,
        stop=["\n\n\n"],
    )
    raw = resp["choices"][0]["message"]["content"]
    return parse_gemma_output(raw)
