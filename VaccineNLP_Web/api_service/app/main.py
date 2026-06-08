import os, hashlib, random, json
from typing import Optional
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal, AnalysisHistory, get_db

import torch
from transformers import AutoTokenizer
from .model_classes import PhoBERTMultitaskClassifier

MODEL_PATH = os.environ.get("PHOBERT_PATH", "/models/phobert_multitask.pt")
phobert = {}

from .preprocess import prepare_text

ID2LABEL = {
    "misinfo":   {0: "Fake", 1: "Real"},
    "stance":    {0: "Favor", 1: "Against", 2: "Neutral"},
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
}


def compute_consistency(misinfo: str, stance: str, sentiment: str) -> str:
    # A — BẤT THƯỜNG (tổ hợp hiếm theo H1 & H3, Gold n=186) → nghi model sai / văn bản lạ
    if stance == "Against" and sentiment == "Positive":       # H1: 0/48 quan sát
        return "unusual"
    if misinfo == "Fake" and stance in ("Favor", "Neutral"):  # H3: Favor 1,9% / Neutral 3,6%
        return "unusual"
    # B — NGUY CƠ CAO (profile chống-vaccine: H1 → 93,8% Phản đối mang Tiêu cực;
    #     H3 → 50% Phản đối chứa Tin giả) → nên rà soát tăng cường
    if stance == "Against" and sentiment == "Negative":
        return "high_risk"
    return "plausible"

def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        phobert["tok"] = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
        model = PhoBERTMultitaskClassifier(num_misinfo=2, num_stance=3, num_sentiment=3)
        state = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Remap heads.X -> head_X if present
        new_state = {}
        for k, v in state.items():
            k_new = k.replace("heads.misinfo.", "head_misinfo.")\
                     .replace("heads.stance.", "head_stance.")\
                     .replace("heads.sentiment.", "head_sentiment.")
            new_state[k_new] = v
        state = new_state

        print("STATE KEYS (Mapped):", list(state.keys())[:10], flush=True)
        print("MODEL KEYS:", list(model.state_dict().keys())[:10], flush=True)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print("[WARN] strict=True error:", e, flush=True)
            res = model.load_state_dict(state, strict=False)
            print("missing:", res.missing_keys, "| unexpected:", res.unexpected_keys, flush=True)
            critical_heads = ["head_misinfo.weight", "head_misinfo.bias", "head_stance.weight", "head_stance.bias", "head_sentiment.weight", "head_sentiment.bias"]
            if any(h in res.missing_keys for h in critical_heads):
                raise RuntimeError("Critical task heads missing from state_dict!")
        model.eval()
        phobert["model"] = model
        print(f"[OK] [api_service] REAL PhoBERT loaded from {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"[WARN] [api_service] Could not load PhoBERT ({type(e).__name__}: {e}) -> SWITCHING TO MOCK MODE", flush=True)
    yield
    phobert.clear()

app = FastAPI(title="VaccineNLP Core API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def call_gemma_explain(row_id: int, text: str, labels: dict):
    db = SessionLocal()
    try:
        try:
            r = requests.post(os.environ["XAI_SERVICE_URL"] + "/api/explain",
                              json={"text": text, "predicted_labels": labels}, timeout=120)
            r.raise_for_status()
            data = r.json()
            gl = data.get("gemma_labels", {})
            data["disagreement"] = {k: (labels.get(k) != gl.get(k)) for k in labels}
            row = db.get(AnalysisHistory, row_id)
            if row:
                row.xai_explanation = data
                row.xai_status = "done"
                db.commit()
        except (requests.exceptions.RequestException, ValueError):
            row = db.get(AnalysisHistory, row_id)
            if row:
                row.xai_status = "failed"
                db.commit()
    finally:
        db.close()

class AnalyzeRequest(BaseModel):
    text: str
    source_url: Optional[str] = None

class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    misinfo_label: str
    misinfo_score: float
    stance_label: str
    stance_score: float
    sentiment_label: str
    sentiment_score: float
    phobert_probs: Optional[dict] = None          # MỚI: phân phối softmax đầy đủ
    consistency_flag: str
    xai_status: str
    xai_explanation: Optional[dict] = None

@torch.no_grad()
def phobert_infer(text: str) -> dict:
    """
    Trả về: {"misinfo": (label, top_score), "stance": (...), "sentiment": (...),
             "_probs": {"misinfo": {"Fake":p,"Real":p}, "stance": {...}, "sentiment": {...}}}
    - Tuple (label, score) GIỮ NGUYÊN để không vỡ các call-site cũ.
    - "_probs": phân phối softmax ĐẦY ĐỦ (thật) cho card phân phối ở frontend.
    """
    if "model" not in phobert:
        # MOCK: tạo phân phối ngẫu nhiên nhưng HỢP LỆ (tổng=1, 1 lớp trội), tránh số bịa cố định.
        out, probs_all = {}, {}
        for task in ("misinfo", "stance", "sentiment"):
            labels = [ID2LABEL[task][i] for i in sorted(ID2LABEL[task])]
            xs = [random.random() for _ in labels]
            xs[random.randrange(len(labels))] += 1.5
            s = sum(xs)
            pr = {lab: round(x / s, 4) for lab, x in zip(labels, xs)}
            lab = max(pr, key=pr.get)
            out[task] = (lab, pr[lab])
            probs_all[task] = pr
        out["_probs"] = probs_all
        return out

    text = prepare_text(text)
    enc = phobert["tok"](text, return_tensors="pt", truncation=True, max_length=256)
    # LƯU Ý: PhoBERTMultitaskClassifier trả về TUPLE (misinfo, stance, sentiment)
    out_logits = phobert["model"](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = {"misinfo": out_logits[0], "stance": out_logits[1], "sentiment": out_logits[2]}

    res, probs_all = {}, {}
    for task in ("misinfo", "stance", "sentiment"):
        prob = torch.softmax(logits[task], dim=-1)[0]
        idx = int(prob.argmax())
        res[task] = (ID2LABEL[task][idx], round(float(prob[idx]), 4))
        probs_all[task] = {ID2LABEL[task][i]: round(float(prob[i]), 4) for i in range(prob.shape[0])}
    res["_probs"] = probs_all
    return res

@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    cached = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if cached:
        return cached

    p = phobert_infer(req.text)
    labels = {k: v[0] for k, v in p.items() if k != "_probs"}
    row = AnalysisHistory(
        source_text=req.text, source_url=req.source_url, text_hash=h,
        misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
        stance_label=p["stance"][0], stance_score=p["stance"][1],
        sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
        phobert_probs=p["_probs"],
        consistency_flag=compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"]),
        xai_status="idle",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

@app.post("/api/analyze-stream")
def analyze_stream(req: AnalyzeRequest, db: Session = Depends(get_db)):
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    cached = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))

    if cached and cached.xai_status == "done":
        def gen_cached():
            phobert_data = {
                "type": "phobert",
                "id": cached.id,
                "misinfo_label": cached.misinfo_label,
                "misinfo_score": cached.misinfo_score,
                "stance_label": cached.stance_label,
                "stance_score": cached.stance_score,
                "sentiment_label": cached.sentiment_label,
                "sentiment_score": cached.sentiment_score,
                "probs": cached.phobert_probs,          # MỚI
                "consistency_flag": cached.consistency_flag,
                "xai_status": cached.xai_status
            }
            yield _sse(phobert_data)

            explanation = cached.xai_explanation or {}
            reasoning = explanation.get("reasoning", "")
            chunk_size = 15
            for i in range(0, len(reasoning), chunk_size):
                yield _sse({"type": "token", "content": reasoning[i:i+chunk_size]})

            yield _sse({"type": "final", **explanation})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen_cached(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    p = phobert_infer(req.text)
    labels = {k: v[0] for k, v in p.items() if k != "_probs"}
    consistency = compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"])

    if cached:
        row = cached
        row.misinfo_label = p["misinfo"][0]
        row.misinfo_score = p["misinfo"][1]
        row.stance_label = p["stance"][0]
        row.stance_score = p["stance"][1]
        row.sentiment_label = p["sentiment"][0]
        row.sentiment_score = p["sentiment"][1]
        row.phobert_probs = p["_probs"]
        row.consistency_flag = consistency
        row.xai_status = "pending"
    else:
        row = AnalysisHistory(
            source_text=req.text, source_url=req.source_url, text_hash=h,
            misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
            stance_label=p["stance"][0], stance_score=p["stance"][1],
            sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
            phobert_probs=p["_probs"],
            consistency_flag=consistency,
            xai_status="pending",
        )
        db.add(row)
    db.commit()
    db.refresh(row)

    row_id = row.id
    probs_snapshot = p["_probs"]

    def gen_live():
        phobert_data = {
            "type": "phobert",
            "id": row_id,
            "misinfo_label": p["misinfo"][0],
            "misinfo_score": p["misinfo"][1],
            "stance_label": p["stance"][0],
            "stance_score": p["stance"][1],
            "sentiment_label": p["sentiment"][0],
            "sentiment_score": p["sentiment"][1],
            "probs": probs_snapshot,                 # MỚI
            "consistency_flag": consistency,
            "xai_status": "pending"
        }
        yield _sse(phobert_data)

        xai_url = os.environ.get("XAI_SERVICE_URL", "http://xai_service:8001")
        try:
            final_data = None
            body = {"text": req.text, "predicted_labels": labels}
            with requests.post(f"{xai_url}/api/explain-stream", json=body, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        if event.get("type") == "token":
                            yield _sse(event)
                        elif event.get("type") == "final":
                            final_data = event
                        elif event.get("type") == "error":
                            yield _sse(event)

            if final_data:
                gl = final_data.get("gemma_labels", {})
                final_data["disagreement"] = {k: (labels.get(k) != gl.get(k)) for k in labels}

                db_inner = SessionLocal()
                try:
                    db_row = db_inner.get(AnalysisHistory, row_id)
                    if db_row:
                        db_row.xai_explanation = final_data
                        db_row.xai_status = "done"
                        db_inner.commit()
                finally:
                    db_inner.close()

                yield _sse(final_data)
            else:
                db_inner = SessionLocal()
                try:
                    db_row = db_inner.get(AnalysisHistory, row_id)
                    if db_row:
                        db_row.xai_status = "failed"
                        db_inner.commit()
                finally:
                    db_inner.close()
                yield _sse({"type": "error", "message": "Failed to generate explanation (no final block)"})

        except Exception as e:
            db_inner = SessionLocal()
            try:
                db_row = db_inner.get(AnalysisHistory, row_id)
                if db_row:
                    db_row.xai_status = "failed"
                    db_inner.commit()
            finally:
                db_inner.close()
            yield _sse({"type": "error", "message": f"Error calling XAI service: {type(e).__name__}"})

        yield "data: [DONE]\n\n"

    return StreamingResponse(gen_live(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
@app.get("/")
def health_check():
    return {"service": "API Core", "status": "Ready", "model_loaded": "model" in phobert}

@app.post("/api/explain-stream")
def explain_stream(req: AnalyzeRequest, db: Session = Depends(get_db)):
    """On-demand XAI: stream Gemma explanation for a text. Reuses cached PhoBERT labels."""
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    row = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if row is None:
        # Text not yet analyzed — run PhoBERT first
        p = phobert_infer(req.text)
        labels = {k: v[0] for k, v in p.items() if k != "_probs"}
        row = AnalysisHistory(
            source_text=req.text, source_url=req.source_url, text_hash=h,
            misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
            stance_label=p["stance"][0], stance_score=p["stance"][1],
            sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
            phobert_probs=p["_probs"], xai_status="pending",
            consistency_flag=compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"]))
        db.add(row)
        db.commit()
        db.refresh(row)
    else:
        labels = {"misinfo": row.misinfo_label, "stance": row.stance_label, "sentiment": row.sentiment_label}
        row.xai_status = "pending"
        db.commit()
    row_id = row.id

    def gen():
        xai_url = os.environ.get("XAI_SERVICE_URL", "http://xai_service:8001")
        final_data = None
        try:
            with requests.post(f"{xai_url}/api/explain-stream",
                               json={"text": req.text, "predicted_labels": labels},
                               stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"): continue
                    payload = line[5:].strip()
                    if payload == "[DONE]": break
                    try: ev = json.loads(payload)
                    except json.JSONDecodeError: continue
                    if ev.get("type") == "token": yield _sse(ev)
                    elif ev.get("type") == "final": final_data = ev
                    elif ev.get("type") == "error": yield _sse(ev)
            dbx = SessionLocal()
            try:
                rr = dbx.get(AnalysisHistory, row_id)
                if final_data:
                    final_data["disagreement"] = {k: (labels.get(k) != final_data.get("gemma_labels", {}).get(k)) for k in labels}
                    if rr:
                        rr.xai_explanation = final_data
                        rr.xai_status = "done"
                        dbx.commit()
                    yield _sse(final_data)
                else:
                    if rr:
                        rr.xai_status = "failed"
                        dbx.commit()
                    yield _sse({"type": "error", "message": "no final block"})
            finally:
                dbx.close()
        except Exception as e:
            dbx = SessionLocal()
            try:
                rr = dbx.get(AnalysisHistory, row_id)
                if rr:
                    rr.xai_status = "failed"
                    dbx.commit()
            finally:
                dbx.close()
            yield _sse({"type": "error", "message": f"XAI: {type(e).__name__}"})
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/analysis/{aid}", response_model=AnalysisResponse)
def get_analysis(aid: int, db: Session = Depends(get_db)):
    row = db.get(AnalysisHistory, aid)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row

# ============================================================================
# TOKEN ATTRIBUTION — Captum Integrated Gradients (nhãn misinfo)
# ============================================================================
class AttributeRequest(BaseModel):
    text: str

def _resolve_embedding_layer(model):
    """Resolve an embedding module for LayerIntegratedGradients without tying to one architecture."""
    for name in ("encoder", "phobert", "roberta", "bert", "model", "backbone"):
        owner = getattr(model, name, None)
        emb = getattr(owner, "embeddings", None) if owner is not None else None
        if emb is not None:
            return emb, f"{name}.embeddings"

    emb = getattr(model, "embeddings", None)
    if emb is not None:
        return emb, "embeddings"

    for getter, label in (
        (getattr(model, "get_input_embeddings", None), "get_input_embeddings()"),
        (getattr(getattr(model, "encoder", None), "get_input_embeddings", None), "encoder.get_input_embeddings()"),
    ):
        if callable(getter):
            try:
                emb = getter()
                if emb is not None:
                    return emb, label
            except Exception:
                pass

    raise RuntimeError("Embedding layer not found (encoder.embeddings / get_input_embeddings)")

def _captum_saliency(text: str):
    """IG trên nhãn misinfo. Trả (tokens, scores_norm, pred_class).
    Port từ compute_captum_saliency (bản Gradio), thích nghi model dạng tuple."""
    from captum.attr import LayerIntegratedGradients
    model = phobert["model"]
    tok = phobert["tok"]
    proc = prepare_text(text)
    enc = tok(proc, return_tensors="pt", truncation=True, max_length=256)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    def forward_fn(ids, mask):
        out = model(input_ids=ids, attention_mask=mask)   # tuple (misinfo, stance, sentiment)
        return out[0]                                      # logits misinfo

    with torch.no_grad():
        pred_class = int(model(input_ids=input_ids, attention_mask=attention_mask)[0].argmax(dim=1))

    # Lấy embeddings layer
    emb_layer, emb_path = _resolve_embedding_layer(model)
    lig = LayerIntegratedGradients(forward_fn, emb_layer)
    baseline = torch.zeros_like(input_ids) + (tok.pad_token_id or 0)
    attributions = lig.attribute(
        inputs=input_ids, baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=pred_class, n_steps=20,
    )
    attr = attributions.sum(dim=-1).squeeze(0).detach()
    norm = attr.abs().max() + 1e-9
    scores = (attr / norm).tolist()
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    return tokens, scores, pred_class, emb_path

@app.post("/api/attribute")
def attribute(req: AttributeRequest):
    if "model" not in phobert:
        raise HTTPException(status_code=503, detail="PhoBERT chưa nạp (đang chạy MOCK) — IG cần model thật.")
    try:
        tokens, scores, pred_class, emb_path = _captum_saliency(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IG lỗi: {type(e).__name__}: {e}")
    SPECIAL = {"<s>", "</s>", "<pad>", "<unk>", "<mask>", "[CLS]", "[SEP]"}
    items = []
    for t, s in zip(tokens, scores):
        if t in SPECIAL:
            continue
        clean = t.replace("@@", "").replace("▁", " ").replace("Ġ", " ").strip()
        if not clean:
            continue
        items.append({"token": clean, "score": round(float(s), 4)})
    return {
        "pred_class": pred_class,
        "pred_label": ID2LABEL["misinfo"][pred_class],
        "embedding_layer": emb_path,
        "tokens": items,
    }

