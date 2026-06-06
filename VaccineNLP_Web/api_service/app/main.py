import os, hashlib, random
from typing import Optional
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal, AnalysisHistory, get_db

import torch
from transformers import AutoTokenizer
from .model_classes import PhoBERTMultitaskClassifier

MODEL_PATH = os.environ.get("PHOBERT_PATH", "/models/phobert_multitask.pt")
phobert = {}

ID2LABEL = {
    "misinfo":   {0: "Real", 1: "Fake"},
    "stance":    {0: "Favor", 1: "Against", 2: "Neutral"},
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
}

# Tam giác: cờ tổ hợp bất thường (STUB)
IMPLAUSIBLE = {("Fake", "Favor", "Positive")} 
def compute_consistency(misinfo: str, stance: str, sentiment: str) -> str:
    return "unusual" if (misinfo, stance, sentiment) in IMPLAUSIBLE else "plausible"

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    phobert["tok"] = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
    try:
        # Đảm bảo tham số num_classes khớp với file checkpoint
        model = PhoBERTMultitaskClassifier(num_misinfo=2, num_stance=3, num_sentiment=3)
        state = torch.load(MODEL_PATH, map_location="cpu")
        # Tuỳ thuộc vào cách lưu lúc train, nếu có key 'model_state_dict' thì dùng dòng dưới:
        # state = state["model_state_dict"] 
        new_state = {}
        for k, v in state.items():
            k_new = k.replace("heads.misinfo.", "head_misinfo.")\
                     .replace("heads.stance.", "head_stance.")\
                     .replace("heads.sentiment.", "head_sentiment.")
            new_state[k_new] = v
        model.load_state_dict(new_state)
        model.eval()
        phobert["model"] = model
        print(f"✅ [api_service] REAL PhoBERT loaded từ {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"⚠️ [api_service] KHÔNG nạp được PhoBERT ({type(e).__name__}: {e}) -> CHUYỂN SANG CHẠY MOCK", flush=True)
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
    consistency_flag: str
    xai_status: str
    xai_explanation: Optional[dict] = None

@torch.no_grad()
def phobert_infer(text: str) -> dict:
    if "model" not in phobert:
        import random
        return {
            "misinfo":   (random.choice(["Fake", "Real"]), 0.99),
            "stance":    (random.choice(["Favor", "Against", "Neutral"]), 0.99),
            "sentiment": (random.choice(["Positive", "Negative", "Neutral"]), 0.99),
        }
    
    enc = phobert["tok"](text, return_tensors="pt", truncation=True, max_length=256)
    # LƯU Ý QUAN TRỌNG: PhoBERTMultitaskClassifier trả về TUPLE (misinfo, stance, sentiment)
    out = phobert["model"](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = {"misinfo": out[0], "stance": out[1], "sentiment": out[2]}
    
    res = {}
    for task in ("misinfo", "stance", "sentiment"):
        prob = torch.softmax(logits[task], dim=-1)[0]
        idx = int(prob.argmax())
        res[task] = (ID2LABEL[task][idx], round(float(prob[idx]), 4))
    return res

@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    cached = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if cached:
        return cached

    p = phobert_infer(req.text)
    labels = {k: v[0] for k, v in p.items()}
    row = AnalysisHistory(
        source_text=req.text, source_url=req.source_url, text_hash=h,
        misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
        stance_label=p["stance"][0], stance_score=p["stance"][1],
        sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
        consistency_flag=compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"]),
        xai_status="pending",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    
    bg.add_task(call_gemma_explain, row.id, req.text, labels)
    return row

@app.get("/")
def health_check():
    return {"service": "API Core", "status": "Ready", "model_loaded": "model" in phobert}

@app.get("/api/analysis/{aid}", response_model=AnalysisResponse)
def get_analysis(aid: int, db: Session = Depends(get_db)):
    row = db.get(AnalysisHistory, aid)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row
