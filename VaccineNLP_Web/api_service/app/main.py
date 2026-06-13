import datetime, os, hashlib, random, json, re
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal, AnalysisHistory, get_db

import torch
from transformers import AutoTokenizer
from .model_classes import PhoBERTMultitaskClassifier

def _load_env_defaults():
    """Load local .env files for dev runs without overriding real environment."""
    here = Path(__file__).resolve()
    candidates = [
        Path.home() / ".config" / "vaccinenlp" / ".env",
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


_load_env_defaults()

def _resolve_phobert_path() -> str:
    explicit = os.environ.get("PHOBERT_PATH", "").strip()
    if explicit:
        return explicit

    here = Path(__file__).resolve()
    candidates = [
        Path("/models/phobert_multitask.pt"),                # Docker volume mount
        here.parents[2] / "models" / "phobert_multitask.pt", # VaccineNLP_Web/models
        Path.cwd() / "models" / "phobert_multitask.pt",
        Path.cwd() / "VaccineNLP_Web" / "models" / "phobert_multitask.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def _resolve_xai_service_url() -> str:
    explicit = os.environ.get("XAI_SERVICE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return "http://xai_service:8001" if _running_in_container() else "http://127.0.0.1:8001"


MODEL_PATH = _resolve_phobert_path()
XAI_SERVICE_URL = _resolve_xai_service_url()
phobert = {}

from .preprocess import prepare_text
from .hardening import canonicalize, obfuscation_report
from .semantic_norm import semantic_normalize, lexicon_hits

ID2LABEL = {
    "misinfo":   {0: "Fake", 1: "Real"},
    "stance":    {0: "Favor", 1: "Against", 2: "Neutral"},
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
}

_LABEL_VI = {
    "misinfo": {"Fake": "Có dấu hiệu tin giả", "Real": "Không phát hiện dấu hiệu sai lệch"},
    "stance": {"Favor": "Ủng hộ", "Against": "Phản đối", "Neutral": "Trung lập"},
    "sentiment": {"Positive": "Tích cực", "Negative": "Tiêu cực", "Neutral": "Trung tính"},
}

MISINFO_REVIEW_TAU = float(os.environ.get("MISINFO_REVIEW_TAU", "0.60"))
MISINFO_DISCLAIMER = "Đây là tín hiệu tự động từ AI, không phải kết luận về tính đúng/sai. Vui lòng đối chiếu nguồn chính thống."

def misinfo_display_label(label_internal: str, score: float) -> str:
    if score < MISINFO_REVIEW_TAU:
        return "Cần kiểm chứng"
    return _LABEL_VI["misinfo"].get(label_internal, label_internal)

# Dynamically bind properties to AnalysisHistory for Pydantic serialization
@property
def _display_label(self) -> str:
    return misinfo_display_label(self.misinfo_label, self.misinfo_score)

@property
def _disclaimer(self) -> str:
    return MISINFO_DISCLAIMER

@property
def _obfuscation(self) -> dict:
    return obfuscation_report(self.source_text)

@property
def _coded_language(self) -> list:
    return lexicon_hits(self.source_text)

AnalysisHistory.display_label = _display_label
AnalysisHistory.disclaimer = _disclaimer
AnalysisHistory.obfuscation = _obfuscation
AnalysisHistory.coded_language = _coded_language



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
            r = requests.post(XAI_SERVICE_URL + "/api/explain",
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

class BatchAnalyzeRequest(BaseModel):
    texts: List[str]
    source_url: Optional[str] = None

class CrawlRequest(BaseModel):
    url: str
    max_items: int = 30

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
    display_label: str
    disclaimer: str
    obfuscation: Optional[dict] = None
    coded_language: Optional[List[dict]] = None

class HistoryItem(AnalysisResponse):
    source_text: str
    source_url: Optional[str] = None
    created_at: Optional[datetime.datetime] = None

class BatchAnalyzeResponse(BaseModel):
    rows: List[HistoryItem]

class CrawlResponse(BaseModel):
    source_type: str
    source_info: str
    count: int
    texts: List[str]
    batch_text: str

@torch.no_grad()
def phobert_infer(text: str) -> dict:
    """
    Trả về: {"misinfo": (label, top_score), "stance": (...), "sentiment": (...),
             "_probs": {"misinfo": {"Fake":p,"Real":p}, "stance": {...}, "sentiment": {...}},
             "obfuscation": dict, "coded_language": list, "raw_misinfo_label": str}
    - Tuple (label, score) GIỮ NGUYÊN để không vỡ các call-site cũ.
    - "_probs": phân phối softmax ĐẦY ĐỦ (thật) cho card phân phối ở frontend.
    """
    canon = canonicalize(text)
    sem = semantic_normalize(canon)
    obf = obfuscation_report(text)
    coded = lexicon_hits(text)

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
        if obf["level"] == "high" and out["misinfo"][0] == "Fake":
            out["raw_misinfo_label"] = "Real"
        else:
            out["raw_misinfo_label"] = out["misinfo"][0]
        out["obfuscation"] = obf
        out["coded_language"] = coded
        return out

    # Chạy trên bản semantic_normalize (MAIN result)
    text_canon = prepare_text(sem)
    enc_canon = phobert["tok"](text_canon, return_tensors="pt", truncation=True, max_length=256)
    out_logits_canon = phobert["model"](input_ids=enc_canon["input_ids"], attention_mask=enc_canon["attention_mask"])
    logits_canon = {"misinfo": out_logits_canon[0], "stance": out_logits_canon[1], "sentiment": out_logits_canon[2]}

    res, probs_all = {}, {}
    for task in ("misinfo", "stance", "sentiment"):
        prob = torch.softmax(logits_canon[task], dim=-1)[0]
        idx = int(prob.argmax())
        res[task] = (ID2LABEL[task][idx], round(float(prob[idx]), 4))
        probs_all[task] = {ID2LABEL[task][i]: round(float(prob[i]), 4) for i in range(prob.shape[0])}
    res["_probs"] = probs_all

    # Chạy thêm trên bản raw để so nhãn misinfo (chỉ khi obf["level"] != "none")
    if obf["level"] == "none":
        res["raw_misinfo_label"] = res["misinfo"][0]
    else:
        text_raw = prepare_text(text)
        enc_raw = phobert["tok"](text_raw, return_tensors="pt", truncation=True, max_length=256)
        out_logits_raw = phobert["model"](input_ids=enc_raw["input_ids"], attention_mask=enc_raw["attention_mask"])
        logits_raw_misinfo = out_logits_raw[0]
        prob_raw_misinfo = torch.softmax(logits_raw_misinfo, dim=-1)[0]
        raw_idx = int(prob_raw_misinfo.argmax())
        res["raw_misinfo_label"] = ID2LABEL["misinfo"][raw_idx]
    res["obfuscation"] = obf
    res["coded_language"] = coded

    return res

def _analyze_and_save(text: str, source_url: Optional[str], db: Session) -> AnalysisHistory:
    text_clean = (text or "").strip()
    if not text_clean:
        raise HTTPException(status_code=400, detail="Text is empty")

    h = hashlib.md5(text_clean.encode("utf-8")).hexdigest()
    cached = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if cached and cached.phobert_probs:
        return cached

    p = phobert_infer(text_clean)
    labels = {k: v[0] for k, v in p.items() if k in ("misinfo", "stance", "sentiment")}
    
    obf = p["obfuscation"]
    if p["raw_misinfo_label"] != labels["misinfo"] or obf["level"] == "high":
        consistency = "evasion_suspected"
    else:
        consistency = compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"])

    if cached:
        row = cached
        row.source_url = row.source_url or source_url
        row.misinfo_label = p["misinfo"][0]
        row.misinfo_score = p["misinfo"][1]
        row.stance_label = p["stance"][0]
        row.stance_score = p["stance"][1]
        row.sentiment_label = p["sentiment"][0]
        row.sentiment_score = p["sentiment"][1]
        row.phobert_probs = p["_probs"]
        row.consistency_flag = consistency
    else:
        row = AnalysisHistory(
            source_text=text_clean,
            source_url=source_url,
            text_hash=h,
            misinfo_label=p["misinfo"][0],
            misinfo_score=p["misinfo"][1],
            stance_label=p["stance"][0],
            stance_score=p["stance"][1],
            sentiment_label=p["sentiment"][0],
            sentiment_score=p["sentiment"][1],
            phobert_probs=p["_probs"],
            consistency_flag=consistency,
            xai_status="idle",
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return row

@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    return _analyze_and_save(req.text, req.source_url, db)

@app.post("/api/batch-analyze", response_model=BatchAnalyzeResponse)
def batch_analyze(req: BatchAnalyzeRequest, db: Session = Depends(get_db)):
    texts = []
    seen = set()
    for raw in req.texts:
        text_clean = (raw or "").strip()
        if not text_clean:
            continue
        h = hashlib.md5(text_clean.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        texts.append(text_clean)
    if not texts:
        raise HTTPException(status_code=400, detail="No valid texts")
    if len(texts) > 100:
        texts = texts[:100]

    rows = [_analyze_and_save(text, req.source_url, db) for text in texts]
    return {"rows": rows}

_NEWS_DOMAINS = [
    "vnexpress", "tuoitre", "thanhnien", "dantri", "vietnamnet",
    "suckhoedoisong", "laodong", "tienphong", "znews", "hanoimoi",
    "baochinhphu", "nhandan", "vov", "vtv",
]

_SPAM_KEYWORDS = [
    "inbox", "ib shop", "giá bao nhiêu", "mua ở đâu", "ship", "freeship",
    "liên hệ zalo", "sđt", "tuyển dụng", "tuyển ctv", "sỉ lẻ", "giá rẻ",
    "thanh lý", "chốt đơn", "nhận hàng", "uy tín", "đặt hàng", "zalo sđt",
    "cam kết", "hiệu quả", "giá sỉ", "giá lẻ", "chuyên sỉ", "tuyển đại lý",
]

_VACCINE_KEYWORDS = [
    "vaccine", "vắc xin", "vacxin", "tiêm", "mũi", "bác sĩ", "bệnh", "y tế",
    "thuốc", "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv", "covid",
    "tiêm chủng", "miễn dịch",
]

def _detect_source(url: str) -> str:
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if any(d in url_lower for d in ["facebook.com", "tiktok.com", "threads.net"]):
        return "apify"
    if any(d in url_lower for d in _NEWS_DOMAINS):
        return "news"
    return "news"

def _collect_apify_tokens() -> List[str]:
    keys = [
        "APIFY_TOKEN", "APIFY_API_TOKEN", "APIFY_TOKEN_1", "APIFY_API_TOKEN_1",
        "APIFY_TOKEN_2", "APIFY_API_TOKEN_2", "APIFY_TOKEN_3", "APIFY_API_TOKEN_3",
        "APIFY_TOKEN_4", "APIFY_API_TOKEN_4", "APIFY_TOKEN_5", "APIFY_API_TOKEN_5",
    ]
    tokens = []
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value and value not in tokens:
            tokens.append(value)
    return tokens

def _is_valid_crawled_text(text: str) -> bool:
    value = (text or "").strip()
    if len(value) < 15 or len(value.split()) < 4:
        return False
    low = value.lower()
    if any(keyword in low for keyword in _SPAM_KEYWORDS):
        return False
    return any(keyword in low for keyword in _VACCINE_KEYWORDS)

def _clean_crawled_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _fetch_news_segments(url: str) -> Tuple[List[str], str]:
    try:
        import trafilatura
    except ImportError as exc:
        raise RuntimeError("api_service chưa cài trafilatura để đọc báo điện tử.") from exc

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        downloaded = response.content
    except Exception:
        downloaded = None

    if not downloaded:
        return [], "Không tải được nội dung bài viết."
    content = trafilatura.extract(downloaded, favor_recall=True) or ""
    content = content.strip()
    if not content:
        return [], "Không trích xuất được phần văn bản chính của bài viết."
    return [content], "Báo điện tử"

def _fetch_youtube_segments(url: str, max_items: int) -> Tuple[List[str], str]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("api_service chưa cài yt-dlp để đọc YouTube.") from exc

    opts = {"quiet": True, "skip_download": True, "getcomments": True, "socket_timeout": 60}
    texts: List[str] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    title = _clean_crawled_text(info.get("title", ""))
    desc = _clean_crawled_text((info.get("description") or "")[:700])
    if title:
        texts.append(f"[TIÊU ĐỀ VIDEO] {title}")
    if desc:
        texts.append(f"[MÔ TẢ VIDEO] {desc}")
    for comment in info.get("comments") or []:
        c_text = _clean_crawled_text(comment.get("text", ""))
        if c_text and _is_valid_crawled_text(c_text):
            texts.append(c_text)
        if len(texts) >= max_items + 2:
            break
    return texts[: max_items + 2], "YouTube"

def _apify_actor_config(kind: str, url: str, max_items: int) -> Tuple[str, dict, str]:
    cap = min(max_items, 15)
    low = url.lower()
    if kind == "youtube_apify" or "youtube.com" in low or "youtu.be" in low:
        return (
            "streamers/youtube-comments-scraper",
            {"startUrls": [{"url": url}], "maxComments": cap},
            "YouTube qua Apify",
        )
    if "facebook.com" in low:
        if "/groups/" in low:
            return (
                "apify/facebook-groups-scraper",
                {
                    "startUrls": [{"url": url}],
                    "maxPosts": 3,
                    "maxComments": cap,
                    "maxCommentsPerPost": 5,
                    "maxPostsPerGroup": 3,
                    "resultsLimit": 15,
                },
                "Facebook Group qua Apify",
            )
        if any(marker in low for marker in ["/posts/", "/permalink/", "comment_id=", "/pfbid"]):
            return (
                "apify/facebook-comments-scraper",
                {"startUrls": [{"url": url}], "maxComments": cap, "resultsLimit": cap},
                "Facebook Post/Comments qua Apify",
            )
        return (
            "apify/facebook-posts-scraper",
            {"startUrls": [{"url": url}], "maxPosts": 3, "maxComments": cap, "maxCommentsPerPost": 5},
            "Facebook Page/Profile qua Apify",
        )
    if "tiktok.com" in low:
        return (
            "clockworks/tiktok-comments-scraper",
            {"postURLs": [url], "maxComments": cap},
            "TikTok Comments qua Apify",
        )
    if "threads.net" in low:
        return (
            "thenetaji/threads-scraper",
            {"startUrls": [{"url": url}], "maxItems": cap},
            "Threads qua Apify",
        )
    raise RuntimeError("URL mạng xã hội này chưa được hỗ trợ.")

def _extract_apify_text(item: dict) -> str:
    for key in ("text", "message", "caption", "comment", "fullText", "description", "messageText", "title", "commentText", "body"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_crawled_text(value)
    return ""

def _fetch_apify_segments(url: str, max_items: int, kind: str = "apify") -> Tuple[List[str], str]:
    tokens = _collect_apify_tokens()
    if not tokens:
        return [], "Thiếu APIFY_TOKEN/APIFY_API_TOKEN trong .env hoặc môi trường chạy."
    try:
        from apify_client import ApifyClient
    except ImportError as exc:
        raise RuntimeError("api_service chưa cài apify-client để gọi Apify.") from exc

    actor_id, run_input, source_info = _apify_actor_config(kind, url, max_items)
    last_error = ""
    for token in tokens:
        try:
            client = ApifyClient(token, timeout_secs=180)
            client.user().get()
            run = client.actor(actor_id).call(run_input=run_input, timeout_secs=180)
            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            texts = []
            for item in items:
                text = _extract_apify_text(item)
                if text and _is_valid_crawled_text(text):
                    texts.append(text)
                if len(texts) >= min(max_items, 15):
                    break
            if texts:
                return texts, source_info
            last_error = "Apify trả về dữ liệu nhưng không có nội dung vaccine hợp lệ."
        except Exception as exc:
            last_error = str(exc)
            continue
    return [], f"Không thu thập được dữ liệu từ Apify. {last_error}".strip()

def _crawl_url_as_list(url: str, max_items: int = 30) -> Tuple[List[str], str, str]:
    url = (url or "").strip()
    if not url:
        return [], "Vui lòng nhập URL.", "unknown"
    if not url.startswith(("http://", "https://")):
        return [], "URL không hợp lệ. URL phải bắt đầu bằng http:// hoặc https://.", "unknown"

    max_items = max(1, min(int(max_items or 30), 50))
    kind = _detect_source(url)
    if kind == "news":
        texts, info = _fetch_news_segments(url)
        return texts[:max_items], info, kind
    if kind == "youtube":
        try:
            texts, info = _fetch_youtube_segments(url, max_items)
            if texts:
                return texts, info, kind
        except Exception:
            texts, info = _fetch_apify_segments(url, max_items, kind="youtube_apify")
            return texts, info, "youtube_apify"
        return [], "Không thu thập được nội dung YouTube phù hợp.", kind
    texts, info = _fetch_apify_segments(url, max_items, kind=kind)
    return texts, info, kind

@app.post("/api/crawl-url", response_model=CrawlResponse)
def crawl_url(req: CrawlRequest):
    try:
        texts, info, kind = _crawl_url_as_list(req.url, req.max_items)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Crawler failed: {type(exc).__name__}") from exc
    if not texts:
        raise HTTPException(status_code=400, detail=info)
    batch_text = "\n\n---\n\n".join(texts)
    return {
        "source_type": kind,
        "source_info": info,
        "count": len(texts),
        "texts": texts,
        "batch_text": batch_text,
    }

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
                "probs": cached.phobert_probs,
                "consistency_flag": cached.consistency_flag,
                "xai_status": cached.xai_status,
                "display_label": misinfo_display_label(cached.misinfo_label, cached.misinfo_score),
                "disclaimer": MISINFO_DISCLAIMER,
                "obfuscation": cached.obfuscation,
                "coded_language": cached.coded_language
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
    labels = {k: v[0] for k, v in p.items() if k in ("misinfo", "stance", "sentiment")}
    
    obf = p["obfuscation"]
    if p["raw_misinfo_label"] != labels["misinfo"] or obf["level"] == "high":
        consistency = "evasion_suspected"
    else:
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
            "xai_status": "pending",
            "display_label": misinfo_display_label(p["misinfo"][0], p["misinfo"][1]),
            "disclaimer": MISINFO_DISCLAIMER,
            "obfuscation": p["obfuscation"],
            "coded_language": p["coded_language"]
        }
        yield _sse(phobert_data)

        xai_url = XAI_SERVICE_URL
        try:
            final_data = None
            body = {"text": req.text, "predicted_labels": labels}
            with requests.post(f"{xai_url}/api/explain-stream", json=body, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
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
    return {"service": "API Core", "status": "Ready", "model_loaded": "model" in phobert, "model_path": MODEL_PATH}

@app.get("/api/history", response_model=List[HistoryItem])
def history(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    return db.scalars(
        select(AnalysisHistory)
        .order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc())
        .limit(limit)
    ).all()



def _build_report_markdown(row: AnalysisHistory) -> str:
    created = row.created_at.isoformat() if row.created_at else ""
    explanation = row.xai_explanation or {}
    reasoning = explanation.get("reasoning") or "Chưa sinh giải thích XAI."
    return f"""# Báo cáo phân tích VaccineNLP

**Mã phân tích:** #{row.id}
**Thời gian:** {created}
**Nguồn:** {row.source_url or "Không có"}

## Văn bản

> {row.source_text}

## Kết quả PhoBERT-v2

| Trục | Nhãn | Độ tin cậy |
|---|---|---:|
| Dấu hiệu sai lệch | {_LABEL_VI["misinfo"].get(row.misinfo_label, row.misinfo_label)} | {row.misinfo_score * 100:.1f}% |
| Thái độ với vaccine | {_LABEL_VI["stance"].get(row.stance_label, row.stance_label)} | {row.stance_score * 100:.1f}% |
| Cảm xúc tổng thể | {_LABEL_VI["sentiment"].get(row.sentiment_label, row.sentiment_label)} | {row.sentiment_score * 100:.1f}% |

**Cờ nhất quán:** `{row.consistency_flag}`

## Phân phối softmax

```json
{json.dumps(row.phobert_probs or {}, ensure_ascii=False, indent=2)}
```

## Giải thích XAI

{reasoning}

---

*Báo cáo được tạo tự động bởi VaccineNLP Web.*
"""

@app.get("/api/report/{aid}.md")
def download_report(aid: int, db: Session = Depends(get_db)):
    row = db.get(AnalysisHistory, aid)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    content = _build_report_markdown(row)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="VaccineNLP_Report_{aid}.md"'},
    )

@app.post("/api/explain-stream")
def explain_stream(req: AnalyzeRequest, db: Session = Depends(get_db)):
    """On-demand XAI: stream Gemma explanation for a text. Reuses cached PhoBERT labels."""
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    row = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if row is None:
        # Text not yet analyzed — run PhoBERT first
        p = phobert_infer(req.text)
        labels = {k: v[0] for k, v in p.items() if k in ("misinfo", "stance", "sentiment")}
        obf = p["obfuscation"]
        if p["raw_misinfo_label"] != labels["misinfo"] or obf["level"] == "high":
            consistency = "evasion_suspected"
        else:
            consistency = compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"])
        row = AnalysisHistory(
            source_text=req.text, source_url=req.source_url, text_hash=h,
            misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
            stance_label=p["stance"][0], stance_score=p["stance"][1],
            sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
            phobert_probs=p["_probs"], xai_status="pending",
            consistency_flag=consistency)
        db.add(row)
        db.commit()
        db.refresh(row)
    else:
        labels = {"misinfo": row.misinfo_label, "stance": row.stance_label, "sentiment": row.sentiment_label}
        row.xai_status = "pending"
        db.commit()
    row_id = row.id

    def gen():
        xai_url = XAI_SERVICE_URL
        final_data = None
        try:
            with requests.post(f"{xai_url}/api/explain-stream",
                               json={"text": req.text, "predicted_labels": labels},
                               stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                r.encoding = "utf-8"
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

