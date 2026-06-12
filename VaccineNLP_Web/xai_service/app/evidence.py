import json
from pathlib import Path
from .embedding_index import build_index, query, index_stats

FACT_KB_PATH = str(Path(__file__).resolve().parents[2] / "data" / "fact_kb.json")
_kb = None  # {id: record}

def _load_kb() -> dict:
    global _kb
    if _kb is None:
        with open(FACT_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _kb = {x["id"]: x for x in data}
    return _kb

def ensure_index() -> None:
    """Build index từ fact_kb nếu chưa có / rỗng / lệch số mục."""
    kb = _load_kb()
    try:
        st = index_stats()
    except Exception:
        st = {"count": 0}
    if st.get("count", 0) < len(kb):
        docs = [
            {
                "id": x["id"],
                "text": x["myth"] + " " + x["fact"],
                "source": (x["sources"][0].get("org", "") if x.get("sources") else ""),
                "url": (x["sources"][0].get("url", "") if x.get("sources") else "")
            }
            for x in kb.values()
        ]
        build_index(docs, rebuild=True)

def retrieve_evidence(text: str, k: int = 3, min_score: float = 0.0) -> list[dict]:
    """Trả bằng chứng KB liên quan nhất để ĐỐI CHIẾU (không phải kết luận)."""
    ensure_index()
    kb = _load_kb()
    out = []
    for h in query(text, k):
        rec = kb.get(h["id"])
        if not rec or h["score"] < min_score:
            continue
        out.append({
            "id": h["id"],
            "topic": rec.get("topic", ""),
            "myth": rec["myth"],
            "fact": rec["fact"],
            "sources": rec.get("sources", []),
            "score": h["score"]
        })
    return out
