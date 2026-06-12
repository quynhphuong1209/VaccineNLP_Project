import os
from .evidence import ensure_index
from .embedding_index import query

ANOMALY_TAU = float(os.environ.get("ANOMALY_TAU", "0.82"))

def embedding_anomaly(text: str) -> dict:
    """Kiểm tra xem văn bản đầu vào có lạc đề (anomaly) so với cơ sở tri thức y văn hay không."""
    try:
        ensure_index()
        results = query(text, k=1)
        if not results:
            return {"is_anomalous": True, "max_similarity": 0.0, "anomaly_tau": ANOMALY_TAU}
        
        max_sim = results[0]["score"]
        is_anomalous = max_sim < ANOMALY_TAU
        return {
            "is_anomalous": is_anomalous,
            "max_similarity": max_sim,
            "anomaly_tau": ANOMALY_TAU
        }
    except Exception:
        # Fallback an toàn nếu có lỗi DB/Index
        return {"is_anomalous": True, "max_similarity": 0.0, "anomaly_tau": ANOMALY_TAU}
