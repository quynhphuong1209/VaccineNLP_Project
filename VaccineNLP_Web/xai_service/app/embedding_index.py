import os
import json
import sqlite3
from pathlib import Path
from typing import Literal
import sqlite_vec
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
INDEX_DB_PATH   = os.environ.get("INDEX_DB_PATH", str(Path(__file__).resolve().parents[2] / "data" / "index.db"))
DEVICE          = os.environ.get("EMBEDDING_DEVICE", "cpu")
_IS_E5 = "e5" in EMBEDDING_MODEL.lower()
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    return _model

def embed(texts: list[str], kind: Literal["query", "passage"]) -> list[list[float]]:
    if _IS_E5:
        # e5 models require "query: " or "passage: " prefix
        texts = [f"{kind}: {t}" for t in texts]
    vecs = _get_model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]

def _connect():
    db = sqlite3.connect(INDEX_DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db

def _dim() -> int:
    return len(embed(["x"], "passage")[0])

def build_index(docs: list[dict], rebuild: bool = False) -> int:
    Path(INDEX_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    if rebuild and Path(INDEX_DB_PATH).exists():
        Path(INDEX_DB_PATH).unlink()
    
    dim = _dim()
    db = _connect()
    cur = db.cursor()
    cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(embedding float[{dim}])")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS docs("
        "rowid INTEGER PRIMARY KEY, "
        "id TEXT UNIQUE, "
        "text TEXT, "
        "source TEXT, "
        "url TEXT, "
        "meta TEXT, "
        "ts TEXT)"
    )
    cur.execute("CREATE TABLE IF NOT EXISTS meta(model TEXT, dim INT, created_at TEXT)")
    cur.execute("DELETE FROM meta")
    cur.execute("INSERT INTO meta VALUES (?, ?, datetime('now'))", (EMBEDDING_MODEL, dim))
    db.commit()
    
    n = upsert(docs, _db=db)
    db.close()
    return n

def upsert(docs: list[dict], _db=None) -> int:
    db = _db or _connect()
    cur = db.cursor()
    vecs = embed([d["text"] for d in docs], "passage")
    for d, v in zip(docs, vecs):
        cur.execute(
            "INSERT OR REPLACE INTO docs(id, text, source, url, meta, ts) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (
                d["id"],
                d["text"],
                d.get("source", ""),
                d.get("url", ""),
                json.dumps(d.get("meta", {}), ensure_ascii=False)
            )
        )
        rid = cur.execute("SELECT rowid FROM docs WHERE id=?", (d["id"],)).fetchone()[0]
        cur.execute(
            "INSERT OR REPLACE INTO vec_docs(rowid, embedding) VALUES (?, ?)",
            (rid, sqlite_vec.serialize_float32(v))
        )
    if _db is None:
        db.commit()
        db.close()
    else:
        db.commit()
    return len(docs)

def query(text: str, k: int = 5) -> list[dict]:
    db = _connect()
    q = embed([text], "query")[0]
    
    # Query using vec_distance_cosine or standard vec0 MATCH distance depending on availability
    # In sqlite-vec, vec_distance_cosine is standard since v0.1.0+
    rows = db.execute(
        "SELECT d.id, d.text, d.source, d.url, vec_distance_cosine(v.embedding, ?) AS dist "
        "FROM vec_docs v "
        "JOIN docs d ON d.rowid = v.rowid "
        "ORDER BY dist "
        "LIMIT ?",
        (sqlite_vec.serialize_float32(q), k)
    ).fetchall()
    
    res = [
        {
            "id": r[0],
            "text": r[1],
            "source": r[2],
            "url": r[3],
            "score": round(1 - r[4], 4)
        }
        for r in rows
    ]
    db.close()
    return res

def index_stats() -> dict:
    db = _connect()
    cnt = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    m = db.execute("SELECT model, dim FROM meta").fetchone()
    db.close()
    return {
        "count": cnt,
        "dim": m[1] if m else None,
        "model": m[0] if m else None,
        "db_path": INDEX_DB_PATH
    }
