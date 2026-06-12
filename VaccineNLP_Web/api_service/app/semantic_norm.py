import os, json, re, unicodedata
from pathlib import Path

LEXICON_PATH = os.environ.get("SEMANTIC_LEXICON_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "semantic_lexicon.json"))
SAFE_CATEGORIES   = {"viết_tắt", "phương_ngữ"}                       # thay được
DETECT_CATEGORIES = {"uyển_ngữ_chống_vaccine", "thuyết_âm_mưu", "lóng"}  # chỉ gắn cờ

_lex = None
def _load() -> list:
    global _lex
    if _lex is None:
        try:
            with open(LEXICON_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)["entries"]
            _lex = []
            for e in entries:
                _lex.append({
                    "variant": unicodedata.normalize("NFC", e.get("variant", "")),
                    "canonical": unicodedata.normalize("NFC", e.get("canonical", "")),
                    "category": e.get("category", ""),
                    "note": e.get("note", "")
                })
        except Exception:
            _lex = []
    return _lex

def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()

def semantic_normalize(text: str) -> str:
    """Thay biến thể RÕ NGHĨA → canonical (cho phân loại). KHÔNG đụng uyển ngữ mơ hồ."""
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    for e in _load():
        if e.get("category") in SAFE_CATEGORIES and e.get("canonical"):
            out = re.sub(rf"(?<!\w){re.escape(e['variant'])}(?!\w)", e["canonical"], out, flags=re.IGNORECASE)
    return out

def lexicon_hits(text: str) -> list[dict]:
    """Phát hiện (KHÔNG thay) uyển ngữ/âm mưu → tín hiệu coded-language."""
    if not text:
        return []
    t = _nfc_lower(text)
    hits = []
    for e in _load():
        if e.get("category") in DETECT_CATEGORIES and _nfc_lower(e["variant"]) in t:
            hits.append({
                "variant": e["variant"],
                "canonical": e["canonical"],
                "category": e["category"]
            })
    return hits
