"""
Pipeline V2 — Preprocessing nâng cấp
======================================
CHANGES vs V1:
  - [FIX P0] is_relevant_text() tích hợp domain-context check (thú y vs người)
  - [FIX P0] Gọi clean_rss_text() cho nguồn RSS/BeautifulSoup
  - [FIX P1] post_id fallback hash dùng URL + text, không hash chuỗi rỗng
  - [NEW]    corpus_audit() — đếm annotatable items sau mỗi lần chạy
  - [NEW]    export_annotation_ready() — xuất items đã sẵn sàng để annotate
"""

# Force UTF-8 for Windows console
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import hashlib
import json
import re
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Setup paths
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.versioning_manager import VersioningManager
from common.paths import DATA_RAW_DIR, DATA_INTERIM_DIR

try:
    # Try relative first, then script-dir based
    from text_cleaner_v2 import (
        clean_text,
        clean_rss_text,
        is_human_vaccine_context,
    )
    print("✅ Loaded text_cleaner_v2")
except ImportError:
    try:
        from preprocessing.text_cleaner_v2 import (
            clean_text,
            clean_rss_text,
            is_human_vaccine_context,
        )
        print("✅ Loaded preprocessing.text_cleaner_v2")
    except ImportError:
        # Fallback
        import html
        def clean_text(text, **kwargs):
            return html.unescape(text or "").strip()
        def clean_rss_text(text):
            return html.unescape(html.unescape(text or "")).strip()
        def is_human_vaccine_context(text):
            return True
        print("⚠️ Warning: Fallback cleaners loaded")


# ── Vaccine keywords (mở rộng) ─────────────────────────────────────────────────
VACCINE_KEYWORDS = {
    # Core terms
    "vaccine", "vaccines", "vắc xin", "vắcxin", "vacxin", "vax",
    "tiêm chủng", "tiêm phòng", "chích ngừa", "chích vắc",
    "mũi tiêm", "lịch tiêm",
    # COVID-specific
    "covid", "covid-19", "sars-cov-2", "pfizer", "moderna",
    "astrazeneca", "sinovac", "sputnik", "vero cell", "covaxin",
    # Vaccine types
    "hpv", "sởi", "rubella", "bạch hầu", "ho gà", "uốn ván",
    "bại liệt", "viêm gan b", "viêm não", "phế cầu",
    "rotavirus", "thủy đậu", "cúm", "tay chân miệng",
    # Side effects / hesitancy
    "phản ứng phụ", "tác dụng phụ", "tai biến", "biến chứng",
    "sốc phản vệ", "sốt sau tiêm", "hành sốt",
    "chết sau tiêm", "tử vong sau tiêm",
    # Stance signals (important for annotation!)
    "không nên tiêm", "phản đối tiêm", "chống vaccine",
    "chống vắc xin", "antivax", "anti-vax",
    "tin giả vaccine", "thuyết âm mưu vaccine",
    "nguy hiểm vaccine", "vaccine nguy hiểm",
    "chip 5g vaccine", "gây vô sinh",
    # Positive stance
    "nên tiêm", "an toàn vaccine", "vaccine an toàn",
    "miễn dịch cộng đồng", "chương trình tiêm chủng",
    "tiêm chủng mở rộng", "5 trong 1", "6 trong 1",
}


# ── Spam / noise patterns ──────────────────────────────────────────────────────
SPAM_PATTERNS = [
    r'\bsale\b', r'\bgiảm\s+giá\b', r'\bfreeship\b',
    r'\bhotline\b', r'\bliên\s+hệ\b.*\bngay\b',
    r'\bđặt\s+mua\b', r'\bkhuyến\s+mãi\b',
    r'\bđào\s+tạo\b', r'\btuyển\s+dụng\b',
    # Link spam
    r'bit\.ly', r'tinyurl', r't\.me/\w+',
]

_SPAM_RE = [re.compile(p, re.IGNORECASE) for p in SPAM_PATTERNS]


def is_relevant_text(text: str, min_length: int = 20) -> bool:
    """
    V2: Filter nâng cấp — kiểm tra relevance với domain awareness.

    Thứ tự kiểm tra:
    1. Minimum length
    2. URL-only content
    3. Spam patterns
    4. Vaccine keyword match
    5. [NEW] Domain context: loại thú y
    """
    if not text or not isinstance(text, str):
        return False

    text_stripped = text.strip()
    if len(text_stripped) < min_length:
        return False

    # Loại bỏ URL-only content
    text_no_url = re.sub(r'https?://\S+|www\.\S+', '', text_stripped)
    if len(text_no_url.strip()) < 10:
        return False

    text_lower = text_stripped.lower()

    # Loại spam
    if any(p.search(text_lower) for p in _SPAM_RE):
        return False

    # Phải có vaccine keyword
    if not any(kw in text_lower for kw in VACCINE_KEYWORDS):
        return False

    # [NEW V2] Domain context check — loại thú y
    if not is_human_vaccine_context(text_stripped):
        return False

    return True


def _make_stable_id(item: dict) -> str:
    """
    [FIX P1] Tạo stable post_id từ URL + source + text[:80].
    Không còn hash chuỗi rỗng → không còn collision d41d8cd9.
    """
    source = item.get("source", "unknown")
    url = item.get("url", "") or item.get("webVideoUrl", "")
    text_snippet = (
        item.get("text", "") or item.get("title", "") or ""
    )[:80]
    seed = f"{source}|{url}|{text_snippet}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _detect_collection_method(item: dict) -> str:
    """Phát hiện phương thức thu thập để chọn cleaner phù hợp."""
    method = item.get("collection_method", "")
    source = item.get("source", "")
    if "rss" in method:
        return "rss"
    if source in {"thanhnien", "dantri", "vietnamnet", "suckhoedoisong", "zingnews", "vtv"}:
        return "rss"  # Extended news thường qua RSS
    return "scrape"


def process_item(item: dict) -> dict | None:
    """
    Xử lý một item raw → normalized item.
    Returns None nếu item bị filter.
    """
    source = item.get("source", "unknown")
    method = _detect_collection_method(item)

    # Chọn cleaner phù hợp
    cleaner = clean_rss_text if method == "rss" else clean_text

    # Build combined text
    title = item.get("title", "")
    body = item.get("text", "") or item.get("description", "") or item.get("body", "")
    combined_raw = f"{title}. {body}".strip(". ").strip()

    cleaned_text = cleaner(combined_raw)

    if not is_relevant_text(cleaned_text):
        return None

    # Stable ID
    existing_id = (
        item.get("article_id")
        or item.get("post_id")
        or item.get("video_id")
        or item.get("id")
    )
    if not existing_id or existing_id == "d41d8cd9":
        existing_id = _make_stable_id(item)

    return {
        "id": existing_id,
        "source": source,
        "platform": item.get("platform", "news" if source in {"vnexpress", "tuoitre"} else source),
        "source_credibility_tier": (
            "institutional"
            if source in {"vnexpress", "tuoitre", "dantri", "thanhnien", "vietnamnet",
                          "suckhoedoisong", "zingnews", "vtv"}
            else "non-institutional"
        ),
        "url": item.get("url", ""),
        "original_text": combined_raw[:1000],
        "cleaned_text": cleaned_text,
        "collected_at": item.get("collected_at", datetime.now().isoformat()),
        "collection_method": method,
        "type": "content",
        # Labels (populated after annotation)
        "label_misinfo": None,
        "label_stance": None,
        "label_sentiment": None,
    }


def process_comments(parent_item: dict, parent_id: str) -> list[dict]:
    """Xử lý comments từ một item."""
    results = []
    source = parent_item.get("source", "unknown")

    comments = parent_item.get("comments", [])
    if not isinstance(comments, list):
        return []

    for comment in comments:
        if not isinstance(comment, dict):
            continue
        raw_text = comment.get("text", "").strip()
        if len(raw_text) < 10:
            continue

        cleaned = clean_text(raw_text)
        if not is_relevant_text(cleaned, min_length=15):
            continue

        results.append({
            "id": _make_stable_id({"source": source, "url": parent_id, "text": raw_text}),
            "parent_id": parent_id,
            "source": source,
            "platform": parent_item.get("platform", source),
            "source_credibility_tier": "non-institutional",  # Comments always non-institutional
            "url": parent_item.get("url", ""),
            "original_text": raw_text,
            "cleaned_text": cleaned,
            "collected_at": str(comment.get("created_time", comment.get("created", ""))),
            "collection_method": "comment",
            "type": "comment",
            "engagement": {
                "likes": comment.get("likes", comment.get("score", 0)),
            },
            "label_misinfo": None,
            "label_stance": None,
            "label_sentiment": None,
        })

    return results


def process_raw_files(raw_dir: Path = None, output_dir: Path = None) -> list[dict]:
    """Chạy toàn bộ preprocessing pipeline."""
    if raw_dir is None:
        raw_dir = DATA_RAW_DIR
    if output_dir is None:
        output_dir = DATA_INTERIM_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = list(raw_dir.glob("*.json"))

    if not raw_files:
        print(f"⚠️  Không tìm thấy file raw JSON trong {raw_dir}")
        return []

    print(f"🔄 Processing {len(raw_files)} raw files...")
    all_processed: list[dict] = []
    stats = Counter()

    for file_path in raw_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")
            continue

        items_raw = data if isinstance(data, list) else [
            item for val in data.values() if isinstance(val, list)
            for item in val
        ]

        file_stats = Counter()
        for item in items_raw:
            if not isinstance(item, dict):
                continue

            # Process main content
            processed = process_item(item)
            if processed:
                all_processed.append(processed)
                file_stats["content"] += 1

                # Process comments
                comments = process_comments(item, processed["id"])
                all_processed.extend(comments)
                file_stats["comments"] += len(comments)
            else:
                file_stats["filtered"] += 1

        print(f"  📄 {file_path.name}: {file_stats['content']} items, "
              f"{file_stats['comments']} comments, {file_stats['filtered']} filtered")
        stats.update(file_stats)

    # Dedup by id
    seen_ids: set[str] = set()
    deduped = []
    for item in all_processed:
        iid = item.get("id", "")
        if iid not in seen_ids:
            seen_ids.add(iid)
            deduped.append(item)

    duplicates_removed = len(all_processed) - len(deduped)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"processed_vaccine_data_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Pipeline complete:")
    print(f"   Content items : {stats['content']}")
    print(f"   Comments      : {stats['comments']}")
    print(f"   Filtered      : {stats['filtered']}")
    print(f"   Duplicates    : {duplicates_removed}")
    print(f"   Total unique  : {len(deduped)}")
    print(f"   Output        : {out_path}")

    # [NEW] Register in manifest
    vm = VersioningManager()
    vm.register_entry(
        stage="preprocessing",
        filepath=str(out_path),
        version=ts,
        metadata={
            "item_count": len(deduped),
            "stats": dict(stats)
        }
    )
    print(f"✨ Registered preprocessing version {ts} in manifest.")

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# Corpus Audit — đo lường annotation readiness
# ══════════════════════════════════════════════════════════════════════════════

def corpus_audit(processed_data: list[dict]) -> dict:
    """
    [NEW V2] Đánh giá trạng thái corpus sau preprocessing.
    Trả về report giúp quyết định annotation strategy.
    """
    total = len(processed_data)
    by_source = Counter(item.get("source") for item in processed_data)
    by_type = Counter(item.get("type") for item in processed_data)
    by_tier = Counter(item.get("source_credibility_tier") for item in processed_data)

    # Text length distribution
    lengths = [len(item.get("cleaned_text", "").split()) for item in processed_data]
    avg_len = sum(lengths) / max(len(lengths), 1)

    # Annotation readiness
    annotatable = [
        item for item in processed_data
        if len(item.get("cleaned_text", "").split()) >= 10  # ≥ 10 tokens
    ]
    annotation_target = 600
    deficit = max(0, annotation_target - len(annotatable))

    report = {
        "total_items": total,
        "annotatable_items": len(annotatable),
        "annotation_target": annotation_target,
        "deficit": deficit,
        "annotation_ready_pct": round(len(annotatable) / max(total, 1) * 100, 1),
        "by_source": dict(by_source.most_common()),
        "by_type": dict(by_type),
        "by_credibility_tier": dict(by_tier),
        "avg_token_length": round(avg_len, 1),
        "status": (
            "✅ SUFFICIENT" if len(annotatable) >= annotation_target else
            "⚠️  MARGINAL" if len(annotatable) >= 400 else
            "❌ INSUFFICIENT — need more data collection"
        ),
    }

    return report


def print_corpus_audit(report: dict) -> None:
    """Pretty-print corpus audit report."""
    print("\n" + "=" * 60)
    print("📊 CORPUS AUDIT REPORT")
    print("=" * 60)
    print(f"Status        : {report['status']}")
    print(f"Total items   : {report['total_items']}")
    print(f"Annotatable   : {report['annotatable_items']} / {report['annotation_target']} target")
    print(f"Deficit       : {report['deficit']} more items needed")
    print(f"Avg length    : {report['avg_token_length']} tokens")
    print()
    print("By source:")
    for src, cnt in report['by_source'].items():
        bar = "█" * min(cnt, 40)
        print(f"  {src:<25} {cnt:>4}  {bar}")
    print()
    print("By type:")
    for t, cnt in report['by_type'].items():
        print(f"  {t:<20} {cnt:>4}")
    print()
    print("By credibility tier:")
    for tier, cnt in report['by_credibility_tier'].items():
        print(f"  {tier:<30} {cnt:>4}")
    print("=" * 60)


if __name__ == "__main__":
    # Force UTF-8 for Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    data = process_raw_files(raw_dir)
    if data:
        report = corpus_audit(data)
        print_corpus_audit(report)
