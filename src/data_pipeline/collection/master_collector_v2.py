"""
Master Collector v2.1 — VaccineNLP-Thesis
==========================================
Orchestrator CLI thu thập đa nguồn.
CHANGELOG v2.1:
  + Thêm Twitter/X collector (twitter_x_collector.py)
  + Fix Bug #4: pipeline_loop.py import
  + Cải thiện Reddit noise filter
  + Thêm retry logic với exponential backoff
  + Cải thiện progress reporting
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
COLLECTOR_DIR = Path(__file__).parent
sys.path.insert(0, str(COLLECTOR_DIR))

from dedup_store import default_store as dedup_store

try:
    import source_catalog
except Exception:
    source_catalog = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

try:
    from common.paths import DATA_RAW_DIR
    RAW_DATA_DIR = DATA_RAW_DIR
except ImportError:
    RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _import_collector(module_name: str):
    """Import collector module, trả về None nếu fail."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        log.warning(f"Cannot import {module_name}: {e}")
        return None


def _dedup_results(items: list[dict]) -> list[dict]:
    """
    Loại bỏ duplicate dùng DedupStore.
    Ưu tiên item có confidence_score cao hơn.
    """
    unique: list[dict] = []
    seen_ids: set[str] = set()

    # Sort by confidence desc để keep best version
    items_sorted = sorted(
        items,
        key=lambda x: x.get("data_confidence_score", 0),
        reverse=True,
    )

    for item in items_sorted:
        url = item.get("url", "")
        text = item.get("text", "")
        source = item.get("source", "unknown")
        global_id = item.get("global_id", "")

        # Dedup by global_id first (faster)
        if global_id and global_id in seen_ids:
            continue

        if dedup_store.is_new(url=url, text=text):
            dedup_store.mark_seen(url=url, text=text, source=source)
            unique.append(item)
            if global_id:
                seen_ids.add(global_id)

    return unique


def _add_global_id(items: list[dict]) -> list[dict]:
    """Thêm global_id cho mỗi item."""
    for item in items:
        source = item.get("source", "unknown")
        post_id = item.get("post_id", "")
        if not post_id:
            text = item.get("text", "")
            post_id = hashlib.md5(text[:80].encode()).hexdigest()[:8]
            item["post_id"] = post_id
        item["global_id"] = f"{source}_{post_id}"
    return items


def _save_combined(all_items: list[dict]) -> Path:
    """Lưu toàn bộ collected items vào 1 file combined."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DATA_DIR / f"combined_vaccine_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    # Tier A — Core sources
    skip_vnexpress: bool = False,
    skip_tuoitre: bool = False,
    skip_reddit: bool = False,
    skip_rss: bool = True,
    # Tier A Extended news
    skip_extended_news: bool = False,
    # Tier B — Social platforms
    skip_youtube: bool = True,
    # Tier C — Needs credentials/token
    skip_twitter_x: bool = True,
    skip_apify: bool = True,
    # Collection limits
    max_articles_per_source: int = 30,
    max_social_per_platform: int = 100,
    max_twitter_per_query: int = 30,
    # Twitter strategy
    twitter_strategy: str = "auto",  # "auto" | "twscrape" | "nitter" | "snscrape"
    # Custom queries
    twitter_queries: list[str] | None = None,
) -> dict:
    """
    Chạy toàn bộ pipeline thu thập.

    Returns:
        dict — {source: count, ..., __total_unique__: int, __duration_sec__: float}
    """
    start_time = time.time()
    print(f"\n🚀 MASTER COLLECTOR v2.1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    all_collected: list[dict] = []
    summary: dict[str, int] = {}
    source_index = 1

    def _run_collector(
        label: str,
        module_name: str,
        collect_fn_name: str,
        collect_kwargs: dict,
        save: bool = True,
    ) -> list[dict]:
        nonlocal source_index
        print(f"\n📌 [{source_index}] {label}")
        source_index += 1

        module = _import_collector(module_name)
        if not module:
            print(f"  ⏭️  Skipped — module not available")
            summary[label.lower().replace(" ", "_")] = 0
            return []

        try:
            collect_fn = getattr(module, collect_fn_name, None)
            if collect_fn is None:
                print(f"  ❌ Function '{collect_fn_name}' not found in {module_name}")
                summary[label.lower().replace(" ", "_")] = 0
                return []

            items = collect_fn(**collect_kwargs)
            items = _add_global_id(items)

            if save and items:
                save_fn = getattr(module, "save_results", None)
                if save_fn:
                    try:
                        save_fn(items)
                    except Exception as e:
                        log.warning(f"Save failed for {label}: {e}")

            count = len(items)
            key = label.lower().replace(" ", "_").replace("/", "_")
            summary[key] = count
            print(f"  ✅ Collected: {count} items")
            return items

        except Exception as e:
            log.error(f"Error in {label}: {e}", exc_info=True)
            key = label.lower().replace(" ", "_")
            summary[key] = 0
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # TIER A — News sources (ổn định nhất)
    # ──────────────────────────────────────────────────────────────────────────

    if not skip_vnexpress:
        items = _run_collector(
            "VnExpress",
            "vnexpress_collector",
            "collect_all",
            {"max_articles_per_query": max_articles_per_source // 4},
        )
        all_collected.extend(items)

    if not skip_tuoitre:
        items = _run_collector(
            "Tuổi Trẻ",
            "tuoitre_collector",
            "collect_all",
            {"max_articles_per_query": max_articles_per_source // 4},
        )
        all_collected.extend(items)

    if not skip_extended_news:
        items = _run_collector(
            "Extended News (6 sources)",
            "extended_news_collector",
            "collect_all",
            {
                "sources": [
                    "dantri", "thanhnien", "vietnamnet",
                    "suckhoedoisong", "zingnews", "vtv",
                ],
                "max_articles_per_source": max_articles_per_source,
            },
        )
        all_collected.extend(items)

    if not skip_reddit:
        items = _run_collector(
            "Reddit",
            "reddit_collector",
            "collect_all",
            {"max_per_query": max_social_per_platform // 4},
        )
        all_collected.extend(items)

    if not skip_rss:
        items = _run_collector(
            "RSS Feeds",
            "rss_collector",
            "collect_all",
            {},
        )
        all_collected.extend(items)

    # ──────────────────────────────────────────────────────────────────────────
    # TIER B — YouTube (không cần API key)
    # ──────────────────────────────────────────────────────────────────────────

    if not skip_youtube:
        catalog_channels = (
            source_catalog.get_platform_urls(
                "youtube", min_priority=9, fallback_min_priority=8, limit=10
            )
            if source_catalog else []
        )
        items = _run_collector(
            "YouTube (yt-dlp)",
            "youtube_v2_collector",
            "collect_all",
            {
                "channels": catalog_channels or None,
                "max_videos_per_query": 10,
                "max_videos_per_channel": 15,
                "max_comments": 50,
            },
        )
        all_collected.extend(items)

    # ──────────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # TIER C — Twitter/X (BỎ QUA - CHIẾN LƯỢC UGC-FIRST V4.0)
    # ──────────────────────────────────────────────────────────────────────────
    summary["twitter_x"] = 0

    # ──────────────────────────────────────────────────────────────────────────
    # TIER C — Apify Social (Facebook + TikTok)
    # ──────────────────────────────────────────────────────────────────────────

    if not skip_apify:
        try:
            import apify_social_collector_v2 as social_v2

            print(f"\n📌 [{source_index}] Apify Social v2 (Facebook + TikTok)")
            source_index += 1

            if not os.getenv("APIFY_API_TOKEN"):
                try:
                    # Fix: Thêm data_collection vào sys.path để import dashboard.collection_service
                    _DC_ROOT = COLLECTOR_DIR.parent
                    if str(_DC_ROOT) not in sys.path:
                        sys.path.insert(0, str(_DC_ROOT))
                    from dashboard.collection_service import load_project_env
                    load_project_env()
                except Exception:
                    pass

            apify_data = social_v2.collect_all(
                limit=max_social_per_platform * 2,
                platforms=["facebook", "tiktok"],
                use_dedup=True,
            )
            apify_data = _add_global_id(apify_data)
            social_v2.save_results(apify_data)
            all_collected.extend(apify_data)
            summary["apify_social"] = len(apify_data)
            print(f"  ✅ Collected: {len(apify_data)} items")

        except Exception as e:
            log.error(f"Apify v2 error: {e}", exc_info=True)
            summary["apify_social"] = 0
            source_index += 1

    else:
        print(f"\n📌 Apify Social — ⏭️ Skipped (dùng --with-apify để bật)")

    # ──────────────────────────────────────────────────────────────────────────
    # Deduplication + Summary
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"🔄 Deduplication: {len(all_collected)} raw items")
    deduped = _dedup_results(all_collected)
    removed = len(all_collected) - len(deduped)
    print(f"   After dedup: {len(deduped)} unique items")
    print(f"   Removed: {removed} duplicates")

    # Source distribution
    source_dist = Counter(item.get("source", "unknown") for item in deduped)
    print(f"\n📊 Source Distribution:")
    for src, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        pct = count / max(len(deduped), 1) * 100
        bar = "█" * int(pct / 3)
        print(f"   {src:<22} {count:>4} ({pct:>5.1f}%) {bar}")

    if deduped:
        out_path = _save_combined(deduped)
        print(f"\n💾 Combined output: {out_path}")

    duration = time.time() - start_time
    summary["__total_unique__"] = len(deduped)
    summary["__duplicates_removed__"] = removed
    summary["__duration_sec__"] = round(duration, 1)
    summary["__timestamp__"] = datetime.now().isoformat()

    print(f"\n{'='*65}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"   Total unique items: {len(deduped)}")
    print(f"   Duration: {duration:.1f}s")
    print(f"{'='*65}\n")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VaccineNLP Master Collector v2.1 — Multi-source data collection"
    )

    # Platform flags
    parser.add_argument("--with-youtube", action="store_true",
                        help="Bật YouTube (yt-dlp, không cần API key)")
    parser.add_argument("--with-twitter", action="store_true",
                        help="Bật Twitter/X (cần credentials hoặc Nitter)")
    parser.add_argument("--with-apify", action="store_true",
                        help="Bật Facebook + TikTok (cần APIFY_API_TOKEN)")
    parser.add_argument("--all", action="store_true",
                        help="Bật tất cả platforms")
    parser.add_argument("--skip-news", action="store_true",
                        help="Bỏ qua tất cả news sources")

    # Twitter-specific
    parser.add_argument("--twitter-strategy", default="auto",
                        choices=["auto", "twscrape", "nitter", "snscrape"],
                        help="Strategy cho Twitter collection (default: auto)")
    parser.add_argument("--twitter-max", type=int, default=30,
                        help="Số tweet tối đa mỗi query (default: 30)")

    # Limits
    parser.add_argument("--max-articles", type=int, default=30,
                        help="Max articles mỗi news source (default: 30)")
    parser.add_argument("--max-social", type=int, default=100,
                        help="Max items mỗi social platform (default: 100)")

    args = parser.parse_args()

    enable_all = args.all
    summary = run_full_pipeline(
        skip_vnexpress=args.skip_news,
        skip_tuoitre=args.skip_news,
        skip_extended_news=args.skip_news,
        skip_reddit=False,
        skip_rss=True,
        skip_youtube=not (args.with_youtube or enable_all),
        skip_twitter_x=not (args.with_twitter or enable_all),
        skip_apify=not (args.with_apify or enable_all),
        max_articles_per_source=args.max_articles,
        max_social_per_platform=args.max_social,
        max_twitter_per_query=args.twitter_max,
        twitter_strategy=args.twitter_strategy,
    )

    print("\nSummary:")
    for k, v in summary.items():
        if not k.startswith("__"):
            print(f"  {k}: {v}")
