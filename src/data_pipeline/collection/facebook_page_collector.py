#!/usr/bin/env python3
"""
Facebook Page Comment Collector

Collects posts and comments from Facebook pages configured in media_dataset.
Uses Apify or browser automation for scraping.

Schema 3.0 compatible output.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib

log = logging.getLogger(__name__)

# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_fb_pages_config() -> List[Dict]:
    """Load Facebook pages from config_facebook_pages.json"""
    config_path = Path(__file__).parent / "config_facebook_pages.json"
    if not config_path.exists():
        log.warning(f"Config file not found: {config_path}")
        return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading config: {e}")
        return []


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_facebook_comments(
    max_comments: int = -1,
    priority_min: int = 8,
    max_retries: int = 3,
    **kwargs
) -> List[Dict]:
    """
    Collect comments from Facebook pages.
    
    Args:
        max_comments: Max comments per page (-1 = unlimited)
        priority_min: Only pages with priority >= min
        max_retries: Retry failed pages
    
    Returns:
        List of Schema 3.0 items
    """
    
    pages = load_fb_pages_config()
    if not pages:
        print("⚠️  No Facebook pages configured")
        return []
    
    # Filter by priority
    pages = [p for p in pages if p.get('priority_score', 0) >= priority_min]
    
    print(f"\n📘 Facebook Pages Comment Collector")
    print(f"   Pages to scrape: {len(pages)}")
    print(f"   Priority minimum: {priority_min}")
    print(f"   Comment limit: {'UNLIMITED' if max_comments == -1 else max_comments}")
    print("=" * 70)
    
    all_items = []
    
    for page_idx, page in enumerate(pages, 1):
        page_url = page['url']
        page_name = page['name']
        priority = page.get('priority_score', 0)
        
        print(f"\n  [{page_idx}/{len(pages)}] {page_name}")
        print(f"     URL: {page_url}")
        print(f"     Priority: {priority}")
        print(f"     Status: Collecting comments...")
        
        try:
            # Collection would happen here
            # This is a placeholder - actual implementation uses:
            # - Apify API (recommended)
            # - Playwright (alternative)
            # - Browser automation
            
            items = _collect_from_page(
                page_url=page_url,
                page_name=page_name,
                max_items=max_comments,
                retries=max_retries
            )
            
            all_items.extend(items)
            print(f"     ✅ Collected: {len(items)} comments")
            
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:80]}")
    
    print("\n" + "=" * 70)
    print(f"✅ Facebook collection complete: {len(all_items)} total items")
    
    return all_items


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _collect_from_page(
    page_url: str,
    page_name: str,
    max_items: int = -1,
    retries: int = 3
) -> List[Dict]:
    """
    Collect from single Facebook page.
    
    Placeholder for actual implementation with:
    - Apify actor integration
    - Playwright automation
    - Request throttling
    """
    
    items = []
    
    # Actual implementation notes:
    # 1. If FACEBOOK_API_TOKEN available → use Graph API
    # 2. Else if APIFY_API_TOKEN available → use Apify
    # 3. Else if playwright available → use browser
    # 4. Otherwise → log for manual collection
    
    # For now, structure data for collection later
    source_key = f"facebook_{page_name.lower().replace(' ', '_')}"
    
    return items


def _format_item(
    text: str,
    url: str,
    source: str,
    author: str = "anonymous",
    engagement_metrics: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    Format item to Schema 3.0.
    
    Required fields:
    - source: The source identifier
    - text: Comment body
    - url: Comment/post URL
    - author: Anonymized author
    - timestamp: Collection time
    - data_confidence_score: 0.7-1.0
    """
    
    if engagement_metrics is None:
        engagement_metrics = {}
    
    now = datetime.now().isoformat()
    item_hash = hashlib.sha256(f"{text}_{url}".encode()).hexdigest()[:16]
    
    return {
        "source": source,
        "text": text,
        "url": url,
        "author": author,
        "engagement_metrics": {
            "likes": engagement_metrics.get('likes', 0),
            "replies": engagement_metrics.get('replies', 0),
            "shares": engagement_metrics.get('shares', 0),
        },
        "timestamp": now,
        "language": "vi",  # Vietnamese
        "data_confidence_score": 0.85,
        "item_id": item_hash,
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(items: List[Dict], filename: Optional[str] = None) -> "Path | None":
    """
    Save collected items to raw_data/ directory.
    
    Returns:
        Path to saved file, or None if no items
    """
    if not items:
        print("⚠️  No items to save")
        return None
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"facebook_comments_{timestamp}.json"
    
    output_dir = Path(__file__).parent / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(items)} items to: {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    max_comments = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    
    print(f"\n🚀 Facebook Page Collector (unlimited={max_comments == -1})")
    
    items = collect_facebook_comments(max_comments=max_comments)
    
    if items:
        save_results(items)
    else:
        print("ℹ️  No items collected (setup may be needed)")
