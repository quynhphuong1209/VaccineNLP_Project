"""
VaccineNLP Data Fetchers Module (app_gradio copy with updates)
=============================================================
URL content fetching utilities with structured Facebook comments scraper and
APIFY feed optimizations.
"""

import logging
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Get Apify tokens from environment/secrets
APIFY_TOKENS = []

# Check with API (e.g. APIFY_API_TOKEN, APIFY_API_TOKEN_2...)
for i in range(1, 6):
    suffix = f"_{i}" if i > 1 else ""
    token = os.environ.get(f"APIFY_API_TOKEN{suffix}") or os.environ.get("APIFY_API_TOKEN")
    if token and token.strip() and token.strip() not in APIFY_TOKENS:
        APIFY_TOKENS.append(token.strip())

# Check without API (e.g. APIFY_TOKEN, APIFY_TOKEN_2...)
for i in range(1, 6):
    suffix = f"_{i}" if i > 1 else ""
    token = os.environ.get(f"APIFY_TOKEN{suffix}") or os.environ.get("APIFY_TOKEN")
    if token and token.strip() and token.strip() not in APIFY_TOKENS:
        APIFY_TOKENS.append(token.strip())


# ─────────────────────────────────────────────────────────────
# URL DETECTION
# ─────────────────────────────────────────────────────────────
def detect_source(url: str) -> str:
    url_lower = url.lower()
    news_domains = [
        "vnexpress", "tuoitre", "thanhnien", "dantri", "vietnamnet",
        "suckhoedoisong", "laodong", "tienphong", "znews", "hanoimoi",
        "baochinhphu", "nhandan", "vov", "vtv"
    ]
    if any(d in url_lower for d in news_domains):
        return "news"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if any(d in url_lower for d in ["facebook.com", "tiktok.com", "threads.net"]):
        return "apify"
    return "news"


# ─────────────────────────────────────────────────────────────
# NEWS FETCHING
# ─────────────────────────────────────────────────────────────
def fetch_news(url: str) -> str:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded, favor_recall=True)
            return content or ""
        return ""
    except Exception as e:
        return f"❌ News fetch error: {e}"


# ─────────────────────────────────────────────────────────────
# YOUTUBE FETCHING
# ─────────────────────────────────────────────────────────────
def fetch_youtube(url: str, max_comments: int = 30) -> str:
    try:
        import yt_dlp
        opts = {"quiet": True, "skip_download": True, "getcomments": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "")
            desc = info.get("description", "")[:1000]
            comments = (info.get("comments") or [])[:max_comments]
            cmt_text = "\n".join([f"- {c.get('text', '')}" for c in comments])
            return f"TIÊU ĐỀ: {title}\n\nMÔ TẢ: {desc}\n\nBÌNH LUẬN ({len(comments)}):\n{cmt_text}"
    except Exception as e:
        return f"❌ YouTube fetch error: {e}"


# ─────────────────────────────────────────────────────────────
# APIFY SCRAPING (SOCIAL MEDIA)
# ─────────────────────────────────────────────────────────────
def fetch_apify(url: str) -> str:
    if not APIFY_TOKENS:
        return "❌ APIFY_TOKEN not configured in environment"
    try:
        from apify_client import ApifyClient
        for token in APIFY_TOKENS:
            try:
                client = ApifyClient(token)
                client.user().get()
                
                if "facebook.com" in url.lower():
                    actor_id = "apify/facebook-posts-scraper"
                elif "tiktok.com" in url.lower():
                    actor_id = "clockworks/tiktok-scraper"
                elif "threads.net" in url.lower():
                    actor_id = "igview-owner/threads-search-scraper"
                else:
                    return "❌ URL not supported"
                
                run = client.actor(actor_id).call(run_input={"startUrls": [{"url": url}]})
                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                if items:
                    texts = [
                        item.get("text", "") or item.get("caption", "") 
                        for item in items[:5]
                    ]
                    return "\n\n".join(t for t in texts if t)
                return "❌ No content returned"
            except Exception:
                continue
        return "❌ All Apify tokens failed"
    except Exception as e:
        return f"❌ Apify error: {e}"


# ─────────────────────────────────────────────────────────────
# COMMENT VALIDATION
# ─────────────────────────────────────────────────────────────
def is_valid_comment(txt: str) -> bool:
    if not txt:
        return False
    txt_strip = txt.strip()
    
    # Filter 1: Minimum length
    if len(txt_strip) < 15 or len(txt_strip.split()) < 4:
        return False
        
    # Filter 2: Spam keywords
    spam_keywords = [
        "inbox", "ib shop", "giá bao nhiêu", "mua ở đâu", "ship", "freeship", 
        "liên hệ zalo", "sđt", "tuyển dụng", "tuyển ctv", "sỉ lẻ", "giá rẻ", 
        "thanh lý", "chốt đơn", "nhận hàng", "uy tín", "đặt hàng", "zalo sđt",
        "cam kết", "hiệu quả", "giá sỉ", "giá lẻ", "chuyên sỉ", "tuyển đại lý"
    ]
    txt_lower = txt_strip.lower()
    if any(kw in txt_lower for kw in spam_keywords):
        return False
        
    # Filter 3: Off-topic
    try:
        from src.preprocessing.text_cleaner_v2 import is_human_vaccine_context
        if not is_human_vaccine_context(txt_strip):
            return False
    except Exception:
        core_keywords = [
            "vaccine", "vắc xin", "tiêm", "mũi", "bác sĩ", "bệnh", "y tế", 
            "thuốc", "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv"
        ]
        if not any(kw in txt_lower for kw in core_keywords):
            return False
            
    return True


# ─────────────────────────────────────────────────────────────
# PATCH 2 — FETCH RAW APIFY ITEMS (NEW)
# ─────────────────────────────────────────────────────────────
def fetch_apify_raw(url: str, max_comments: int = 30):
    """Gọi Apify cho FB và trả về (raw_items, source_display, error).
    KHÔNG làm phẳng — để thread_parser.parse_apify() dựng cấu trúc."""
    if not APIFY_TOKENS:
        return [], "", "❌ APIFY_TOKEN chưa cấu hình"
    from apify_client import ApifyClient
    apify_max_cmt = min(max_comments, 100)   # giữ trần để tiết kiệm credit
    u = url.lower()
    POST_SIG = ("/permalink/", "/posts/", "/pfbid", "story_fbid=", "fbid=",
                "comment_id=", "/photo")
    if any(s in u for s in POST_SIG):
        actor_id = "apify/facebook-comments-scraper"
        run_input = {"startUrls": [{"url": url}], "resultsLimit": apify_max_cmt}
        src = "💬 FB Post + Comments (Tier 3)"
    elif "/groups/" in u:
        actor_id = "apify/facebook-groups-scraper"
        run_input = {"startUrls": [{"url": url}], "resultsLimit": 5,
                     "maxCommentsPerPost": apify_max_cmt}
        src = "👥 FB Group Feed (Tier 3)"
    else:
        actor_id = "apify/facebook-posts-scraper"
        run_input = {"startUrls": [{"url": url}], "resultsLimit": 3,
                     "maxCommentsPerPost": apify_max_cmt}
        src = "📱 FB Page Feed (Tier 3)"

    last_err = ""
    for token in APIFY_TOKENS:
        try:
            client = ApifyClient(token)
            client.user().get()
            logger.info(f"🚀 Apify {actor_id} ⟵ {url}")
            run = client.actor(actor_id).call(run_input=run_input)
            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            if items:
                return items, src, ""
        except Exception as ex:
            last_err = str(ex)
            logger.warning(f"Apify token failed: {ex}")
            continue
    return [], src, f"❌ Apify không trả dữ liệu ({last_err})"


# ─────────────────────────────────────────────────────────────
# UNIFIED LIST FETCHER WITH PATCH 1
# ─────────────────────────────────────────────────────────────
def fetch_url_as_list(url: str, max_comments: int = 30) -> Tuple[List[str], str]:
    if not url or not url.strip():
        return [], "❌ Vui lòng nhập URL"
    
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return [], "❌ URL không hợp lệ"

    kind = detect_source(url)
    texts = []
    info = ""

    # Route 1: News websites
    if kind == "news":
        content = fetch_news(url)
        if content and not content.startswith("❌"):
            texts = [content]
            info = "📰 Báo điện tử (Tier 1, ~2s)"
        else:
            return [], content

    # Route 2: YouTube
    elif kind == "youtube":
        try:
            import yt_dlp
            opts = {"quiet": True, "skip_download": True, "getcomments": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                title = info_dict.get("title", "")
                desc = info_dict.get("description", "") or ""
                comments = info_dict.get("comments") or []
                
                if title:
                    texts.append(f"[TIÊU ĐỀ VIDEO] {title}")
                if desc.strip():
                    texts.append(f"[MÔ TẢ VIDEO] {desc.strip()[:500]}")
                
                for c in comments:
                    c_text = c.get("text", "").strip()
                    if c_text and is_valid_comment(c_text):
                        texts.append(c_text)
                    if len(texts) >= max_comments + 2:
                        break
                info = "🎬 YouTube (Tier 2, ~10s)"
        except Exception as e:
            logger.warning(f"yt_dlp failed, trying Apify fallback: {e}")
            kind = "youtube_apify"

    # Route 3: Apify (Facebook, TikTok, Threads, YouTube fallback)
    if kind in ("apify", "youtube_apify"):
        if not APIFY_TOKENS:
            return [], "❌ APIFY_TOKEN chưa được setup trong HF Spaces Secrets"
        
        last_err = ""
        actor_id = ""
        
        try:
            from apify_client import ApifyClient
            
            for token in APIFY_TOKENS:
                try:
                    client = ApifyClient(token)
                    client.user().get()
                    
                    # Limit Apify calls to avoid token burn
                    apify_max_cmt = min(max_comments, 100)

                    # Smart actor selection based on URL
                    if kind == "youtube_apify" or "youtube.com" in url.lower() or "youtu.be" in url.lower():
                        actor_id = "streamers/youtube-comments-scraper"
                        run_input = {"startUrls": [{"url": url}], "maxComments": apify_max_cmt}
                        source_display = "🎬 YouTube (Apify Fallback Scraper - Tier 3)"
                    
                    # PATCH 1: Sửa khối router FB actor
                    elif "facebook.com" in url.lower():
                        u = url.lower()
                        # Bài CỤ THỂ (kể cả nằm trong group) → comments-scraper để lấy ĐÚNG bài + comments
                        POST_SIG = ("/permalink/", "/posts/", "/pfbid", "story_fbid=", "fbid=",
                                    "comment_id=", "/photo")
                        is_specific_post = any(s in u for s in POST_SIG)

                        if is_specific_post:
                            actor_id = "apify/facebook-comments-scraper"
                            run_input = {"startUrls": [{"url": url}], "resultsLimit": apify_max_cmt}
                            source_display = "💬 FB Post + Comments (Tier 3)"
                        elif "/groups/" in u:                       # CHỈ feed group (không có bài cụ thể)
                            actor_id = "apify/facebook-groups-scraper"
                            run_input = {"startUrls": [{"url": url}], "resultsLimit": 5,
                                         "maxCommentsPerPost": apify_max_cmt}   # ⚠️ KHÔNG dùng maxPosts (đã deprecated)
                            source_display = "👥 FB Group Feed (Tier 3)"
                        else:                                       # feed page/profile
                            actor_id = "apify/facebook-posts-scraper"
                            run_input = {"startUrls": [{"url": url}], "resultsLimit": 3,
                                         "maxCommentsPerPost": apify_max_cmt}
                            source_display = "📱 FB Page Feed (Tier 3)"
                    
                    elif "tiktok.com" in url.lower():
                        actor_id = "clockworks/tiktok-comments-scraper"
                        run_input = {
                            "postURLs": [url],
                            "maxComments": apify_max_cmt
                        }
                        source_display = "🎵 TikTok Comments (Apify Scraper - Tier 3)"
                    
                    elif "threads.net" in url.lower():
                        actor_id = "thenetaji/threads-scraper"
                        run_input = {
                            "startUrls": [{"url": url}],
                            "maxItems": apify_max_cmt
                        }
                        source_display = "🧵 Threads (Apify Scraper - Tier 3)"
                    
                    else:
                        return [], "❌ URL không được hỗ trợ"
                    
                    logger.info(f"🚀 Running Apify Actor: {actor_id} for URL: {url}")
                    run = client.actor(actor_id).call(run_input=run_input)
                    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                    
                    if items:
                        for item in items:
                            txt = (
                                item.get("text", "") or 
                                item.get("message", "") or 
                                item.get("caption", "") or 
                                item.get("comment", "") or 
                                item.get("fullText", "") or 
                                item.get("description", "") or 
                                item.get("messageText", "") or
                                item.get("title", "") or
                                item.get("commentText", "") or
                                item.get("body", "")
                            )
                            if txt and txt.strip():
                                clean_txt = txt.strip()
                                if is_valid_comment(clean_txt):
                                    texts.append(clean_txt)
                            if len(texts) >= apify_max_cmt:
                                break
                        
                        if texts:
                            info = source_display
                            break
                
                except Exception as ex:
                    last_err = str(ex)
                    logger.warning(f"Apify token failed for actor {actor_id}: {ex}")
                    continue
            
            if not texts:
                err_detail = f" (Chi tiết lỗi: {last_err})" if last_err else ""
                return [], f"❌ Không thu thập được bài viết/bình luận nào từ Apify{err_detail}"
        
        except Exception as e:
            return [], f"❌ Apify error: {e}"

    return texts, info


def fetch_url(url: str, max_comments: int = 30) -> Tuple[str, str]:
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        return "", info
    return "\n\n".join(texts), f"**Nguồn:** {info}"
