"""
VaccineNLP Data Fetchers Module
===============================
URL content fetching utilities for news, YouTube, Facebook, TikTok, and other social media.

Extracted from: app_gradio/app.py
"""

import logging
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Get Apify tokens from environment/secrets
APIFY_TOKENS = []
for i in range(1, 4):
    token = os.environ.get(f"APIFY_TOKEN_{i}") or os.environ.get("APIFY_TOKEN")
    if token:
        APIFY_TOKENS.append(token.strip())


# ─────────────────────────────────────────────────────────────
# URL DETECTION
# ─────────────────────────────────────────────────────────────
def detect_source(url: str) -> str:
    """Detect URL source type to determine appropriate fetcher.
    
    Args:
        url: URL to analyze
        
    Returns:
        Source type: "news", "youtube", or "apify"
    """
    url_lower = url.lower()
    
    # Vietnamese news domains
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
    """Fetch article content from news website using Trafilatura.
    
    Args:
        url: News article URL
        
    Returns:
        Extracted article text, or error message if fetch fails
    """
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
    """Fetch YouTube video title, description, and comments.
    
    Args:
        url: YouTube video URL
        max_comments: Maximum number of comments to fetch
        
    Returns:
        Formatted string with video metadata and comments
    """
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
    """Fetch content from social media (Facebook, TikTok, Threads) via Apify actors.
    
    Args:
        url: Social media URL
        
    Returns:
        Scraped content text, or error message if fetch fails
    """
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
    """Smart filter for social media comments/posts.
    
    Filters out:
    - Too short messages ("chấm", "hóng", emoji only)
    - Spam (sales, recruitment, ads)
    - Off-topic content (not vaccine-related)
    
    Args:
        txt: Comment text to validate
        
    Returns:
        True if comment is valid, False if should be filtered
    """
    if not txt:
        return False
    txt_strip = txt.strip()
    
    # Filter 1: Minimum length (remove "lol", "icon only", etc)
    if len(txt_strip) < 15 or len(txt_strip.split()) < 4:
        return False
        
    # Filter 2: Spam keywords (sales, recruitment, ads)
    spam_keywords = [
        "inbox", "ib shop", "giá bao nhiêu", "mua ở đâu", "ship", "freeship", 
        "liên hệ zalo", "sđt", "tuyển dụng", "tuyển ctv", "sỉ lẻ", "giá rẻ", 
        "thanh lý", "chốt đơn", "nhận hàng", "uy tín", "đặt hàng", "zalo sđt",
        "cam kết", "hiệu quả", "giá sỉ", "giá lẻ", "chuyên sỉ", "tuyển đại lý"
    ]
    txt_lower = txt_strip.lower()
    if any(kw in txt_lower for kw in spam_keywords):
        return False
        
    # Filter 3: Off-topic (must be vaccine-related)
    try:
        from src.preprocessing.text_cleaner_v2 import is_human_vaccine_context
        if not is_human_vaccine_context(txt_strip):
            return False
    except Exception:
        # Fallback: keyword filter if text_cleaner_v2 unavailable
        core_keywords = [
            "vaccine", "vắc xin", "tiêm", "mũi", "bác sĩ", "bệnh", "y tế", 
            "thuốc", "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv"
        ]
        if not any(kw in txt_lower for kw in core_keywords):
            return False
            
    return True


# ─────────────────────────────────────────────────────────────
# UNIFIED FETCHER
# ─────────────────────────────────────────────────────────────
def fetch_url_as_list(url: str, max_comments: int = 30) -> Tuple[List[str], str]:
    """Fetch content from URL and return as list of text segments (posts/comments).
    
    Supports:
    - Vietnamese news websites (Trafilatura)
    - YouTube (yt_dlp)
    - Facebook, TikTok, Threads (Apify actors)
    
    Args:
        url: URL to fetch from
        max_comments: Maximum number of comments/posts to retrieve
        
    Returns:
        Tuple of (list of text segments, source info string)
    """
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
                    if len(texts) >= max_comments + 2:  # +2 for title & desc
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
                    apify_max_cmt = min(max_comments, 15)

                    # Smart actor selection based on URL
                    if kind == "youtube_apify" or "youtube.com" in url.lower() or "youtu.be" in url.lower():
                        actor_id = "streamers/youtube-comments-scraper"
                        run_input = {"startUrls": [{"url": url}], "maxComments": apify_max_cmt}
                        source_display = "🎬 YouTube (Apify Fallback Scraper - Tier 3)"
                    
                    elif "facebook.com" in url.lower():
                        if "/groups/" in url.lower():
                            actor_id = "apify/facebook-groups-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxPosts": 3,
                                "maxComments": apify_max_cmt,
                                "maxCommentsPerPost": 5,
                                "maxPostsPerGroup": 3,
                                "resultsLimit": 15
                            }
                            source_display = "👥 Facebook Group (Apify Scraper - Tier 3)"
                        elif "/posts/" in url.lower() or "/permalink/" in url.lower() or "comment_id=" in url.lower() or "/pfbid" in url.lower():
                            actor_id = "apify/facebook-comments-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxComments": apify_max_cmt,
                                "resultsLimit": apify_max_cmt
                            }
                            source_display = "💬 Facebook Post/Comments (Apify Scraper - Tier 3)"
                        else:
                            actor_id = "apify/facebook-posts-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxPosts": 3,
                                "maxComments": apify_max_cmt,
                                "maxCommentsPerPost": 5
                            }
                            source_display = "📱 Facebook Page/Profile (Apify Scraper - Tier 3)"
                    
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
                            # Capture various text fields from different actors
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
    """Main fetcher dispatcher - combines all segments into single text (backward compatibility).
    
    Args:
        url: URL to fetch from
        max_comments: Maximum comments to retrieve
        
    Returns:
        Tuple of (combined text, source info)
    """
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        return "", info
    return "\n\n".join(texts), f"**Nguồn:** {info}"
