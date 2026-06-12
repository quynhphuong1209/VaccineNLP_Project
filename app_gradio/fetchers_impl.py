"""Local-only multi-source URL fetchers (news / YouTube / social via Apify).

IMPORTANT — DO NOT SHIP THIS FILE TO A PUBLIC HUGGING FACE SPACE.
Hugging Face's abuse detection flags Spaces that scrape YouTube / social media
(it statically matches things like Apify social-media actor ids and yt_dlp comment
extraction). `app.py` imports this module *optionally*: when it is absent the URL
fetchers become inert no-ops and the deployed `app.py` carries no scraping code.

Keep this module out of any Space upload (it is excluded from the deploy allow-list
and listed in the Space `.hfignore`). It is meant for local data-collection runs.
"""
import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Apify tokens (multiple naming conventions supported).
APIFY_TOKENS: List[str] = []
_POSSIBLE_APIFY_KEYS = [
    "APIFY_TOKEN", "APIFY_API_TOKEN", "APIFY_TOKEN_1", "APIFY_API_TOKEN_1",
    "APIFY_TOKEN_2", "APIFY_API_TOKEN_2", "APIFY_TOKEN_3", "APIFY_API_TOKEN_3",
    "APIFY_TOKEN_4", "APIFY_API_TOKEN_4", "APIFY_TOKEN_5", "APIFY_API_TOKEN_5",
]
for _k in _POSSIBLE_APIFY_KEYS:
    _val = os.environ.get(_k, "").strip()
    if _val and _val not in APIFY_TOKENS:
        APIFY_TOKENS.append(_val)


def detect_source(url: str) -> str:
    """Detect URL source type."""
    url_lower = url.lower()
    news_domains = ["vnexpress", "tuoitre", "thanhnien", "dantri", "vietnamnet",
                    "suckhoedoisong", "laodong", "tienphong", "znews", "hanoimoi",
                    "baochinhphu", "nhandan", "vov", "vtv"]
    if any(d in url_lower for d in news_domains):
        return "news"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if any(d in url_lower for d in ["facebook.com", "tiktok.com", "threads.net"]):
        return "apify"
    return "news"


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


def fetch_apify(url: str) -> str:
    if not APIFY_TOKENS:
        return "❌ APIFY_TOKEN chưa được setup"
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
                    return "❌ URL không được hỗ trợ"
                run = client.actor(actor_id).call(run_input={"startUrls": [{"url": url}]})
                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                if items:
                    texts = [item.get("text", "") or item.get("caption", "") for item in items[:5]]
                    return "\n\n".join(t for t in texts if t)
                return "❌ Không trả về content"
            except Exception:
                continue
        return "❌ Tất cả Apify tokens đều thất bại"
    except Exception as e:
        return f"❌ Apify error: {e}"


def _is_valid_comment(txt: str) -> bool:
    """Màng lọc thông minh đầu vào cho comments/posts mạng xã hội."""
    if not txt:
        return False
    txt_strip = txt.strip()
    if len(txt_strip) < 15 or len(txt_strip.split()) < 4:
        return False
    spam_keywords = [
        "inbox", "ib shop", "giá bao nhiêu", "mua ở đâu", "ship", "freeship",
        "liên hệ zalo", "sđt", "tuyển dụng", "tuyển ctv", "sỉ lẻ", "giá rẻ",
        "thanh lý", "chốt đơn", "nhận hàng", "uy tín", "đặt hàng", "zalo sđt",
        "cam kết", "hiệu quả", "giá sỉ", "giá lẻ", "chuyên sỉ", "tuyển đại lý",
    ]
    txt_lower = txt_strip.lower()
    if any(kw in txt_lower for kw in spam_keywords):
        return False
    try:
        from src.preprocessing.text_cleaner_v2 import is_human_vaccine_context
        if not is_human_vaccine_context(txt_strip):
            return False
    except Exception:
        core_kws = ["vaccine", "vắc xin", "tiêm", "mũi", "bác sĩ", "bệnh", "y tế",
                    "thuốc", "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv"]
        if not any(kw in txt_lower for kw in core_kws):
            return False
    return True


def fetch_url_as_list(url: str, max_comments: int = 30) -> Tuple[List[str], str]:
    """Fetch content from URL and return as a list of text segments + source info."""
    if not url or not url.strip():
        return [], "❌ Vui lòng nhập URL"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return [], "❌ URL không hợp lệ"

    kind = detect_source(url)
    texts: List[str] = []
    info = ""

    if kind == "news":
        content = fetch_news(url)
        if content and not content.startswith("❌"):
            texts = [content]
            info = "📰 Báo điện tử (Tier 1, ~2s)"
        else:
            return [], content

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
                    if c_text and _is_valid_comment(c_text):
                        texts.append(c_text)
                    if len(texts) >= max_comments + 2:
                        break
                info = "🎬 YouTube (Tier 2, ~10s)"
        except Exception as e:
            logger.warning(f"yt_dlp failed, trying Apify fallback: {e}")
            kind = "youtube_apify"

    if kind in ("apify", "youtube_apify"):
        if not APIFY_TOKENS:
            return [], "❌ APIFY_TOKEN chưa được setup"
        last_err = ""
        actor_id = ""
        try:
            from apify_client import ApifyClient
            for token in APIFY_TOKENS:
                try:
                    client = ApifyClient(token)
                    client.user().get()
                    apify_max_cmt = min(max_comments, 15)

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
                                "resultsLimit": 15,
                            }
                            source_display = "👥 Facebook Group (Apify Scraper - Tier 3)"
                        elif "/posts/" in url.lower() or "/permalink/" in url.lower() or "comment_id=" in url.lower() or "/pfbid" in url.lower():
                            actor_id = "apify/facebook-comments-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxComments": apify_max_cmt,
                                "resultsLimit": apify_max_cmt,
                            }
                            source_display = "💬 Facebook Post/Comments (Apify Scraper - Tier 3)"
                        else:
                            actor_id = "apify/facebook-posts-scraper"
                            run_input = {
                                "startUrls": [{"url": url}],
                                "maxPosts": 3,
                                "maxComments": apify_max_cmt,
                                "maxCommentsPerPost": 5,
                            }
                            source_display = "📱 Facebook Page/Profile (Apify Scraper - Tier 3)"
                    elif "tiktok.com" in url.lower():
                        actor_id = "clockworks/tiktok-comments-scraper"
                        run_input = {"postURLs": [url], "maxComments": apify_max_cmt}
                        source_display = "🎵 TikTok Comments (Apify Scraper - Tier 3)"
                    elif "threads.net" in url.lower():
                        actor_id = "thenetaji/threads-scraper"
                        run_input = {"startUrls": [{"url": url}], "maxItems": apify_max_cmt}
                        source_display = "🧵 Threads (Apify Scraper - Tier 3)"
                    else:
                        return [], "❌ URL không được hỗ trợ"

                    logger.info(f"Running Apify Actor: {actor_id} for URL: {url}")
                    run = client.actor(actor_id).call(run_input=run_input)
                    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

                    if items:
                        for item in items:
                            txt = (
                                item.get("text", "")
                                or item.get("message", "")
                                or item.get("caption", "")
                                or item.get("comment", "")
                                or item.get("fullText", "")
                                or item.get("description", "")
                                or item.get("messageText", "")
                                or item.get("title", "")
                                or item.get("commentText", "")
                                or item.get("body", "")
                            )
                            if txt and txt.strip():
                                clean_txt = txt.strip()
                                if _is_valid_comment(clean_txt):
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
    """Main fetcher dispatcher (backward compatibility)."""
    texts, info = fetch_url_as_list(url, max_comments)
    if not texts:
        return "", info
    return "\n\n".join(texts), f"**Nguồn:** {info}"
