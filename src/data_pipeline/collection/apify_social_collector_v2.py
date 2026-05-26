"""
apify_social_collector_v2.py — PHIÊN BẢN 3.1
=============================================
API Contract (khớp với collection_service.py):

  collect_facebook(token, post_urls, max_comments, min_likes, timeout_secs, dedup_store)
  collect_tiktok(token, hashtags, max_per_tag, comments_per_video, max_replies,
                 min_video_comments, min_likes, timeout_secs, geo, dedup_store)
  collect_all(limit, platforms, min_likes, auto_discover_urls,
              facebook_urls, tiktok_hashtags, geo, use_dedup)

THAY ĐỔI SO VỚI V3.0:
  + collect_tiktok: thêm comments_per_video, max_replies, min_video_comments
  + collect_tiktok: chiến lược 2 phase (lấy video IDs trước, rồi comments)
  + Tất cả API phải khớp với collection_service.py dòng 335-360
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

try:
    from spam_filter import is_spam
except ImportError:
    def is_spam(text: str) -> bool: return False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

_HERE        = Path(__file__).parent
try:
    from common.paths import DATA_RAW_DIR
    RAW_DATA_DIR = DATA_RAW_DIR
except ImportError:
    RAW_DATA_DIR = _HERE.parent / "raw_data"

# ── Actor IDs ──────────────────────────────────────────────────────────────────
ACTOR_FACEBOOK_COMMENTS  = "apify/facebook-comments-scraper"
ACTOR_FACEBOOK_POSTS     = "apify/facebook-posts-scraper"
ACTOR_FACEBOOK_SEARCH    = "apify/facebook-search-scraper" 
ACTOR_FACEBOOK_GROUPS    = "apify/facebook-groups-scraper"
ACTOR_TIKTOK             = "clockworks/free-tiktok-scraper"
ACTOR_TIKTOK_COMMENTS    = "clockworks/tiktok-scraper"
ACTOR_THREADS            = "igview-owner/threads-search-scraper"
ACTOR_YOUTUBE            = "streamers/youtube-scraper"

CONFIG_DIR = _HERE / "actor_configs"

def load_platform_config(platform: str) -> dict:
    """Tải cấu hình từ JSON template hoặc trả về mặc định."""
    path = CONFIG_DIR / f"{platform}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Lỗi load config {platform}: {e}")
    return {}

# ── Session Logging ────────────────────────────────────────────────────────────
class CollectionSessionLog:
    """Ghi nhận log chi tiết cho mỗi URL thu thập để hậu kiểm (Architect's checkpoint)."""
    def __init__(self, platform: str):
        self.platform = platform
        self.logs = []
        self.start_time = time.time()

    def add(self, url: str, status: str, count: int = 0, error: str = ""):
        self.logs.append({
            "url": url,
            "status": status,
            "count": count,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def save(self):
        log_dir = Path("logs/collection_sessions")
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.platform}_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = log_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "platform": self.platform,
                "duration": time.time() - self.start_time,
                "items": self.logs
            }, f, ensure_ascii=False, indent=2)
        return path

# ── Architect's Slang & Controversial Keywords (Thesis v4.2) ───────────────────
# Nhóm 1: Trải nghiệm tiêu cực / Side-effects
SLANG_NEGATIVE = ["kiếp nạn", "bị hành", "vật lên vật xuống", "sốt li bì", "đơ tay", "sợ ngang", "ác mộng"]
# Nhóm 2: Quan điểm trái chiều / Bóc phốt
SLANG_CONTROVERSIAL = ["chê nha", "phốt tiêm chủng", "hút máu", "lùa gà", "tiền mất tật mang"]
# Nhóm 3: Dấu hiệu Misinfo / Thuận tự nhiên
SLANG_MISINFO_TRIGGERS = ["để tự nhiên", "thuận tự nhiên", "chữa lành", "hệ miễn dịch tự nhiên", "bài thuốc dân gian", "không cần tiêm"]

ALL_DISCOVERY_KEYWORDS = ["vaccine", "vắc xin", "tiêm chủng"] + SLANG_NEGATIVE + SLANG_CONTROVERSIAL + SLANG_MISINFO_TRIGGERS

# ── TikTok Hashtags (25 tags, 6 nhóm) ─────────────────────────────────────────
TIKTOK_HASHTAGS: dict[str, list[str]] = {
    "general"     : ["tiemchung", "vacxin", "tiemphong", "lichtiemchung", "muitiemvacxin"],
    "covid"       : ["vacxincovid", "vacxinastrazeneca", "vacxinpfizer", "vacxinmoderna",
                     "vacxinsinovac", "muicovid", "covid19vietnam"],
    "side_effects": ["phanungphuvacxin", "tacduongphuvacxin", "taibienvacxin",
                     "socphanve", "hanhsotsauvacxin", "bienchangsauvacxin"],
    "children"    : ["vacxintrem", "tiemchungtrem", "vacxin5trong1",
                     "vacxin6trong1", "lichtiem0120", "chamsoctrem"],
    "specific"    : ["vacxinsoi", "vacxinhpv", "vacxincum", "viennaovacxin", "soimoivietnam"],
    "stance"      : ["khongtiemvacxin", "chongvacxin", "antivax",
                     "suthattevacxin", "vacxinantoankong"],
}
TIKTOK_HASHTAGS_FLAT: list[str] = [t for g in TIKTOK_HASHTAGS.values() for t in g]

# ── Vaccine Keywords ──────────────────────────────────────────────────────────
VACCINE_KEYWORDS: frozenset[str] = frozenset([
    "vaccine", "vắc xin", "vắcxin", "vacxin", "vx",
    "tiêm chủng", "tiêm phòng", "chích ngừa", "chích vắc",
    "covid", "pfizer", "astrazeneca", "moderna", "sinovac",
    "sputnik", "vero cell", "5 trong 1", "6 trong 1",
    "hpv", "cúm", "sởi", "rubella", "bại liệt",
    "viêm não", "thương hàn", "ho gà", "bạch hầu",
    "phản ứng phụ", "tác dụng phụ", "tai biến", "biến chứng",
    "sốc phản vệ", "sốt sau tiêm", "hành sốt",
    "chết sau tiêm", "tử vong sau tiêm",
    "không nên tiêm", "phản đối tiêm", "chống vaccine",
    "tiêm chủng mở rộng", "chương trình tiêm chủng",
    "chích", "mũi tiêm", "mũi covid", "lịch tiêm",
])

# ── PII Regex ─────────────────────────────────────────────────────────────────
_RE_PHONE   = re.compile(r"(\+84|0)[0-9]{8,10}")
_RE_PROFILE = re.compile(r"https?://(?:www\.)?(?:facebook|tiktok)\.com/[^\s\"']+")
_RE_HTML    = re.compile(r"<[^>]+>")
_RE_MENTION = re.compile(r"@[\w.]+")
_RE_URL_GEN = re.compile(r"https?://\S+")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sha(v: str, n: int = 16) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()[:n]


def _anonymize(text: str) -> str:
    text = _RE_PHONE.sub("[SĐT ẨN]", text)
    text = _RE_PROFILE.sub("[PROFILE ẨN]", text)
    text = _RE_HTML.sub("", text)
    text = _RE_MENTION.sub(lambda m: "@" + _sha(m.group(0)), text)
    text = _RE_URL_GEN.sub("[URL ẨN]", text)
    return text.strip()


def _is_vaccine_relevant(text: str, min_tokens: int = 2) -> bool:
    if not text:
        return False
    
    # Giảm từ 5 xuống 2 để bắt được các video ngắn trên TikTok/FB
    if len(text.split()) < min_tokens:
        log.debug(f"Bỏ qua nội dung quá ngắn (<{min_tokens} từ): {text[:30]}...")
        return False
        
    relevant = any(kw in text.lower() for kw in VACCINE_KEYWORDS)
    if not relevant:
        log.debug(f"Bỏ qua nội dung không liên quan vaccine: {text[:30]}...")
    return relevant


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _get_dedup_store():
    try:
        from dedup_store import default_store
        return default_store
    except ImportError:
        log.debug("dedup_store.py không tìm thấy — dedup bị tắt")
        return None


def get_apify_client(token_override: str | None = None):
    """
    Tạo ApifyClient với cơ chế fallback tự động.
    Thứ tự ưu tiên: token_override > APIFY_API_TOKEN > APIFY_API_TOKEN_BACKUP
    """
    try:
        from apify_client import ApifyClient
    except ImportError:
        log.error("apify-client chưa cài. Chạy: pip install apify-client")
        return None, None

    # 1. Thử token override nếu có
    if token_override:
        return ApifyClient(token_override), token_override

    # 2. Thử primary token
    primary_token = os.getenv("APIFY_API_TOKEN", "").strip()
    if primary_token:
        client = ApifyClient(primary_token)
        try:
            # Kiểm tra xem token còn sống không (gọi nhẹ 1 API)
            client.user().get()
            return client, primary_token
        except Exception as e:
            log.warning(f"Apify primary token lỗi hoặc hết hạn: {e}")

    # 3. Thử backup token
    backup_token = os.getenv("APIFY_API_TOKEN_BACKUP", "").strip()
    if backup_token:
        log.info("Đang chuyển sang sử dụng Apify BACKUP token...")
        client = ApifyClient(backup_token)
        try:
            client.user().get()
            return client, backup_token
        except Exception as e:
            log.error(f"Apify backup token cũng lỗi: {e}")

    return None, None


# ── Normalizers ───────────────────────────────────────────────────────────────

def _norm_fb(raw: dict) -> dict | None:
    text_raw = raw.get("text", raw.get("commentText", raw.get("message", "")))
    if is_spam(text_raw):
        return None
    text     = _anonymize(text_raw)
    if not _is_vaccine_relevant(text):
        return None
    cid      = str(raw.get("id", raw.get("commentId", "")))
    post_url = raw.get("postUrl", raw.get("url", ""))
    return {
        "source"                 : "facebook",
        "platform"               : "facebook",
        "source_credibility_tier": "non-institutional",
        "post_id"                : _sha(cid) if cid else _sha(text[:40]),
        "text"                   : text,
        "url"                    : post_url,
        "engagement_metrics"     : {
            "likes"   : _safe_int(raw.get("likesCount", raw.get("likes", 0))),
            "replies" : _safe_int(raw.get("replyCount", 0)),
            "comments": 0,
            "views"   : 0,
            "shares"  : 0,
        },
        "timestamp"              : raw.get("date", raw.get("timestamp", "")),
        "collected_at"           : datetime.now().isoformat(),
        "collection_method"      : "apify-facebook-comments-scraper",
        "data_confidence_score"  : 0.75,
        "language"               : "vi",
    }


def _norm_tiktok(raw: dict, hashtag: str = "") -> dict | None:
    text_raw = (
        raw.get("text", "")
        or raw.get("desc", "")
        or raw.get("commentText", "")
        or ""
    )
    if is_spam(text_raw):
        return None
    text = _anonymize(text_raw)
    if not _is_vaccine_relevant(text):
        return None
    vid = str(raw.get("id", raw.get("videoId", raw.get("aweme_id", ""))))
    url = raw.get("webVideoUrl", raw.get("shareUrl", ""))
    stats = raw.get("stats", {})
    return {
        "source"                 : "tiktok",
        "platform"               : "tiktok",
        "source_credibility_tier": "non-institutional",
        "post_id"                : _sha(vid) if vid else _sha(text[:40]),
        "text"                   : text,
        "url"                    : url,
        "engagement_metrics"     : {
            "likes"   : _safe_int(raw.get("diggCount",    stats.get("diggCount",    0))),
            "comments": _safe_int(raw.get("commentCount", stats.get("commentCount", 0))),
            "views"   : _safe_int(raw.get("playCount",    stats.get("playCount",    0))),
            "shares"  : _safe_int(raw.get("shareCount",   stats.get("shareCount",   0))),
        },
        "timestamp"              : str(raw.get("createTime", raw.get("createTimeISO", ""))),
        "collected_at"           : datetime.now().isoformat(),
        "collection_method"      : "apify-tiktok-scraper",
        "query_hashtag"          : hashtag,
        "data_confidence_score"  : 0.70,
        "language"               : "vi",
    }


def _norm_youtube(raw: dict) -> dict | None:
    """Chuẩn hóa dữ liệu YouTube (từ apify/youtube-scraper)."""
    desc = raw.get("description", "")
    subs = ""
    if raw.get("subtitles"):
        subs = "\n[SUBTITLES]: " + " ".join([s.get("text", "") for s in raw.get("subtitles", [])[:100]])
    
    text_raw = f"{desc}\n{subs}".strip()
    
    if is_spam(text_raw): return None
    text = _anonymize(text_raw)
    if not _is_vaccine_relevant(text, min_tokens=1): return None
    
    vid = raw.get("id", "")
    comment_threads = []
    for c in raw.get("comments", []):
        c_text = _anonymize(c.get("text", ""))
        if _is_vaccine_relevant(c_text, min_tokens=1):
            comment_threads.append({
                "author": "anonymous",
                "text": c_text,
                "is_relevant": True,
                "likes": _safe_int(c.get("likes")),
                "replies": [
                    {"author": "anonymous", "text": _anonymize(r.get("text", ""))} 
                    for r in c.get("replies", [])
                ]
            })

    return {
        "source": "youtube",
        "platform": "youtube",
        "source_credibility_tier": "mixed",
        "post_id": vid,
        "text": text,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "engagement_metrics": {
            "likes": _safe_int(raw.get("likes")),
            "views": _safe_int(raw.get("viewCount")),
            "comments": _safe_int(raw.get("commentCount")),
        },
        "comment_threads": comment_threads,
        "timestamp": raw.get("date", ""),
        "collected_at": datetime.now().isoformat(),
        "collection_method": "apify-youtube-scraper",
        "data_confidence_score": 0.85,
        "language": "vi",
    }


def _norm_threads(raw: dict) -> dict | None:
    """Chuẩn hóa dữ liệu Threads (từ apify/threads-scraper)."""
    text_raw = raw.get("text", "")
    if is_spam(text_raw): return None
    text = _anonymize(text_raw)
    if not _is_vaccine_relevant(text, min_tokens=1): return None
    
    code = raw.get("code", "")
    return {
        "source": "threads",
        "platform": "threads",
        "source_credibility_tier": "non-institutional",
        "post_id": code,
        "text": text,
        "url": f"https://www.threads.net/post/{code}",
        "engagement_metrics": {
            "likes": _safe_int(raw.get("like_count")),
            "replies": _safe_int(raw.get("reply_count")),
        },
        "timestamp": raw.get("taken_at_iso", ""),
        "collected_at": datetime.now().isoformat(),
        "collection_method": "apify-threads-scraper",
        "data_confidence_score": 0.8,
        "language": "vi",
    }


# ── Facebook Collector ────────────────────────────────────────────────────────

def collect_facebook(
    token: str,
    post_urls: list[str],
    max_comments: int = 9999,
    min_likes: int = 0,
    timeout_secs: int = 900,
    dedup_store: any = None,
) -> list[dict]:
    """
    Thu thập Facebook comments với URL Router thông minh.
    """
    if not post_urls:
        return []

    client, active_token = get_apify_client(token)
    if not client:
        log.error("Không thể khởi tạo ApifyClient (thiếu hoặc sai token)")
        return []

    results = []
    session_log = CollectionSessionLog("facebook")

    # 1. URL Router Logic
    routes = {"groups": [], "posts": [], "search": []}
    for url in post_urls:
        if "/search/" in url:
            routes["search"].append(url)
        elif "/posts/" in url or "/share/p/" in url or "/permalink/" in url:
            routes["posts"].append(url)
        elif "/groups/" in url:
            routes["groups"].append(url)
        else:
            routes["posts"].append(url)

    # 2. Xử lý Search URLs (Nhóm 3) -> Discover & Filter -> Đẩy vào Nhóm 2
    # if routes["search"]:
    #     log.info(f"🔍 [ROUTING] Phát hiện {len(routes['search'])} Search URLs")
    #     for s_url in routes["search"]:
    #         try:
    #             import urllib.parse
    #             query = s_url.split("q=")[-1] if "q=" in s_url else "vắc xin"
    #             query = urllib.parse.unquote(query)
    #             
    #             run_input = {
    #                 "searchMode": "posts",
    #                 "searchTerms": [query],
    #                 "resultsLimit": 50,
    #                 "sortOrder": "latest"
    #             }
    #             run = client.actor(ACTOR_FACEBOOK_SEARCH).call(run_input=run_input, timeout_secs=timeout_secs)
    #             dataset_id = run.get("defaultDatasetId", "")
    #             
    #             items = list(client.dataset(dataset_id).iterate_items())
    #             # Token Guard
    #             if len(items) == 0:
    #                 log.warning(f"⚠️ [TOKEN GUARD] Cạn kiệt dữ liệu hoặc sai Actor cho Search: {s_url}")
    #                 continue
    #             
    #             passed_count = 0
    #             for item in items:
    #                 comments_count = item.get("comments", 0)
    #                 url = item.get("url", "") or item.get("facebookUrl", "")
    #                 if url and comments_count > 20:
    #                     routes["posts"].append(url)
    #                     passed_count += 1
    #             log.info(f"  -> Trích xuất được {passed_count} bài viết tiềm năng (comments > 20) từ {query}")
    #         except Exception as e:
    #             log.error(f"  ❌ Lỗi Search URL {s_url}: {e}")

    # Helper xử lý dữ liệu chung
    def process_and_dedup(items_list, source_urls):
        if len(items_list) == 0:
            log.warning(f"⚠️ [TOKEN GUARD] Cạn kiệt dữ liệu hoặc sai Actor cho batch: {source_urls}")
            return 0
        
        found = 0
        for raw in items_list:
            item = _norm_fb(raw)
            if item:
                if item["engagement_metrics"]["likes"] < min_likes:
                    continue
                if dedup_store and not dedup_store.is_new(url=item["url"], text=item["text"]):
                    continue
                if dedup_store:
                    dedup_store.mark_seen(url=item["url"], text=item["text"], source="facebook")
                results.append(item)
                found += 1
        return found

    # 3. Xử lý Groups (Nhóm 1)
    # if routes["groups"]:
    #     log.info(f"👥 [ROUTING] Xử lý {len(routes['groups'])} Group URLs bằng facebook-groups-scraper")
    #     for url in routes["groups"]:
    #         log.info(f"  Scraping Group: {url}")
    #         run_input = {
    #             "startUrls": [{"url": url}],
    #             "maxComments": max_comments,
    #             "viewOption": "TOP_POSTS"
    #         }
    #         try:
    #             run = client.actor(ACTOR_FACEBOOK_GROUPS).call(run_input=run_input, timeout_secs=timeout_secs)
    #             items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    #             
    #             found = process_and_dedup(items, [url])
    #             session_log.add(url, "SUCCESS", count=found)
    #         except Exception as e:
    #             log.error(f"  ❌ Group error: {e}")
    #             session_log.add(url, "FAILED", error=str(e))

    # 4. Xử lý Posts (Nhóm 2)
    if routes["posts"]:
        log.info(f"📝 [ROUTING] Xử lý {len(routes['posts'])} Post URLs bằng facebook-comments-scraper")
        batch_size = 2
        for i in range(0, len(routes["posts"]), batch_size):
            batch = routes["posts"][i : i + batch_size]
            log.info(f"  Batch {i//batch_size + 1}: Scrapping {len(batch)} Posts...")
            
            run_input = {
                "startUrls": [{"url": url} for url in batch],
                "maxComments": max_comments,
                "maxReplies": 20,
                "viewOption": "RANKED_THREADED"
            }
            try:
                run = client.actor(ACTOR_FACEBOOK_COMMENTS).call(run_input=run_input, timeout_secs=timeout_secs)
                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                
                found = process_and_dedup(items, batch)
                for url in batch:
                    session_log.add(url, "SUCCESS", count=found // len(batch))
            except Exception as e:
                log.error(f"  ❌ Batch error: {e}")
                for url in batch:
                    session_log.add(url, "FAILED", error=str(e))

    session_log.save()
    log.info(f"  ✅ Facebook: {len(results)} items collected")
    return results


def discover_facebook_posts(
    token       : str,
    keywords    : list[str],
    pages       : list[str] = None,
    limit       : int = 50,
    timeout_secs: int = 300
) -> list[str]:
    """
    Search post URLs có tương tác cao (Viral) dựa trên keyword hoặc danh sách trang.
    GIAI ĐOẠN 1: Discovery (Mở rộng: Cả Page và Personal Profiles qua Search)
    """
    client, active_token = get_apify_client(token)
    if not client: return []

    post_urls = []
    
    # Nếu có danh sách trang cụ thể, dùng post lookup của trang đó
    # Nếu không, dùng Global Search cho toàn bộ FB
    if pages:
        log.info(f"FB Discovery: Target scanning {len(pages)} specific pages/profiles")
        run_input = {
            "startUrls": [{"url": f"https://www.facebook.com/{p}"} for p in pages],
            "resultsLimit": limit,
            "searchTerms": keywords,
        }
        actor_id = ACTOR_FACEBOOK_POSTS
    else:
        log.info(f"FB Discovery: Global Viral Search for keywords {keywords}")
        run_input = {
            "searchMode": "posts",
            "searchTerms": keywords,
            "resultsLimit": limit,
            "sortOrder": "latest", # Hoặc 'top' nếu actor hỗ trợ
        }
        actor_id = ACTOR_FACEBOOK_SEARCH

    try:
        run = client.actor(actor_id).call(run_input=run_input, timeout_secs=timeout_secs)
        dataset_id = run.get("defaultDatasetId", "")
        for item in client.dataset(dataset_id).iterate_items():
            url = item.get("url", "") or item.get("facebookUrl", "")
            if url and url not in post_urls:
                post_urls.append(url)
    except Exception as e:
        log.error(f"FB Discovery error: {e}")
    
    return post_urls


# ── TikTok Collector ──────────────────────────────────────────────────────────

def collect_tiktok(
    token: str,
    hashtags: list[str] | None = None,
    max_per_tag: int | None = None,
    comments_per_video: int | None = None,
    max_replies: int | None = None,
    min_video_comments: int | None = None,
    min_likes: int | None = None,
    timeout_secs: int = 600,
    geo: str = "VN",
    dedup_store: any = None,
) -> list[dict]:
    """
    Thu thập dữ liệu TikTok theo mô hình 2 GIAI ĐOẠN (Reclamation Model).
    """
    conf = load_platform_config("tiktok")
    m_per_tag = max_per_tag or conf.get("max_per_tag", 20)
    c_per_vid = comments_per_video or conf.get("comments_per_video", 30)
    m_replies = max_replies or conf.get("max_replies", 5)
    batch_size = conf.get("batch_size", 3)
    client, active_token = get_apify_client(token)
    if not client:
        log.error("TikTok: Không thể khởi tạo ApifyClient")
        return []

    if hashtags is None:
        hashtags = TIKTOK_HASHTAGS_FLAT

    log.info(f"🚀 TikTok Reclamation: {len(hashtags)} hashtags, Stage 1 Discovery...")
    video_urls = []
    session_log = CollectionSessionLog("tiktok")

    # STAGE 1: Discovery
    for tag in hashtags:
        run_input = {
            "hashtags": [tag],
            "resultsPerPage": m_per_tag,
            "shouldDownloadVideos": conf.get("shouldDownloadVideos", False),
        }
        try:
            run = client.actor(ACTOR_TIKTOK).call(run_input=run_input, timeout_secs=300)
            dataset_id = run.get("defaultDatasetId")
            if dataset_id:
                for raw in client.dataset(dataset_id).iterate_items():
                    url = raw.get("webVideoUrl")
                    if url:
                        video_urls.append(url)
        except Exception as e:
            log.error(f"  ❌ Discovery error in #{tag}: {e}")

    video_urls = list(set(video_urls))
    log.info(f"  Tìm thấy {len(video_urls)} video URLs. Bắt đầu Stage 2 (Deep Comments)...")

    # STAGE 2: Deep Extraction
    results = []
    for i in range(0, len(video_urls), batch_size):
        batch = video_urls[i : i + batch_size]
        log.info(f"  Extracted Batch {i//batch_size + 1}/{len(video_urls)//batch_size + 1}...")
        
        run_input = {
            "postURLs": batch,
            "maxComments": c_per_vid * len(batch),
            "maxRepliesPerComment": m_replies,
        }
        try:
            run = client.actor(ACTOR_TIKTOK_COMMENTS).call(run_input=run_input, timeout_secs=timeout_secs)
            batch_count = 0
            for raw in client.dataset(run["defaultDatasetId"]).iterate_items():
                item = _norm_tiktok(raw)
                if item:
                    if item["engagement_metrics"]["likes"] < min_likes:
                        continue
                    if dedup_store:
                        if not dedup_store.is_new(url=item["url"], text=item["text"]):
                            continue
                        dedup_store.mark_seen(url=item["url"], text=item["text"], source="tiktok")
                    results.append(item)
                    batch_count += 1
            
            for url in batch:
                session_log.add(url, "SUCCESS", count=batch_count // len(batch))
        except Exception as e:
            log.error(f"  ❌ Extraction error: {e}")
            for url in batch:
                session_log.add(url, "FAILED", error=str(e))

    session_log.save()
    log.info(f"  ✅ TikTok: {len(results)} items collected")
    return results



def collect_threads(
    token       : str,
    queries     : list[str],
    limit        : int | None = None,
    timeout_secs: int = 120,
    dedup_store : any = None,
) -> list[dict]:
    """Thu thập dữ liệu Threads sử dụng cấu trúc slang keywords."""
    client, active_token = get_apify_client(token)
    if not client: return []

    conf = load_platform_config("threads")
    # Gộp slang nếu query trống
    active_queries = queries if queries else conf.get("slang_keywords", [])
    max_items = limit or conf.get("maxItems", 50)
    
    log.info(f"🚀 Threads Scraping: {len(active_queries)} queries, mode: {conf.get('mode', 'search')}")
    results = []
    
    for q in active_queries:
        run_input = {
            "searchQueries": [q],
            "maxItems": max(5, max_items // len(active_queries)) if active_queries else 10,
            "mode": conf.get("mode", "search")
        }
        try:
            run = client.actor(ACTOR_THREADS).call(run_input=run_input, timeout_secs=timeout_secs)
            for raw in client.dataset(run["defaultDatasetId"]).iterate_items():
                item = _norm_threads(raw)
                if item:
                    if dedup_store and not dedup_store.is_new(url=item["url"], text=item["text"]):
                        continue
                    results.append(item)
                    if dedup_store:
                        dedup_store.mark_seen(url=item["url"], text=item["text"], source="threads")
        except Exception as e:
            log.error(f"  ❌ Threads error for '{q}': {e}")
            
    return results


def collect_youtube(
    token: str,
    video_urls: list[str] | None = None,
    queries: list[str] | None = None,
    max_comments: int | None = None,
    timeout_secs: int = 600,
    dedup_store: any = None,
) -> list[dict]:
    """Thu thập YouTube 2 giai đoạn: Discovery (nếu no URLs) -> Deep Extraction."""
    client, active_token = get_apify_client(token)
    if not client: return []

    conf = load_platform_config("youtube")
    urls = video_urls or []
    
    # Giai đoạn 1: Discovery (nếu không có URL)
    if not urls and queries:
        log.info(f"🚀 YouTube Discovery: {len(queries)} keywords...")
        for q in queries:
            try:
                run = client.actor(ACTOR_YOUTUBE).call(run_input={
                    "searchKeywords": q, 
                    "maxResults": conf.get("maxResults", 5)
                }, timeout_secs=300)
                for raw in client.dataset(run["defaultDatasetId"]).iterate_items():
                    if raw.get("id"): urls.append(raw["id"])
            except Exception as e:
                log.error(f"  ❌ YT Discovery error: {e}")
    
    urls = list(set(urls))
    if not urls: return []

    # Giai đoạn 2: Deep Extraction
    log.info(f"🚀 YouTube Extraction: {len(urls)} videos...")
    results = []
    batch_size = conf.get("batchSize", 2)
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        run_input = {
            "videoUrls": [f"https://www.youtube.com/watch?v={vid}" if len(vid)==11 else vid for vid in batch],
            "maxComments": max_comments or conf.get("maxComments", 100),
            "maxRepliesPerComment": conf.get("maxRepliesPerComment", 10),
            "downloadSubtitles": conf.get("downloadSubtitles", True),
            "saveSubsToText": conf.get("saveSubsToText", True),
        }
        try:
            run = client.actor(ACTOR_YOUTUBE).call(run_input=run_input, timeout_secs=timeout_secs)
            for raw in client.dataset(run["defaultDatasetId"]).iterate_items():
                item = _norm_youtube(raw)
                if item:
                    if dedup_store and not dedup_store.is_new(url=item["url"], text=item["text"]):
                        continue
                    results.append(item)
                    if dedup_store:
                        dedup_store.mark_seen(url=item["url"], text=item["text"], source="youtube")
        except Exception as e:
            log.error(f"  ❌ YT Extraction batch error: {e}")
            
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def collect_all(
    limit              : int = 500,
    platforms          : list[Literal["facebook", "tiktok", "threads"]] | None = None,
    min_likes          : int = 0,
    auto_discover_urls : bool = True,
    facebook_urls      : list[str] | None = None,
    facebook_pages     : list[str] | None = None,
    keywords           : list[str] | None = None,
    tiktok_hashtags    : list[str] | None = None,
    geo                : str = "VN",
    use_dedup          : bool = True,
) -> list[dict]:
    """
    Hàm chính — thu thập từ Facebook và/hoặc TikTok.

    Args:
        limit             : Tổng items tối đa
        platforms         : ['facebook'], ['tiktok'], hoặc cả hai
        min_likes         : Engagement filter
        auto_discover_urls: Tự tìm FB URLs qua Google nếu không có
        facebook_urls     : Override FB URL list
        tiktok_hashtags   : Override hashtag list (None = 25 mặc định)
        geo               : Country code cho TikTok
        use_dedup         : Dùng dedup_store để chống trùng xuyên phiên

    Raises:
        ValueError: Nếu APIFY_API_TOKEN chưa cấu hình
    """
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    # Nếu token chưa có sẵn, get_apify_client sẽ tự lo ở dưới
    
    if platforms is None:
        platforms = ["facebook", "tiktok", "threads", "youtube"]

    dedup        = _get_dedup_store() if use_dedup else None
    per_platform = max(1, limit // len(platforms))
    all_results  : list[dict] = []
    
    queries = keywords or ALL_DISCOVERY_KEYWORDS

    if "facebook" in platforms:
        urls = facebook_urls or []
        if (not urls or auto_discover_urls) and facebook_pages:
            log.info(f"Facebook Discovery Phase starting for pages: {facebook_pages}")
            discovered_urls = discover_facebook_posts(token, queries, facebook_pages)
            urls.extend(discovered_urls)
            urls = list(set(urls))

        if urls:
            fb_items = collect_facebook(
                token=token, post_urls=urls,
                max_comments=per_platform, min_likes=min_likes, dedup_store=dedup,
            )
            all_results.extend(fb_items)

    if "threads" in platforms:
        th_items = collect_threads(
            token=token, queries=queries, limit=per_platform, dedup_store=dedup
        )
        all_results.extend(th_items)

    if "youtube" in platforms:
        yt_items = collect_youtube(
            token=token, queries=queries, 
            max_comments=per_platform, dedup_store=dedup
        )
        all_results.extend(yt_items)

    if "tiktok" in platforms:
        tags         = tiktok_hashtags or TIKTOK_HASHTAGS_FLAT
        tt_items = collect_tiktok(
            token=token, hashtags=tags[:5], # Giới hạn tags discovery cho demo
            comments_per_video=30, dedup_store=dedup, min_likes=min_likes
        )
        all_results.extend(tt_items)

    if dedup:
        log.info(f"Dedup store stats: {dedup.stats()}")

    return all_results


def save_results(data: list[dict], filename: str | None = None) -> Path | None:
    if not data:
        return None
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DATA_DIR / (filename or f"apify_social_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(data)} items → {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== Apify Social Collector v2 (API v3.1) ===\n")
    
    from dotenv import load_dotenv
    load_dotenv() # Load from current directory
    
    token = os.getenv("APIFY_API_TOKEN", "")
    if not token:
        # Thử load tuyệt đối nếu chạy từ thư mục khác
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        dotenv_path = project_root / ".env"
        print(f"DEBUG: Looking for .env at {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
        token = os.getenv("APIFY_API_TOKEN", "")

    if not token:
        print("⚠  APIFY_API_TOKEN chưa cấu hình")
        print("   1. Đăng ký: https://apify.com")
        print("   2. Thêm vào .env: APIFY_API_TOKEN=apify_api_xxx")
    else:
        print(f"✔ Token OK: ...{token[-6:]}")
        print(f"✔ {len(TIKTOK_HASHTAGS_FLAT)} hashtags sẵn sàng ({len(TIKTOK_HASHTAGS)} nhóm)")
        print("\nBắt đầu test với TikTok (geo=VN)...\n")
        data = collect_all(limit=30, platforms=["tiktok"], use_dedup=False)
        save_results(data)
        print(f"\n✅ Hoàn thành: {len(data)} items")
