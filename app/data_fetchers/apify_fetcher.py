"""
Apify fetcher — Tầng 3 (Slow: 30-120s)
Facebook/TikTok/Threads với 5-token rotation
"""
import re
import hashlib
import time
from datetime import datetime
import streamlit as st

ACTOR_FACEBOOK_COMMENTS = "apify/facebook-comments-scraper"
ACTOR_FACEBOOK_POSTS    = "apify/facebook-posts-scraper"
ACTOR_TIKTOK_COMMENTS   = "clockworks/tiktok-scraper"
ACTOR_THREADS           = "igview-owner/threads-search-scraper"

def is_facebook_url(url: str) -> bool:
    return bool(re.search(r'facebook\.com|fb\.com|fb\.watch', url, re.I))

def is_tiktok_url(url: str) -> bool:
    return bool(re.search(r'tiktok\.com', url, re.I))

def is_threads_url(url: str) -> bool:
    return bool(re.search(r'threads\.net', url, re.I))


def _get_apify_tokens() -> list[str]:
    """Lấy list token từ file .env (ưu tiên) hoặc Streamlit secrets, fallback empty."""
    tokens = []
    
    # 1. Thử load từ file .env ở thư mục gốc dự án
    try:
        import os
        from pathlib import Path
        
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            # Đọc thủ công file .env để tránh phụ thuộc vào thư viện python-dotenv ngoài
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ["APIFY_API_TOKEN", "APIFY_API_TOKEN_2", "APIFY_API_TOKEN_3", "APIFY_API_TOKEN_4", "APIFY_API_TOKEN_5"]:
                            if v and v not in tokens:
                                tokens.append(v)
    except Exception:
        pass

    # 2. Nếu không có gì từ .env, thử streamlit secrets
    if not tokens:
        try:
            st_tokens = st.secrets.get("APIFY_TOKENS", [])
            if isinstance(st_tokens, str):
                st_tokens = [st_tokens]
            tokens = [t for t in st_tokens if t and t.strip()]
        except Exception:
            pass
            
    # Lọc trùng và rỗng
    final_tokens = []
    for t in tokens:
        if t and t.strip() and t not in final_tokens:
            final_tokens.append(t.strip())
            
    return final_tokens

def _get_client_with_rotation():
    """
    Trả về (client, token_used) với rotation tự động.
    Thử từng token, ngừng ở token đầu hợp lệ.
    """
    try:
        from apify_client import ApifyClient
    except ImportError as e:
        st.error(f"Không thể import apify_client: {e}")
        return None, None
    
    tokens = _get_apify_tokens()
    if not tokens:
        st.warning("⚠️ Không tìm thấy Apify token nào trong .env hoặc secrets!")
        return None, None
    
    errors = []
    candidates = []
    
    for i, token in enumerate(tokens):
        try:
            client = ApifyClient(token)
            # Test nhẹ: gọi user().get() để verify token còn sống
            client.user().get()
            return client, f"token_{i+1}_of_{len(tokens)}"
        except Exception as e:
            err_msg = str(e)
            # Kiểm tra lỗi xác thực thực tế (401, unauthorized)
            is_auth_error = False
            if "401" in err_msg or "unauthorized" in err_msg.lower() or "invalid token" in err_msg.lower():
                is_auth_error = True
                
            errors.append(f"Token {i+1} ({token[:8]}...): {err_msg}")
            
            if not is_auth_error:
                # Nếu chỉ là lỗi kết nối mạng, SSL, hoặc lỗi không phải Unauthenticated,
                # vẫn giữ client này làm ứng viên để tiếp tục gọi actor
                candidates.append((client, f"token_{i+1}_of_{len(tokens)} (fallback-network)"))
                
    if candidates:
        return candidates[0]
        
    if errors:
        st.error("❌ Lỗi xác thực tất cả Apify tokens:\n- " + "\n- ".join(errors))
    return None, None


@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 phút
def fetch_facebook(url: str, max_comments: int = 30, timeout: int = 180) -> dict | None:
    """Facebook post + comments qua Apify."""
    client, token_id = _get_client_with_rotation()
    if not client:
        return {'error': 'Không có Apify token hợp lệ (đã thử 5 token)', 'url': url}
    
    try:
        run_input = {
            "startUrls": [{"url": url}],
            "maxComments": max_comments,
            "maxReplies": 5,
            "viewOption": "RANKED_THREADED",
        }
        run = client.actor(ACTOR_FACEBOOK_COMMENTS).start(
            run_input=run_input,
            wait_for_finish=timeout,
        )
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else (
            getattr(run, "default_dataset_id", None) or (
                run.dict().get("defaultDatasetId") if hasattr(run, "dict") else run.model_dump().get("defaultDatasetId")
            )
        )
        items = list(client.dataset(dataset_id).iterate_items())
        
        if not items:
            return {'error': 'Post không có comments hoặc bị private', 'url': url}
        
        # Trích xuất nội dung bài viết gốc cực kỳ chính xác cho cả Group và Page
        post_text = ""
        for it in items:
            p_text = it.get('postTitle') or it.get('postText') or it.get('description') or ''
            p_text = p_text.strip()
            if p_text:
                post_text = p_text
                break
                
        comments = []
        is_fallback_post = False
        
        if not post_text:
            post_text = items[0].get('text', '').strip()
            is_fallback_post = True
            
        for idx, it in enumerate(items):
            if idx == 0 and is_fallback_post:
                continue
            txt = (it.get('text') or it.get('commentText') or '').strip()
            if txt and txt != post_text:
                comments.append({
                    'text':   txt,
                    'author': 'anonymous',
                    'likes':  it.get('likesCount', 0) or 0,
                })
        
        return {
            'kind':         'facebook',
            'title':        'Facebook Post',
            'text':         post_text[:1500] if post_text else (comments[0]['text'] if comments else ''),
            'comments':     comments[:max_comments],
            'url':          url,
            'extracted_at': datetime.now().isoformat(),
            'fetcher':      f'apify/{token_id}',
        }
    except Exception as e:
        return {'error': f'Apify error: {str(e)[:200]}', 'url': url}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_tiktok(url: str, max_comments: int = 30, timeout: int = 180) -> dict | None:
    """TikTok video + comments qua Apify."""
    client, token_id = _get_client_with_rotation()
    if not client:
        return {'error': 'Không có Apify token hợp lệ', 'url': url}
    
    try:
        run_input = {
            "postURLs": [url],
            "maxComments": max_comments,
            "maxRepliesPerComment": 5,
        }
        run = client.actor(ACTOR_TIKTOK_COMMENTS).start(
            run_input=run_input,
            wait_for_finish=timeout,
        )
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else (
            getattr(run, "default_dataset_id", None) or (
                run.dict().get("defaultDatasetId") if hasattr(run, "dict") else run.model_dump().get("defaultDatasetId")
            )
        )
        items = list(client.dataset(dataset_id).iterate_items())
        
        if not items:
            return {'error': 'Video không có comments hoặc bị private', 'url': url}
        
        video_text = items[0].get('text') or items[0].get('desc', '') or ''
        comments = []
        for it in items[1:] if len(items) > 1 else items:
            txt = (it.get('text') or it.get('commentText') or '').strip()
            if txt:
                comments.append({
                    'text':   txt,
                    'author': 'anonymous',
                    'likes':  it.get('diggCount', 0) or 0,
                })
        
        return {
            'kind':         'tiktok',
            'title':        'TikTok Video',
            'text':         video_text[:1500],
            'comments':     comments[:max_comments],
            'url':          url,
            'extracted_at': datetime.now().isoformat(),
            'fetcher':      f'apify/{token_id}',
        }
    except Exception as e:
        return {'error': f'Apify error: {str(e)[:200]}', 'url': url}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_threads(url: str, timeout: int = 180) -> dict | None:
    """Threads post qua Apify."""
    client, token_id = _get_client_with_rotation()
    if not client:
        return {'error': 'Không có Apify token hợp lệ', 'url': url}
    
    try:
        # Threads actor cần search query — extract code từ URL
        # https://www.threads.net/@user/post/CODE
        code_match = re.search(r'/post/([^/?]+)', url)
        if not code_match:
            return {'error': 'URL Threads không hợp lệ', 'url': url}
        
        run_input = {
            "searchQueries": [url],
            "maxItems": 5,
            "mode": "url",
        }
        run = client.actor(ACTOR_THREADS).start(
            run_input=run_input,
            wait_for_finish=timeout,
        )
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else (
            getattr(run, "default_dataset_id", None) or (
                run.dict().get("defaultDatasetId") if hasattr(run, "dict") else run.model_dump().get("defaultDatasetId")
            )
        )
        items = list(client.dataset(dataset_id).iterate_items())
        
        if not items:
            return {'error': 'Không lấy được Threads post', 'url': url}
        
        main = items[0]
        return {
            'kind':         'threads',
            'title':        'Threads Post',
            'text':         (main.get('text') or '')[:1500],
            'comments':     [],   # Threads actor ít hỗ trợ comments
            'url':          url,
            'extracted_at': datetime.now().isoformat(),
            'fetcher':      f'apify/{token_id}',
        }
    except Exception as e:
        return {'error': f'Apify error: {str(e)[:200]}', 'url': url}
