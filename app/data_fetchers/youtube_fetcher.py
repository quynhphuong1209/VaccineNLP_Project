"""
YouTube fetcher — Tầng 2 (Fast: 5-15s)
Lấy title + description + top N comments (không cần API key)
"""
import re
from datetime import datetime
import streamlit as st

def is_youtube_url(url: str) -> bool:
    return bool(re.search(r'(youtube\.com|youtu\.be)', url, re.I))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_youtube(url: str, max_comments: int = 30, timeout: int = 30) -> dict | None:
    """Lấy YouTube content + top comments bằng yt-dlp."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return {'error': 'yt-dlp chưa cài', 'url': url}
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'getcomments': True,
        'extractor_args': {
            'youtube': {
                'comment_sort': ['top'],
                'max_comments': [str(max_comments), str(max_comments), '0', '0'],
            }
        },
        'socket_timeout': timeout,
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        comments = []
        for c in (info.get('comments') or [])[:max_comments]:
            txt = (c.get('text') or '').strip()
            if txt:
                comments.append({
                    'text':   txt,
                    'author': c.get('author', 'anonymous'),
                    'likes':  c.get('like_count', 0) or 0,
                })
        
        # Text chính = title + description (xếp đầu, ngắn gọn) — dùng cho phân tích
        # Reasoning: model phân tích trên text 600-1000 chars là tối ưu
        main_text = info.get('title', '').strip()
        desc = (info.get('description', '') or '').strip()
        if desc:
            main_text = f"{main_text}. {desc[:800]}".strip()
        
        return {
            'kind':         'youtube',
            'title':        info.get('title', ''),
            'text':         main_text,
            'description':  desc,
            'channel':      info.get('uploader', ''),
            'view_count':   info.get('view_count', 0),
            'like_count':   info.get('like_count', 0),
            'comments':     comments,
            'url':          url,
            'extracted_at': datetime.now().isoformat(),
            'fetcher':      'yt-dlp',
        }
    except Exception as e:
        return {'error': str(e), 'url': url}
