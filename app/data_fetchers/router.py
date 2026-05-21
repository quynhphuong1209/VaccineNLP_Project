"""
Router — Phân loại URL và dispatch đến fetcher phù hợp.
Cung cấp UI metadata: tốc độ, có comments không, có cần token không.
"""
from urllib.parse import urlparse

from .news_fetcher    import fetch_news_article, is_supported_news_url
from .youtube_fetcher import fetch_youtube,      is_youtube_url
from .apify_fetcher   import (
    fetch_facebook, fetch_tiktok, fetch_threads,
    is_facebook_url, is_tiktok_url, is_threads_url,
)

FETCHER_INFO = {
    'news': {
        'tier':           1,
        'estimated_secs': '1-3s',
        'has_comments':   False,
        'needs_token':    False,
        'icon':           '📰',
        'label':          'Báo điện tử',
    },
    'youtube': {
        'tier':           2,
        'estimated_secs': '5-15s',
        'has_comments':   True,
        'needs_token':    False,
        'icon':           '📺',
        'label':          'YouTube',
    },
    'facebook': {
        'tier':           3,
        'estimated_secs': '30-90s',
        'has_comments':   True,
        'needs_token':    True,
        'icon':           '📘',
        'label':          'Facebook (Apify)',
    },
    'tiktok': {
        'tier':           3,
        'estimated_secs': '60-120s',
        'has_comments':   True,
        'needs_token':    True,
        'icon':           '🎵',
        'label':          'TikTok (Apify)',
    },
    'threads': {
        'tier':           3,
        'estimated_secs': '30-60s',
        'has_comments':   False,
        'needs_token':    True,
        'icon':           '🧵',
        'label':          'Threads (Apify)',
    },
}

def detect_source(url: str) -> str | None:
    """Trả về kind: 'news'/'youtube'/'facebook'/'tiktok'/'threads'/None."""
    if not url or not url.startswith(('http://', 'https://')):
        return None
    if is_supported_news_url(url): return 'news'
    if is_youtube_url(url):         return 'youtube'
    if is_facebook_url(url):        return 'facebook'
    if is_tiktok_url(url):          return 'tiktok'
    if is_threads_url(url):         return 'threads'
    return 'news'   # generic fallback dùng trafilatura

def fetch(url: str, max_comments: int = 30) -> dict | None:
    """Dispatcher chính."""
    kind = detect_source(url)
    if kind == 'news':     return fetch_news_article(url)
    if kind == 'youtube':  return fetch_youtube(url, max_comments=max_comments)
    if kind == 'facebook': return fetch_facebook(url, max_comments=max_comments)
    if kind == 'tiktok':   return fetch_tiktok(url, max_comments=max_comments)
    if kind == 'threads':  return fetch_threads(url)
    return None
