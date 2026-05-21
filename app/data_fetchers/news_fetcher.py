"""
News fetcher — Tầng 1 (Instant: 1-3s)
Hỗ trợ 15+ báo VN qua trafilatura
"""
import re
import requests
from urllib.parse import urlparse
from datetime import datetime
import streamlit as st

SUPPORTED_DOMAINS = {
    'vnexpress.net', 'tuoitre.vn', 'dantri.com.vn', 'thanhnien.vn',
    'vietnamnet.vn', 'vietnamplus.vn', 'suckhoedoisong.vn', 'laodong.vn',
    'tienphong.vn', 'znews.vn', 'hanoimoi.vn', 'baochinhphu.vn',
    'vtv.vn', 'vov.vn', 'soha.vn', 'kenh14.vn', 'cafef.vn',
    'nhandan.vn', 'sggp.org.vn', 'qdnd.vn',
}

def is_supported_news_url(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        return any(d in domain for d in SUPPORTED_DOMAINS)
    except Exception:
        return False

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_article(url: str, timeout: int = 10) -> dict | None:
    """Lấy bài báo bằng trafilatura. Cache 1 giờ."""
    try:
        import trafilatura
    except ImportError:
        return {'error': 'trafilatura chưa cài', 'url': url}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        }
        # Tải bằng requests trước để tránh bot-block và redirect không mong muốn
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.encoding = 'utf-8'
            downloaded = r.text if r.status_code == 200 else None
        except Exception:
            downloaded = None
            
        # Fallback dùng trafilatura.fetch_url nếu requests fail
        if not downloaded:
            downloaded = trafilatura.fetch_url(url)
        
        import json as _json
        
        title = ""
        text = ""
        author = None
        date = None
        fetcher_name = "trafilatura"
        
        result_json = trafilatura.extract(
            downloaded,
            output_format='json',
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
        )
        
        if result_json:
            try:
                data = _json.loads(result_json)
                text = (data.get('text') or '').strip()
                title = (data.get('title') or '').strip()
                author = data.get('author')
                date = data.get('date')
            except Exception:
                pass

        # Fallback bằng BeautifulSoup nếu trafilatura thất bại hoặc văn bản quá ngắn
        if len(text) < 150:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(downloaded, 'html.parser')
                
                # Trích xuất tiêu đề
                title_tag = soup.find('h1') or soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ""
                
                # Loại bỏ các thành phần không mong muốn
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
                    tag.decompose()
                
                # Trích xuất các thẻ p
                paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
                # Lọc các dòng văn bản rác hoặc quá ngắn
                valid_p = [p for p in paragraphs if len(p.split()) > 5]
                text = "\n\n".join(valid_p)
                fetcher_name = "beautifulsoup-fallback"
            except Exception as e:
                pass
                
        if len(text) < 100:
            return {'error': 'Nội dung trích xuất quá ngắn (<100 ký tự)', 'url': url}
            
        domain = urlparse(url).netloc.lower().replace('www.', '')
        return {
            'kind':         'news',
            'title':        title,
            'text':         text,
            'author':       author,
            'date':         date,
            'source':       domain,
            'url':          url,
            'word_count':   len(text.split()),
            'comments':     [],
            'extracted_at': datetime.now().isoformat(),
            'fetcher':      fetcher_name,
        }
    except Exception as e:
        return {'error': str(e), 'url': url}
