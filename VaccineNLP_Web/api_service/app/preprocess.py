import html
import re
import unicodedata
from typing import Dict

# ── Emoji pattern ──────────────────────────────────────────────────────────────
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# ── Teen code (mở rộng) ────────────────────────────────────────────────────────
TEEN_CODE_MAP = {
    # Phủ định
    "ko": "không", "k": "không", "hk": "không", "hem": "không", "kg": "không",
    "kp": "không phải", "kh": "không",
    # Đại từ
    "mk": "mình", "mik": "mình", "mh": "mình",
    "bn": "bạn", "ad": "admin",
    # Thời gian
    "bh": "bây giờ", "bjo": "bây giờ",
    # Động từ
    "dc": "được", "đc": "được", "đk": "được",
    "ns": "nói", "nc": "nói chuyện",
    "lm": "làm", "ms": "mới", "oy": "rồi", "r": "rồi",
    # Y tế - VACCINE DOMAIN SPECIFIC
    "vx": "vắc xin", "vc": "vaccine", "vcn": "vaccine",
    "bs": "bác sĩ", "bv": "bệnh viện", "bvien": "bệnh viện",
    "yt": "y tế", "sk": "sức khỏe",
    "pup": "phản ứng phụ", "tdp": "tác dụng phụ",
    "tc": "tiêm chủng", "tp": "tiêm phòng",
    # Cảm xúc / thái độ
    "cx": "cũng", "j": "gì", "z": "vậy", "v": "vậy",
    "ok": "được", "oke": "được", "okie": "được",
    "ah": "à", "nha": "nhé", "nhe": "nhé",
    "tks": "cảm ơn", "camon": "cảm ơn",
    "xl": "xin lỗi", "klq": "không liên quan",
    "bt": "bình thường", "bth": "bình thường",
    "tl": "trả lời", "rep": "trả lời",
    "ck": "chồng", "vk": "vợ", "gd": "gia đình",
}

FOOTER_BOILERPLATE_PATTERNS = [
    r'sắp\s+có\s+thuốc\s+khiến\s+tế\s+bào\s+ung\s+thư\s+biến\s+mất',
    r'mắc\s+những\s+bệnh\s+này,\s+bạn\s+tuyệt\s+đối\s+không\s+nên\s+ăn\s+thịt\s+bò',
    r'thai\s+phụ\s+nguy\s+hiểm\s+khi\s+cùng\s+mắc\s+sốt\s+xuất\s+huyết\s+và\s+cúm\s+a',
    r'chuyên\s+gia\s+sẽ\s+giải\s+đáp\s+chi\s+tiết',
    r'xem\s+thêm\b', r'tin\s+liên\s+quan\b', r'bài\s+viết\s+cùng\s+chủ\s+đề\b',
    r'đọc\s+thêm\b', r'click\s+để\s+xem\b', r'nhấn\s+vào\s+đây\b',
]

def decode_html_entities(text: str) -> str:
    if not text:
        return text
    text = html.unescape(text)
    if "&amp;" in text or "&#" in text:
        text = html.unescape(text)
    return text

def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)

def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)

def remove_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub(" ", text)

def emoji_to_text(text: str) -> str:
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        return remove_emojis(text)

def process_hashtags(text: str) -> str:
    def split_hashtag(match):
        tag = match.group(0)[1:]
        split = re.sub(r'([A-Z])', r' \1', tag)
        return split.strip()
    return re.sub(r'#\S+', split_hashtag, text)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_footer_noise(text: str) -> str:
    for pattern in FOOTER_BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text

def replace_teen_code(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in TEEN_CODE_MAP:
            result.append(TEEN_CODE_MAP[lower])
        else:
            result.append(word)
    return " ".join(result)

def remove_special_chars(text: str) -> str:
    return re.sub(
        r'[^\w\s.,!?;:\-\'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộ'
        r'ơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
        '',
        text,
        flags=re.IGNORECASE | re.UNICODE,
    )

def _clean(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = decode_html_entities(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = process_hashtags(text)
    text = emoji_to_text(text)
    text = text.lower()
    text = replace_teen_code(text)
    text = remove_special_chars(text)
    text = remove_footer_noise(text)
    return normalize_whitespace(text)

def prepare_text(text: str) -> str:
    cleaned = _clean(text)
    try:
        from underthesea import word_tokenize
        return word_tokenize(cleaned, format="text")
    except ImportError:
        print("⚠️ underthesea not installed. Word segmenter fallback to space separation.", flush=True)
        return cleaned
