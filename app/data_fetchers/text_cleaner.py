"""
Text Cleaner — Rút gọn từ src/data_pipeline/preprocessing/text_cleaner_v2.py
Chỉ giữ 4 hàm cần thiết cho App: decode_html_entities, clean_text, clean_rss_text, is_human_vaccine_context
GIỮ NGUYÊN logic gốc.
"""

import html
import re
import unicodedata

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

# ── Domain context patterns ────────────────────────────────────────────────────
HUMAN_HEALTH_SIGNALS = [
    r'\btiêm\s+chủng\b', r'\btiêm\s+phòng\b', r'\bvắc\s*xin\s+covid\b',
    r'\bvaccine\s+covid\b', r'\bphản\s+ứng\s+phụ\b', r'\btác\s+dụng\s+phụ\b',
    r'\bchương\s+trình\s+tiêm\b', r'\bmiễn\s+dịch\s+cộng\s+đồng\b',
    r'\bbộ\s+y\s+tế\b', r'\btrẻ\s+em\b', r'\btrẻ\s+sơ\s+sinh\b',
    r'\bngười\s+lớn\b', r'\bsởi\b', r'\bhpv\b', r'\bbạch\s+hầu\b',
    r'\bviêm\s+gan\b', r'\buốn\s+ván\b', r'\bho\s+gà\b', r'\bbại\s+liệt\b',
    r'\bphế\s+cầu\b', r'\brotavirus\b', r'\bthủy\s+đậu\b',
    r'\bkháng\s+thể\b', r'\bmiễn\s+dịch\b.*\bngười\b',
    r'\bantivax\b', r'\bchống\s+vaccine\b', r'\btai\s+biến\b',
    r'\bsốc\s+phản\s+vệ\b', r'\bgói\s+tiêm\b', r'\blịch\s+tiêm\b',
]

VETERINARY_SIGNALS = [
    r'\bthú\s+y\b', r'\bgia\s+súc\b', r'\bgia\s+cầm\b', r'\bchăn\s+nuôi\b',
    r'\bgà\s+đẻ\b', r'\bgà\s+mái\b', r'\bheo\b', r'\blợn\b',
    r'\bbò\b.*\btiêm\b', r'\btrại\s+nuôi\b', r'\bđàn\s+gà\b',
    r'\bthuốc\s+thú\s+y\b', r'\bkháng\s+sinh\b.*\bchăn\b',
    r'\bsản\s+xuất\s+trứng\b', r'\btrại\s+gà\b',
]

IRRELEVANT_SIGNALS = [
    r'\bshowbiz\b', r'\bdiễn\s+viên\b', r'\bnghỉ\s+dưỡng\b', r'\bdu\s+lịch\b',
    r'\bca\s+sĩ\b', r'\bnghệ\s+sĩ\b', r'\bphim\b', r'\bgiai\s+trí\b',
    r'\bngười\s+mẫu\b', r'\bhoa\s+hậu\b', r'\bphát\s+hành\b.*\balbum\b',
    r'\bconcert\b', r'\bcát\s+xê\b', r'\blâm\s+khánh\s+chi\b', r'\bphú\s+quốc\b',
]

FOOTER_BOILERPLATE_PATTERNS = [
    r'sắp\s+có\s+thuốc\s+khiến\s+tế\s+bào\s+ung\s+thư\s+biến\s+mất',
    r'mắc\s+những\s+bệnh\s+này,\s+bạn\s+tuyệt\s+đối\s+không\s+nên\s+ăn\s+thịt\s+bò',
    r'thai\s+phụ\s+nguy\s+hiểm\s+khi\s+cùng\s+mắc\s+sốt\s+xuất\s+huyết\s+và\s+cúm\s+a',
    r'chuyên\s+gia\s+sẽ\s+giải\s+đáp\s+chi\s+tiết',
    r'xem\s+thêm\b', r'tin\s+liên\s+quan\b', r'bài\s+viết\s+cùng\s+chủ\s+đề\b',
    r'đọc\s+thêm\b', r'click\s+để\s+xem\b', r'nhấn\s+vào\s+đây\b',
]

_HUMAN_SIGNALS_RE = [re.compile(p, re.IGNORECASE) for p in HUMAN_HEALTH_SIGNALS]
_VET_SIGNALS_RE = [re.compile(p, re.IGNORECASE) for p in VETERINARY_SIGNALS]
_NOISE_SIGNALS_RE = [re.compile(p, re.IGNORECASE) for p in IRRELEVANT_SIGNALS]


# ══════════════════════════════════════════════════════════════════════════════
# Core cleaning functions
# ══════════════════════════════════════════════════════════════════════════════

def decode_html_entities(text: str) -> str:
    """Decode HTML entities TRƯỚC bất kỳ xử lý nào."""
    if not text:
        return text
    text = html.unescape(text)
    if "&amp;" in text or "&#" in text:
        text = html.unescape(text)
    return text


def _remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def _remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)


def _remove_emojis(text: str, replace_with: str = " ") -> str:
    return EMOJI_PATTERN.sub(replace_with, text)


def _emoji_to_text(text: str) -> str:
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        return _remove_emojis(text)


def _process_hashtags(text: str) -> str:
    def split_hashtag(match):
        tag = match.group(0)[1:]
        split = re.sub(r'([A-Z])', r' \1', tag)
        return split.strip()
    return re.sub(r'#\S+', split_hashtag, text)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _remove_footer_noise(text: str) -> str:
    for pattern in FOOTER_BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def _replace_teen_code(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in TEEN_CODE_MAP:
            result.append(TEEN_CODE_MAP[lower])
        else:
            result.append(word)
    return " ".join(result)


def _remove_special_chars(text: str) -> str:
    return re.sub(
        r'[^\w\s.,!?;:\-\'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộ'
        r'ơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
        '',
        text,
        flags=re.IGNORECASE | re.UNICODE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Domain relevance check
# ══════════════════════════════════════════════════════════════════════════════

def is_human_vaccine_context(text: str) -> bool:
    """
    Phân biệt vaccine y tế người vs vaccine thú y / không liên quan.
    Logic:
      1. Nếu có tín hiệu thú y → False
      2. Nếu có ít nhất 1 tín hiệu y tế người → True
      3. Fallback → True
    """
    text_lower = text.lower()
    vet_hits = sum(1 for p in _VET_SIGNALS_RE if p.search(text_lower))
    human_hits = sum(1 for p in _HUMAN_SIGNALS_RE if p.search(text_lower))
    noise_hits = sum(1 for p in _NOISE_SIGNALS_RE if p.search(text_lower))

    if noise_hits >= 1 and human_hits == 0:
        return False
    if noise_hits >= 2:
        return False
    if vet_hits > 0 and human_hits == 0:
        return False
    if vet_hits >= 2 and human_hits <= 1:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main cleaning pipelines
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(
    text: str,
    do_decode_html: bool = True,
    do_lowercase: bool = True,
    do_remove_urls: bool = True,
    do_remove_html: bool = True,
    do_emoji_to_text: bool = True,
    do_remove_emojis: bool = False,
    do_normalize_unicode: bool = True,
    do_replace_teen_code: bool = True,
    do_remove_special: bool = True,
    do_process_hashtags: bool = True,
) -> str:
    """Pipeline tiền xử lý văn bản tiếng Việt — V2."""
    if not text or not isinstance(text, str):
        return ""
    if do_decode_html:
        text = decode_html_entities(text)
    if do_normalize_unicode:
        text = _normalize_unicode(text)
    if do_remove_urls:
        text = _remove_urls(text)
    if do_remove_html:
        text = _remove_html_tags(text)
    if do_process_hashtags:
        text = _process_hashtags(text)
    if do_emoji_to_text:
        text = _emoji_to_text(text)
    elif do_remove_emojis:
        text = _remove_emojis(text)
    if do_lowercase:
        text = text.lower()
    if do_replace_teen_code:
        text = _replace_teen_code(text)
    if do_remove_special:
        text = _remove_special_chars(text)
    text = _remove_footer_noise(text)
    return _normalize_whitespace(text)


def clean_rss_text(text: str) -> str:
    """Wrapper cho RSS/BeautifulSoup content (double-encoded entities)."""
    if not text:
        return ""
    text = html.unescape(text)
    if '&' in text and ';' in text:
        text = html.unescape(text)
    return clean_text(text)
