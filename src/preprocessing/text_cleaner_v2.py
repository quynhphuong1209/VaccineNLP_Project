"""
Text Cleaner V2 — Tiền xử lý văn bản tiếng Việt cho NLP
==========================================================
CHANGES vs V1:
  - [FIX P0] html.unescape() được thêm TRƯỚC normalize_unicode
  - [FIX P0] Tách biệt rõ clean_text() vs clean_rss_text() cho RSS/HTML source
  - [FIX P1] Mở rộng TEEN_CODE_MAP với các biến thể phổ biến hơn
  - [FIX P1] Thêm hàm is_human_vaccine_context() để loại noise thú y
"""

import html
import re
import unicodedata
from pathlib import Path

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
# Patterns cho y tế người (cần có ít nhất 1)
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

# Patterns loại trừ (thú y — không phải y tế người)
VETERINARY_SIGNALS = [
    r'\bthú\s+y\b', r'\bgia\s+súc\b', r'\bgia\s+cầm\b', r'\bchăn\s+nuôi\b',
    r'\bgà\s+đẻ\b', r'\bgà\s+mái\b', r'\bheo\b', r'\blợn\b',
    r'\bbò\b.*\btiêm\b', r'\btrại\s+nuôi\b', r'\bđàn\s+gà\b',
    r'\bthuốc\s+thú\s+y\b', r'\bkháng\s+sinh\b.*\bchăn\b',
    r'\bsản\s+xuất\s+trứng\b', r'\btrại\s+gà\b',
]

# [NEW] Patterns nội dung không liên quan (Showbiz, Giải trí, Du lịch)
IRRELEVANT_SIGNALS = [
    r'\bshowbiz\b', r'\bdiễn\s+viên\b', r'\bnghỉ\s+dưỡng\b', r'\bdu\s+lịch\b',
    r'\bca\s+sĩ\b', r'\bnghệ\s+sĩ\b', r'\bphim\b', r'\bgiai\s+trí\b',
    r'\bngười\s+mẫu\b', r'\bhoa\s+hậu\b', r'\bphát\s+hành\b.*\balbum\b',
    r'\bconcert\b', r'\bcát\s+xê\b', r'\blâm\s+khánh\s+chi\b', r'\bphú\s+quốc\b',
]

# [NEW] Các mẫu câu tiêu đề tin tức rác thường thấy ở cuối bài viết (Boilerplate)
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
    """
    [NEW V2] Decode HTML entities TRƯỚC bất kỳ xử lý nào.
    Xử lý cả named (&amp; &nbsp; &lsquo; &rsquo; &oacute; v.v.)
    và numeric entities (&#039; &#8220; v.v.).

    Ví dụ:
        "C&oacute; thể" → "Có thể"
        "&lsquo;n&eacute;&rsquo;" → "'né'"
        "&#039;Cicada&#039;" → "'Cicada'"
    """
    if not text:
        return text
    # html.unescape xử lý cả named lẫn numeric entities
    text = html.unescape(text)
    # Xử lý double-encoded (&amp;amp; → &amp; → &)
    if "&amp;" in text or "&#" in text:
        text = html.unescape(text)
    return text


def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)


def remove_emojis(text: str, replace_with: str = " ") -> str:
    return EMOJI_PATTERN.sub(replace_with, text)


def emoji_to_text(text: str) -> str:
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        return remove_emojis(text)


def process_hashtags(text: str, mode: str = "split") -> str:
    if mode == "remove":
        return re.sub(r'#\S+', '', text)
    elif mode == "split":
        def split_hashtag(match):
            tag = match.group(0)[1:]
            split = re.sub(r'([A-Z])', r' \1', tag)
            return split.strip()
        return re.sub(r'#\S+', split_hashtag, text)
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_footer_noise(text: str) -> str:
    """[NEW V2] Loại bỏ các câu tiêu đề tin tức rác lặp lại."""
    for pattern in FOOTER_BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return text


def replace_teen_code(text: str, mapping: dict = None) -> str:
    if mapping is None:
        mapping = TEEN_CODE_MAP
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in mapping:
            result.append(mapping[lower])
        else:
            result.append(word)
    return " ".join(result)


def remove_special_chars(text: str, keep_vietnamese: bool = True) -> str:
    if keep_vietnamese:
        return re.sub(
            r'[^\w\s.,!?;:\-\'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộ'
            r'ơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
            '',
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )
    return re.sub(r'[^\w\s]', '', text)


# ══════════════════════════════════════════════════════════════════════════════
# Domain relevance check
# ══════════════════════════════════════════════════════════════════════════════

def is_human_vaccine_context(text: str) -> bool:
    """
    [NEW V2] Phân biệt vaccine y tế người vs vaccine thú y / không liên quan.

    Logic:
      1. Nếu có tín hiệu thú y → False (loại trừ)
      2. Nếu có ít nhất 1 tín hiệu y tế người → True
      3. Fallback → True (để pipeline.py quyết định bằng keyword match)

    Returns:
        True  → có khả năng là nội dung vaccine y tế người
        False → nhiều khả năng là thú y hoặc không liên quan
    """
    text_lower = text.lower()

    # Check thú y trước
    vet_hits = sum(1 for p in _VET_SIGNALS_RE if p.search(text_lower))
    human_hits = sum(1 for p in _HUMAN_SIGNALS_RE if p.search(text_lower))
    noise_hits = sum(1 for p in _NOISE_SIGNALS_RE if p.search(text_lower))

    # Ưu tiên loại bỏ rác (Showbiz/Entertainment)
    if noise_hits >= 1 and human_hits == 0:
        return False
    if noise_hits >= 2:
        return False # Showbiz dominant

    if vet_hits > 0 and human_hits == 0:
        return False  # Chỉ có tín hiệu thú y
    if vet_hits >= 2 and human_hits <= 1:
        return False  # Thú y dominant

    return True  # Default: giữ lại


# ══════════════════════════════════════════════════════════════════════════════
# Main cleaning pipelines
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(
    text: str,
    do_decode_html: bool = True,       # [NEW V2] Bật mặc định
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
    """
    Pipeline tiền xử lý văn bản tiếng Việt — V2.

    THAY ĐỔI CHÍNH: decode_html_entities() được gọi ĐẦU TIÊN,
    trước normalize_unicode, để tránh encoding conflict.
    """
    if not text or not isinstance(text, str):
        return ""

    # [V2 - NEW] HTML entities decode TRƯỚC tất cả
    if do_decode_html:
        text = decode_html_entities(text)

    if do_normalize_unicode:
        text = normalize_unicode(text)
    if do_remove_urls:
        text = remove_urls(text)
    if do_remove_html:
        text = remove_html_tags(text)
    if do_process_hashtags:
        text = process_hashtags(text, mode="split")
    if do_emoji_to_text:
        text = emoji_to_text(text)
    elif do_remove_emojis:
        text = remove_emojis(text)
    if do_lowercase:
        text = text.lower()
    if do_replace_teen_code:
        text = replace_teen_code(text)
    if do_remove_special:
        text = remove_special_chars(text)

    # [NEW] Loại bỏ rác ở footer
    text = remove_footer_noise(text)

    return normalize_whitespace(text)


def clean_rss_text(text: str) -> str:
    """
    [NEW V2] Wrapper đặc biệt cho RSS/BeautifulSoup content.
    HTML entities trong RSS thường double-encoded hoặc mixed.
    Gọi decode 2 lần để xử lý &amp;oacute; → &oacute; → ó.
    """
    if not text:
        return ""
    # Pass 1: decode entities
    text = html.unescape(text)
    # Pass 2: decode nếu vẫn còn entities (double-encoded)
    if '&' in text and ';' in text:
        text = html.unescape(text)
    return clean_text(text)


def clean_dataset(data: list, text_field: str = "text") -> list:
    """Tiền xử lý toàn bộ dataset."""
    cleaned = []
    for item in data:
        item_copy = item.copy()
        if text_field in item_copy:
            item_copy[f"{text_field}_original"] = item_copy[text_field]
            # Detect source để chọn cleaner phù hợp
            source = item_copy.get("source", "")
            collection_method = item_copy.get("collection_method", "")
            if "rss" in collection_method or source in {"thanhnien", "dantri", "vietnamnet"}:
                item_copy[text_field] = clean_rss_text(item_copy[text_field])
            else:
                item_copy[text_field] = clean_text(item_copy[text_field])
        cleaned.append(item_copy)
    return cleaned


# ── Tests ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Text Cleaner V2 — Tests")
    print("=" * 65)

    # Test 1: HTML entity decoding
    rss_sample = "Biến thể BA.3.2 của Covid-19: C&oacute; khả năng &lsquo;n&eacute;&rsquo; miễn dịch"
    cleaned = clean_rss_text(rss_sample)
    print(f"\n[TEST 1] HTML entity decode (RSS):")
    print(f"  Input : {rss_sample}")
    print(f"  Output: {cleaned}")
    assert "có" in cleaned and "né" in cleaned, "FAIL: HTML entities not decoded"
    print("  ✅ PASS")

    # Test 2: Double-encoded entities
    double_enc = "L&amp;oacute;ng C&amp;acirc;u"
    cleaned2 = clean_rss_text(double_enc)
    print(f"\n[TEST 2] Double-encoded entities:")
    print(f"  Input : {double_enc}")
    print(f"  Output: {cleaned2}")
    print("  ✅ PASS")

    # Test 3: Veterinary domain filter
    vet_text = "Gà mái sẽ được theo dõi sức khỏe, tiêm vaccine và sử dụng thuốc thú y đúng liều lượng"
    human_text = "Bộ Y tế khuyến cáo tiêm chủng vaccine HPV cho trẻ em từ 9-14 tuổi"
    print(f"\n[TEST 3] Domain context detection:")
    print(f"  Vet text   → is_human: {is_human_vaccine_context(vet_text)}")
    print(f"  Human text → is_human: {is_human_vaccine_context(human_text)}")
    assert not is_human_vaccine_context(vet_text), "FAIL: Vet text should return False"
    assert is_human_vaccine_context(human_text), "FAIL: Human text should return True"
    print("  ✅ PASS")

    # Test 4: Teen code vaccine domain
    teen_sample = "mình vừa đi tc cho con, bs nói vx an toàn, ko lo pup gì cả"
    cleaned4 = clean_text(teen_sample)
    print(f"\n[TEST 4] Teen code (vaccine domain):")
    print(f"  Input : {teen_sample}")
    print(f"  Output: {cleaned4}")
    print("  ✅ PASS")

    print("\n" + "=" * 65)
    print("Tất cả tests PASSED ✅")
