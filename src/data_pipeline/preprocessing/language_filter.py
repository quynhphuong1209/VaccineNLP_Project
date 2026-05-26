"""
Language Filter — Phát hiện ngôn ngữ và lọc texts theo ngôn ngữ
Sử dụng langdetect hoặc regex-based fallback
"""
import re
from pathlib import Path


# Vietnamese character ranges (Unicode)
VN_CHARS = re.compile(
    r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
    re.IGNORECASE
)


def detect_language(text, engine="regex"):
    """
    Phát hiện ngôn ngữ của văn bản.
    
    Args:
        text: Văn bản cần phát hiện
        engine: "langdetect" hoặc "regex" (fallback)
    
    Returns:
        str — Mã ngôn ngữ (vi, en, unknown)
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return "unknown"
    
    if engine == "langdetect":
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42  # Reproducible results
            lang = detect(text)
            return lang
        except ImportError:
            print("⚠️ langdetect not installed. Run: pip install langdetect")
            return detect_language(text, engine="regex")
        except Exception:
            return "unknown"
    
    # Regex-based: check for Vietnamese characters
    vn_matches = len(VN_CHARS.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    
    if total_alpha == 0:
        return "unknown"
    
    vn_ratio = vn_matches / total_alpha
    
    if vn_ratio > 0.05:  # >5% Vietnamese chars → likely Vietnamese
        return "vi"
    elif re.search(r'[a-zA-Z]', text):
        return "en"
    else:
        return "unknown"


def filter_by_language(texts, target_lang="vi", engine="regex"):
    """
    Lọc danh sách texts theo ngôn ngữ.
    
    Args:
        texts: List[str] hoặc List[dict] (với key 'text')
        target_lang: Ngôn ngữ mục tiêu ("vi", "en")
        engine: "langdetect" hoặc "regex"
    
    Returns:
        Tuple[List, List] — (matched, filtered_out)
    """
    matched = []
    filtered_out = []
    
    for item in texts:
        if isinstance(item, dict):
            text = item.get("text", "") or item.get("cleaned_text", "") or ""
        else:
            text = str(item)
        
        lang = detect_language(text, engine=engine)
        
        if isinstance(item, dict):
            item_copy = item.copy()
            item_copy["detected_language"] = lang
            if lang == target_lang:
                matched.append(item_copy)
            else:
                filtered_out.append(item_copy)
        else:
            if lang == target_lang:
                matched.append(item)
            else:
                filtered_out.append(item)
    
    return matched, filtered_out


def split_by_language(data, text_field="text", engine="regex"):
    """
    Tách dataset thành các nhóm ngôn ngữ.
    
    Returns:
        Dict[str, List] — {"vi": [...], "en": [...], "unknown": [...]}
    """
    result = {}
    
    for item in data:
        if isinstance(item, dict):
            text = item.get(text_field, "")
        else:
            text = str(item)
        
        lang = detect_language(text, engine=engine)
        
        if lang not in result:
            result[lang] = []
        
        if isinstance(item, dict):
            item_copy = item.copy()
            item_copy["detected_language"] = lang
            result[lang].append(item_copy)
        else:
            result[lang].append(item)
    
    # Summary
    print(f"📊 Language Distribution:")
    for lang, items in sorted(result.items()):
        print(f"  {lang}: {len(items)} items")
    
    return result


# ============================================================
if __name__ == "__main__":
    samples = [
        "Vaccine COVID rất an toàn cho trẻ em",
        "The vaccine is safe and effective",
        "Vắc xin bảo vệ cộng đồng khỏi bệnh truyền nhiễm",
        "I got my booster shot yesterday",
        "12345",
        "Tiêm chủng đầy đủ để bảo vệ sức khỏe",
    ]
    
    print("🌐 Language Filter — Test")
    print("=" * 60)
    for s in samples:
        lang = detect_language(s)
        print(f"  [{lang:3s}] {s[:60]}")
    
    print("\n📊 Filter for Vietnamese:")
    vi_texts, other = filter_by_language(samples, target_lang="vi")
    print(f"  Vietnamese: {len(vi_texts)} | Other: {len(other)}")
