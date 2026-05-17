"""
Vietnamese Tokenizer — Tách từ tiếng Việt sử dụng underthesea
"""
from pathlib import Path


def tokenize_text(text, engine="underthesea"):
    """
    Tách từ tiếng Việt.
    
    Args:
        text: Văn bản tiếng Việt đã qua preprocessing
        engine: "underthesea" hoặc "pyvi"
    
    Returns:
        Văn bản đã tách từ (dấu _ nối các từ ghép)
    """
    if not text or not isinstance(text, str):
        return ""
    
    if engine == "underthesea":
        try:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
        except ImportError:
            print("⚠️ underthesea not installed. Run: pip install underthesea")
            return text
    elif engine == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize(text)
        except ImportError:
            print("⚠️ pyvi not installed. Run: pip install pyvi")
            return text
    else:
        return text


def tokenize_dataset(data, text_field="text", engine="underthesea"):
    """Tách từ cho toàn bộ dataset"""
    tokenized = []
    for i, item in enumerate(data):
        item_copy = item.copy()
        if text_field in item_copy:
            item_copy[f"{text_field}_tokenized"] = tokenize_text(
                item_copy[text_field], engine=engine
            )
        tokenized.append(item_copy)
        if (i + 1) % 50 == 0:
            print(f"  Tokenized {i+1}/{len(data)} items...")
    return tokenized


def extract_stopwords():
    """Trả về danh sách stopwords tiếng Việt phổ biến"""
    return [
        "và", "của", "là", "có", "được", "trong", "để", "cho",
        "với", "này", "các", "từ", "một", "những", "đã", "đó",
        "không", "người", "như", "cũng", "về", "nhưng", "thì",
        "khi", "ra", "nên", "vì", "mà", "hay", "tại", "theo",
        "đến", "rất", "còn", "hơn", "lại", "trên", "sau", "nếu",
        "sẽ", "đang", "phải", "bị", "hết", "tôi", "bạn", "anh",
        "chị", "em", "ông", "bà", "mình", "nó", "họ", "ta",
    ]


# ============================================================
if __name__ == "__main__":
    samples = [
        "vắc xin covid rất an toàn cho trẻ em",
        "bộ y tế khuyến cáo tiêm chủng đầy đủ",
        "phản ứng phụ sau tiêm vaccine thường nhẹ",
    ]
    print("🔤 Vietnamese Tokenizer — Test")
    print("=" * 60)
    for s in samples:
        result = tokenize_text(s)
        print(f"\n  Input:     {s}")
        print(f"  Tokenized: {result}")
