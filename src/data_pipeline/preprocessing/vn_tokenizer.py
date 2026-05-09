# pyrefly: ignore [missing-import]
from pyvi import ViTokenizer

class VietnameseTokenizer:
    """Module tách từ (Word Segmentation) dành riêng cho tiếng Việt"""
    
    def tokenize(self, text):
        if not text:
            return ""
        
        # Sử dụng pyvi để tách từ
        # Kết quả: "học sinh học sinh học" -> "học_sinh học sinh_học"
        return ViTokenizer.tokenize(text)

if __name__ == "__main__":
    tokenizer = VietnameseTokenizer()
    test_text = "Dự án phân loại thông tin về vắc xin đang được triển khai."
    print(f"Gốc: {test_text}")
    print(f"Tách: {tokenizer.tokenize(test_text)}")
