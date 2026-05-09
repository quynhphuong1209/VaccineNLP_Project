import re
import unicodedata

class TextCleaner:
    """Module làm sạch văn bản tiếng Việt chuyên sâu"""
    
    def __init__(self):
        # Regex để tìm link web, email, và các ký tự đặc biệt
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<.*?>')

    def normalize_unicode(self, text):
        """Đưa văn bản về chuẩn Unicode dựng sẵn (NFC) để tránh lỗi font tiếng Việt"""
        return unicodedata.normalize('NFC', text)

    def clean(self, text):
        if not text:
            return ""
        
        # 1. Chuyển về chữ thường
        text = text.lower()
        
        # 2. Xóa mã HTML
        text = self.html_pattern.sub(r'', text)
        
        # 3. Xóa Link web và Email
        text = self.url_pattern.sub(r'', text)
        text = self.email_pattern.sub(r'', text)
        
        # 4. Chuẩn hóa Unicode
        text = self.normalize_unicode(text)
        
        # 5. Xóa các ký tự thừa, khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    cleaner = TextCleaner()
    test_text = "Vắc-xin này rất tốt! <br> Xem thêm tại: https://example.com 😊"
    print(f"Trước: {test_text}")
    print(f"Sau:   {cleaner.clean(test_text)}")
