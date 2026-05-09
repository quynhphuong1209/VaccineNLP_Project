import json
import os
from langdetect import detect, DetectorFactory
from src.common import paths

# Đảm bảo kết quả phát hiện ngôn ngữ ổn định
DetectorFactory.seed = 0

class VNDataReclaimer:
    """Lọc dữ liệu để chỉ giữ lại nội dung tiếng Việt"""
    
    def __init__(self):
        self.input_file = paths.PROCESSED_DATA_DIR / "corpus_1856_unlabeled.json"
        self.output_file = paths.GOLD_DATA_DIR / "reclaimed_master_pool_vn_clean.json"

    def is_vietnamese(self, text):
        if not text or len(text) < 10:
            return False
        try:
            return detect(text) == 'vi'
        except:
            return False

    def run(self):
        print(f"🔍 Đang lọc dữ liệu tiếng Việt từ {self.input_file}...")
        
        if not os.path.exists(self.input_file):
            print(f"⚠️ Không tìm thấy file đầu vào. Hãy chắc chắn bạn đã có dữ liệu ở {self.input_file}")
            return

        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vn_only_data = [item for item in data if self.is_vietnamese(item.get('text', ''))]
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(vn_only_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Đã lọc xong! Giữ lại {len(vn_only_data)}/{len(data)} bài viết tiếng Việt.")
        print(f"💾 Kết quả lưu tại: {self.output_file}")

if __name__ == "__main__":
    reclaimer = VNDataReclaimer()
    reclaimer.run()
