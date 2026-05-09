import os
import json
from src.common import paths
from src.data_pipeline.preprocessing.text_cleaner_v2 import TextCleaner
from src.data_pipeline.preprocessing.vn_tokenizer import VietnameseTokenizer

class PreprocessingPipeline:
    """Dây chuyền tiền xử lý dữ liệu: Load -> Clean -> Tokenize -> Save"""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer = VietnameseTokenizer()
        self.raw_dir = paths.RAW_DATA_DIR
        self.output_dir = paths.DATA_DIR / "02_processed"
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, input_filename, output_filename):
        input_path = self.raw_dir / input_filename
        output_path = self.output_dir / output_filename
        
        print(f"🧹 Đang xử lý file: {input_filename}...")
        
        processed_data = []
        
        # Giả định dữ liệu đầu vào là JSON list
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for item in raw_data:
                text = item.get('text', '')
                # Bước 1: Làm sạch
                cleaned_text = self.cleaner.clean(text)
                # Bước 2: Tách từ
                tokenized_text = self.tokenizer.tokenize(cleaned_text)
                
                item['text_processed'] = tokenized_text
                processed_data.append(item)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Đã lưu dữ liệu sạch tại: datasets/02_processed/{output_filename}")
        else:
            print(f"⚠️ Không tìm thấy file đầu vào: {input_path}")

    def run_full_pipeline(self):
        # Ví dụ xử lý một file mặc định
        self.process_file("raw_posts.json", "cleaned_posts.json")

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run_full_pipeline()
