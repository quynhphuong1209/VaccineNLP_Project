import torch
from transformers import AutoTokenizer
from src.modeling.phobert_multitask_trainer import PhoBertMultitaskTrainer
from src.data_pipeline.preprocessing.text_cleaner_v2 import TextCleaner
from src.data_pipeline.preprocessing.vn_tokenizer import VietnameseTokenizer

class VaccineInferenceAPI:
    """API hợp nhất để thực hiện dự đoán cho bất kỳ văn bản nào"""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer_vn = VietnameseTokenizer()
        # Giả lập load model (trong thực tế sẽ load từ paths.MODEL_DIR)
        print("🤖 Hệ thống Inference đã sẵn sàng.")

    def predict_all(self, raw_text):
        # 1. Làm sạch & Tách từ
        cleaned = self.cleaner.clean(raw_text)
        tokenized = self.tokenizer_vn.tokenize(cleaned)
        
        # 2. Dự đoán (Giả lập kết quả dựa trên logic đơn giản cho bản demo)
        print(f"🔮 Đang phân tích: {tokenized}")
        
        # Kết quả mẫu
        return {
            "text": raw_text,
            "predictions": {
                "misinfo": "Tin giả" if "thuốc độc" in cleaned else "Không phải tin giả",
                "stance": "Phản đối" if "không tiêm" in cleaned else "Ủng hộ",
                "sentiment": "Tiêu cực" if "sợ" in cleaned else "Tích cực"
            },
            "confidence": 0.98
        }

if __name__ == "__main__":
    api = VaccineInferenceAPI()
    result = api.predict_all("Đừng tiêm vaccine, nó là thuốc độc đấy!")
    print(result)
