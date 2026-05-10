import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from src.common import paths
from src.modeling.phobert_multitask_trainer import VaccineMultitaskModel

class VaccineInferenceAPI:
    """Unified API for real-time inference using the trained PhoBERT Multitask model"""
    
    def __init__(self, model_version="phobert-multitask-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = paths.MODEL_DIR / model_version / "pytorch_model.bin"
        self.tokenizer_name = "vinai/phobert-base"
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Load Model
        self.model = VaccineMultitaskModel(model_name=self.tokenizer_name)
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"✅ Loaded model weights from: {self.model_path}")
        else:
            print(f"⚠️ Model weights not found at {self.model_path}. Running with random initialization (for testing).")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Taxonomy mapping
        self.taxonomy = {
            'misinfo': {1: "Tin giả", 2: "Chính xác", 0: "Không liên quan"},
            'stance': {1: "Ủng hộ", 2: "Phản đối", 0: "Trung lập", 3: "Không rõ"},
            'sentiment': {2: "Tiêu cực", 0: "Trung tính", 1: "Tích cực"}
        }

    def predict(self, raw_text):
        # 1. Preprocess (Word Segmentation)
        segmented_text = ViTokenizer.tokenize(raw_text)
        
        # 2. Tokenize
        inputs = self.tokenizer(
            segmented_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        ).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            
            # Extract Probabilities
            p_m = F.softmax(logits['misinfo'], dim=1)[0]
            p_st = F.softmax(logits['stance'], dim=1)[0]
            p_se = F.softmax(logits['sentiment'], dim=1)[0]
            
        # 4. Get Labels and Confidence
        m_idx = p_m.argmax().item()
        st_idx = p_st.argmax().item()
        se_idx = p_se.argmax().item()
        
        return {
            "text": raw_text,
            "segmented_text": segmented_text,
            "results": {
                "misinfo": {
                    "label": self.taxonomy['misinfo'].get(m_idx, "N/A"),
                    "confidence": p_m[m_idx].item()
                },
                "stance": {
                    "label": self.taxonomy['stance'].get(st_idx, "N/A"),
                    "confidence": p_st[st_idx].item()
                },
                "sentiment": {
                    "label": self.taxonomy['sentiment'].get(se_idx, "N/A"),
                    "confidence": p_se[se_idx].item()
                }
            }
        }

if __name__ == "__main__":
    api = VaccineInferenceAPI()
    test_text = "Tiêm vaccine là cách tốt nhất để bảo vệ sức khỏe cộng đồng."
    result = api.predict(test_text)
    print(f"📥 Input: {test_text}")
    print(f"🔮 Prediction: {result['results']}")
