import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from preprocessing.vn_tokenizer import tokenize_text
from modeling.phobert_multitask_trainer import VaccineMultitaskModel
from src.common.paths import MODELS_DIR, CONFIGS_DIR, ensure_src_in_sys_path
ensure_src_in_sys_path()

CORE_VACCINE_KEYWORDS = [
    "vaccine", "vắc xin", "vắc-xin", "tiêm", "mũi", "phế cầu", "astra", 
    "pfizer", "vnvc", "phản vệ", "sốt", "đề kháng", "chích", "ngừa",
    "covid", "covid-19", "sars-cov-2", "corona", "dịch bệnh", "f0"
]

class VaccineInferencePipeline:
    """
    Decoupled Inference Engine for the VaccineNLP Multi-task model.
    Handles Preprocessing -> Forward Pass -> Results Mapping.
    """
    def __init__(self, model_dir=None, device=None):
        self.configs_dir = CONFIGS_DIR
        # Default to a subdir in MODELS_DIR if no path provided
        if model_dir is None:
            model_dir = MODELS_DIR / "production_v1"

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.taxonomy = self._load_taxonomy()
        
        # Initialize model and tokenizer (Baseline model has 2 misinfo, 3 stance, 3 sentiment classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = VaccineMultitaskModel(num_misinfo=3, num_stance=4, num_sentiment=3)
        
        weights_path = Path(model_dir) / "pytorch_model.bin"
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

    def _load_taxonomy(self):
        tax_path = self.configs_dir / "taxonomy.json"
        if not tax_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at: {tax_path}")
            
        with open(tax_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict(self, text, threshold=None):
        """
        Analyze text and return a structured dictionary of results.
        """
        if not text:
            return None
        
        if threshold is None:
            threshold = self.taxonomy["inference"]["default_threshold"]

        # 1. Preprocessing (Tokenization using project wrapper)
        clean_text = tokenize_text(text)
        
        # [NEW] Pre-filter OOD logic
        text_lower = text.lower()
        is_relevant = any(kw in text_lower for kw in CORE_VACCINE_KEYWORDS)
        
        if not is_relevant:
            # Skip model, return Rác/Lạc đề directly
            results = self._map_outputs(None, None, None, threshold, force_ood=True)
            results["keywords"] = self._extract_keywords(text)
            return results

        # 2. Preparation
        inputs = self.tokenizer(
            clean_text, 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 3. Forward Pass
        with torch.no_grad():
            m_logits, st_logits, se_logits = self.model(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
        
        # 4. Post-processing & Thresholding
        results = self._map_outputs(m_logits, st_logits, se_logits, threshold)
        
        # 5. Keywords extraction (Logic can be refined later)
        results["keywords"] = self._extract_keywords(text)
        
        return results

    def _map_outputs(self, m_logits, st_logits, se_logits, threshold, force_ood=False):
        tax = self.taxonomy["tasks"]
        
        # Handle OOD force
        if force_ood:
            return {
                "relevance": {"label": tax["relevance"]["labels"][1], "confidence": 1.0, "is_confident": True},
                "misinfo": {"label": "N/A", "confidence": 0.0, "is_confident": False},
                "stance": {"label": "N/A", "confidence": 0.0, "is_confident": False},
                "sentiment": {"label": "N/A", "confidence": 0.0, "is_confident": False}
            }

        # Softmax for probabilities
        p_m = torch.softmax(m_logits, dim=1)[0]
        p_s = torch.softmax(st_logits, dim=1)[0]
        p_se = torch.softmax(se_logits, dim=1)[0]
        
        idx_m = torch.argmax(p_m).item()
        idx_s = torch.argmax(p_s).item()
        idx_se = torch.argmax(p_se).item()
        
        # Threshold logic for Master Architecture with index safety
        def get_label(tax_key, idx):
            labels = tax[tax_key]["labels"]
            return labels[idx] if idx < len(labels) else "N/A"

        return {
            "relevance": {
                "label": tax["relevance"]["labels"][0],
                "confidence": 1.0, # Keywords matched
                "is_confident": True
            },
            "misinfo": {
                "label": get_label("misinformation", idx_m),
                "confidence": p_m[idx_m].item(),
                "is_confident": p_m[idx_m].item() >= threshold
            },
            "stance": {
                "label": get_label("stance", idx_s),
                "confidence": p_s[idx_s].item(),
                "is_confident": p_s[idx_s].item() >= threshold
            },
            "sentiment": {
                "label": get_label("sentiment", idx_se),
                "confidence": p_se[idx_se].item(),
                "is_confident": p_se[idx_se].item() >= threshold
            }
        }

    def _extract_keywords(self, text, top_n=5):
        """Standard keyword extraction logic (Placeholder for advanced NLP)"""
        import re
        from collections import Counter
        words = tokenize_text(text).split()
        # Remove common short words/stopwords
        words = [w for w in words if len(w) > 3]
        vax_keywords = ["vắc_xin", "tiêm_chủng", "pfizer", "astrazeneca", "moderna", "phản_ứng"]
        
        counts = Counter(words)
        results = []
        for w, c in counts.most_common(20):
            score = c
            if any(vk in w.lower() for vk in vax_keywords):
                score *= 3 # Priority for vaccine-related terms
            results.append((w, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return [w.replace("_", " ") for w, s in results[:top_n]]

if __name__ == "__main__":
    # Test script
    import sys
    ensure_src_in_sys_path()
        
    print("Testing VaccineInferencePipeline...")
    # Add dummy test here if needed
