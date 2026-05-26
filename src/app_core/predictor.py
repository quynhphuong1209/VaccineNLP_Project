"""
VaccineNLP Model Predictor Module
==================================
Shared model loading, inference, and prediction caching logic for both Streamlit and Gradio apps.

Extracted from: app/streamlit_demo.py and app_gradio/app.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_DIR = PROJECT_ROOT / "app"
XAI_CACHE_PATH = APP_DIR / "xai_cache.json"

MODEL_CONFIGS = {
    "PhoBERT-v2": {
        "repo_id": "hung2903/phobert-vaccine-multitask", 
        "base_repo": "vinai/phobert-base-v2",
        "type": "phobert"
    },
    "XLM-R-v1": {
        "repo_id": "hung2903/xlmr-vaccine-multitask", 
        "base_repo": "xlm-roberta-base",
        "type": "xlm-roberta"
    }
}

# ─────────────────────────────────────────────────────────────
# VACCINE MULTITASK MODEL CLASS
# ─────────────────────────────────────────────────────────────
class VaccineMultitaskModel(nn.Module):
    """Multitask model with shared PhoBERT encoder and task-specific heads."""

    def __init__(self, model_name="vinai/phobert-base-v2",
                 num_misinfo=2, num_stance=3, num_sentiment=3, token=None):
        from transformers import AutoConfig, AutoModel
        super(VaccineMultitaskModel, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name, token=token, trust_remote_code=True)
        # Load encoder with maximum RAM-saving technique
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            token=token, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        hidden_size = self.config.hidden_size
        # Flexible structure: Support both ModuleDict and individual layers
        self.heads = nn.ModuleDict({
            "misinfo": nn.Linear(hidden_size, num_misinfo),
            "stance": nn.Linear(hidden_size, num_stance),
            "sentiment": nn.Linear(hidden_size, num_sentiment)
        })
        # Backward compatibility aliases if needed
        self.head_misinfo = self.heads["misinfo"]
        self.head_stance = self.heads["stance"]
        self.head_sentiment = self.heads["sentiment"]
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        return (
            self.heads["misinfo"](pooled_output),
            self.heads["stance"](pooled_output),
            self.heads["sentiment"](pooled_output),
        )


# ─────────────────────────────────────────────────────────────
# TEMPERATURE SCALING PARAMETERS
# ─────────────────────────────────────────────────────────────
def load_temperature_params() -> Dict[str, Dict[str, float]]:
    """Load Temperature Scaling parameters from experiments/results/.
    Falls back to T=1.0 (no scaling) if file not found."""
    params = {
        'phobert_v2': {'misinfo': 1.0, 'stance': 1.0, 'sentiment': 1.0},
        'xlmr_v1':    {'misinfo': 1.0, 'stance': 1.0, 'sentiment': 1.0},
    }
    paths = [
        PROJECT_ROOT / "experiments" / "results" / "temperature_params.json",
        PROJECT_ROOT / "app" / "temperature_params.json",
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p) as f:
                    loaded = json.load(f)
                params.update(loaded)
                params['_source'] = 'LIVE'
                break
            except Exception as e:
                print(f"Warning: Could not load temperature params from {p}: {e}")
                params['_source'] = 'fallback'
                break
    
    if '_source' not in params:
        params['_source'] = 'fallback'
    return params


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_model(model_key: str = "PhoBERT-v2", hf_token: Optional[str] = None) -> Tuple[Optional[nn.Module], Optional[Any], bool]:
    """Load model BERT/XLM-R from Hugging Face with mmap technique.
    
    Args:
        model_key: Model name ("PhoBERT-v2" or "XLM-R-v1")
        hf_token: Hugging Face token (optional)
        
    Returns:
        Tuple of (model, tokenizer, success_flag)
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    
    # Clean up RAM before loading
    gc.collect()
    
    cfg = MODEL_CONFIGS[model_key]
    
    # Validate token
    if hf_token:
        hf_token = hf_token.strip() if isinstance(hf_token, str) else None
    else:
        hf_token = None
    
    try:
        # Check local model checkpoint first (Resilient Offline Mode)
        local_dir_name = "phobert_v2" if model_key == "PhoBERT-v2" else "xlmr_v1"
        local_model_path = PROJECT_ROOT / "experiments" / "models" / local_dir_name / "best_model.pt"
        if local_model_path.exists():
            model_path = str(local_model_path)
        else:
            # Download checkpoint file (.pt)
            model_path = hf_hub_download(repo_id=cfg["repo_id"], filename="best_model.pt", token=hf_token)
            
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_repo"], token=hf_token, trust_remote_code=True)
        model = VaccineMultitaskModel(model_name=cfg["base_repo"], token=hf_token)
        
        # Load weights (mmap=True helps save RAM)
        state = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        new_state = { 
            (k.replace("head_", "heads.") if k.startswith("head_") and "heads." not in k else k): v 
            for k, v in state.items() 
        }
        model.load_state_dict(new_state, strict=False)
        model.eval()
        
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, tokenizer, True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        gc.collect()
        return None, None, False


# ─────────────────────────────────────────────────────────────
# XAI CACHE LOADING
# ─────────────────────────────────────────────────────────────
def load_xai_cache() -> Dict[str, str]:
    """Load pre-built XAI reasoning cache (text → reasoning mapping)."""
    if XAI_CACHE_PATH.exists():
        try:
            with open(XAI_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load XAI cache: {e}")
            return {}
    return {}


# ─────────────────────────────────────────────────────────────
# TEXT UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def is_mostly_english(txt: str) -> bool:
    """Detect if text is mostly English using common English words."""
    common_en_words = {
        "the", "and", "of", "to", "a", "in", "is", "that", "it", "he", 
        "was", "for", "on", "are", "as", "with", "his", "they", "i", 
        "at", "be", "this", "have", "from"
    }
    words = set(txt.lower().split())
    return len(words.intersection(common_en_words)) > 2


def translate_to_vietnamese(txt: str) -> str:
    """Translate text to Vietnamese using Google Translate API (free, stable).
    
    Args:
        txt: Text to translate
        
    Returns:
        Vietnamese translated text, or original text if translation fails
    """
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=vi&dt=t&q=" + urllib.parse.quote(txt)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode('utf-8'))
            translated_sentences = [sentence[0] for sentence in res[0] if sentence[0]]
            return "".join(translated_sentences)
    except Exception as e:
        print(f"⚠️  Translation error: {e}")
        return txt


# ─────────────────────────────────────────────────────────────
# XAI REASONING VALIDATION
# ─────────────────────────────────────────────────────────────
def is_reasoning_corrupted(txt: str) -> bool:
    """Check if XAI reasoning is corrupted, has repeated words, or is garbage
    generated by small models.
    
    Args:
        txt: Reasoning text to validate
        
    Returns:
        True if reasoning appears corrupted, False otherwise
    """
    if not txt:
        return True
        
    corrupt_patterns = [
        "[Phan hieu]", "[Tinh tinh]", "[Phan loai]", "[Phan tich]", "[Mot van mot loi chay]",
        "[Phan doi]", "[Tin gia]", "<Trung tinh>", "<Tieu cuc>", "<Tich cuc>", "<Trung lap>", 
        "<Ung ho>", "<Phan doi>",
        "=== KẾT QUẢ ===", "=== GIẢI THÍCH ===", "=== PHÂN TÍCH ===",
        "Phan doi HOẶC", "Tin gia HOẶC", "Trung lap HOẶC", "Chinh xac HOẶC",
        "Sentiment: <", "Stance: <", "Misinformation: <", "Phan tich: <",
        "Nội Dung: Không", "Thông Tin: Không", "Không tim thay ket qua",
        "details>Chi tiet</details>"
    ]
    
    for pattern in corrupt_patterns:
        if pattern in txt:
            return True
            
    # Detect too-short or meaningless reasoning
    txt_lower = txt.lower()
    if len(txt) < 100:
        if any(kw in txt_lower for kw in ["không có thông tin", "không chứa thông tin", "chỉ là một câu"]):
            return True
            
    return False


# ─────────────────────────────────────────────────────────────
# PREDICTION WITH CACHING
# ─────────────────────────────────────────────────────────────
def predict_cached(text: str, model_key: str = "PhoBERT-v2", hf_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Predict class labels and probabilities for input text.
    
    Args:
        text: Input text to classify
        model_key: Model to use ("PhoBERT-v2" or "XLM-R-v1")
        hf_token: Hugging Face token (optional)
        
    Returns:
        Dictionary with predictions and probabilities, or None if error
    """
    model, tokenizer, success = load_model(model_key, hf_token)
    if not success or model is None:
        return None

    # Tokenize text
    processed_text = text
    if "phobert" in model_key.lower():
        try:
            from underthesea import word_tokenize
            processed_text = word_tokenize(text, format="text")
        except Exception as e:
            print(f"⚠️  Warning: Tokenization error: {e}")
    
    # Prepare input
    enc = tokenizer(processed_text, truncation=True, max_length=256, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Get predictions
    with torch.no_grad():
        logits_m, logits_st, logits_se = model(enc["input_ids"], enc["attention_mask"])

    # Load temperature parameters
    T_params = load_temperature_params()
    model_key_lower = model_key.lower().replace('-', '_').replace(' ', '_')
    
    # Map: "PhoBERT-v2" → "phobert_v2"; "XLM-R-v1" → "xlmr_v1"
    if 'phobert' in model_key_lower:
        Ts = T_params.get('phobert_v2', {'misinfo': 1.0, 'stance': 1.0, 'sentiment': 1.0})
    elif 'xlm' in model_key_lower:
        Ts = T_params.get('xlmr_v1', {'misinfo': 1.0, 'stance': 1.0, 'sentiment': 1.0})
    else:
        Ts = {'misinfo': 1.0, 'stance': 1.0, 'sentiment': 1.0}

    # Raw probabilities (original)
    p_mis_raw = F.softmax(logits_m, dim=1).cpu().numpy()[0]
    p_st_raw  = F.softmax(logits_st, dim=1).cpu().numpy()[0]
    p_sen_raw = F.softmax(logits_se, dim=1).cpu().numpy()[0]

    # Calibrated probabilities (Temperature Scaling)
    p_mis_cal = F.softmax(logits_m / Ts['misinfo'], dim=1).cpu().numpy()[0]
    p_st_cal  = F.softmax(logits_st / Ts['stance'], dim=1).cpu().numpy()[0]
    p_sen_cal = F.softmax(logits_se / Ts['sentiment'], dim=1).cpu().numpy()[0]

    # Model predictions
    pred_m = int(torch.argmax(logits_m, dim=1))
    pred_st = int(torch.argmax(logits_st, dim=1))
    pred_se = int(torch.argmax(logits_se, dim=1))

    return {
        'misinfo': {
            'pred': pred_m, 
            'prob_raw': p_mis_raw.tolist(), 
            'prob_cal': p_mis_cal.tolist()
        },
        'stance': {
            'pred': pred_st, 
            'prob_raw': p_st_raw.tolist(), 
            'prob_cal': p_st_cal.tolist()
        },
        'sentiment': {
            'pred': pred_se, 
            'prob_raw': p_sen_raw.tolist(), 
            'prob_cal': p_sen_cal.tolist()
        }
    }
