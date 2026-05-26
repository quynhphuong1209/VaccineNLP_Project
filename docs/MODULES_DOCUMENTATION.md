# VaccineNLP App Core Module Documentation

**Version:** 1.0.0  
**Date:** May 26, 2026  
**Module Location:** `src/app_core/`

## Overview

The `app_core` module contains all shared business logic extracted from the Streamlit and Gradio applications. This module enables both frontends to use a **single source of truth** for model loading, predictions, XAI reasoning, and data fetching.

## Architecture

```
src/app_core/
├── __init__.py          # Module exports
├── predictor.py         # Model loading, inference, predictions
├── xai_engine.py        # Explainability, reasoning generation
├── fetchers.py          # Data fetching from URLs, social media
└── README.md            # Module documentation
```

## Module 1: Predictor (`predictor.py`)

Handles model loading, inference, temperature scaling, and text utilities.

### Classes

#### `VaccineMultitaskModel(nn.Module)`

Multitask neural network model with shared encoder and task-specific heads.

**Parameters:**
- `model_name` (str): Pre-trained model name ("vinai/phobert-base-v2" or "xlm-roberta-base")
- `num_misinfo` (int): Number of misinformation classes (default: 2)
- `num_stance` (int): Number of stance classes (default: 3)
- `num_sentiment` (int): Number of sentiment classes (default: 3)
- `token` (str, optional): Hugging Face API token

**Example:**
```python
from src.app_core.predictor import VaccineMultitaskModel

model = VaccineMultitaskModel(
    model_name="vinai/phobert-base-v2",
    num_misinfo=2,
    num_stance=3,
    num_sentiment=3
)

# Model has 3 prediction heads:
# - heads["misinfo"]: Predicts misinformation (0=Tin giả, 1=Chính xác)
# - heads["stance"]: Predicts vaccine stance (0=Ủng hộ, 1=Phản đối, 2=Trung lập)
# - heads["sentiment"]: Predicts sentiment (0=Tiêu cực, 1=Trung tính, 2=Tích cực)
```

### Functions

#### `load_model(model_key="PhoBERT-v2", hf_token=None)`

Load pre-trained multitask model from Hugging Face with optimized RAM usage.

**Parameters:**
- `model_key` (str): Model selection - "PhoBERT-v2" or "XLM-R-v1"
- `hf_token` (str, optional): Hugging Face token for private repos

**Returns:**
- `Tuple[Optional[nn.Module], Optional[Any], bool]`: (model, tokenizer, success_flag)

**Example:**
```python
from src.app_core import predictor

model, tokenizer, success = predictor.load_model("PhoBERT-v2", hf_token="hf_...")
if success:
    print("✅ Model loaded successfully")
else:
    print("❌ Model loading failed")
```

**Behavior:**
- First checks local checkpoint: `experiments/models/{model_name}/best_model.pt`
- Falls back to Hugging Face download if not found
- Uses memory-efficient loading (mmap=True, low_cpu_mem_usage=True)
- Cleans RAM after loading (gc.collect(), torch.cuda.empty_cache())

---

#### `load_xai_cache()`

Load pre-built XAI reasoning cache from disk.

**Returns:**
- `Dict[str, str]`: Cache mapping text → reasoning explanation

**Example:**
```python
xai_cache = predictor.load_xai_cache()
print(f"Cache contains {len(xai_cache)} entries")
```

**File Location:** `app/xai_cache.json` (single shared cache for both apps)

---

#### `load_temperature_params()`

Load calibrated temperature scaling parameters from experiments/results.

**Returns:**
- `Dict[str, Dict[str, float]]`: Temperature parameters for each model and task

**Example:**
```python
temp_params = predictor.load_temperature_params()
print(temp_params['phobert_v2']['misinfo'])  # Output: 1.0 or calibrated value
```

**File Locations:** 
- `experiments/results/temperature_params.json` (priority)
- `app/temperature_params.json` (fallback)

---

#### `predict_cached(text, model_key="PhoBERT-v2", hf_token=None)`

Make multitask predictions with temperature scaling and optional reasoning lookup.

**Parameters:**
- `text` (str): Input text to classify
- `model_key` (str): Model to use ("PhoBERT-v2" or "XLM-R-v1")
- `hf_token` (str, optional): Hugging Face token

**Returns:**
- `Optional[Dict[str, Any]]`: Prediction dict with structure:
```json
{
  "misinfo": {
    "pred": 0,                      // Predicted label (0 or 1)
    "prob_raw": [0.15, 0.85],      // Raw softmax probabilities
    "prob_cal": [0.12, 0.88]       // Temperature-calibrated probabilities
  },
  "stance": {
    "pred": 1,
    "prob_raw": [0.2, 0.6, 0.2],
    "prob_cal": [0.15, 0.75, 0.1]
  },
  "sentiment": {
    "pred": 0,
    "prob_raw": [0.8, 0.15, 0.05],
    "prob_cal": [0.85, 0.1, 0.05]
  }
}
```

**Example:**
```python
from src.app_core import predictor

text = "Vắc xin an toàn và hiệu quả"
result = predictor.predict_cached(text, "PhoBERT-v2")

if result:
    print(f"Misinformation: {result['misinfo']['pred']} (confidence: {max(result['misinfo']['prob_cal'])})")
    print(f"Stance: {result['stance']['pred']}")
    print(f"Sentiment: {result['sentiment']['pred']}")
```

**Pipeline:**
1. Load model and tokenizer
2. Preprocess text (PhoBERT: word tokenization via underthesea)
3. Get raw logits from all 3 heads
4. Apply temperature scaling for calibration
5. Convert to probabilities (softmax)
6. Return predictions and confidence scores

---

#### `is_mostly_english(txt)`

Detect if text is primarily English using common word frequencies.

**Parameters:**
- `txt` (str): Text to analyze

**Returns:**
- `bool`: True if >50% English content, False if Vietnamese

**Example:**
```python
print(predictor.is_mostly_english("Hello world"))  # True
print(predictor.is_mostly_english("Xin chào Việt Nam"))  # False
```

**Usage:** Used internally to decide whether to translate reasoning to Vietnamese.

---

#### `translate_to_vietnamese(txt)`

Translate text to Vietnamese using Google Translate API (free, stable).

**Parameters:**
- `txt` (str): Text to translate

**Returns:**
- `str`: Vietnamese translation, or original text if translation fails

**Example:**
```python
vn_text = predictor.translate_to_vietnamese("This vaccine is safe and effective")
print(vn_text)  # "Vắc xin này an toàn và hiệu quả"
```

**Note:** Uses `googleapis.com` (official Google service), not selenium or browser automation.

---

#### `is_reasoning_corrupted(txt)`

Validate if XAI reasoning is corrupted or malformed.

**Parameters:**
- `txt` (str): Reasoning text to validate

**Returns:**
- `bool`: True if corrupted, False if valid

**Example:**
```python
# Detects these as corrupted:
predictor.is_reasoning_corrupted("[Phan hieu]")  # True (malformed token)
predictor.is_reasoning_corrupted("không có thông tin")  # True (too short/generic)
predictor.is_reasoning_corrupted("Lý luận chi tiết về......")  # False (valid)
```

---

## Module 2: XAI Engine (`xai_engine.py`)

Explainability and AI reasoning generation.

### Functions

#### `find_xai_reasoning(text, cache)`

Look up explanation with prioritized caching: hard-coded → cache file.

**Parameters:**
- `text` (str): Input text to find reasoning for
- `cache` (Dict[str, str]): Cache dictionary from `load_xai_cache()`

**Returns:**
- `Optional[str]`: Explanation string, or None if not found

**Example:**
```python
from src.app_core import xai_engine, predictor

text = "Vắc xin gây tự kỷ"
cache = predictor.load_xai_cache()
reasoning = xai_engine.find_xai_reasoning(text, cache)

if reasoning:
    print(f"Found reasoning:\n{reasoning}")
```

**Priority Levels:**
1. **Exact match** in hard-coded demo cache (highest quality)
2. **Fuzzy match** in hard-coded cache
3. **Exact match** in cache file (validated)
4. **Fuzzy match** in cache file (validated)
5. **None** if not found anywhere

---

#### `generate_smart_fallback(result)`

Generate AI-like fallback explanation when all APIs fail.

**Parameters:**
- `result` (Dict): Prediction dict from `predict_cached()`

**Returns:**
- `str`: Vietnamese explanation mimicking Gemma reasoning

**Example:**
```python
from src.app_core import xai_engine

result = {
    "misinfo": {"pred": 0},    # Misinformation (0=giả)
    "stance": {"pred": 1},     # Stance: phản đối (1)
    "sentiment": {"pred": 0}   # Sentiment: tiêu cực (0)
}

fallback = xai_engine.generate_smart_fallback(result)
print(fallback)
# Output: "Dựa trên các đặc trưng ngôn ngữ, hệ thống nhận diện đây là nội dung có rủi ro cao về tin giả..."
```

**Usage:** Called automatically if both cache lookup and API calls fail.

---

#### `query_gemma_api(short_text, hf_token=None)`

Query multi-tier Gemma API system for automated reasoning generation.

**Parameters:**
- `short_text` (str): Input text (will be truncated to 1000 chars)
- `hf_token` (str, optional): Hugging Face API token

**Returns:**
- `Optional[str]`: Generated reasoning in Vietnamese, or None if all APIs fail

**Example:**
```python
from src.app_core import xai_engine

text = "Tôi không cho con tiêm vì vắc xin gây hại"
reasoning = xai_engine.query_gemma_api(text[:1000], hf_token="hf_...")

if reasoning:
    print(reasoning)
```

**Model Priority:**
1. **Custom Gemma-4-E4B** (hung2903/gemma-4-E4B-unsloth-vaccine-xai) - Optimized for Vietnamese
2. **Google Gemma-2-2b-it** - General fallback
3. **Mistral-7B-Instruct-v0.3** - Final fallback

**Prompt Engineering:** Each model uses specialized prompts:
- Gemma-4: Full Vietnamese, QLoRA template
- Gemma-2/Mistral: Standard Vietnamese prompts

---

#### `compute_captum_saliency(text, model_key, model=None, tokenizer=None)`

Compute token-level attribution for visual explanation via Integrated Gradients.

**Parameters:**
- `text` (str): Input text
- `model_key` (str): Model to use
- `model` (nn.Module, optional): Pre-loaded model to save time
- `tokenizer` (optional): Pre-loaded tokenizer

**Returns:**
- `Tuple[List[str], List[float], int]`: (tokens, attribution_scores, pred_class)

**Example:**
```python
from src.app_core import xai_engine

tokens, attr_scores, pred_class = xai_engine.compute_captum_saliency(
    "Vắc xin an toàn",
    "PhoBERT-v2"
)

for token, score in zip(tokens, attr_scores):
    print(f"{token}: {score:.3f}")  # Higher score = more important for prediction
```

**Requirements:** `pip install captum`

**Visualization:** Used to render token heatmaps showing which tokens influenced the prediction.

---

#### `is_reasoning_corrupted(txt)`

Validate reasoning quality. Same as in predictor module.

---

## Module 3: Fetchers (`fetchers.py`)

Data fetching from various sources (news, YouTube, social media).

### Functions

#### `detect_source(url)`

Determine the type of content source from URL.

**Parameters:**
- `url` (str): URL to analyze

**Returns:**
- `str`: Source type ("news", "youtube", or "apify")

**Example:**
```python
from src.app_core import fetchers

print(fetchers.detect_source("https://vnexpress.net/..."))  # "news"
print(fetchers.detect_source("https://youtube.com/watch?v=..."))  # "youtube"
print(fetchers.detect_source("https://facebook.com/..."))  # "apify"
```

---

#### `fetch_url_as_list(url, max_comments=30)`

Fetch content from URL and return as individual text segments.

**Parameters:**
- `url` (str): URL to fetch from
- `max_comments` (int): Maximum comments/posts to retrieve

**Returns:**
- `Tuple[List[str], str]`: (text_segments, source_info)

**Example:**
```python
from src.app_core import fetchers

texts, info = fetchers.fetch_url_as_list(
    "https://www.facebook.com/...",
    max_comments=30
)

print(f"Fetched {len(texts)} segments from {info}")
for i, text in enumerate(texts[:3]):
    print(f"\n--- Segment {i+1} ---")
    print(text[:200])
```

**Supported Sources:**

| Source | Method | Speed | Notes |
|--------|--------|-------|-------|
| News (VN sites) | Trafilatura | ~2s | vnexpress, tuoitre, dantri, etc. |
| YouTube | yt_dlp | ~10s | Title, description, comments |
| Facebook | Apify actor | ~30s | Groups, pages, posts, comments |
| TikTok | Apify actor | ~30s | Video comments |
| Threads | Apify actor | ~30s | Posts and comments |

**Error Handling:**
- Returns empty list + error message if URL invalid
- Falls back through fetchers if one fails
- Example: YouTube yt_dlp fails → tries Apify as backup

---

#### `fetch_url(url, max_comments=30)`

Convenience wrapper: combines segments into single text (backward compatibility).

**Parameters:**
- `url` (str): URL to fetch
- `max_comments` (int): Maximum content to retrieve

**Returns:**
- `Tuple[str, str]`: (combined_text, source_info)

**Example:**
```python
from src.app_core import fetchers

combined_text, source = fetchers.fetch_url("https://youtube.com/watch?v=...")
print(f"Content ({source}):\n{combined_text[:500]}")
```

---

#### `is_valid_comment(txt)`

Smart filter for social media comments/posts.

**Parameters:**
- `txt` (str): Comment text to validate

**Returns:**
- `bool`: True if valid comment, False if should be filtered

**Filters Out:**
1. **Too short** (<15 chars or <4 words)
2. **Spam**: Sales, recruitment, ads
3. **Off-topic**: Not vaccine-related (uses keyword matching)

**Example:**
```python
print(fetchers.is_valid_comment("check out my website"))  # False (spam)
print(fetchers.is_valid_comment("ok"))  # False (too short)
print(fetchers.is_valid_comment("Vắc xin phòng bệnh sởi rất hiệu quả"))  # True
```

---

## Usage in Applications

### Streamlit App Example

```python
from src.app_core import predictor, xai_engine, fetchers
import streamlit as st

# 1. Load model once (cached)
@st.cache_resource
def init_model():
    model, tokenizer, ok = predictor.load_model("PhoBERT-v2", hf_token=st.secrets.HF_TOKEN)
    return model, tokenizer

# 2. Get user input
text = st.text_area("Enter text to analyze")

# 3. Make prediction
result = predictor.predict_cached(text, "PhoBERT-v2", hf_token=st.secrets.HF_TOKEN)

# 4. Get explanation
xai_cache = predictor.load_xai_cache()
reasoning = xai_engine.find_xai_reasoning(text, xai_cache)
if not reasoning:
    reasoning = xai_engine.generate_smart_fallback(result)

# 5. Display results
st.write(f"Misinformation: {result['misinfo']['pred']}")
st.write(f"Reasoning: {reasoning}")
```

### Gradio App Example

```python
from src.app_core import predictor, xai_engine, fetchers

def classify_and_explain(text, url=None):
    # Fetch from URL if provided
    if url:
        segments, source = fetchers.fetch_url_as_list(url, max_comments=10)
        text = "\n".join(segments)
    
    # Predict
    result = predictor.predict_cached(text, "PhoBERT-v2")
    
    # Explain
    xai_cache = predictor.load_xai_cache()
    reasoning = xai_engine.find_xai_reasoning(text, xai_cache)
    if not reasoning:
        reasoning = xai_engine.generate_smart_fallback(result)
    
    return result, reasoning

# Use in Gradio
interface = gr.Interface(
    fn=classify_and_explain,
    inputs=[gr.Textbox(label="Text"), gr.Textbox(label="URL")],
    outputs=[gr.JSON(label="Predictions"), gr.Textbox(label="Explanation")]
)
```

---

## Configuration & Environment

### Required Environment Variables

```bash
# Hugging Face API
export HF_TOKEN="hf_..."

# Apify (for social media fetching - optional)
export APIFY_TOKEN="apify_..."
export APIFY_TOKEN_2="apify_..."  # Backup tokens
export APIFY_TOKEN_3="apify_..."
```

### Model Configs

Located in `src/app_core/predictor.py`:

```python
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
```

---

## Performance Considerations

### Model Loading
- **First load**: ~10-30 seconds (download + init)
- **Subsequent loads**: <1 second (cached in memory)
- **Memory usage**: ~2-3 GB per model

### Predictions
- **Latency**: 200-500ms per prediction (depending on GPU)
- **Batch processing**: 50-100 texts per second on GPU

### Caching Strategy
- **XAI cache**: Load once at startup (~2MB)
- **Temperature params**: Load once (~1KB)
- **Model**: Cache with `@st.cache_resource` or similar

### Optimization Tips
```python
# Pre-load once at startup
model, tokenizer, _ = predictor.load_model("PhoBERT-v2")
xai_cache = predictor.load_xai_cache()

# Reuse for multiple predictions
for text in texts:
    result = predictor.predict_cached(text, "PhoBERT-v2")
    reasoning = xai_engine.find_xai_reasoning(text, xai_cache)
```

---

## Error Handling

### Common Issues

**Issue:** Model download fails  
**Solution:** Provide `hf_token` or set `HF_TOKEN` environment variable

**Issue:** Translation fails  
**Solution:** Function falls back to original text; not critical

**Issue:** Apify tokens invalid  
**Solution:** Social media fetching disabled; falls back to simpler methods

**Issue:** XAI API timeouts  
**Solution:** Auto-generates fallback explanation using predictions

---

## Testing

```python
# Test predictor
from src.app_core import predictor

text = "Vắc xin gây tự kỷ"
result = predictor.predict_cached(text, "PhoBERT-v2")
assert result is not None
assert 'misinfo' in result
assert 'stance' in result
assert 'sentiment' in result

# Test XAI engine
from src.app_core import xai_engine

cache = predictor.load_xai_cache()
reasoning = xai_engine.find_xai_reasoning(text, cache)
fallback = xai_engine.generate_smart_fallback(result)
assert isinstance(fallback, str)

# Test fetchers
from src.app_core import fetchers

texts, info = fetchers.fetch_url_as_list("https://vnexpress.net/...")
assert isinstance(texts, list)
assert isinstance(info, str)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | May 26, 2026 | Initial release - extracted from apps |
| 0.9.0 | May 20, 2026 | Beta testing |

---

## Authors & Attribution

- **Extracted from:** `app/streamlit_demo.py` and `app_gradio/app.py`
- **Refactored by:** VaccineNLP Team
- **Purpose:** Enable code reuse and reduce maintenance burden
- **License:** Same as parent project

---

**Last Updated:** May 26, 2026  
**Status:** ✅ Production Ready
