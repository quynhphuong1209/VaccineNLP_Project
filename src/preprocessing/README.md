# 🧹 Text Preprocessing Module (src/preprocessing/) - v1.2

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Active & Optimized

## Mục Đích

Cung cấp các công cụ xử lý văn bản (Text Preprocessing) để chuẩn bị dữ liệu cho các models:
- 🧼 Text cleaning (HTML removal, normalization)
- 🔤 Vietnamese tokenization (VN-specific NLP)
- 🗣️ Language detection & filtering
- 🗂️ Ontology mapping (text → taxonomy categories)
- 🔄 Data pipeline orchestration

---

## 📋 Danh Sách Modules

### 1️⃣ **pipeline.py** - Preprocessing Pipeline Orchestrator

**Mục Đích:** Điều phối toàn bộ quy trình tiền xử lý từ dữ liệu thô đến dữ liệu sẵn sàng cho huấn luyện

**Kiến Trúc:**
```
Input: Raw JSON
    ↓
[1. Load Data]
    ↓
[2. Text Cleaning] (HTML removal, normalization)
    ↓
[3. Language Detection] (Keep VN only)
    ↓
[4. Tokenization] (Vietnamese-aware)
    ↓
[5. Validation] (Quality checks)
    ↓
[6. Save Output] (Processed JSON/JSONL)
    ↓
Output: Cleaned Data
```

**Sử Dụng:**
```python
from src.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
result = pipeline.process(
    input_file='datasets/01_raw/apify_raw.json',
    output_file='datasets/02_processed/cleaned.json'
)
```

**Configuration:**
```python
config = {
    'remove_html': True,
    'normalize_unicode': True,
    'min_length': 10,
    'max_length': 2048,
    'language': 'vi',
    'tokenize': True
}
```

---

### 2️⃣ **text_cleaner_v2.py** - Text Cleaning Module

**Mục Đích:** Làm sạch và chuẩn hóa văn bản

**Chức Năng:**
```python
# Chứa các phương thức:
- remove_html()          # Remove HTML tags & entities
- normalize_unicode()    # Fix Vietnamese diacritics
- remove_urls()          # Remove URLs
- remove_mentions()      # Remove @mentions
- remove_hashtags()      # Remove #hashtags
- normalize_whitespace() # Fix whitespace issues
- lowercase()            # Convert to lowercase
- remove_emojis()        # Remove emoji characters
```

**Ví dụ:**
```python
from src.preprocessing.text_cleaner_v2 import TextCleaner

cleaner = TextCleaner()
text = "<p>Xin chào @user 😊 #vaccine</p>"
cleaned = cleaner.clean(text)
# Output: "xin chào user vaccine"
```

---

### 3️⃣ **vn_tokenizer.py** - Vietnamese Tokenizer

**Mục Đích:** Tokenize text theo Vietnamese linguistic rules

**Chức Năng:**
```python
# Sử dụng:
- pyvi.ViTokenizer (Vietnamese word segmentation)
- underthesea (POS tagging, NER)
- fasttext (Language detection)
```

**Ví dụ:**
```python
from src.preprocessing.vn_tokenizer import VietnameseTokenizer

tokenizer = VietnameseTokenizer()

# Word tokenization
tokens = tokenizer.tokenize("Vắc xin là tốt cho sức khoẻ")
# Output: ['Vắc xin', 'là', 'tốt cho', 'sức khoẻ']

# POS tagging
pos_tags = tokenizer.pos_tag(tokens)
# Output: [('Vắc xin', 'N'), ('là', 'V'), ...]
```

---

### 4️⃣ **language_filter.py** - Language Detection & Filtering

**Mục Đích:** Lọc chỉ giữ lại dữ liệu tiếng Việt

**Chức Năng:**
```python
# Detect language
- fasttext language identification
- Threshold-based filtering (confidence > 0.95)
- Filter out non-Vietnamese content
```

**Ví Dụ:**
```python
from src.preprocessing.language_filter import LanguageFilter

filter = LanguageFilter(language='vi', threshold=0.95)

texts = [
    "Vắc xin rất an toàn",
    "Vaccine is safe",
    "백신은 안전합니다"
]

results = filter.filter(texts)
# Output: [("Vắc xin rất an toàn", 'vi', 0.98)]
```

---

### 5️⃣ **ontology_mapper.py** - Taxonomy Mapping Module

**Mục Đích:** Map text (hoặc labels) vào taxonomy categories

**Chức Năng:**
```python
# Mapping strategy:
1. Keyword-based matching (exact/fuzzy)
2. Semantic similarity (embedding-based)
3. Classification-based (using fine-tuned model)

# Taxonomy levels:
- Level 1: Vaccine (chủ đề chính)
- Level 2: Sub-topics (adverse effects, efficacy, myths, etc.)
- Level 3: Specific aspects (antigen, side effects, duration, etc.)
```

**Ví Dụ:**
```python
from src.preprocessing.ontology_mapper import OntologyMapper

mapper = OntologyMapper(taxonomy_file='configs/taxonomy.json')

text = "Vắc xin gây tê liệt"
category = mapper.map(text)
# Output: {
#     'topic': 'vaccine',
#     'subtopic': 'adverse_effects',
#     'aspect': 'neurological'
# }
```

**Taxonomy Structure (từ configs/taxonomy.json):**
```json
{
  "vaccine": {
    "adverse_effects": ["tê liệt", "sốt cao", "sốc nặng", ...],
    "efficacy": ["hiệu quả", "bảo vệ", "miễn dịch", ...],
    "myths": ["chứa chip", "làm vô sinh", "biến đổi ADN", ...],
    "safety": ["an toàn", "kiểm định", "chính thức", ...]
  }
}
```

---

### 6️⃣ **preprocess_external_data.py** - External Data Preprocessing

**Mục Đích:** Xử lý các nguồn dữ liệu bên ngoài (VFND API, scraping, etc.)

**Chức Năng:**
```python
# Hỗ trợ:
- VFND (Vietnam Fact-check Network) API
- Custom JSON/CSV sources
- Real-time social media data
- Web scraping outputs
```

**Ví Dụ:**
```python
from src.preprocessing.preprocess_external_data import ExternalDataProcessor

processor = ExternalDataProcessor()

# Process VFND data
vfnd_data = processor.process_vfnd_api(api_key='...')

# Standardize to common format
standardized = processor.standardize_format(vfnd_data)
```

---

## 🏗️ Preprocessing Pipeline Flow

```
Stage 1: Collection
├─ Raw data từ Apify, Facebook, VFND, etc.
└─ Stored in datasets/01_raw/

Stage 2: Cleaning  
├─ HTML removal
├─ Unicode normalization
├─ Emoji/URL removal
└─ Stored in datasets/02_interim/

Stage 3: Analysis
├─ Language detection
├─ Length check
├─ Duplicate detection
└─ Quality scoring

Stage 4: Tokenization
├─ Vietnamese word segmentation
├─ POS tagging (optional)
└─ Token statistics

Stage 5: Enrichment
├─ Ontology mapping
├─ Taxonomy classification
├─ Metadata extraction

Stage 6: Validation
├─ Format validation
├─ Data quality checks
├─ Statistical profiling

Final Output: datasets/02_processed/
```

---

## 🔧 Configuration

**File:** `configs/preprocessing.json` (optional)

```json
{
  "cleaning": {
    "remove_html": true,
    "remove_urls": true,
    "remove_emojis": true,
    "normalize_unicode": true,
    "min_length": 10,
    "max_length": 2048
  },
  "filtering": {
    "language": "vi",
    "confidence_threshold": 0.95,
    "remove_duplicates": true
  },
  "tokenization": {
    "method": "pyvi",
    "pos_tagging": false,
    "ner_tagging": false
  },
  "mapping": {
    "enable_ontology": true,
    "taxonomy_file": "configs/taxonomy.json",
    "strategy": "keyword"
  }
}
```

---

## 📊 Processing Statistics

```
Input: 1,856 raw posts
  ├─ Valid Vietnamese: 1,842 (99.2%)
  ├─ Invalid/Noise: 14 (0.8%)
  
After cleaning:
  ├─ Avg text length: 450 chars
  ├─ Min length: 12 chars
  ├─ Max length: 2,045 chars
  ├─ Duplicates removed: 23
  
After tokenization:
  ├─ Avg tokens: 85 words
  ├─ Unique tokens: 12,340
  ├─ Vocabulary size: 9,850
  
Final output: 1,819 posts (✅ GOLD layer)
```

---

## 🚀 Usage Examples

### Quick Start
```python
from src.preprocessing.pipeline import PreprocessingPipeline

# Initialize
pipeline = PreprocessingPipeline()

# Process single file
pipeline.process(
    input_file='datasets/01_raw/apify_raw.json',
    output_file='datasets/02_processed/cleaned.json'
)

# Process with custom config
config = {
    'remove_html': True,
    'normalize_unicode': True,
    'language': 'vi'
}
pipeline.process(input_file='...', config=config)
```

### Batch Processing
```python
import glob

# Process all raw files
for raw_file in glob.glob('datasets/01_raw/*.json'):
    pipeline.process(raw_file, 'datasets/02_processed/')
```

### Integration with Training
```python
# In training script
from src.preprocessing.pipeline import PreprocessingPipeline
from src.modeling.dataset_loader import DatasetLoader

# Preprocess
pipeline = PreprocessingPipeline()
pipeline.process('datasets/01_raw/apify_raw.json', 
                 'datasets/02_processed/cleaned.json')

# Load for training
loader = DatasetLoader('datasets/02_processed/cleaned.json')
train_dl, val_dl, test_dl = loader.get_dataloaders()
```

---

## 📈 Monitoring & Logging

All preprocessing operations are logged to:
```
experiments/logs/preprocessing.log
```

**Log Levels:**
- `INFO`: Progress updates, file processing
- `WARNING`: Data quality issues, filtering
- `ERROR`: Processing failures, invalid data

---

## 🔐 Best Practices

- ✅ Always validate input data before processing
- ✅ Keep original raw data in `01_raw/` (immutable)
- ✅ Use relative paths (via `src.common.paths`)
- ✅ Log all transformations
- ✅ Save processing statistics

- ❌ Don't modify raw data directly
- ❌ Don't hardcode paths
- ❌ Don't skip validation steps

---

**📅 Updated:** May 21, 2026 (v1.2)
**👤 Owner:** VaccineNLP Team  
**🔬 Framework:** PyVi + Underthesea + FastText  
**📊 Integration:** Part of Medallion Architecture (Bronze → Silver → Gold)
