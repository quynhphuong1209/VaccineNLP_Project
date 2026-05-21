# 🗣️ Data Engineering Pipeline (src/data_pipeline/) - v1.2

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Active & Modular

**Mục đích:** Đóng vai trò là Proof of Work minh chứng cho khả năng tự động hóa và làm sạch dữ liệu. Các module này được tách biệt hoàn toàn khỏng môi trường Huấn luyện (Phase 5) của `VaccineNLP_Clean_V1` để đảm bảo không gây xung Đột thư viện (dependency conflicts).

---

## 🛠️ Cấu Trúc

```
src/data_pipeline/
├── __init__.py
├── README.md                              ← You are here
├── collection/                             # 📢 Data Collection Module
│   ├── __init__.py
│   ├── master_collector_v2.py              # 🧏 Orchestrator (main entry)
│   ├── apify_social_collector_v2.py       # Apify API integration
│   ├── facebook_page_collector.py          # Facebook Pages API
│   ├── vfnd_fact_checker.py                # VFND API integration
│   ├── actor_configs/                     # 🧄 Config templates
│   │   ├── facebook_config.json
│   │   ├── tiktok_config.json
│   │   ├── youtube_config.json
│   │   ├── rss_config.json
│   │   ├── threads_config.json
│   └── logging/
│       ├── collection.log
│       ├── errors.log
│       └── api_calls.log
│
├── preprocessing/                         # 🔧 Data Cleaning Module
│   ├── __init__.py
│   ├── pipeline.py                        # 🔄 Orchestrator
│   ├── text_cleaner_v2.py                 # HTML/URL removal
│   ├── language_filter.py                 # Language detection
│   └── vn_tokenizer.py                    # Vietnamese word segmentation
│
└── __pycache__/
```

---

## 📢 **collection/** - Data Collection Module

### Mục Đích
Lấy dữ liệu từ nhiều nguồn: Facebook Pages, TikTok, YouTube, RSS, VFND, etc.

### 🧏 **master_collector_v2.py** - Main Orchestrator

**Chuc năng:**
- Orchestrate multiple data sources
- Handle parallel collection
- Aggregate outputs
- Error recovery

**Kiến trúc:**
```
MasterCollector
  ├─ ApifySocialCollector
  ├─ FacebookPageCollector
  ├─ VFNDFactChecker
  ├─ RSSFeedCollector
  └─ OutputAggregator
       └─ Unified Format (JSON)
```

**Usage:**
```python
from src.data_pipeline.collection.master_collector_v2 import MasterCollector

collector = MasterCollector(config_file='configs/seeds.json')
results = collector.collect(
    platforms=['facebook', 'tiktok'],
    limit=5000,
    parallel=True
)

# Save to Bronze layer
collector.save(results, 'datasets/01_raw/apify_raw.json')
```

### 📂 **apify_social_collector_v2.py** - Apify Integration

Connect to Apify API for multi-platform scraping

**Supported Platforms:**
- 🔘 Facebook (pages, posts, comments)
- 🤺 TikTok (videos, comments, profiles)
- 📺 YouTube (videos, comments)
- 💿 RSS (feeds)
- 🔗 Threads

### 🟣️ **facebook_page_collector.py** - Facebook API

Direct Facebook Graph API integration for dedicated collection.

### ✔️ **vfnd_fact_checker.py** - VFND Database

Fetch fact-checked claims from Vietnam Fact-check Network.

### 🧄 **actor_configs/** - Configuration Templates

JSON configs for each Apify actor:
- Platform-specific field mappings
- Rate limiting settings
- Proxy configuration
- Error handling parameters

**Example (facebook_config.json):**
```json
{
  "startUrls": [{"url": "https://www.facebook.com/page_name"}],
  "fieldsToExtract": ["url", "message", "reactions", "comments_count"],
  "limitPages": 1,
  "limitPosts": 100,
  "dateFrom": "2024-01-01",
  "dateUntil": "2026-05-21"
}
```

---

## 🔧 **preprocessing/** - Data Cleaning Module

### Mục Đích
Cléaning và chuang hoà dữ liệu từ Bronze → Silver layer

### 🔄 **pipeline.py** - Main Preprocessing Pipeline

**Stages:**
```
1. Load raw data
   ↓
2. Text cleaning (HTML, URLs, etc.)
   ↓
3. Language detection (Vietnamese only)
   ↓
4. Tokenization
   ↓
5. Validation
   ↓
6. Save cleaned data
```

**Usage:**
```python
from src.data_pipeline.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
result = pipeline.process(
    input_file='datasets/01_raw/apify_raw.json',
    output_file='datasets/02_processed/cleaned.json'
)

print(f"Processed: {result['count']} records")
print(f"Removed: {result['duplicates_removed']} duplicates")
```

### 🖜️ **text_cleaner_v2.py** - Text Cleaning

**Operations:**
- Remove HTML tags
- Remove URLs
- Remove @mentions, #hashtags
- Normalize Unicode (Vietnamese diacritics)
- Remove emojis
- Fix whitespace
- Lowercase conversion

**Usage:**
```python
from src.data_pipeline.preprocessing.text_cleaner_v2 import TextCleaner

cleaner = TextCleaner()
clean_text = cleaner.clean("<p>Xin chào @user 😊 #vaccine</p>")
# Output: "xin chào user vaccine"
```

### 🌐 **language_filter.py** - Language Detection

Filter to keep only Vietnamese content (confidence > 0.95)

**Usage:**
```python
from src.data_pipeline.preprocessing.language_filter import LanguageFilter

filter = LanguageFilter(language='vi', threshold=0.95)
vi_only = filter.filter(texts)  # Returns Vietnamese texts
```

### 🔔 **vn_tokenizer.py** - Vietnamese Tokenization

Word segmentation using PyVi + Underthesea

**Usage:**
```python
from src.data_pipeline.preprocessing.vn_tokenizer import VietnameseTokenizer

tokenizer = VietnameseTokenizer()
tokens = tokenizer.tokenize("Vắc xin là tốt cho sức khoẻ")
# Output: ['Vắc xin', 'là', 'tốt', 'cho', 'sức khoẻ']
```

---

## 📄 Pipeline Configuration

**File:** `src/data_pipeline/config.yaml`

```yaml
collection:
  apify:
    api_token: ${APIFY_TOKEN}
    actors:
      facebook: "sKQR3V7snxWGGw9ZX"
      tiktok: "eFmohrhm5bjVqx7u5"
  concurrency: 4
  timeout: 300

preprocessing:
  language: "vi"
  confidence_threshold: 0.95
  remove_duplicates: true
  min_length: 10
  max_length: 2048

output:
  raw_dir: "datasets/01_raw/"
  processed_dir: "datasets/02_processed/"
  format: "json"
```

---

## 📄 Workflow Integration

```
║ Data Collection & Preprocessing Workflow ║

📢 COLLECTION
   master_collector_v2.py
   ↑ ↑ ↑ ↑ ↑
   Facebook | TikTok | YouTube | RSS | VFND
   ↓ ↓ ↓ ↓ ↓
   datasets/01_raw/apify_raw.json
   (🟄 BRONZE Layer)
        ↓
  🔧 PREPROCESSING
   pipeline.py
   ↓ ↓ ↓ ↓
   Clean | Filter | Tokenize | Validate
   ↓ ↓ ↓ ↓
   datasets/02_processed/cleaned.json
   (🟡 SILVER Layer)
        ↓
  🎨 NEXT STAGES
   Auto-labeling → Expert review → Training
```

---

## 📈 Statistics

| Stage | Records | Processing | Status |
|---|---|---|---|
| Collection | ~5,000/run | Multi-source | 📂 |  
| Cleaning | 4,980 | ~2min | ✅ |
| Language filter | 4,956 | Instant | ✅ |
| Tokenization | 4,956 | ~1min | ✅ |
| **Output** | **4,956** | **~3min total** | **✅ Ready** |

---

## 🔍 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| API quota exceeded | Too many requests | Reduce batch_size, add delay |
| Encoding errors | Unicode mismatch | Use text_cleaner_v2 normalize_unicode() |
| Slow processing | Large files | Use generator-based processing |
| Duplicate records | Collection overlap | Enable remove_duplicates flag |

---

**📅 Updated:** May 21, 2026 | Phiên bản 1.2
**👤 Owner:** VaccineNLP Data Engineering Team
**🔧 Integration:** Part of Medallion Architecture (Bronze → Silver → Gold)
