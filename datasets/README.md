# 📊 Thư Mục Dữ Liệu (datasets/)

## Mục Đích
Quản lý dữ liệu theo **Medallion Architecture**: Bronze (Raw) → Silver (Cleaned) → Gold (Validated)

Đảm bảo tracking rõ ràng chất lượng và giai đoạn xử lý của mỗi dataset.

---

## 🏗️ Kiến Trúc Medallion

```
📊 DATASETS
├── 🔴 BRONZE (01_raw/)          - Dữ liệu thô từ nguồn
├── 🟡 SILVER (02_interim/, 02_processed/, 04_silver_labels/) - Đã xử lý cơ bản
└── 🟢 GOLD (03_processed/)      - Xác nhận chất lượng cao
```

---

## 🔴 **01_raw/** - BRONZE LAYER

### Mục Đích
Lưu trữ dữ liệu thô từ các nguồn (Apify, APIs) mà chưa xử lý

### Đặc Điểm
- ❌ Chưa làm sạch (uncleaned)
- ❌ Chưa validate (unvalidated)
- ❌ Có thể có duplicates, errors, encoding issues
- ✅ Lưu giữ gốc để audit trail

### Files
- `target_urls_fb.txt` - Danh sách URLs Facebook cần crawl

### Format Data
```
Text format: Một URL mỗi dòng
https://www.facebook.com/page1
https://www.facebook.com/page2
```

### Sử Dụng
```python
# Đọc target URLs
with open('datasets/01_raw/target_urls_fb.txt') as f:
    urls = [line.strip() for line in f]

# Crawl từ URLs này
for url in urls:
    data = collect_from(url)
    save_to('datasets/01_raw/apify_raw.json')
```

---

## 🟡 **02_interim/** - SILVER LAYER (Work In Progress)

### Mục Đích
Dữ liệu đang trong quá trình xử lý/làm sạch

### Đặc Điểm
- ⚙️ Đang apply cleaning pipeline
- ⚙️ Language detection in progress
- ⚙️ Tokenization happening
- 🔄 Có thể bị thay đổi/xóa khi hoàn tất

### Sử Dụng
- Temporary staging area trong pipeline
- Không để lâu hạn
- Tự động migrate sang 02_processed/ khi done

---

## 🟡 **02_processed/** - SILVER LAYER (Cleaned, Unlabeled)

### Mục Đích
Dữ liệu đã qua cleaning pipeline nhưng chưa có nhãn chuyên gia

### Đặc Điểm
- ✅ Text cleaned (HTML removed, normalized)
- ✅ Language filtered (Vietnamese only)
- ✅ Tokenized
- ❌ Chưa có labels từ chuyên gia
- 📊 Ready for labeling/annotation

### Files
- `corpus_1856_unlabeled.json` - 1,856 bài viết đã làm sạch, chưa nhãn

### Format
```json
{
  "data": [
    {
      "id": "post_001",
      "text": "Nội dung bài viết đã được làm sạch...",
      "source": "facebook",
      "url": "https://...",
      "timestamp": "2024-01-15T10:30:00Z",
      "language": "vi",
      "platform": "facebook",
      "metadata": {
        "likes": 100,
        "comments": 50,
        "shares": 20
      }
    }
  ]
}
```

### Sử Dụng
```python
import json

# Load unlabeled corpus
with open('datasets/02_processed/corpus_1856_unlabeled.json') as f:
    corpus = json.load(f)

# Dùng để:
# 1. EDA (Exploratory Data Analysis)
# 2. Gửi để annotation
# 3. Active learning sampling
```

---

## 🟢 **03_processed/** - GOLD LAYER (Validated)

### Mục Đích
Dữ liệu chất lượng cao, đã xác nhận bởi chuyên gia, **Master Dataset**

### Đặc Điểm
- ✅ Đã làm sạch + xác nhận chất lượng
- ✅ Chỉ Vietnamese content
- ✅ Có labels từ chuyên gia
- ✅ Duplicate-free
- 🏆 Production-ready
- 🔒 Không thay đổi

### Files
- `reclaimed_master_pool_vn_clean.json` - Master dataset cuối cùng

### Format
```json
{
  "metadata": {
    "total_samples": 8500,
    "language": "vietnamese",
    "quality": "gold",
    "verified_by": "expert_team",
    "date": "2024-03-15"
  },
  "data": [
    {
      "id": "vn_gold_001",
      "text": "Nội dung đã xác nhận...",
      "label": "misinformation",
      "label_confidence": 0.95,
      "annotation_notes": "Các lý do tại sao...",
      "verified_date": "2024-03-10",
      "source": "facebook",
      "url": "https://...",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Sử Dụng
```python
# Load gold standard dataset
import json

with open('datasets/03_processed/reclaimed_master_pool_vn_clean.json') as f:
    gold_data = json.load(f)

# Sử dụng để train/test models
for sample in gold_data['data']:
    text = sample['text']
    label = sample['label']
    
    # Training pipeline
    train_model(text, label)
```

---

## 🟡 **04_silver_labels/** - Auto-Labeled Data

### Mục Đích
Dữ liệu được gán nhãn tự động bằng mô hình/heuristic (chưa verified)

### Đặc Điểm
- 🤖 Nhãn từ mô hình/heuristic
- ⚠️ Chưa verify bởi chuyên gia
- 📊 Có confidence scores
- 🔄 Có thể update khi improve models

### Files
- `annotated_v6.jsonl` - JSONL format (Line-delimited JSON)

### Format (JSONL)
```jsonl
{"text": "nội dung 1", "label": "misinformation", "confidence": 0.87, "model_v": "phobert_v6"}
{"text": "nội dung 2", "label": "safe", "confidence": 0.92, "model_v": "phobert_v6"}
{"text": "nội dung 3", "label": "misleading", "confidence": 0.78, "model_v": "phobert_v6"}
```

### Sử Dụng
```python
# Load JSONL efficiently
data = []
with open('datasets/04_silver_labels/annotated_v6.jsonl') as f:
    for line in f:
        record = json.loads(line)
        data.append(record)

# Filter by confidence
high_conf = [r for r in data if r['confidence'] > 0.9]
```

---

## ⏱️ **temp/** - Temporary/Experiment Data

### Mục Đích
Dữ liệu tạm thời, thử nghiệm, có thể xóa

### Files
| File | Mục Đích | Trạng Thái |
|------|---------|----------|
| `apify_raw_v5.json` | Apify crawl v5 | ⛔ Deprecated |
| `apify_raw_v6.json` | Apify crawl v6 | ✅ Current |
| `apify_raw_v6_5.json` | Apify crawl v6.5 | 🧪 Testing |
| `apify_real_data_v4.json` | Real data Apify v4 | ⛔ Old |
| `targeted_257_posts.txt` | 257 targeted posts | 🧪 Experiment |
| `youtube_cleaned_verified.json` | YouTube verified | ✅ Quality check |

### Sử Dụng
```python
# Load temp data cho testing
import json

with open('datasets/temp/youtube_cleaned_verified.json') as f:
    yt_data = json.load(f)
    
# Verify quality, then migrate to 02_processed/
quality_score = assess_quality(yt_data)
if quality_score > 0.9:
    migrate_to_processed(yt_data)
```

### 🗑️ Cleanup Policy
- Xóa files >3 tháng tuổi nếu không cần
- Migrate data chất lượng cao sang 02_processed/
- Archive old versions nếu cần audit trail

---

## 📈 Data Statistics

### Dataset Size Overview
| Layer | Total | Files | Status |
|-------|-------|-------|--------|
| Bronze | ~50K posts | Multiple | Raw |
| Silver (Interim) | ~10K posts | Processing | WIP |
| Silver (Processed) | 1,856 posts | 1 | Unlabeled |
| Silver (Labels) | 5,000+ posts | 1 | Auto-labeled |
| Gold | 8,500 posts | 1 | Validated ✅ |

---

## 🔄 Data Flow Pipeline

```
Raw Sources (Web, API, Crawl)
          ↓
    [01_raw/] 
    Apify output, API responses
          ↓
    [Cleaning Pipeline]
    - Remove HTML
    - Normalize text
    - Language filter
          ↓
    [02_interim/]
    Processing in progress
          ↓
    [02_processed/]
    Cleaned, unlabeled corpus
          ↓
    [Annotation/Labeling]
    - Expert review
    - Quality assurance
          ↓
    [03_processed/]
    Gold standard (MASTER)
    
    Parallel: [04_silver_labels/]
    Auto-labeled for quick feedback
```

---

## ✅ Best Practices

1. **Không sửa Gold data** (03_processed/) - immutable
2. **Version control** - Sử dụng naming convention: `data_v1.json`, `data_v2.json`
3. **Metadata tracking** - Lưu info: source, date, processing steps
4. **Backup** - Gold datasets backup regularly
5. **Cleanup temp** - Xóa temp/ files khi không cần

---

## 🔐 Data Privacy

- ✅ Chỉ lưu public posts (Facebook public pages)
- ✅ Không lưu PII (personal identifiable information)
- ✅ GDPR compliant (right to be forgotten support)
- ✅ Anonymized when needed

---

**📅 Cập nhật:** April 2026  
**🏗️ Architecture:** Medallion (Bronze-Silver-Gold)  
**📊 Total Size:** ~100K posts across all layers
