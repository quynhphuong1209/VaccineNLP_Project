# 📊 Thư Mục Dữ Liệu (datasets/) - v1.2

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Complete

## 🎯 Mục Đích

Quản lý dữ liệu theo **Medallion Architecture**: Bronze → Silver → Gold.

---

## 🏗️ Kiến Trúc Medallion

```
🔴 BRONZE (01_raw/)              - Dữ liệu thô từ nguồn
        ↓
🟡 SILVER (02_interim/)          - Đang xử lý
        ↓
🟡 SILVER (02_processed/)        - Đã làm sạch & unlabeled
        ↓
🟡 SILVER (04_silver_labels/)    - Auto-labeled data
        ↓
🟢 GOLD (03_processed/)          - Expert-reviewed (Ground Truth)
        ↓
🔵 MODEL-READY (05_model_ready/) - Tokenized for training
```

---

## 🔴 **01_raw/** - BRONZE LAYER

### Mục Đích
Lưu trữ dữ liệu thô từ các nguồn (Apify, APIs) mà chưa xử lý

### Đặc Điểm
- ❌ Chưa làm sạch (uncleaned)
- ❌ Chưa validate (unvalidated) 
- ❌ Có duplicates, errors, encoding issues
- ✅ Immutable - Lưu giữ gốc để audit trail

### Files
- `target_urls_fb.txt` - Danh sách URLs Facebook cần crawl
- `apify_raw.json` - Dữ liệu thô từ Apify

### Sử Dụng
```python
with open('datasets/01_raw/target_urls_fb.txt') as f:
    urls = [line.strip() for line in f]
```

---

## 🟡 **02_processed/** - SILVER PROCESSED LAYER

### Mục Đích
Dữ liệu đã làm sạch, chuẩn hóa, sẵn sàng cho annotation

### Files Chính
- `corpus_1856_unlabeled.json` - Corpus chính (1856 samples)

### Đặc Điểm
- ✅ Làm sạch (HTML removed, normalized)
- ✅ Duplicates removed
- ✅ Language filtered (Vietnamese only)
- ❌ Chưa labeled

---

## 🟡 **04_silver_labels/** - AUTO-LABELED DATA

### Mục Đích
Dữ liệu auto-labeled bởi LLM với confidence scores

### Files
- `annotated_v6.jsonl` - Latest auto-labels (JSONL format)
- `label_confidence_scores.json` - Confidence per label

---

## 🟢 **03_processed/** - GOLD LAYER

### Mục Đích
Dữ liệu xác nhận chất lượng cao (Expert-reviewed Ground Truth) - Nguồn dữ liệu cuối cùng cho benchmark

### Files Chính
- `reclaimed_master_pool_vn_clean.json` - Master dataset (500 samples) - Training data
- `gold_test_set_240.json` - Benchmark test set (240 samples) - Final evaluation
- `gold_metadata.json` - Metadata & annotation statistics

### Đặc Điểm
- ✅ Human-reviewed by experts (100% validated)
- ✅ High-confidence labels (inter-annotator agreement > 0.85)
- ✅ Balanced class distribution
- ✅ No duplicates or data leakage
- 🏅 Ready for final evaluation & publication

### Sử Dụng
```python
import json

with open('datasets/03_processed/reclaimed_master_pool_vn_clean.json') as f:
    gold_data = json.load(f)

print(f"Total Gold samples: {len(gold_data)}")
print(f"Fields: {gold_data[0].keys()}")
```

---

## 🔵 **05_model_ready/** - TRAINING-READY DATA

### Mục Đích
Dữ liệu đã convert sang format tối ưu cho training

### Files
- `train_split.pt` - Training set (PyTorch format)
- `val_split.pt` - Validation set
- `test_split.pt` - Test set

### Sử Dụng
```python
import torch
from torch.utils.data import DataLoader

train_dataset = torch.load('datasets/05_model_ready/train_split.pt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## � **05_model_ready/** - TRAINING-READY DATA

### Mục Đích
Dữ liệu đã convert sang format tối ưu cho training, inference, và đánh giá

### Cấu Trúc
```
05_model_ready/
├── train/              # 80% training data (400 samples)
├── val/                # 10% validation (50 samples)  
├── test/               # 10% test set (50 samples)
├── benchmark/          # 240 samples - Independent evaluation set
├── metadata.json       # Split information & statistics
└── README.md          # Format specifications
```

### Files Per Split
- `train_tokenized.pt` - PyTorch tensors (train split)
- `val_tokenized.pt` - PyTorch tensors (validation split)
- `test_tokenized.pt` - PyTorch tensors (test split)
- `*.json` - JSON format (if needed for analysis)

### Sử Dụng
```python
import torch
from torch.utils.data import DataLoader

# Load PyTorch tensors
train_dataset = torch.load('datasets/05_model_ready/train_tokenized.pt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load JSON
import json
with open('datasets/05_model_ready/metadata.json') as f:
    metadata = json.load(f)
    print(f"Split sizes: {metadata['sizes']}")
    print(f"Label distribution: {metadata['label_dist']}")
```

### Benchmark Set (05_model_ready/benchmark/)
Danh sách independent test set cho final evaluation:
```json
{
  "total_samples": 240,
  "misinfo_samples": 28,
  "correct_samples": 158,
  "unlabeled_samples": 54,
  "characteristics": {
    "avg_length": 245,
    "min_length": 50,
    "max_length": 1200
  }
}
```

---

## 📈 Data Pipeline Summary

| Layer | Samples | Process | Output |
|---|---|---|---|
| 01_raw/ | ~3000 | Raw collection | Unstructured |
| 02_processed/ | 1856 | Cleaning & filtering | Clean text |
| 04_silver_labels/ | 1500 | Auto-labeling (LLM) | Provisional labels |
| 03_processed/ | 500 | Expert review | Gold standard |
| 05_model_ready/ | 500 | Tokenization | Model tensors |

### Quality Metrics
| Layer | Status | Quality | Confidence |
|---|---|---|---|
| BRONZE | Raw | ⭐ | - |
| SILVER | Cleaned | ⭐⭐⭐ | 60% |
| GOLD | Expert-reviewed | ⭐⭐⭐⭐⭐ | 95%+ |
| MODEL-READY | Tokenized | ⭐⭐⭐⭐⭐ | 100% |

---

## 🔗 Related Documentation

- [Dataset Card](../docs/02_DATASET_CARD.md) - Detailed schema & statistics
- [Methodology](../docs/03_METHODOLOGY.md) - Data collection & annotation process
- [Pipeline Architecture](../docs/01_PIPELINE_ARCHITECTURE.md) - Full system design

---

*Cập nhật: 21/05/2026 | Phiên bản 1.3 | Status: ✅ Complete & Production-Ready*
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
