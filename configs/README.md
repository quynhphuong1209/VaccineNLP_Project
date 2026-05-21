# ⚙️ Cấu Hình Hệ Thống (configs/) - v1.1

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Maintained

## 🎯 Mục Đích
Tập trung quản lý tất cả file cấu hình JSON để điều khiển hành vi của các thành phần hệ thống. Giúp dễ dàng thay đổi cấu hình mà không cần sửa code.

---

## 📋 Danh Sách Files

### 🔗 `facebook.json` - Facebook Pages Configuration
**Mục đích:** Cấu hình Facebook Pages API

**Nội dung:**
```json
{
  "access_token": "YOUR_FACEBOOK_TOKEN",
  "page_ids": [
    "page_id_1",
    "page_id_2"
  ],
  "fields": ["message", "created_time", "likes", "comments"],
  "rate_limit": 100,
  "timeout": 30
}
```

**Sử dụng bởi:**
- `src/data_pipeline/collection/facebook_page_collector.py`

---

### 🌱 `seeds.json` - Initial Data Sources
**Mục đích:** Danh sách URL/account seed ban đầu để khởi động crawling

**Nội dung:**
```json
{
  "facebook_pages": ["URL1", "URL2"],
  "tiktok_accounts": ["@account1", "@account2"],
  "youtube_channels": ["channel_id_1"],
  "rss_feeds": ["feed_url_1"],
  "keywords": ["vaccine", "vaccination"]
}
```

**Sử dụng bởi:**
- `src/data_pipeline/collection/master_collector_v2.py`

---

### 📚 `taxonomy.json` - Vaccine Ontology
**Mục đích:** Bộ phân loại chủ đề vaccine (ontology mapping)

**Nội dung:**
```json
{
  "categories": {
    "adverse_effects": {
      "description": "Tác dụng phụ, phản ứng phụ",
      "keywords": ["side effects", "adverse", "reaction"]
    },
    "efficacy": {
      "description": "Hiệu quả, độ bảo vệ",
      "keywords": ["effectiveness", "protection", "immunity"]
    },
    "myths": {
      "description": "Thông tin sai lệch, tin giả",
      "keywords": ["myth", "fake", "hoax", "false claim"]
    },
    "misinformation": {
      "description": "Thông tin gây hiểu nhầm",
      "keywords": ["misleading", "manipulation"]
    }
  }
}
```

**Sử dụng bởi:**
- `src/preprocessing/ontology_mapper.py`
- `src/modeling/phobert_multitask_trainer.py`

---

## 🔑 Biến Môi Trường (Environment Variables)

Để sử dụng configs, thiết lập biến môi trường trong `.env`:

```bash
FACEBOOK_TOKEN=your_token
APIFY_TOKEN=your_token
DATA_ROOT=path/to/datasets
MODEL_ROOT=path/to/experiments/models
```

---

## 📊 Cách Sử Dụng Configs trong Code

```python
import json
from src.common.paths import CONFIG_DIR

# Load config
with open(CONFIG_DIR / 'facebook.json') as f:
    fb_config = json.load(f)

# Sử dụng
token = fb_config['access_token']
```

---

## 🔄 Phiên Bản Cấu Hình

| File | Phiên Bản | Cập Nhật | Ghi Chú |
|------|---------|---------|--------|
| facebook.json | v1.0 | 2024-01 | Cấu hình mới |
| seeds.json | v2.1 | 2024-02 | Thêm Threads |
| taxonomy.json | v3.0 | 2024-03 | Expand categories |

---

**🔐 Lưu ý:** Không commit `.env` hoặc tokens vào Git. Sử dụng `.env.template` để document các biến cần thiết.
