# 🔧 Sổ Đăng Ký Đường Dẫn & Quản Lý Phiên Bản (src/common/)

**Cập nhật:** 02/06/2026 · Phiên bản: 2.0 · Trạng thái: ✅ Đồng bộ thực tế 100%

Thư mục `src/common/` chứa các tệp tin cấu trúc nền tảng (Shared Infrastructure) chịu trách nhiệm quản lý đường dẫn tập tin nhất quán trong toàn dự án và theo dõi tính tương thích phiên bản của dữ liệu và mô hình.

---

## 🗂️ Danh Sách Modules (File Directory)

### 1️⃣ **paths.py** — Sổ Đăng Ký Đường Dẫn Hệ Thống (Central Path Registry)
Tệp tin này định nghĩa toàn bộ cấu trúc đường dẫn thư mục và tệp tin trong VaccineNLP dựa trên thư viện tiêu chuẩn `pathlib`. Việc tập trung đường dẫn giúp dự án triệt tiêu hoàn toàn lỗi đường dẫn tương đối khi chạy ở các thư mục làm việc khác nhau.

**Các Hằng Số Đường Dẫn Thực Tế (Khớp 100% với `paths.py`):**
```python
import os
from pathlib import Path

# Thư mục gốc dự án (VaccineNLP_Clean_V1)
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# Thư mục cấu hình và tài liệu
CONFIGS_DIR = ROOT_DIR / "configs"
DOCS_DIR = ROOT_DIR / "docs"

# Phân tầng dữ liệu theo Kiến trúc Medallion
DATA_RAW_DIR = ROOT_DIR / "datasets" / "01_raw"                 # Bronze Data (Dữ liệu thô từ Apify)
DATA_UNLABELED_DIR = ROOT_DIR / "datasets" / "02_processed"     # Silver Data (Đã sạch, chưa gán nhãn)
DATA_GOLD_DIR = ROOT_DIR / "datasets" / "03_processed"          # Gold Data (Benchmark 186 mẫu có HITL)
DATA_SILVER_DIR = ROOT_DIR / "datasets" / "04_silver_labels"    # Silver Data (Gán nhãn yếu từ LLM 1.670 mẫu)
DATA_MODEL_READY_DIR = ROOT_DIR / "datasets" / "05_model_ready" # Model-Ready Data (Đã tách từ pyvi & phân chia split)
DATA_TEMP_DIR = ROOT_DIR / "datasets" / "temp"                  # Thư mục làm việc tạm thời

# Thư mục quản lý thực nghiệm và checkpoints
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
```

**Hàm Tiện Ích Đăng Ký:**
- `ensure_src_in_sys_path()`: Tự động phát hiện và thêm thư mục `src/` vào biến môi trường hệ thống `sys.path` để mọi module con có thể dễ dàng import lẫn nhau bằng cú pháp tuyệt đối `from src.modeling...` mà không sợ lỗi `ModuleNotFoundError`.

**Cách sử dụng chuẩn trong code:**
```python
from src.common.paths import DATA_GOLD_DIR, MODELS_DIR

# 1. Đọc tệp Gold Test Set chuẩn xác
gold_file = DATA_GOLD_DIR / "benchmark_test_set_v3.jsonl"

# 2. Lưu checkpoint mô hình
model_save_path = MODELS_DIR / "phobert_v2"
```

---

### 2️⃣ **versioning_manager.py** — Bộ Quản Lý Phiên Bản & Tính Tương Thích
Đảm bảo tính tái sản xuất (Reproducibility) khoa học bằng cách đóng gói snapshot dữ liệu và theo dõi tính tương thích giữa dữ liệu huấn luyện và mô hình.

- **Class `VersioningManager`:**
  - `create_version(self, data_path: Path, version_tag: str) -> Path`: Tạo bản sao lưu được đánh dấu phiên bản của tệp dữ liệu (ví dụ: `reclaimed_master_pool_vn_clean_v3.0.json`).
  - `get_latest_version(self, dir_path: Path, pattern: str) -> Path`: Tìm tệp tin có số hiệu phiên bản cao nhất trong thư mục chỉ định.
  - `check_compatibility(self, model_version: str, data_version: str) -> bool`: Quét tệp kê khai (Manifest) của mô hình để kiểm tra xem cấu trúc nhãn của tập dữ liệu huấn luyện có tương thích với số lượng đầu ra của mạng neural PhoBERT-v2 hay không (ví dụ: ngăn chặn việc nạp dữ liệu Stance 4 lớp vào mô hình PhoBERT v3 chỉ hỗ trợ 3 lớp).

---

## 🎯 Quy Tắc An Toàn (Constraints)

1. **Tuyệt đối CẤM hardcode đường dẫn cục bộ:** Không sử dụng các chuỗi tuyệt đối như `"D:\VaccineNLP_Clean_V1\datasets..."` hoặc các chuỗi tương đối như `"../../datasets/"`. Luôn sử dụng hằng số import từ `src.common.paths`.
2. **Khởi tạo tự động:** Khi thư mục `src.common.paths` được import, hệ thống sẽ tự động quét và khởi tạo các thư mục trống còn thiếu (Bronze, Silver, Gold, Models, Results) trên hệ đĩa cứng để tránh lỗi hệ thống ghi file `FileNotFoundError`.

---

*VaccineNLP Infrastructure Team · HUPH 2026*
