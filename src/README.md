# 💻 Thư Mục Mã Nguồn Lõi (src/)

**Cập nhật:** 02/06/2026 · Phiên bản: 2.0 · Trạng thái: ✅ Hoàn tất hệ thống hóa

Thư mục `src/` chứa toàn bộ mã nguồn lõi của dự án VaccineNLP, được thiết kế theo nguyên tắc lập trình hướng cấu trúc (Modular Architecture), có tính tái sử dụng cao và tuân thủ chặt chẽ nguyên lý DRY (Don't Repeat Yourself).

---

## 🗂️ Cấu Trúc Tổng Quan (Source Code Directory)

```
src/
├── __init__.py                  # Đánh dấu Python Package
│
├── 📂 app_core/                 # ⭐ NEW: Shared Runtime Engines
│   ├── __init__.py              # Khởi tạo exports
│   ├── predictor.py             # Động cơ tải mô hình phân loại & Temperature Scaling
│   ├── xai_engine.py            # Động cơ lập luận Chain-of-Thought giải thích y khoa
│   ├── fetchers.py              # Bộ cào dữ liệu liên kết xã hội & parser bài viết/bình luận
│   └── README.md                # Hướng dẫn runtime apps
│
├── 📂 common/                   # 🔧 Quản lý Đường dẫn & Phiên bản chung
│   ├── __init__.py
│   ├── paths.py                 # Centralized Path Registry (Sổ đăng ký đường dẫn chung)
│   ├── versioning_manager.py    # Quản lý versioning dữ liệu/mô hình
│   └── README.md
│
├── 📂 preprocessing/            # 🧹 Pipeline Tiền Xử Lý Dữ Liệu
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrator phối hợp pipeline 6 giai đoạn
│   ├── text_cleaner_v2.py       # Giải thuật làm sạch HTML, teen-code và diacritics
│   ├── vn_tokenizer.py          # Tokenizer tách từ chuẩn hóa pyvi + underthesea
│   ├── language_filter.py       # Bộ lọc ngôn ngữ fasttext (giữ tiếng Việt p > 0.95)
│   ├── ontology_mapper.py       # Ánh xạ từ khóa vào bộ phân loại chủ đề vaccine
│   ├── preprocess_external_data.py # Xử lý chuẩn hóa nguồn dữ liệu ngoại (VFND)
│   └── README.md
│
├── 📂 modeling/                 # 🤖 Động Cơ Huấn Luyện & Đánh Giá
│   ├── __init__.py
│   ├── dataset_loader.py        # PyTorch Dataset & DataLoader
│   ├── phobert_multitask_trainer.py # Kiến trúc mạng 1 Encoder + 3 Heads huấn luyện PhoBERT
│   ├── llm_inference_engine.py  # Động cơ inference mô hình Gemma-4 QLoRA
│   ├── inference.py             # Unified Inference API tích hợp hệ thống Hybrid
│   ├── error_analysis.py        # Module tính toán FP/FN phân tích định tính
│   └── README.md
│
└── 📂 data_pipeline/            # 🔄 Dữ liệu thu thập Apify
    ├── README.md
    ├── collection/              # apify_social_collector_v2.py, master_collector_v2.py
    └── preprocessing/           # Ngôn ngữ lọc và chuẩn hóa thô
```

---

## ⚙️ Logic Hoạt Động & Sự Tương Tác Giữa Các Modules

Toàn bộ hệ thống VaccineNLP vận hành theo một luồng dữ liệu khép kín từ lúc thu thập cho đến khi trực quan hóa kết quả:

```
 Dữ liệu Mạng Xã Hội
        │
        ▼ (cào quét qua Apify)
 src/data_pipeline/ (Thu thập dữ liệu thô Bronze)
        │
        ▼ (làm sạch Unicode, teen-code, tách từ, lọc tiếng Việt)
 src/preprocessing/ (Xử lý dữ liệu Silver Raw)
        │
        ▼ (nạp vào PyTorch DataLoader)
 src/modeling/ (Huấn luyện mô hình PhoBERT-v2 / Gemma QLoRA)
        │
        ▼ (lưu checkpoints vào experiments/models/)
 src/app_core/ (Shared Runtime Engines quản lý load model & XAI)
        │
        ▼ (kết xuất ra giao diện)
 app/ & app_gradio/ (Streamlit slide deck & Gradio HF Space)
```

1. **Giai đoạn Thu thập (data_pipeline):** Scraper cào bài viết và bình luận từ các nguồn công khai Facebook, YouTube, diễn đàn, lưu vào tầng Bronze.
2. **Giai đoạn Tiền xử lý (preprocessing):** Loại bỏ nhiễu HTML, chuẩn hóa bộ gõ tiếng Việt dựng sẵn, loại bỏ emoji gây loãng, phát hiện và tách từ ghép (vd: `vắc_xin`, `an_toàn`), lọc giữ lại nội dung thuần Việt với độ tin cậy > 95%.
3. **Giai đoạn Huấn luyện (modeling):** Phục vụ việc học trọng số mô hình. Mạng đa nhiệm PhoBERT-v2 học song song 3 trục thông tin qua hàm loss tổng hợp có cân bằng trọng số lớp để triệt tiêu ảnh hưởng lệch mẫu.
4. **Giai đoạn Runtime (app_core):** Cung cấp các API dùng chung cho ứng dụng Web. Khi có chuỗi văn bản mới nhập vào, `app_core` sẽ gọi mô hình PhoBERT-v2 dự đoán nhãn thô, nhân với tham số nhiệt độ $T$ để hiệu chuẩn xác suất y tế, đồng thời gọi Gemma CoT lý giải y học dựa trên bằng chứng khoa học.

---

## 🎯 Nguyên Tắc Lập Trình (Best Practices)

Để duy trì tính sạch sẽ và cấu trúc khoa học của mã nguồn, mọi đóng góp hoặc sửa đổi code trong tương lai phải tuân thủ nghiêm ngặt các quy tắc sau:

- **1. Centralized Path Import:** Tuyệt đối không hardcode đường dẫn chuỗi (vd: `datasets/03_processed/...`). Luôn import hằng số từ sổ đăng ký `src.common.paths` (vd: `from src.common.paths import DATA_GOLD_DIR`).
- **2. Modularity & DRY:** Một hàm chỉ làm một việc duy nhất (Single Responsibility). Nếu một khối logic xuất hiện ở cả Streamlit và Gradio, khối logic đó **bắt buộc** phải được chuyển vào `src/app_core/` để dùng chung.
- **3. Cấu hình tách biệt:** Mọi bộ tham số hyperparameter, nhãn lớp hay trọng số nghịch đảo lớp phải được tải từ thư mục `configs/` dưới dạng tệp JSON, không viết trực tiếp (hardcode) trong code logic.
- **4. Type Hints & Docstrings:** Mọi hàm khai báo phải có chú thích kiểu dữ liệu (Type Hints) đầu vào/đầu ra và docstring mô tả chi tiết mục đích, tham số truyền vào và dữ liệu trả về.

---

*VaccineNLP Core Development Team · HUPH 2026*
