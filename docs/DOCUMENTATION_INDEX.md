# 📚 Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam - Tài Liệu Hệ Thống Hoàn Chỉnh

**Cập nhật:** April 23, 2026 (Lần 3 - FINAL Complete)
**Phiên bản:** v1.3 - Full Documentation Coverage
**Ngôn ngữ:** Tiếng Việt + Code  
**Notebooks:** 8 files (4 unique + variations)  
**Total READMEs:** 18 files ✅

---

## 🤖 Hugging Face Model Hub

| Model ID | Task | Size |
| :--- | :--- | :--- |
| [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai) | XAI / CoT | 4B (4-bit) |
| [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask) | Multitask Classification | 540MB |
| [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask) | Multitask Baseline | 1.1GB |

---

## 🎯 Mục Đích Tài Liệu (Updated v1.2)

Tài liệu này cung cấp **cây thư mục chi tiết, mô tả từng file, hướng dẫn chức năng, và trạng thái hiện tại** cho toàn bộ dự án VaccineNLP.

Ngoài ra, tài liệu được **cập nhật toàn bộ lần 2** để phản ánh tất cả các thay đổi cơ cấu dữ liệu và notebooks mới.

1.  **PhoBERT Multitask (Encoder)**: Sử dụng kỹ thuật *Weighted CrossEntropy* và `class_weight='balanced'` để xử lý triệt để bài toán mất cân bằng dữ liệu. Mô hình hiện có sẵn tại: [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask).
2.  **XLM-R Multitask (Baseline)**: Mô hình đa ngôn ngữ phục vụ so sánh baseline. Có sẵn tại: [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask).
3.  **Gemma-4 4B QLoRA (Decoder)**: Huấn luyện theo định dạng Chat Template, bắt chước khả năng lý luận của LLM annotator. Quá trình Inference đánh giá F1-score được thực thi trên Kaggle Cloud để tận dụng phần cứng. Tại bước này, dự án áp dụng cơ chế **Robust Parsing** kết hợp với **Prompt Engineering** cực kỳ chi tiết nhằm chuẩn hóa định dạng kết quả, đồng thời thực hiện **Raw Response Logging** để đảm bảo không thất thoát dữ liệu do sự bất ổn định về cấu trúc trả lời của mô hình ngôn ngữ lớn. Mô hình có sẵn tại: [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai).
- ✅ Tìm kiếm tệp cần thiết nhanh chóng
- ✅ Biết mỗi folder làm gì
- ✅ Theo dõi data flow (Bronze → Silver → Gold)
- ✅ Xem cây thư mục chi tiết

---

## 📋 Danh Sách Tài Liệu Được Tạo & Cập Nhật (v1.3 - FINAL)

### 🌟 Tài Liệu Chính (4 files)

| Tệp | Vị Trí | Mục Đích | Phiên Bản |
|------|--------|---------|----------|
| **ARCHITECTURE.md** | .agent/ | 🏗️ Blueprint & giải pháp toàn bộ dự án | v3.2 ✅ |
| **FOLDER_STRUCTURE.md** | Root | 📊 Cây thư mục toàn bộ + mô tả chi tiết | v1.1 ✅ |
| **DOCUMENTATION_INDEX.md** | Root | 📚 Index tài liệu (tài liệu này) | v1.3 ✅ |
| **README.md** | Root | 📄 Mô tả dự án tổng quát | v1.1 ✅ |

### 📂 README.md cho Các Thư Mục Chính (8 files)

| Thư Mục | README | Mô Tả | Phiên Bản |
|---------|--------|------|----------|
| **docs/** | docs/README.md | 📚 Tài liệu khoa học & future roadmap (4 docs) | v1.1 ✅ |
| **notebooks/** | notebooks/README.md | 📔 Jupyter notebooks (4 types + 8 files) | v1.1 ✅ |
| **experiments/** | experiments/README.md | 🧪 Experiment tracking, MLflow, checkpoints | v1.0 ✅ |
| **datasets/** | datasets/README.md | 📊 Medallion Architecture (Bronze-Silver-Gold) | v1.0 ✅ |
| **configs/** | configs/README.md | ⚙️ Cấu hình Facebook, Seeds, Taxonomy | v1.0 ✅ |
| **app/** | app/README.md | 📱 Streamlit Dashboard & XAI Demo | v1.2 ✅ |
| **src/** | src/README.md | 💻 Mã nguồn chính (high-level overview) | v1.0 ✅ |

### 🔧 README.md cho Các Subfolders (8 files)

| Subfolder | README | Mô Tả | Phiên Bản |
|-----------|--------|------|----------|
| **src/common/** | src/common/README.md | 🔧 Shared utilities (paths, versioning) | v1.0 ✅ |
| **src/data_pipeline/** | src/data_pipeline/README.md | 🔄 Data collection & preprocessing archive | v1.0 ✅ |
| **src/modeling/** | src/modeling/README.md | 🤖 Model training & inference | v1.0 ✅ |
| **src/preprocessing/** | src/preprocessing/README.md | 🧹 Text preprocessing (NEW) | v1.0 ✅ NEW |
| **docs/** (files) | 4 technical docs | 📋 Pipeline, Dataset, Methodology, Future Works | v1.0 ✅ |

### 📊 Total Status: **20 Markdown Files** ✅

--- (v1.2)

```
📁 VaccineNLP_Clean_V1
│
├── 🏗️ ARCHITECTURE.md              ← Blueprint dự án (v3.1 - UPDATED)
├── 📋 FOLDER_STRUCTURE.md          ← Cây thư mục chi tiết (v1.0)
├── 📚 DOCUMENTATION_INDEX.md       ← Tài liệu này (v1.2 - UPDATED)
├── 📄 README.md                    ← Mô tả dự án (v1.0)
│
├── 📂 app/                              # 📱 DASHBOARD (NEW v1.2)
│   └── streamlit_demo.py                 # 🚀 Giao diện Demo & Benchmark
│
├── 📂 .agent/
│   └── ARCHITECTURE.md             ← Blueprint toàn bộ hạ tầng
│
├── 📂 docs/
│   ├── 01_PIPELINE_ARCHITECTURE.md
│   ├── 02_DATASET_CARD.md
│   ├── 03_METHODOLOGY.md
│   └── README.md
│
├── 📂 notebooks/
│   ├── 01_phobert_multitask_training.ipynb
│   ├── 02_gemma4_4b_qlora_training.ipynb
│   ├── 03_gemma4_inference_eval.ipynb
│   ├── vaccine-nlp-gemma-eval.ipynb (variations)
│   └── README.md
│

├── 📂 configs/
│   ├── facebook.json, seeds.json, taxonomy.json
│   └── README.md
│
├── 📂 datasets/
│   ├── 01_raw/, 02_interim/, 02_processed/
│   ├── 03_processed/, 04_silver_labels/, 05_model_ready/
│   └── README.md
│
├── 📂 experiments/
│   ├── models/, results/
│   └── README.md
│
└── 📂 src/
    ├── common/
    │   ├── paths.py, versioning_manager.py
    │   └── README.md
    ├── data_pipeline/
    │   ├── collection/, preprocessing/
    │   └── README.md
    ├── modeling/
    │   ├── dataset_loader.py
    │   ├── phobert_multitask_trainer.py
    │   ├── llm_inference_engine.py
    │   └── README.md
    └── README.md
    └── 📂 preprocessing/
        └── Various modules
```

---

## 🔍 Hướng Dẫn Sử Dụng Tài Liệu

### Nếu bạn muốn...

#### **Hiểu toàn bộ dự án**
1. Đọc [README.md](README.md) - Overview
2. Đọc [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) - Cây thư mục chi tiết

#### **Hiểu cấu trúc dữ liệu**
1. Đọc [datasets/README.md](datasets/README.md) - Medallion architecture
2. Tìm file dữ liệu trong `datasets/01_raw/`, `02_processed/`, `03_processed/`

#### **Chạy data collection**
1. Đọc [configs/README.md](configs/README.md) - Setup cấu hình
2. Xem [src/data_pipeline/README.md](src/data_pipeline/README.md) - Pipeline details

#### **Huấn luyện mô hình**
1. Đọc [notebooks/README.md](notebooks/README.md) - Jupyter workflows
2. Chạy [notebooks/01_phobert_multitask_training.ipynb](notebooks/01_phobert_multitask_training.ipynb)
3. Xem kết quả trong [experiments/README.md](experiments/README.md)

#### **Chạy inference**
1. Đọc [src/modeling/README.md](src/modeling/README.md) - Model APIs
2. Dùng inference endpoint từ LLM server

#### **Tìm hiểu code source**
1. Đọc [src/README.md](src/README.md) - Overview
2. Đọc các subfolders:
   - [src/common/README.md](src/common/README.md) - Utilities
   - [src/data_pipeline/README.md](src/data_pipeline/README.md) - Data pipeline
   - [src/modeling/README.md](src/modeling/README.md) - Model training

#### **Hiểu phương pháp luận**
1. Đọc [docs/README.md](docs/README.md) - Overview
2. Xem [docs/01_PIPELINE_ARCHITECTURE.md](docs/01_PIPELINE_ARCHITECTURE.md) - Architecture
3. Xem [docs/02_DATASET_CARD.md](docs/02_DATASET_CARD.md) - Dataset info
4. Xem [docs/03_METHODOLOGY.md](docs/03_METHODOLOGY.md) - Research methodology

---

## 📚 Chi Tiết Từng README

### 1. FOLDER_STRUCTURE.md (ROOT)
**📊 Cây thư mục chi tiết + mô tả chức năng**
- Cây thư mục đầy đủ với ASCII art
- Mô tả từng folder
- Data flow diagram
- Key conventions

### 2. configs/README.md
**⚙️ Quản lý cấu hình**
- facebook.json - Facebook Pages API config
- seeds.json - Initial data sources
- taxonomy.json - Vaccine ontology
- Environment variables setup

### 3. datasets/README.md
**📊 Dữ liệu phân lớp Medallion**
- 🔴 01_raw/ - Bronze layer (thô)
- 🟡 02_interim/ - Silver layer (processing)
- 🟡 02_processed/ - Silver layer (cleaned)
- 🟢 03_processed/ - Gold layer (validated)
- 🟡 04_silver_labels/ - Auto-labeled
- ⏱️ temp/ - Temporary files
- Data statistics & flow

### 4. docs/README.md
**📚 Tài liệu khoa học**
- 01_PIPELINE_ARCHITECTURE.md - System architecture
- 02_DATASET_CARD.md - Dataset documentation
- 03_METHODOLOGY.md - Research methodology
- Writing guidelines & templates

### 5. experiments/README.md
**🧪 Tracking & Model Management**
- models/ - Trained model artifacts
- mlruns/ - MLflow experiment tracking
- logs/ - Training logs
- results/ - Experiment results
- Comparison tables & metrics

### 6. notebooks/README.md
**📔 Jupyter Training Notebooks**
- 01_phobert_multitask_training.ipynb
  - PhoBERT multitask (classification + CoT)
  - Training, evaluation, visualization
- 02_gemma4_4b_qlora_training.ipynb
  - Gemma-4 4B QLoRA fine-tuning
  - LLM-assisted Annotation Engine
- Best practices for notebook development

### 7. src/README.md
**💻 Source Code Overview**
- High-level overview of all modules
- common/ - Shared utilities
- data_pipeline/ - Collection & preprocessing
- modeling/ - Training & inference
- preprocessing/ - Additional preprocessing
- Import patterns & examples

### 8. src/common/README.md
**🔧 Shared Utilities**
- paths.py - Centralized path management
  - ROOT_DIR, DATA_DIR, MODEL_DIR, LOG_DIR
  - Data layer directories (BRONZE, SILVER, GOLD)
  - Specific file paths
- versioning_manager.py - Version tracking
  - Create versions
  - Compatibility checking
  - Snapshot management

### 10. src/data_pipeline/README.md
**🔄 Data Collection & Preprocessing**
- Pipeline architecture diagram
- collection/ - Data sources
  - master_collector_v2.py - Orchestrator
  - facebook_page_collector.py - Facebook API
  - apify_social_collector_v2.py - Multi-platform
  - actor_configs/ - Platform configs
- preprocessing/ - Data cleaning
  - pipeline.py - Orchestrator
  - text_cleaner_v2.py - HTML/unicode cleaning
  - language_filter.py - Language detection
  - vn_tokenizer.py - Vietnamese tokenization
- Quality metrics & data flow
- Usage examples

### 11. src/modeling/README.md
**🤖 Model Training & Inference**
- Model architectures (PhoBERT, Gemma-4)
- dataset_loader.py - PyTorch DataLoaders
- phobert_multitask_trainer.py - PhoBERT training
- llm_inference_engine.py - Gemma-4 inference
- inference.py - Unified API
- Training workflow
- Server implementation (Flask)

---

## 🔗 Mối Liên Kết Giữa Tài Liệu

```
FOLDER_STRUCTURE.md (Tổng quan)
    │
    ├─→ configs/README.md (Cấu hình)
    │
    ├─→ datasets/README.md (Dữ liệu)
    │   └─→ docs/README.md (Document data)
    │
    ├─→ experiments/README.md (Training)
    │   └─→ notebooks/README.md (Run training)
    │
    └─→ src/README.md (Code)
        ├─→ src/common/README.md (Utilities)
        ├─→ src/data_pipeline/README.md (Pipeline)
        └─→ src/modeling/README.md (Models)
```

---

## 📈 Data Flow Diagram (Mối Liên Kết)

```
configs/
(cấu hình)
    ↓
datasets/01_raw/
(Bronze - thô)
    ↓
src/data_pipeline/preprocessing/
    ↓
datasets/02_processed/
(Silver - làm sạch)
    ↓
notebooks/
(Training)
    ↓
src/modeling/
    ↓
experiments/
(Gold - kết quả)
    ↓
✅ Predictions + CoT Explanations
```

---

## ✅ Danh Sách Kiểm Tra - Bạn Có Tất Cả Tài Liệu Cần Thiết Không?

### Tài Liệu Chính
- ✅ FOLDER_STRUCTURE.md (ROOT)
- ✅ README.md (ROOT - đã có)

### README.md cho Folders Chính
- ✅ configs/README.md
- ✅ datasets/README.md
- ✅ docs/README.md
- ✅ experiments/README.md
- ✅ notebooks/README.md
- ✅ src/README.md

### README.md cho Subfolders
- ✅ src/common/README.md
- ✅ src/data_pipeline/README.md
- ✅ src/modeling/README.md

---

## 🎓 Hướng Dẫn Nhanh

### Bạn là người mới?
1. Đọc: `README.md` (overview)
2. Đọc: `FOLDER_STRUCTURE.md` (cây thư mục)
3. Đọc: `docs/01_PIPELINE_ARCHITECTURE.md` (kiến trúc)

### Bạn muốn chạy dự án?
1. Đọc: `configs/README.md` (setup)

### Bạn muốn phát triển?
1. Đọc: `src/README.md` (code overview)
2. Đọc: Subfolders (`common/`, `data_pipeline/`, `modeling/`)
3. Bắt đầu sửa code

### Bạn muốn huấn luyện mô hình?
1. Đọc: `notebooks/README.md`
2. Mở: `notebooks/01_phobert_multitask_training.ipynb`
3. Chạy: Training cells
4. Xem: `experiments/README.md` để hiểu kết quả

---

## 📞 Liên Hệ & Hỗ Trợ

Nếu có thắc mắc:
1. **Tìm kiếm** trong tài liệu (Ctrl+F)
2. **Kiểm tra** README.md của folder liên quan
3. **Xem** code comments trong source files
4. **Chạy** examples trong các README

---

## 📚 Tài Nguyên Bổ Sung

### Dokumentation được tham khảo:
- [Hugging Face Model Card](https://huggingface.co/docs/hub/datasets)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## 🔐 Bảo Mật & Riêng Tư

- ✅ Không commit `.env` files
- ✅ Không lưu API keys trong code
- ✅ Dùng `.env.template` cho documentation
- ✅ Chỉ lưu public posts (Facebook public pages)
- ✅ GDPR compliant

---

## 📊 Thống Kê Dự Án

| Thống Kê | Số Liệu |
|----------|---------|
| **Tổng folders** | 12 main + sub |
| **Tổng README.md** | 12 files |
| **Tài liệu khoa học** | 3 documents |
| **Notebooks** | 8 notebooks (4 unique + variations) |
| **Dữ liệu** | ~8,000-10,000 posts |

---

**📅 Cập nhật:** April 23, 2026 (Lần 3 - FINAL Complete)  
**👤 Tác giả:** VaccineNLP Research Team  
**📄 Phiên bản:** v1.2
**🔗 License:** CC-BY-4.0

---

## 🙏 Cảm Ơn

Cảm ơn bạn đã sử dụng VaccineNLP! 

Dự án này là kết quả của nỗ lực nghiên cứu sâu trong lĩnh vực:
- 🔬 NLP y tế (Medical NLP)
- 🧠 Explainable AI (XAI)
- 📊 LLM-assisted Annotation
- 🌐 Misinformation Detection

Nếu dự án này hữu ích cho bạn, hãy cite và chia sẻ! 🌟
