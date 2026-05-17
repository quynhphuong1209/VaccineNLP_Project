# 📁 Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam (Folder Structure v1.1 - Updated 23/04/2026)

## 📊 Tổng Quan Kiến Trúc

```
VaccineNLP_Clean_V1/
│
├── 📄 README.md                          # 📝 Tài liệu chính của dự án (v1.1)
├── 📄 FOLDER_STRUCTURE.md                # 📋 Tài liệu này - Hướng dẫn cây thư mục chi tiết (v1.1)
├── 📄 DOCUMENTATION_INDEX.md             # 📚 Index tài liệu + hướng dẫn sử dụng (v1.3)
├── 📄 ARCHITECTURE.md                    # 🏗️ Blueprint toàn bộ dự án & tài liệu status (v3.2)
├── 📄 requirements.txt                   # 📦 Danh sách thư viện Python cần thiết
├── .env                                  # 🔐 Biến môi trường cục bộ (không commit)
├── .env.template                         # 📋 Template biến môi trường
├── .venv/                                # 🐍 Virtual Environment Python
├── .agent/                               # 🤖 Agent Configuration & Blueprints (v3.2)
│   ├── ARCHITECTURE.md                   # 🏗️ Detailed architecture (v3.2 - Final)
│   ├── rules/                            # Hard constraints & guidelines
│   ├── skills/                           # Core NLP & ML skills
│   └── workflows/                        # Research workflows
├── .kaggle/                              # 📊 Kaggle API credentials (không commit)
├── .kaggle-outputs/                      # 📊 Kaggle competition outputs
│
├── 📂 configs/                           # ⚙️ THƯ MỤC CẤU HÌNH
│   ├── facebook.json                     # 🔗 Cấu hình Facebook Pages (URLs, access tokens)
│   ├── seeds.json                        # 🌱 Danh sách seed URLs ban đầu cho thu thập dữ liệu
│   ├── taxonomy.json                     # 📚 Bộ phân loại chủ đề vaccine (ontology)
│   └── [README]                          # Mô tả: Tập trung các file cấu hình JSON cho các thành phần hệ thống
│
├── 📂 app/                              # 📱 THƯ MỤC ỨNG DỤNG (STREAMLIT DASHBOARD & DEMO)
│   ├── streamlit_demo.py                 # 🚀 Streamlit App chính (3 Features: Classification, XAI, Benchmark)
│   ├── xai_cache.json                    # 💾 XAI Cache: Lưu sẵn 186 reasoning từ Teacher Model
│   ├── requirements_demo.txt             # 📦 Dependencies cho demo
│   ├── README.md                         # 📋 Hướng dẫn chạy demo (v1.3 - 23/04)
│   ├── README_demo.md                    # 📚 Extended demo documentation
│   └── [README]                          # Mô tả: Giao diện Streamlit + XAI Cache
│
├── 📂 datasets/                          # 📊 THƯ MỤC DỮ LIỆU (Medallion Architecture)
│   ├── 01_raw/                           # 🔴 BRONZE - Dữ liệu thô từ nguồn
│   │   ├── target_urls_fb.txt           # Danh sách URL Facebook cần crawl
│   │   └── [README]                      # Mô tả: Dữ liệu gốc trực tiếp từ Apify/APIs
│   │
│   ├── 02_interim/                       # 🟡 SILVER - Dữ liệu trung gian đang xử lý
│   │   └── [README]                      # Mô tả: Dữ liệu sau khi làm sạch cơ bản (cleaning phase)
│   │
│   ├── 02_processed/                     # 🟡 SILVER - Dữ liệu đã xử lý lần 1
│   │   ├── corpus_1856_unlabeled.json   # Corpus 1,856 bài viết chưa nhãn từ quá trình first-pass
│   │   └── [README]                      # Mô tả: Dữ liệu sau preprocessing, chưa có nhãn chuyên gia
│   │
│   ├── 03_processed/                     # 🟢 GOLD - Dữ liệu đã xác nhận (Public Benchmark)
│   │   ├── benchmark_test_set.jsonl     # 🎯 Bộ đề thi Benchmark (186 mẫu) - ĐÃ PUBLIC
│   │   ├── reclaimed_master_pool_vn_clean.json  # Master dataset được xác thực (chỉ VN, đã làm sạch)
│   │   └── [README]                             # Mô tả: Dữ liệu chất lượng cao, dùng cho đánh giá cuối cùng
│   │
│   ├── 04_silver_labels/                 # 🟡 SILVER - Nhãn tự động sinh
│   │   ├── annotated_v6.jsonl           # Dataset v6 đã được gán nhãn (JSONL format)
│   │   └── [README]                      # Mô tả: Dữ liệu có nhãn từ mô hình/heuristic
│   │
│   ├── temp/                             # ⏱️ TỰA THỜI - Dữ liệu tạm thời (có thể xóa)
│   │   └── [README]                       # Mô tả: Thư mục tạm để lưu các bản thử nghiệm
│   │
│   └── __init__.py                       # Đánh dấu thư mục là Python package
│
├── 📂 docs/                              # 📚 THƯ MỤC TÀI LIỆU (DOCUMENTATION)
│   ├── 01_PIPELINE_ARCHITECTURE.md      # 🏗️ Kiến trúc Pipeline
│   ├── 02_DATASET_CARD.md               # 📋 Dataset Card (Metadata & Stats)
│   ├── 03_METHODOLOGY.md                # 🔬 Phương pháp luận (Experiments & Metrics)
│   └── [README]                          # Mô tả: Tài liệu kỹ thuật chuẩn học thuật
│
├── 📂 experiments/                       # 🧪 THƯ MỤC THỰC NGHIỆM & KẾT QUẢ
│   ├── results/                          # 📈 Báo cáo kết quả (JSON/CSV) - ĐÃ PUBLIC
│   │   ├── benchmark_xai_cache.json     # Cache lý luận của Gemma-4 cho Dashboard
│   │   ├── gemma_v2_results.json        # Metrics F1 chốt hạ của Gemma-4
│   │   ├── phobert_v2_results.json      # Metrics F1 chốt hạ của PhoBERT-v2
│   │   └── xlmr_v1_results.json         # Metrics F1 chốt hạ của XLM-R-v1
│   │
│   └── models/                           # 🤖 Lưu trữ trọng số mô hình (Git Ignored)
│       └── [README]                      # Mô tả: Checkpoints, weights (Tải từ Hugging Face)
│
├── 📂 notebooks/                         # 📔 THƯ MỤC JUPYTER NOTEBOOKS
│   ├── 01_phobert_multitask_training.ipynb
│   │   # Huấn luyện PhoBERT cho multitask learning (Classification + CoT generation)
│   │   # Chứa: EDA, data loading, model architecture, training loop, evaluation
│   │
│   ├── 02_gemma4_4b_qlora_training.ipynb
│   │   # Huấn luyện Gemma-4 4B với QLoRA (4-bit quantization) để Knowledge Distillation
│   │   # Chứa: Cấu hình QLoRA, loading teacher predictions, fine-tuning, inference
│   │
│   ├── 03_gemma4_inference_eval.ipynb
│   │   # Inference & Evaluation cho Gemma-4 model
│   │   # Chứa: Model loading, batch inference, metrics evaluation, result analysis
│   │   # Notebook Status: 📔 Updated (v1.1)
│   │
│   ├── vaccine-nlp-gemma-eval.ipynb (multiple versions)
│   │   # Evaluation notebooks cho Gemma trên vaccine dataset
│   │   # Versions: v5, variations for different experiments
│   │   # Status: 📔 Multiple versions maintained
│   │
│   ├── kernel-metadata.json              # 🔧 Jupyter kernel configuration
│   ├── kernel-metadata-bak.json          # 🔧 Backup kernel metadata
│   ├── kernel-metadata-test.json         # 🔧 Test kernel metadata
│   ├── test_push.py                      # 🧪 Script kiểm tra push functionality
│   │   └── [Various debug notebooks]
│   │
│   └── [README]                          # 📋 Mô tả: Kịch bản huấn luyện tương tác, EDA, thử nghiệm (v1.1)
│

├── 📂 src/                               # 💻 THƯ MỤC MÃ NGUỒN CHÍNH
│   ├── __init__.py                       # Đánh dấu thư mục là Python package
│   │
│   ├── 📂 common/                        # 🔧 SHARED UTILITIES (Các tiện ích chung)
│   │   ├── __init__.py
│   │   ├── paths.py                      # 📍 Quản lý đường dẫn tập trung
│   │   │   # Định nghĩa: BASE_DIR, DATA_DIR, MODEL_DIR, LOG_DIR
│   │   │   # Sử dụng: Tất cả modules để truy cập paths nhất quán
│   │   │
│   │   ├── versioning_manager.py         # 📌 Quản lý version dữ liệu & mô hình
│   │   │   # Chức năng: Version tracking, compatibility checking
│   │   │   # Output: Version metadata files
│   │   │
│   │   └── [README]                      # Mô tả: Utilities dùng chung toàn dự án
│   │
│   ├── 📂 data_pipeline/                 # 🔄 PROOF OF WORK - Pipeline Thu thập & Tiền xử lý
│   │   ├── README.md                     # Tài liệu pipeline (xem chi tiết dưới)
│   │   │
│   │   ├── 📂 collection/                # 🕷️ PHASE 1: DATA COLLECTION (Thu thập dữ liệu)
│   │   │   ├── master_collector_v2.py    # 🎯 Collector chính - Điều phối tất cả nguồn
│   │   │   │   # Tích hợp: FacebookCollector, ApifySocialCollector
│   │   │   │   # Output: Unified JSON format
│   │   │   │
│   │   │   ├── facebook_page_collector.py # 📘 Facebook Pages Collector
│   │   │   │   # Sử dụng: Facebook Graph API
│   │   │   │   # Config: configs/facebook.json (access token, page IDs)
│   │   │   │   # Output: posts, comments, engagement metrics
│   │   │   │
│   │   │   ├── apify_social_collector_v2.py # 🔗 Apify Multi-Platform Collector
│   │   │   │   # Nền tảng: Facebook, TikTok, YouTube, RSS feeds
│   │   │   │   # Actor: Apify actors + custom routing logic
│   │   │   │   # Output: posts, videos, comments, metadata
│   │   │   │
│   │   │   ├── 📂 actor_configs/        # ⚙️ Cấu hình Apify Actors
│   │   │   │   ├── config_facebook_pages.json     # Config Facebook actor
│   │   │   │   ├── config_tiktok_accounts.json    # Config TikTok actor
│   │   │   │   ├── config_web_sources.json        # Config Web scraper
│   │   │   │   ├── config_youtube_channels.json   # Config YouTube actor
│   │   │   │   ├── rss_feeds.json                 # Danh sách RSS feeds
│   │   │   │   ├── threads.json                   # Config Threads platform
│   │   │   │   ├── tiktok.json                    # TikTok-specific config
│   │   │   │   └── youtube.json                   # YouTube-specific config
│   │   │   │
│   │   │   └── [README]                  # Mô tả: Data collection modules
│   │   │
│   │   ├── 📂 preprocessing/             # 🧹 PHASE 2: DATA PREPROCESSING (Tiền xử lý)
│   │   │   ├── pipeline.py               # 📋 Pipeline chính - Orchestrator
│   │   │   │   # Flow: Load -> Clean -> Tokenize -> Filter -> Validate -> Save
│   │   │   │   # Error handling: Logging, rollback
│   │   │   │
│   │   │   ├── text_cleaner_v2.py        # 🧼 Text Cleaning Module
│   │   │   │   # Chức năng: Remove HTML, normalize unicode, lowercase (tuỳ cấu hình)
│   │   │   │   # Xử lý: Emoji removal, whitespace normalization, accent handling
│   │   │   │
│   │   │   ├── language_filter.py        # 🗣️ Language Detection & Filtering
│   │   │   │   # Sử dụng: fasttext, langdetect libraries
│   │   │   │   # Output: Filter non-Vietnamese content
│   │   │   │
│   │   │   ├── vn_tokenizer.py           # 🔤 Vietnamese Tokenizer
│   │   │   │   # Sử dụng: pyvi, underthesea for Vietnamese NLP
│   │   │   │   # Output: Tokenized text, part-of-speech tags
│   │   │   │
│   │   │   └── [README]                  # Mô tả: Preprocessing modules
│   │   │
│   │   └── README.md                     # 📝 Pipeline overview & architecture
│   │
│   ├── 📂 modeling/                      # 🤖 PHASE 3-5: MODELING & INFERENCE
│   │   ├── __init__.py
│   │   │
│   │   ├── dataset_loader.py             # 📂 Data Loading Module
│   │   │   # Chức năng: Load từ JSON/JSONL, PyTorch DataLoader, batching
│   │   │   # Hỗ trợ: Train/val/test split, data augmentation
│   │   │
│   │   ├── phobert_multitask_trainer.py  # 🧠 PhoBERT Multitask Trainer
│   │   │   # Kiến trúc: PhoBERT base + 2 classification heads
│   │   │   # Task 1: Binary/Multi-class classification (misinfo detection)
│   │   │   # Task 2: CoT generation (Chain-of-Thought reasoning)
│   │   │   # Loss: Combined loss từ 2 tasks
│   │   │   # Output: model checkpoints + predictions
│   │   │
│   │   ├── llm_inference_engine.py       # 🚀 LLM Inference Engine
│   │   │   # Mô hình: Gemma-4 4B (QLoRA fine-tuned)
│   │   │   # Tính năng: Batch inference, streaming response
│   │   │   # Deployment: Flask/FastAPI server
│   │   │   # Performance: GPU optimization, caching
│   │   │
│   │   ├── inference.py                  # 🔮 Unified Inference Module
│   │   │   # Chức năng: High-level API cho PhoBERT + Gemma inference
│   │   │   # Input: Text | Output: Classification + CoT explanation
│   │   │
│   │   └── [README]                      # Mô tả: Modeling, training, inference modules
│   │
│   ├── 📂 preprocessing/                 # 🧹 ACTIVE PREPROCESSING MODULE (Tiền xử lý văn bản)
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # 📋 Main preprocessing orchestrator
│   │   ├── text_cleaner_v2.py            # 🧼 HTML removal, Unicode normalization
│   │   ├── vn_tokenizer.py               # 🔤 Vietnamese tokenization (pyvi, underthesea)
│   │   ├── language_filter.py            # 🗣️ Language detection (fasttext)
│   │   ├── ontology_mapper.py            # 🗂️ Ontology Mapping (text → taxonomy)
│   │   ├── preprocess_external_data.py   # 🌐 External data (VFND API, etc.)
│   │   └── [README]                      # 📋 Mô tả: Text preprocessing module (v1.0 - NEW 23/04)
│   │
│   ├── 📂 data_pipeline/                 # 🗄️ ARCHIVE: PROOF OF WORK (Thu thập & Tiền xử lý Phase 1-2)
│   │   ├── collection/                   # 🕷️ Data collection (archived)
│   │   ├── preprocessing/                # 🧹 Data preprocessing (archived)
│   │   └── [README]                      # Mô tả: Archive of data collection & preprocessing pipeline
│   │
│   └── [README]                          # Mô tả: Source code modules
│
└── [ROOT README]                         # Mô tả: VaccineNLP project overview, setup instructions
```

---

## 📋 Hướng Dẫn Chi Tiết Từng Thư Mục

### 🔴 **1. configs/** - THÀNH PHẦN CẤU HÌNH

**Mục đích:** Lưu trữ tất cả file cấu hình JSON để điều khiển hành vi của các thành phần hệ thống.

| File | Chức Năng | Ví dụ |
|------|---------|-------|
| `facebook.json` | Cấu hình Facebook Pages API (access tokens, page IDs, fields lấy) | Pages để crawl: vaccine-related pages |
| `seeds.json` | Danh sách URL/account seed ban đầu để crawling | URLs khởi đầu từ FB, TikTok, YouTube |
| `taxonomy.json` | Bộ phân loại chủ đề vaccine (ontology) | Categories: adverse-effects, efficacy, myths, etc. |

**Sử dụng:** Được import bởi `src/data_pipeline/collection/*.py`

---

### 📊 **2. datasets/** - MEDALLION DATA ARCHITECTURE

**Mục đích:** Quản lý dữ liệu phân lớp từ Raw (Bronze) → Cleaned (Silver) → Validated (Gold)

#### 🔴 **01_raw/** - BRONZE LAYER
- **Dữ liệu:** Thô từ nguồn gốc (Apify, APIs, crawlers)
- **Đặc điểm:** Chưa làm sạch, có thể có lỗi, duplicates, encoding issues
- **Files:**
  - `target_urls_fb.txt` - Danh sách URLs Facebook cần crawl

#### 🟡 **02_interim/** - SILVER LAYER (WIP)
- **Dữ liệu:** Đang trong quá trình xử lý
- **Xử lý:** Text cleaning, language detection, tokenization
- **Output:** Chuẩn bị cho next phase

#### 🟡 **02_processed/** - SILVER LAYER (Cleaned, Unlabeled)
- **Dữ liệu:** Đã làm sạch nhưng chưa có nhãn chuyên gia
- **Files:**
  - `corpus_1856_unlabeled.json` - 1,856 bài viết đã làm sạch
- **Format:** JSON chứa: text, source, timestamp, url, metadata

#### 🟢 **03_processed/** - GOLD LAYER (Validated)
- **Dữ liệu:** Chất lượng cao, xác minh bởi chuyên gia
- **Files:**
  - `reclaimed_master_pool_vn_clean.json` - Master dataset (VN only, high-quality)
- **Đặc điểm:** Bộ dữ liệu cuối cùng để huấn luyện mô hình

#### 🟡 **04_silver_labels/** - Auto-Labeled Data
- **Dữ liệu:** Có nhãn từ mô hình/heuristic
- **Files:**
  - `annotated_v6.jsonl` - JSONL format (1 record/dòng)
- **Format:** `{text, label, confidence, model_version}`

#### ⏱️ **temp/** - Temporary/Experiment Data
- **Dữ liệu:** Tạm thời, có thể xóa sau khi thực nghiệm xong
- **Files:**
  - `apify_raw_v5.json`, `v6.json`, `v6_5.json` - Các phiên bản thử Apify
  - `apify_real_data_v4.json` - Dữ liệu thực từ Apify v4
  - `targeted_257_posts.txt` - 257 bài viết mục tiêu
  - `youtube_cleaned_verified.json` - YouTube data đã verify

---

### 📚 **3. docs/** - DOCUMENTATION & RESEARCH

**Mục đích:** Lưu trữ tài liệu kỹ thuật, hướng dẫn, và báo cáo khoa học

| File | Nội Dung |
|------|---------|
| `01_PIPELINE_ARCHITECTURE.md` | 🏗️ Kiến trúc toàn bộ pipeline (Data Flow Diagram, Components) |
| `02_DATASET_CARD.md` | 📊 Dataset Card (Statistics, distributions, data quality metrics) |
| `03_METHODOLOGY.md` | 🔬 Phương pháp luận (Experimental setup, evaluation metrics, baselines) |

---

### 🧪 **4. experiments/** - EXPERIMENTAL TRACKING

**Mục đích:** Lưu trữ kết quả thực nghiệm, mô hình, metrics

```
experiments/
└── models/
    ├── phobert_multitask_v1/
    │   ├── checkpoint-100/
    │   ├── checkpoint-500/
    │   └── pytorch_model.bin
    └── gemma4_qlora_v1/
        ├── adapter_model.bin
        ├── training_args.bin
        └── logs/
```

---

### 📔 **5. notebooks/** - INTERACTIVE DEVELOPMENT

**Mục đích:** Kịch bản huấn luyện, EDA, thử nghiệm

| Notebook | Mục Đích |
|----------|---------|
| `01_phobert_multitask_training.ipynb` | 🧠 PhoBERT training cho multitask (classification + CoT) |
| `02_gemma4_4b_qlora_training.ipynb` | 💡 Gemma-4 4B QLoRA training (Knowledge Distillation) |

**Công cụ:** Jupyter, GPU environment, MLflow integration



### 💻 **7. src/** - CORE SOURCE CODE

#### **src/common/** - Shared Utilities

| Module | Chức Năng |
|--------|---------|
| `paths.py` | 📍 Centralized path management (BASE_DIR, DATA_DIR, MODEL_DIR) |
| `versioning_manager.py` | 📌 Version tracking cho data & models |

#### **src/data_pipeline/** - Data Collection & Preprocessing

**collection/** - Data Sources
- `master_collector_v2.py` - Orchestrator chính
- `facebook_page_collector.py` - Facebook Graph API
- `apify_social_collector_v2.py` - Multi-platform (Apify)
- `actor_configs/` - JSON configs cho mỗi nền tảng

**preprocessing/** - Data Cleaning
- `pipeline.py` - Orchestrator (Load → Clean → Tokenize → Filter)
- `text_cleaner_v2.py` - HTML removal, normalization
- `language_filter.py` - Language detection
- `vn_tokenizer.py` - Vietnamese tokenization

#### **src/modeling/** - Model Training & Inference

| Module | Chức Năng |
|--------|---------|
| `dataset_loader.py` | 📂 PyTorch DataLoader, batching |
| `phobert_multitask_trainer.py` | 🧠 PhoBERT training (2 heads) |
| `llm_inference_engine.py` | 🚀 Gemma-4 inference server |
| `inference.py` | 🔮 Unified high-level API |

#### **src/preprocessing/** - Additional Preprocessing

| Module | Chức Năng |
|--------|---------|
| `pipeline.py` | Tiền xử lý pipeline |
| `preprocess_external_data.py` | Xử lý dữ liệu ngoài |
| `ontology_mapper.py` | 🗂️ Map text → taxonomy |

---

## 🎯 FLOW DỮ LIỆU (Data Flow)

```
Raw Sources (FB, TikTok, YT, RSS)
          ↓
    [apify_multi_router.py]
          ↓
    datasets/01_raw/ (Bronze)
          ↓
[src/data_pipeline/preprocessing/pipeline.py]
          ↓
datasets/02_interim/ → datasets/02_processed/ (Silver - Cleaned)
          ↓
[Labeling/Annotation]
          ↓
datasets/03_processed/ (Gold - Validated)
datasets/04_silver_labels/
          ↓
[src/modeling/phobert_multitask_trainer.py]
          ↓
experiments/models/phobert_* (Checkpoints)
          ↓
[src/modeling/llm_inference_engine.py]
          ↓
✅ Predictions + CoT Explanations
```

---

## 📌 KEY CONVENTIONS

1. **Paths:** Tất cả imports path sử dụng `src/common/paths.py`
2. **Versioning:** Mô hình và dataset có version tracking qua `versioning_manager.py`
3. **MLflow:** Tất cả experiments tracked bằng MLflow
4. **Data Format:** JSON cho metadata, JSONL cho large datasets
5. **Logs:** Toàn bộ logs lưu trong `experiments/` với timestamp

---

## 🚀 HƯỚNG DẪN SỬ DỤNG

### Notebook Training
```bash
jupyter notebook notebooks/01_phobert_multitask_training.ipynb
```

---

**📅 Cập nhật lần cuối:** April 2026  
**👤 Tác giả:** VaccineNLP Research Team  
**📄 Phiên bản:** v1.0 - Clean Version
