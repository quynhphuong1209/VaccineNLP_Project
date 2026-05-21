# 📚 Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam - Tài Liệu Hệ Thống Hoàn Chỉnh

**Cập nhật:** 21/05/2026 (Phase 6 — Final)
**Phiên bản:** v2.0
**Ngôn ngữ:** Tiếng Việt + Code
**Notebooks:** 5 files (numbered 01–04)
**Total Documentation Files:** 20+ files ✅

---

## 🤖 Hugging Face Model Hub

| Model ID | Task | Size |
| :--- | :--- | :--- |
| [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai) | XAI / CoT Reasoning | 4B (4-bit QLoRA) |
| [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask) | Multitask Classification | 540MB |
| [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask) | Multitask Baseline | 1.1GB |

---

## 📊 Benchmark Results (Nguồn: `experiments/results/*.json`)

| Mô hình | Misinfo (F1) | Stance (F1) | Sentiment (F1) |
|---|:---:|:---:|:---:|
| **PhoBERT-v2** | 0.6996 | **0.6640** | 0.7266 |
| **Gemma-4-4B** | 0.6377 | 0.6264 | **0.7700** |
| **XLM-R-v1** | **0.7038** | 0.6224 | 0.6866 |

---

## 🎯 Mục Đích Tài Liệu

Tài liệu này cung cấp **cây thư mục chi tiết, mô tả từng file, hướng dẫn chức năng, và trạng thái hiện tại** cho toàn bộ dự án VaccineNLP.

- ✅ Tìm kiếm tệp cần thiết nhanh chóng
- ✅ Biết mỗi folder làm gì
- ✅ Theo dõi data flow (Bronze → Silver → Gold)
- ✅ Xem cây thư mục chi tiết

---

## 📋 Danh Sách Tài Liệu (v2.0)

### 🌟 Tài Liệu Chính

| Tệp | Vị Trí | Mục Đích | Phiên Bản |
|------|--------|---------|----------|
| **README.md** | Root | 🏠 Trang chủ dự án + benchmarks + figures | ✅ Final |
| **FOLDER_STRUCTURE.md** | docs/ | 📊 Cây thư mục public (auto-generated) | ✅ Final |
| **DOCUMENTATION_INDEX.md** | docs/ | 📚 Index tài liệu (tài liệu này) | v2.0 ✅ |

### 📚 Tài Liệu Khoa Học (docs/)

| Tệp | Mô Tả | Phiên Bản |
|------|--------|----------|
| **FINAL_TECHNICAL_REPORT.md** | Báo cáo kỹ thuật tổng thể + TO-DO luận văn + post-hoc calibration và XAI | ✅ Final |
| **01_PIPELINE_ARCHITECTURE.md** | Kiến trúc Medallion + HuggingFace Hub + Multi-source fetcher | v3.1 ✅ |
| **02_DATASET_CARD.md** | Dataset Card: taxonomy, schema, thống kê | v2.0 ✅ |
| **03_METHODOLOGY.md** | 5 phases nghiên cứu + benchmark results sau khi hiệu chuẩn (Temperature Scaling) và Captum XAI | v2.0 ✅ |
| **04_FUTURE_WORKS_XAI.md** | Real-time XAI với LM Studio (frozen) | v1.0 ✅ |
| **DEPLOYMENT_GUIDE.md** | Hướng dẫn deploy ứng dụng lên Streamlit Community Cloud | v1.0 ✅ |
| **TAXONOMY_CHANGE_LOG.md** | Lịch sử thay đổi hệ thống nhãn | ✅ |

*Ghi chú: Phase 6, 7, và 8 (Hiệu chuẩn độ tin cậy, Giải thích AI chuẩn khoa học Captum, và Thu thập đa nguồn) đã được tích hợp đầy đủ và đồng bộ xuyên suốt tài liệu.*

### 📔 Notebooks (Kaggle — 5 files)

| File | Mục Đích | Platform |
|------|----------|----------|
| `01_vaccinenlp-phobert-v2-multitask.ipynb` | PhoBERT Multi-task Training | Kaggle T4x2 |
| `02_vaccinenlp-xlm-r-v1-multitask-classifier.ipynb` | XLM-R Baseline Training | Kaggle T4x2 |
| `03A_vaccinenlp-gemma-4-training.ipynb` | Gemma-4 4B QLoRA Training | Kaggle T4x2 |
| `03B_vaccinenlp-gemma-4-inference.ipynb` | Gemma-4 4B Benchmark Inference | Kaggle T4x2 |
| `04_vaccinenlp-model-benchmark-report.ipynb` | Unified Benchmark Report + Figures | Kaggle |

### 📂 README.md cho Các Thư Mục

| Thư Mục | Mô Tả |
|---------|--------|
| **app/** | Streamlit Demo App + XAI cache |
| **configs/** | Cấu hình taxonomy, class weights |
| **datasets/** | Medallion Architecture (Bronze-Silver-Gold) |
| **experiments/** | Results, figures, benchmark report |
| **notebooks/** | Kaggle training notebooks |
| **src/** | Source code overview |
| **src/common/** | Shared utilities (paths.py) |
| **src/data_pipeline/** | Data collection archive (Proof of Work) |
| **src/modeling/** | Model training & inference scripts |
| **src/preprocessing/** | Text preprocessing modules |

---

## 📁 Cây Thư Mục (Cập nhật 20/05/2026)

```
📁 VaccineNLP_Clean_V1/
│
├── app/                              # 🖥️ Streamlit Demo Application
│   ├── streamlit_demo.py             # Main demo app (offline XAI)
│   └── xai_cache.json                # Pre-computed XAI explanations
│
├── configs/                          # ⚙️ Project configurations
│   ├── class_weights_v2.json
│   └── taxonomy.json
│
├── datasets/                         # 📦 Medallion Data Architecture
│   ├── 03_processed/                 # Gold Test Set (186 mẫu)
│   └── 05_model_ready/              # Final train/test splits
│
├── docs/                             # 📚 Scientific Documentation
│   ├── 01_PIPELINE_ARCHITECTURE.md
│   ├── 02_DATASET_CARD.md
│   ├── 03_METHODOLOGY.md
│   ├── 04_FUTURE_WORKS_XAI.md
│   ├── FINAL_TECHNICAL_REPORT.md
│   ├── FOLDER_STRUCTURE.md
│   ├── TAXONOMY_CHANGE_LOG.md
│   └── DOCUMENTATION_INDEX.md        # ← Tài liệu này
│
├── experiments/                      # 📈 Results & Figures
│   └── results/
│       ├── phobert_v2_results.json
│       ├── xlmr_v1_results.json
│       ├── gemma_v3_results.json
│       ├── gemma_inference_results_v3.jsonl
│       ├── benchmark_report.md
│       └── figures/                  # 6 publication-ready charts (PNG)
│
├── notebooks/                        # 📔 Kaggle Training Notebooks
│   ├── 01_vaccinenlp-phobert-v2-multitask.ipynb
│   ├── 02_vaccinenlp-xlm-r-v1-multitask-classifier.ipynb
│   ├── 03A_vaccinenlp-gemma-4-training.ipynb
│   ├── 03B_vaccinenlp-gemma-4-inference.ipynb
│   └── 04_vaccinenlp-model-benchmark-report.ipynb
│
├── src/                              # 🐍 Source Code
│   ├── common/                       # Shared utilities (paths.py)
│   ├── data_pipeline/                # Proof of Work (Phase 1 archive)
│   ├── modeling/                     # Core training & inference
│   └── preprocessing/               # Text processing (pyvi)
│
└── README.md                         # 🏠 Main project page + benchmarks
```

---

## 🔍 Hướng Dẫn Sử Dụng Tài Liệu

### Nếu bạn muốn...

#### **Hiểu toàn bộ dự án**
1. Đọc `README.md` — Overview + Benchmarks + Figures
2. Đọc `docs/FOLDER_STRUCTURE.md` — Cây thư mục sanitized

#### **Hiểu dữ liệu**
1. Đọc `docs/02_DATASET_CARD.md` — Taxonomy + Statistics
2. Xem `datasets/README.md` — Medallion Architecture

#### **Xem kết quả thực nghiệm**
1. Đọc `experiments/results/benchmark_report.md` — Full per-class breakdown
2. Xem `experiments/results/figures/` — 6 biểu đồ chuẩn luận văn
3. Đọc `docs/03_METHODOLOGY.md` — Phương pháp luận

#### **Huấn luyện mô hình**
1. Đọc `notebooks/README.md` — Notebook workflows
2. Upload notebook lên Kaggle → Chạy trên GPU T4x2
3. Tải output về `experiments/results/`

#### **Chạy Demo**
1. `streamlit run app/streamlit_demo.py`
2. Xem `app/README.md` cho hướng dẫn chi tiết

#### **Cập nhật README benchmarks**
1. Tải JSON mới về `experiments/results/`
2. Chạy: `python scratch/update_readme_benchmarks.py`

---

## 📈 Data Flow Diagram

```
configs/taxonomy.json
       ↓
datasets/01_raw/ (Bronze - thô)
       ↓
src/preprocessing/ (Triple Filter Pipeline)
       ↓
datasets/04_silver_labels/ (Silver - LLM-labeled)
       ↓
datasets/03_processed/ (Gold - Human-validated, 186 mẫu)
       ↓
datasets/05_model_ready/ (Model-ready train/test)
       ↓
notebooks/ (Training trên Kaggle)
       ↓
experiments/results/ (JSON metrics + Figures)
       ↓
README.md (Auto-updated benchmarks)
```

---

## 🔐 Bảo Mật & Riêng Tư

- ✅ Không commit `.env`, `.agent/`, `scratch/`, `_archive/`
- ✅ Không lưu API keys trong code
- ✅ Dùng `src/common/paths.py` thay vì hardcode paths
- ✅ Chỉ lưu public posts (Facebook public pages)
- ✅ GDPR compliant

---

## 📊 Thống Kê Dự Án

| Thống Kê | Số Liệu |
|----------|---------|
| **Tổng documentation files** | 20+ files |
| **Tài liệu khoa học (docs/)** | 7 documents |
| **Notebooks** | 5 notebooks (Kaggle) |
| **Models** | 3 (PhoBERT, XLM-R, Gemma-4) |
| **Gold Test Set** | 186 mẫu Human-validated |
| **Figures** | 6 publication-ready charts |
| **HuggingFace Models** | 3 public repositories |

---

**📅 Cập nhật:** 20/05/2026 (Phase 6 — Final)
**👤 Tác giả:** Kim Mạnh Hưng | MSSV: 2211090016
**📄 Phiên bản:** v2.0
**🔗 License:** CC-BY-4.0
