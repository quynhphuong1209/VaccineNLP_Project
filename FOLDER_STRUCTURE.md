# 📁 Cây Thư Mục Dự Án VaccineNLP (v2.0 - Updated May 2026)

Tệp này cung cấp sơ đồ cây thư mục hoàn chỉnh, mô tả chi tiết vai trò của từng thư mục và tệp tin trong toàn bộ hệ thống VaccineNLP.

---

## 🏛️ Sơ đồ cây thư mục chi tiết

```text
VaccineNLP_ĐỒ_ÁN/
│
├── 📄 README.md                          # 📝 Tài liệu chính giới thiệu tổng quan dự án
├── 📄 FOLDER_STRUCTURE.md                # 📋 Sơ đồ cây thư mục chi tiết này (v2.0)
├── 📄 DOCUMENTATION_INDEX.md             # 📚 Chỉ mục lục giới thiệu tài liệu & hướng dẫn sử dụng
├── 📄 ARCHITECTURE.md                    # 🏗️ Sơ đồ thiết kế kiến trúc kỹ thuật hệ thống
├── 📄 requirements.txt                   # 📦 Danh sách các thư viện Python phụ thuộc
├── .env.template                         # 📋 Template khai báo biến môi trường mẫu
└── .gitattributes                        # ⚙️ Thuộc tính cấu hình Git
│
├── 📂 configs/                           # ⚙️ THƯ MỤC CẤU HÌNH HỆ THỐNG
│   ├── facebook.json                     # 🔗 Cấu hình các Facebook Fanpage/Group quét tin
│   ├── seeds.json                        # 🌱 Danh sách URLs gợi ý cho bộ quét báo chí
│   ├── taxonomy.json                     # 🏷️ Bộ phân loại chủ đề vaccine (Ontology)
│   ├── class_weights_v2.json             # ⚖️ Trọng số lớp cân bằng mẫu cho hàm loss
│   └── README.md                         # 📘 Mô tả chi tiết vai trò của các tệp cấu hình
│
├── 📂 app/                               # 📱 THƯ MỤC ỨNG DỤNG (STREAMLIT DASHBOARD)
│   ├── streamlit_demo.py                 # 🐍 File mã nguồn chính khởi chạy giao diện
│   ├── xai_cache.json                    # 💾 Bộ đệm lưu trữ giải thích từ Gemma-4 XAI
│   └── README.md                         # 📘 Hướng dẫn cấu trúc 6 tab và cách chạy local
│
├── 📂 datasets/                          # 📊 THƯ MỤC DỮ LIỆU (MEDALLION DATA PIPELINE)
│   ├── 01_raw/                           # 🔴 BRONZE - Dữ liệu thô từ Crawler
│   ├── 02_interim/                       # 🟡 SILVER - Dữ liệu đã chuẩn hóa, tách từ tiếng Việt
│   ├── 03_processed/                     # 🟢 GOLD - Dữ liệu nhãn chuẩn y khoa HUPH (n=186)
│   └── README.md                         # 📘 Mô tả quy trình quản lý và cấu trúc dữ liệu
│
├── 📂 docs/                              # 📚 THƯ MỤC TÀI LIỆU KỸ THUẬT
│   ├── 01_pipeline_architecture.md       # ⚙️ Tài liệu thiết kế luồng xử lý dữ liệu
│   └── README.md                         # 📘 Chỉ mục tài liệu kỹ thuật chuẩn học thuật
│
├── 📂 experiments/                       # 🧪 THƯ MỤC THỰC NGHIỆM
│   ├── results/                          # 📊 Thống kê các chỉ số F1-Score & ma trận nhầm lẫn
│   └── README.md                         # 📘 Thống kê kết quả và đường dẫn model checkpoints
│
├── 📂 notebooks/                         # 📔 THƯ MỤC RESEARCH NOTEBOOKS
│   ├── 01-phobert-multitask-training.ipynb  # 🔬 Notebook huấn luyện PhoBERT-v2 MTL
│   ├── 02-gemma4-4b-qlora-training.ipynb    # 🔬 Notebook tinh chỉnh QLoRA Gemma-4 4B
│   ├── vaccine-nlp-eval-final-t4.ipynb      # 🔬 Notebook đánh giá mô hình thực nghiệm
│   ├── vaccinenlp-model-benchmark-report.ipynb # 🔬 Notebook báo cáo so sánh chéo khoa học
│   └── README.md                         # 📘 Hướng dẫn chạy các notebook trên Kaggle/Colab
│
├── 📂 scripts/                           # 🔧 THƯ MỤC SCRIPTS TỰ ĐỘNG HÓA
│   ├── unify_pipeline.py                 # 🔌 Kịch bản hợp nhất chạy toàn bộ pipeline
│   ├── download_models.py                # 📥 Tải mô hình từ HuggingFace Hub
│   ├── extract_urls_from_apify.py        # 🕸️ Tiền lọc và định cấu trúc dữ liệu thô
│   ├── vaccine_nlp_eval_final_t4.py      # 📈 Script đánh giá F1 thực nghiệm
│   └── README.md                         # 📘 Danh sách scripts tự động hóa dòng lệnh (CLI)
│
└── 📂 src/                               # 💻 THƯ MỤC MÃ NGUỒN CHÍNH (CORE MODULAR LIBRARY)
    ├── common/                           # 🛠️ Trình quản lý paths, logs, seeds dùng chung
    ├── data_pipeline/                    # 🚜 Pipeline thu thập, tiền xử lý và gán nhãn
    ├── modeling/                         # 🤖 Kiến trúc PhoBERT đa nhiệm & Gemma QLoRA XAI
    └── README.md                         # 📘 Giới thiệu nguyên tắc thiết kế Modular của thư viện
```