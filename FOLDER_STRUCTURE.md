# 📁 Cây Thư Mục Dự Án VaccineNLP (v1.0 - Updated May 2026)

## 📊 Tổng Quan Kiến Trúc

```
VaccineNLP_ĐỒ_ÁN/
│
├── 📄 README.md                          # 📝 Tài liệu chính của dự án (tiếng Anh/Việt)
├── 📄 FOLDER_STRUCTURE.md                # 📋 Tài liệu này - Hướng dẫn cây thư mục chi tiết (v1.0)
├── 📄 DOCUMENTATION_INDEX.md             # 📚 Index tài liệu + hướng dẫn sử dụng (v1.2)
├── 📄 ARCHITECTURE.md                    # 🏗️ Blueprint toàn bộ dự án & tài liệu status (v3.1)
├── 📄 requirements.txt                   # 📦 Danh sách thư viện Python cần thiết
├── .env.template                         # 📋 Template biến môi trường
├── .kaggle/                              # 📊 Kaggle API credentials
│
├── 📂 configs/                           # ⚙️ THƯ MỤC CẤU HÌNH
│   ├── facebook.json                     # 🔗 Cấu hình Facebook Pages
│   ├── seeds.json                        # 🌱 Danh sách seed URLs
│   └── README.md                         # Mô tả cấu hình hệ thống
│
├── 📂 app/                               # 📱 THƯ MỤC ỨNG DỤNG (DASHBOARD)
│   └── README.md                         # Giao diện tương tác và trực quan hóa
│
├── 📂 datasets/                          # 📊 THƯ MỤC DỮ LIỆU
│   ├── 01_raw/                           # 🔴 BRONZE - Dữ liệu thô
│   ├── 02_interim/                       # 🟡 SILVER - Đang xử lý
│   ├── 03_processed/                     # 🟢 GOLD - Dữ liệu đã xác nhận
│   └── README.md                         # Quản lý dữ liệu Medallion
│
├── 📂 docs/                              # 📚 THƯ MỤC TÀI LIỆU
│   ├── 01_PIPELINE_ARCHITECTURE.md       # Kiến trúc Pipeline
│   └── README.md                         # Tài liệu kỹ thuật chuẩn học thuật
│
├── 📂 experiments/                       # 🧪 THƯ MỤC THỰC NGHIỆM
│   ├── results/                          # 📈 Báo cáo kết quả (F1, Metrics)
│   └── models/                           # 🤖 Trọng số mô hình (PhoBERT, Gemma)
│
├── 📂 notebooks/                         # 📔 THƯ MỤC NOTEBOOKS
│   └── README.md                         # Kịch bản huấn luyện tương tác
│
├── 📂 scripts/                           # 🔧 THƯ MỤC SCRIPTS
│   └── README.md                         # Công cụ tự động hóa
│
└── 📂 src/                               # 💻 THƯ MỤC MÃ NGUỒN CHÍNH
    ├── common/                           # Tiện ích dùng chung
    ├── data_pipeline/                    # Pipeline thu thập & xử lý
    └── modeling/                         # Huấn luyện & Inference
```