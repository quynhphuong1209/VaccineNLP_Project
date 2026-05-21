# 📚 Thư Mục Tài Liệu (docs/)

**Cập nhật:** 21/05/2026 | Phase 6 — Thesis Finalization
**Trạng thái:** ✅ Complete (6 technical docs + 1 roadmap)

---

## Mục Đích

Lưu trữ tài liệu kỹ thuật, báo cáo khoa học, phương pháp luận, và thiết kế hệ thống cho dự án VaccineNLP. Đây là nơi chứa toàn bộ **kiến trúc, dữ liệu, kết quả thực nghiệm, và lộ trình tương lai** của luận văn.

---

## 📋 Danh Sách Tài Liệu (8 files)

| # | Tệp | Mô Tả | Phiên Bản |
|---|------|--------|----------|
| 1 | **FINAL_TECHNICAL_REPORT.md** | Báo cáo kỹ thuật tổng thể — xương sống luận văn | ✅ Final |
| 2 | **01_PIPELINE_ARCHITECTURE.md** | Kiến trúc Medallion + HuggingFace Hub + Benchmark | v3.1 |
| 3 | **02_DATASET_CARD.md** | Dataset Card: taxonomy 3 trục, schema JSONL, thống kê Gold Test Set | v2.0 |
| 4 | **03_METHODOLOGY.md** | Phương pháp 5 phases + Benchmark Results + Novelty | v2.0 |
| 5 | **04_FUTURE_WORKS_XAI.md** | Blueprint: Real-time XAI với LM Studio (frozen) | v1.0 |
| 6 | **DEPLOYMENT_GUIDE.md** | Hướng dẫn deploy ứng dụng lên Streamlit Community Cloud | v1.0 |
| 7 | **FOLDER_STRUCTURE.md** | Cây thư mục public sanitized (auto-generated) | ✅ Final |
| 8 | **DOCUMENTATION_INDEX.md** | Master index toàn bộ tài liệu dự án | v2.0 |

> Ngoài ra còn có `TAXONOMY_CHANGE_LOG.md` ghi lại lịch sử thay đổi hệ thống nhãn.

---

## 📊 Benchmark Results (Tóm tắt)

Nguồn: `experiments/results/*.json` — Kaggle LIVE run 20/05/2026.

| Mô hình | Misinfo (F1) | Stance (F1) | Sentiment (F1) |
|---|:---:|:---:|:---:|
| **PhoBERT-v2** | **0.7079** | **0.7107** | **0.7260** |
| **Gemma-4-4B** | 0.6925 | 0.5818 | 0.7196 |
| **XLM-R-v1** | 0.5823 | 0.4217 | 0.1842 |

---

## 🔗 Liên kết Nhanh

| Nếu bạn muốn... | Đọc |
|-----------------|------|
| Hiểu toàn bộ dự án | `README.md` (root) |
| Xem cây thư mục | `FOLDER_STRUCTURE.md` |
| Xem dữ liệu chi tiết | `02_DATASET_CARD.md` |
| Xem phương pháp nghiên cứu | `03_METHODOLOGY.md` |
| Xem kiến trúc hệ thống | `01_PIPELINE_ARCHITECTURE.md` |
| Xem kết quả chi tiết | `experiments/results/benchmark_report.md` |
| Xem thiết kế tương lai | `04_FUTURE_WORKS_XAI.md` |

---

## 🔐 Lưu ý Bảo mật

- ✅ Tài liệu public: kiến trúc, phương pháp luận, kết quả
- ❌ Không lưu API keys, credentials
- ❌ Không đề cập `scratch/`, `scripts/`, `_archive/` trong docs public

---

*Cập nhật: 20/05/2026 | Owner: Kim Mạnh Hưng | MSSV: 2211090016*
