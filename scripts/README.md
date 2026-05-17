# 🔧 Thư Mục Công Cụ & Scripts Tự Động Hóa (Scripts)

Thư mục này chứa các tệp kịch bản Python (`.py`) dùng để tự động hóa toàn bộ luồng công việc từ thu thập dữ liệu mạng xã hội, tải mô hình, chạy huấn luyện dòng lệnh, cho tới đánh giá thực nghiệm chéo.

## 📂 Các Scripts chính và Vai trò

1.  **`unify_pipeline.py`**:
    *   *Nhiệm vụ*: Kịch bản hợp nhất điều hành toàn bộ pipeline của dự án từ đầu đến cuối (Crawl ➔ Preprocess ➔ Train ➔ Eval).
2.  **`download_models.py`**:
    *   *Nhiệm vụ*: Tải nhanh các trọng số mô hình PhoBERT-v2, XLM-R-v1, Gemma-4 từ HuggingFace về thư mục lưu trữ cục bộ hoặc bộ đệm hệ thống.
3.  **`extract_urls_from_apify.py`**:
    *   *Nhiệm vụ*: Trích xuất, tiền lọc và định cấu trúc các bài viết từ kết quả quét định dạng JSON thô của Apify về thư mục dữ liệu Bronze (`datasets/01_raw`).
4.  **`01_phobert_multitask_training.py` & `02_gemma4_4b_qlora_training.py`**:
    *   *Nhiệm vụ*: Phiên bản mã nguồn dạng dòng lệnh (CLI script) để huấn luyện PhoBERT và Gemma-4 trực tiếp qua console, thích hợp chạy nền trong các phiên SSH dài.
5.  **`vaccine_nlp_eval_final_t4.py` & `vaccinenlp_model_benchmark_report.py`**:
    *   *Nhiệm vụ*: Chạy đánh giá chéo hiệu năng F1-Score và trích xuất ma trận nhầm lẫn chuẩn xác từ Gold Test Set.
6.  **`reclaim_vn_only.py` & `robust_fetch_vfnd.py`**:
    *   *Nhiệm vụ*: Dọn dẹp dữ liệu, loại bỏ bài viết không phải tiếng Việt và quét bổ sung các bài viết từ tập dữ liệu tin giả Việt Nam (VFND).

---

## 🚀 Cách chạy các công cụ tự động hóa

Bạn có thể chạy các script trực tiếp bằng Python tại terminal ở thư mục gốc của dự án:

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Ví dụ: Tải các mô hình từ HuggingFace
python scripts/download_models.py

# Ví dụ: Chạy toàn bộ pipeline tích hợp
python scripts/unify_pipeline.py
```
