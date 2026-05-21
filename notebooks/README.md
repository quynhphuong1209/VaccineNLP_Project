# 📔 Thư Mục Notebooks Nghiên Cứu (Research Jupyter Notebooks)

Thư mục này chứa toàn bộ các Jupyter Notebook (`.ipynb`) dùng để nghiên cứu, thực nghiệm huấn luyện mô hình qua GPU NVIDIA T4 trên các nền tảng đám mây Kaggle và Google Colab.

## 📂 Các Notebooks chính trong dự án

1.  **`01-phobert-multitask-training.ipynb`**:
    *   *Nhiệm vụ*: Huấn luyện mô hình phân loại đa nhiệm PhoBERT-v2 bằng PyTorch. Thiết lập kiến trúc chia sẻ đặc trưng (Shared-bottom Encoder) với 3 đầu ra phân loại độc lập.
2.  **`02-gemma4-4b-qlora-training.ipynb`**:
    *   *Nhiệm vụ*: Huấn luyện tinh chỉnh mô hình sinh ngôn ngữ lớn Gemma-4 4B bằng kỹ thuật QLoRA qua thư viện Unsloth nhằm tối ưu hóa 4-bit và tiết kiệm VRAM.
3.  **`vaccine-nlp-eval-final-t4.ipynb`**:
    *   *Nhiệm vụ*: Script đánh giá tổng hợp, kiểm thử chéo và trích xuất chỉ số thống kê F1-Score trên tập kiểm thử vàng Gold Test Set.
4.  **`vaccinenlp-model-benchmark-report.ipynb`**:
    *   *Nhiệm vụ*: Tạo lập báo cáo khoa học so sánh chéo hiệu năng F1 giữa PhoBERT-v2, XLM-R-v1 và Gemma-4 4B.
5.  **`gemma-e4b-it.ipynb`**:
    *   *Nhiệm vụ*: Thử nghiệm giao tiếp suy luận nhanh với mô hình Gemma-4.

---

## 🚀 Cách chạy các Notebooks trên Kaggle/Colab

Tất cả các notebook đều được cấu hình sẵn để nhận diện môi trường Kaggle tự động:
1.  **Bước 1**: Đẩy tệp `.ipynb` tương ứng lên tài khoản Kaggle Notebook hoặc Google Colab của bạn.
2.  **Bước 2**: Bật môi trường tăng tốc GPU (khuyên dùng **GPU T4 x2** trên Kaggle hoặc **GPU L4/T4** trên Colab).
3.  **Bước 3**: Cài đặt các thư viện cần thiết bằng cách chạy ô cài đặt đầu tiên trong notebook (Unsloth, Transformers, Peft, v.v.).
4.  **Bước 4**: Run All để thực thi toàn bộ tiến trình huấn luyện hoặc đánh giá. Trọng số mô hình sau khi kết thúc sẽ tự động được đóng gói để sẵn sàng đẩy lên HuggingFace Hub.
