# 🧪 Thư Mục Thực Nghiệm & Kết Quả Đánh Giá (Experiments)

Thư mục này dùng để lưu trữ các bảng kết quả F1-Score chi tiết, báo cáo đánh giá thực nghiệm (evaluation reports) và quản lý các checkpoint trọng số mô hình trong quá trình nghiên cứu.

## 📂 Cấu trúc thư mục con

*   **`results/`**:
    *   Chứa các tệp báo cáo số liệu, ma trận nhầm lẫn (confusion matrices), và biểu đồ F1-Score trích xuất từ tập kiểm thử vàng Gold Test Set.
    *   Các số liệu này được nạp động vào giao diện Dashboard Streamlit để vẽ biểu đồ so sánh F1 và ma trận nhầm lẫn chéo.

---

## 🤖 Định vị các Checkpoint Mô hình huấn luyện (Models)

Trong quá trình thực nghiệm, các trọng số mô hình tối ưu đã được đẩy lên nền tảng **HuggingFace** để phục vụ việc chia sẻ và tải nhanh trong ứng dụng local hoặc đám mây:

1.  **PhoBERT-v2 Multitask Classifier** (Mô hình phân loại đa nhiệm tối ưu nhất):
    *   *HuggingFace ID*: `quynhphuong1209/phobert-multitask` hoặc `hung2903/phobert-vaccine-multitask`
    *   *Kiến trúc*: RoBERTa-base (Vietnamese-specific)
    *   *Nhiệm vụ*: Dự đoán cứng nhãn Misinfo, Stance, Sentiment.
2.  **XLM-R-v1 Multitask Classifier** (Baseline đa ngôn ngữ):
    *   *HuggingFace ID*: `quynhphuong1209/xlmr-multitask` hoặc `hung2903/xlmr-vaccine-multitask`
    *   *Kiến trúc*: XLM-RoBERTa-base
    *   *Nhiệm vụ*: Phân loại so sánh chéo hiệu năng.
3.  **Gemma-4-4B XAI Reasoning Engine** (Mô hình lý luận sinh giải thích):
    *   *HuggingFace ID*: `quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai` hoặc `hung2903/gemma-4-E4B-unsloth-vaccine-xai`
    *   *Kiến trúc*: Gemma-2-4B-Instruct (Fine-tuned QLoRA 4-bit)
    *   *Nhiệm vụ*: Suy luận chuỗi lý do gán nhãn và sinh văn bản hướng dẫn xử lý khủng hoảng.
