# 📚 Thư Mục Tài Liệu Kỹ Thuật (Technical Documentation)

Thư mục này lưu trữ các bản thiết kế kiến trúc hệ thống, báo cáo phân tích khoa học và tài liệu lý thuyết nền tảng phục vụ cho đồ án nghiên cứu.

## 📄 Nội dung chính

1.  **`01_PIPELINE_ARCHITECTURE.md`**:
    *   *Nội dung*: Mô tả chi tiết 4 giai đoạn xử lý dữ liệu của hệ thống bao gồm: Thu thập (Crawler Phase), Chuẩn hóa cấu trúc (Standardization Phase), Tiền xử lý ngôn ngữ tự nhiên tiếng Việt (NLP Preprocessing), và Phân lớp lưu trữ theo kiến trúc Medallion (Bronze ➔ Silver ➔ Gold).
2.  **`README.md`**: (Tệp tin này) - Mục lục giới thiệu tài liệu kỹ thuật của dự án.

---

## 🏛️ Tóm tắt các Phân hệ Kiến trúc chính của Dự án

Dự án **VaccineNLP** bao gồm 3 trụ cột kỹ thuật chính:

### 1. Luồng Dữ liệu Medallion (Data Pipeline)
*   **Bronze (Dữ liệu thô)**: Dữ liệu thu thập trực tiếp từ các Facebook Fanpages, báo chí chính thống và diễn đàn qua bộ quét Apify.
*   **Silver (Dữ liệu chuẩn hóa)**: Dữ liệu đã được làm sạch, tách từ (Word Segmentation) bằng thư viện `pyvi` hoặc `underthesea`, và loại bỏ các bài viết rác (spam).
*   **Gold (Tập Gold Test Set)**: Tập dữ liệu 186 mẫu kiểm thử vàng được gán nhãn thủ công chính xác bởi các chuyên gia Y tế công cộng của HUPH, dùng làm thước đo đánh giá hiệu năng khoa học.

### 2. Mô hình phân loại Đa nhiệm (Discriminator - PhoBERT-v2)
*   Được tinh chỉnh (fine-tuned) trên cấu trúc Encoder tiên tiến `vinai/phobert-base-v2`.
*   Huấn luyện theo cơ chế **Multi-task Learning (MTL)** giúp dự đoán đồng thời 3 đầu ra nhãn từ một văn bản duy nhất, giúp tiết kiệm tối đa tài nguyên tính toán và bộ nhớ.

### 3. Trợ lý Giải thích Y khoa (Generative - Gemma-4 4B)
*   Được tinh chỉnh QLoRA 4-bit trên nền tảng **Gemma-2-4B-IT** (phiên bản Unsloth tối ưu).
*   Đảm nhận vai trò **Explainable AI (AI có khả năng giải thích)**, trích xuất chuỗi lập luận (chain-of-thought) chỉ rõ từ khóa độc hại của tin giả và đưa ra cẩm nang xử lý thông tin kịp thời cho cán bộ y tế.
