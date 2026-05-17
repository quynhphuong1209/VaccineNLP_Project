# BÁO CÁO KỸ THUẬT TỔNG THỂ VÀ KẾ HOẠCH LUẬN VĂN: Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam

## PHẦN A: TỔNG HỢP TIẾN TRÌNH KỸ THUẬT (TECHNICAL PIPELINE)

### 1. Giai đoạn 1: Thu thập và Tiền xử lý Dữ liệu (Data Pipeline)
- **Quy mô dữ liệu:** Từ tập thô (Bronze) -> Tập tinh chế (Silver) -> Tập kiểm thử (Gold Test Set gồm 186 mẫu).
- **Công cụ sử dụng:** pandas (xử lý dataframe), underthesea (word segmentation cho tiếng Việt), Regular Expressions (chuẩn hóa từ lóng).

### 2. Giai đoạn 2: Gán nhãn hỗ trợ bởi LLM & Human-in-the-Loop (Annotation)
- **Mô hình Oracle:** Sử dụng Gemma 31B để tạo nhãn yếu (Weak labels) và lý luận (Rationales).
- **Đánh giá đồng thuận (Cohen's Kappa):** Được đo lường giữa LLM và Chuyên gia y tế trên 182 mẫu test.
  - Misinformation: 0.0525
  - Stance: 0.2895
  - Sentiment: 0.4712
  - Trung bình: 0.2711
- **Luận điểm rút ra:** Kappa trục Misinfo rất thấp chứng tỏ AI chưa thể thay thế chuyên gia trong việc xác định tính chính xác của thông tin y khoa phức tạp.

### 3. Giai đoạn 3: Động cơ Phân loại (Discriminator - PhoBERT-v2)
- **Vai trò:** Mô hình SLM (Small Language Model) tối ưu cho tốc độ và triển khai thực tế.
- **Kiến trúc:** Multi-task Learning với 3 heads độc lập. Dựa trên pre-trained `vinai/phobert-base-v2`.
- **Kết quả Benchmark (Macro F1):**
  - Misinformation: 0.4547
  - Stance: 0.6608
  - Sentiment: 0.7325

### 4. Giai đoạn 4: Động cơ Giải thích (Explainability - Gemma-4 4B)
- **Vai trò:** Mô hình sinh văn bản cung cấp lý luận minh bạch (Chain-of-Thought).
- **Kỹ thuật huấn luyện:** QLoRA 4-bit Quantization (Rank 16) thông qua framework Unsloth.
- **Kiểm định Parse Failure:** Đạt tỷ lệ thành công 97.8% (182 mẫu OK), tỷ lệ lỗi định dạng 2.2% (4 mẫu Fail) do mô hình sinh câu mào đầu ngoài lề.
- **Kết quả Benchmark (Macro F1):**
  - Misinformation: 0.44
  - Stance: 0.62
  - Sentiment: 0.66

### 5. Giai đoạn 5: Kiến trúc Hybrid System (Deployment)
- **Bản chất:** Dual-Student Hybrid (PhoBERT dự đoán + Gemma giải thích).
- **Triển khai:** Ứng dụng Web Streamlit.
- **Cơ chế an toàn (Safe Mode):** Tích hợp XAI Cache để chạy offline 100%, bảo vệ bằng Fallback cứng (ngừng app nếu thiếu file weights).

---

## PHẦN B: DANH MỤC CÔNG VIỆC LUẬN VĂN (THESIS TO-DO LIST)

### Chương 1 & 2: Mở đầu và Tổng quan tài liệu
- [ ] Định nghĩa bài toán Tin giả Vắc-xin tại VN.
- [ ] Review các bài báo về SLM vs LLM, Knowledge Distillation, LLM-assisted Annotation.

### Chương 3: Phương pháp nghiên cứu
- [ ] Vẽ sơ đồ kiến trúc 3 tầng: Oracle (31B) -> Knowledge Base -> Dual-Student.
- [ ] Viết chi tiết quy trình Human-in-the-Loop và chèn số liệu Kappa.
- [ ] Mô tả thiết kế mạng Multi-task PhoBERT và hàm Loss.

### Chương 4: Thực nghiệm và Đánh giá
- [ ] Mô tả thiết lập tham số (Hyperparameters) của PhoBERT và QLoRA Gemma.
- [ ] Kẻ bảng so sánh F1-Score của 3 mô hình (XLM-R, PhoBERT, Gemma-4).
- [ ] Đưa tỷ lệ Parse Failure (2.2%) vào để đánh giá rủi ro của LLM.

### Chương 5: Thảo luận Kết quả (Discussion)
- [ ] Giải thích nguyên nhân "Student (PhoBERT) vượt Teacher (Gemma)" về điểm F1 (Encoder vs Decoder, Domain-specific vs Multilingual).
- [ ] Biện luận giá trị của Kiến trúc Hybrid: PhoBERT cho Accuracy, Gemma cho Transparency.
- [ ] Hạn chế của đề tài (Closed-world assumption, yêu cầu phần cứng).

### Phụ lục & Chuẩn bị Bảo vệ
- [x] Upload Code lên GitHub (Đã xong).
- [ ] Upload Models lên HuggingFace.
- [ ] Quay Video Demo Offline của Streamlit App.
- [ ] Trả lời 5 câu hỏi Q&A phòng thủ (chuẩn bị bởi Cố vấn học thuật).
