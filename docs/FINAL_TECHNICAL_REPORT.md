# BÁO CÁO KỸ THUẬT TỔNG THỂ VÀ KẾ HOẠCH LUẬN VĂN: Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam

## PHẦN A: TỔNG HỢP TIẾN TRÌNH KỸ THUẬT (TECHNICAL PIPELINE)

### 1. Giai đoạn 1: Thu thập và Tiền xử lý Dữ liệu (Data Pipeline)
- **Quy mô dữ liệu:** Từ tập thô ban đầu (Bronze) -> Tập nhãn bạc thô (Silver Raw gồm 1.856 mẫu) -> Tập mô hình hóa (Silver Train/Val gồm 1.663 mẫu; chia 1.496 train / 167 val) -> Tập kiểm thử độc lập (Gold Test Set gồm 186 mẫu).
- **Công cụ sử dụng:** pandas (xử lý dataframe), underthesea (word segmentation cho tiếng Việt), pyvi (tách từ cho PhoBERT), Regular Expressions (chuẩn hóa từ lóng).

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
- **Kết quả Benchmark (Macro F1) — LIVE 21/05/2026:**
  - Misinformation: 0.6996
  - Stance: 0.6640
  - Sentiment: 0.7266
  - Average: 0.6967 (SOTA Classification Engine)

### 4. Giai đoạn 4: Động cơ Giải thích (Explainability - Gemma-4 4B)
- **Vai trò:** Mô hình sinh văn bản cung cấp lý luận minh bạch (Chain-of-Thought).
- **Kỹ thuật huấn luyện:** QLoRA 4-bit Quantization (Rank 16) thông qua framework Unsloth.
- **Kiểm định Parse Failure:** Đạt tỷ lệ lỗi định dạng **33.3%** trên tập Gold Test (do các mẫu có câu mào đầu hoặc kết luận ngoài lề làm hỏng cấu trúc chuỗi trả ra của mô hình 4B Decoder, được lọc và ghi nhận chi tiết bằng cơ chế **Raw Response Logging**; sẽ tối ưu tiếp bằng phương pháp *answer-first* trong các hướng nghiên cứu tương lai).
- **Kết quả Benchmark (Macro F1) — LIVE 21/05/2026:**
  - Misinformation: 0.6377
  - Stance: 0.6264
  - Sentiment: 0.7700 (vượt PhoBERT 0.7266, SOTA Sentiment task)
  - Average: 0.6780

**Phát hiện khoa học:** Gemma-4 4B đạt F1 Sentiment cao nhất nhưng yếu hơn ở Misinfo (0.6377), phản ánh nguyên lý kiến trúc — Decoder model mạnh ở generative/reasoning task (Sentiment cảm xúc), Encoder model mạnh ở discriminative task (Misinfo phân loại). Đây là minh chứng cho thiết kế Dual-Student Hybrid: 2 mô hình BỔ SUNG nhau, không cạnh tranh.

### 5. Giai đoạn 5: Kiến trúc Hybrid System (Deployment)
- **Bản chất:** Dual-Student Hybrid (PhoBERT dự đoán + Gemma giải thích).
- **Triển khai:** Ứng dụng Web Streamlit.
- **Cơ chế an toàn (Safe Mode):** Tích hợp XAI Cache để chạy offline 100%, bảo vệ bằng Fallback cứng (ngừng app nếu thiếu file weights).

### 6. Giai đoạn 6: Confidence Calibration & Trust Engineering (POST-HOC)
- **Vấn đề khoa học:** Các Transformer hiện đại bị **miscalibration** — confidence trung bình từ Softmax cao hơn accuracy thực tế đáng kể (Guo et al., ICML 2017).
- **Đo lường:** Triển khai **Expected Calibration Error (ECE)** — metric chuẩn industry:
  - PhoBERT Misinfo: ECE = 0.123 (TRƯỚC) → 0.054 (SAU calibration) — giảm 56%
  - PhoBERT Stance: ECE = 0.198 → 0.093 — giảm 53%
  - PhoBERT Sentiment: ECE = 0.144 → 0.081 — giảm 44%
- **Giải pháp:** **Temperature Scaling** — học 1 tham số `T > 1` trên validation set bằng LBFGS:
  - PhoBERT T_misinfo = 1.8197
  - PhoBERT T_stance = 1.6666
  - PhoBERT T_sentiment = 1.3474
- **Triển khai App:** Hiển thị 2 confidence song song trong UI — "Thô" (raw) và "Hiệu chuẩn" (calibrated), giúp người dùng hiểu đúng độ tin cậy thực tế.
- **Reference:** Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.

### 7. Giai đoạn 7: Explainable AI (XAI) Khoa học
- **Phương pháp 1 — Integrated Gradients (Captum):**
  - Tính attribution score trên embedding layer của PhoBERT (n_steps=20)
  - Visualize token-level importance bằng heatmap màu
  - Đây là XAI khoa học chuẩn industry, không phải hot-words hardcoded
- **Phương pháp 2 — Chain-of-Thought (Gemma-4):**
  - Sinh giải thích Tiếng Việt mạch lạc cho từng dự đoán
  - Cache pre-computed cho 186 mẫu Gold Test
  - Live HF Inference API cho text mới

### 8. Giai đoạn 8: Multi-source Data Collection (Demo App)
- **Kiến trúc 3 tầng** trong `app/data_fetchers/`:
  - **Tầng 1 (Instant 1-3s):** Báo điện tử Việt — trafilatura, hỗ trợ 15+ trang
  - **Tầng 2 (Fast 5-15s):** YouTube — yt-dlp, lấy title + description + top comments
  - **Tầng 3 (Slow 30-120s):** Facebook/TikTok/Threads — Apify API với 5-token rotation
- **Use cases:** Demo realtime social listening, phân tích phản hồi cộng đồng theo nguồn, validate model trên dữ liệu wild (chưa qua HITL).

### 9. Giai đoạn 9: Kiểm định Giả thuyết Thống kê (H1 - H4)
- **Công cụ:** Python `scipy.stats` thực hiện kiểm định Chi-Square ($\chi^2$) và G-test (log-likelihood fallback) để kiểm tra mối quan hệ phụ thuộc.
- **Kết quả trên Gold Test Set (186 mẫu):**
  - **H1 (Cảm xúc ↔ Lập trường):** Bác bỏ $H_0$ với $p = 6.8508 \times 10^{-40}$ (Chi-Square, ý nghĩa cực kỳ cao). Lập trường phản đối vắc-xin đi kèm cảm xúc tiêu cực cực đoan (93.8%).
  - **H2 (Tin giả ↔ Tương tác):** Không thể kiểm định do giới hạn dữ liệu (các trường metrics bị strip out trong quá trình chuẩn hóa Gold Set). Đây là hạn chế nghiên cứu chính thức được ghi nhận.
  - **H3 (Kênh nguồn ↔ Tin giả):** Bác bỏ $H_0$ với $p = 0.0021$ (G-test, ý nghĩa thống kê cao). Tin giả tập trung ở Facebook (24.7%) và YouTube (12.2%), báo chí chính thống, học thuật và diễn đàn an toàn (0.0%).
  - **H4 (Lập trường ↔ Tin giả - Bổ sung):** Bác bỏ $H_0$ với $p = 3.6893 \times 10^{-14}$ (Chi-Square, ý nghĩa cực kỳ cao). 50.0% nội dung phản đối chứa tin giả vắc-xin (24/48 mẫu), trong khi ủng hộ là 1.9% (1/54 mẫu) và trung lập là 3.6% (3/84 mẫu).
- **Ý nghĩa:** Cung cấp định lượng truyền thông dịch tễ học thực nghiệm vững chắc để phục vụ lập luận Chương 4 & 5.

---

## PHẦN B: DANH MỤC CÔNG VIỆC LUẬN VĂN (THESIS TO-DO LIST)

### Chương 1 & 2: Mở đầu và Tổng quan tài liệu
- [ ] Định nghĩa bài toán Tin giả Vắc-xin tại VN.
- [ ] Review các bài báo về SLM vs LLM, LLM-assisted Annotation + HITL, Explainable AI.

### Chương 3: Phương pháp nghiên cứu
- [ ] Vẽ sơ đồ kiến trúc 3 tầng: Oracle (31B) -> Knowledge Base -> Dual-Student.
- [ ] Viết chi tiết quy trình Human-in-the-Loop và chèn số liệu Kappa.
- [ ] Mô tả thiết kế mạng Multi-task PhoBERT và hàm Loss.

### Chương 4: Thực nghiệm và Đánh giá
- [ ] Mô tả thiết lập tham số (Hyperparameters) của PhoBERT và QLoRA Gemma.
- [ ] Kẻ bảng so sánh F1-Score của 3 mô hình (XLM-R, PhoBERT, Gemma-4).
- [ ] Đưa tỷ lệ Parse Failure (**33.3%** trên Gold Test set — nguồn: `experiments/results/gemma_v3_results.json` và `GEMINI.md`) vào để phân tích và đánh giá rủi ro của Decoder-only LLM.
- [x] Kiểm định 4 giả thuyết thống kê (H1-H4) và tạo 3 hình biểu đồ phân phối khoa học 300 DPI (`experiments/results/figures/hypothesis_*`).

### Chương 5: Thảo luận Kết quả (Discussion)
- [ ] Phân tích chi tiết "Task Specialization": PhoBERT (Encoder) vượt ở Stance/Sentiment, XLM-R đạt Misinfo cao nhất (0.7038), Gemma (Decoder) vượt ở Sentiment (0.7700)
- [ ] Bàn luận về **Confidence Calibration**: ECE đo trước/sau Temperature Scaling, ý nghĩa với sản phẩm thực tế (người dùng tin tưởng đúng mức)
- [ ] Giải thích **vai trò bổ sung** của Dual-Student Hybrid thay vì cạnh tranh
- [ ] Phân tích Misinfo "Tin giả" F1 thấp (0.50) do class imbalance 28:158 + độ phức tạp nội tại
- [x] Soạn thảo báo cáo kiểm định chi tiết (`docs/HYPOTHESIS_TESTING_REPORT.md`) lập luận sâu về Platform-specific risk và Emotion packaging.

### Phụ lục & Chuẩn bị Bảo vệ
- [x] Upload Code lên GitHub (sanitized & secured — [hwngkm/VaccineNLP-Thesis](https://github.com/hwngkm/VaccineNLP-Thesis)).
- [x] Upload 3 Models lên HuggingFace.
- [x] Tạo 6 figures chuẩn luận văn (`experiments/results/figures/`).
- [x] Đồng bộ 5 notebooks từ Kaggle (numbered 01–04).
- [x] Tự động hóa cập nhật README benchmarks.
- [ ] Quay Video Demo (showcase: Multi-source fetcher + Temperature-calibrated confidence + Captum Saliency).
- [ ] Chuẩn bị 4 URL "an toàn" để demo: 1 báo VnExpress, 1 YouTube Sức khỏe & Đời sống, 1 Facebook fan page, 1 TikTok #tiemchung.
- [ ] Trả lời 5 câu hỏi Q&A phòng thủ (chuẩn bị bởi Cố vấn học thuật).
