# 🧬 03. Phương pháp Nghiên cứu (Research Methodology)

Tài liệu này chi tiết hóa quy trình 5 giai đoạn (Phases) của dự án Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam, từ thu thập dữ liệu thô đến triển khai mô hình đa nhiệm và AI giải thích được (XAI).

---

## 3.1. Tổng quan Kiến trúc (System Architecture)

Hệ thống VaccineNLP tuân thủ **Medallion Architecture**, đảm bảo tính toàn vẹn và khả năng truy xuất của dữ liệu qua 5 Phase cốt lõi:

*   **Phase 1**: Data Harvesting (Thu hoạch đa nguồn).
*   **Phase 2**: Preprocessing & Cleaning (Làm sạch sâu).
*   **Phase 3**: Taxonomy Definition (Định nghĩa hệ thống nhãn đa chiều).
*   **Phase 4**: LLM Inference & HITL (Gán nhãn tự động & Tinh chỉnh chuyên gia).
*   **Phase 5**: Explainable AI Distillation & Training (Chưng cất tri thức và huấn luyện mô hình Student).

---

## 3.2. Phase 1 & 2: Thu thập và Tiền xử lý

1.  **Nguồn cốt lõi (Core)**: Facebook Groups/Pages (cung cấp độ phủ và ngữ cảnh thảo luận sâu).
2.  **Nguồn bổ trợ (Low-yield/Reclaim)**: YouTube, TikTok, Forums (chứa nhiều nhiễu, được lọc thô và tái chế vào tập Bronze).
3.  **Nguồn ngoại lai (External)**: Tích hợp *Vietnamese Fake News Dataset (VFND)* và **MiSoVac** để làm phong phú tập huấn luyện với các mẫu đã được xác thực chuyên gia.

**Quy trình sạch**: Toàn bộ dữ liệu đi qua **Triple Filter Pipeline** bao gồm: Chuẩn hóa Unicode, Dịch Teen-code, và Lọc Domain y tế (loại bỏ vắc-xin thú y).

---

## 3.3. Phase 3 & 4: Gán nhãn LLM và Human-in-the-Loop

Dự án sử dụng **Gemma-4 31B** làm Teacher Model để gán nhãn cho toàn bộ 1,856 câu.

*   **Silver Data (90% Train)**: Giữ nguyên nhãn của LLM để tận dụng quy mô lớn, chấp nhận tỷ lệ nhiễu cực nhỏ như một dạng Regularization tự nhiên.
*   **Gold Data (10% Benchmark)**: Áp dụng quy trình **Human-in-the-Loop (HITL)**. Chuyên gia y tế tiến hành kiểm duyệt mù (Blind review), tinh chỉnh các sai lệch về văn hóa và ngữ cảnh địa phương để tạo ra bộ "Đề thi quốc gia" (Ground Truth) khách quan tuyệt đối.

---

## 3.4. Phase 5: Huấn luyện và Đánh giá

Tiến hành **Benchmark Showdown** giữa hai kiến trúc Student chính:

1.  **PhoBERT Multitask (Encoder)**: Sử dụng kỹ thuật *Weighted CrossEntropy* và `class_weight='balanced'` để xử lý triệt để bài toán mất cân bằng dữ liệu. Mô hình hiện có sẵn tại: [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask).
2.  **XLM-R Multitask (Baseline)**: Mô hình đa ngôn ngữ phục vụ so sánh baseline. Có sẵn tại: [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask).
3.  **Gemma-4 4B QLoRA (Decoder)**: Huấn luyện theo định dạng Chat Template, bắt chước khả năng lý luận của Teacher Model. Quá trình Inference đánh giá F1-score được thực thi trên Kaggle Cloud để tận dụng phần cứng. Tại bước này, dự án áp dụng cơ chế **Robust Parsing** kết hợp với **Prompt Engineering** cực kỳ chi tiết nhằm chuẩn hóa định dạng kết quả, đồng thời thực hiện **Raw Response Logging** để đảm bảo không thất thoát dữ liệu do sự bất ổn định về cấu trúc trả lời của mô hình ngôn ngữ lớn. Mô hình có sẵn tại: [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai).

---

## 3.5. Tính Đột phá và Ứng dụng Thực tiễn (Scientific Novelty)

1.  **Explainable AI (XAI) via CoT Distillation**: Giải quyết bài toán "Hộp đen" trong y tế. Mô hình 4B được huấn luyện để học "Chuỗi lý luận" (Chain-of-Thought) từ mô hình 31B, giúp bác sĩ hiểu rõ TẠI SAO AI lại đưa ra kết luận đó.
2.  **Zero-Cost Local Deployment**: Kiến trúc QLoRA 4-bit giúp đưa AI giải thích lên được các Local Server tại CDC hoặc VNVC, đảm bảo bảo mật dữ liệu công dân và chi phí vận hành tiệm cận bằng 0.
3.  **Linguistic Camouflage Detection**: Phân tích lý luận của AI giúp phát hiện các "mật mã" ngụy trang ngôn ngữ (vd: nước cất, sinh tố, chốt,...) mà cộng đồng anti-vaccine thường dùng để lách các bộ lọc mạng xã hội.

---

## 3.6. Cơ sở Lý thuyết (Literature Review)

Dự án kế thừa tri thức luận từ các nghiên cứu tiền đề quan trọng:
*   **UIT-ViCoV19QA**: Định hình cấu trúc QA y tế và các dạng thảo luận về dịch bệnh tại Việt Nam.
*   **ViGoEmotions**: Truyền cảm hứng cho chiến lược sử dụng LLM làm Annotator chuyên gia cho các dữ liệu văn bản tiếng Việt có độ khó cao.
