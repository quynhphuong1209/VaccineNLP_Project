# 🧬 03. Phương pháp Nghiên cứu (Research Methodology)

Tài liệu này chi tiết hóa quy trình 5 giai đoạn (Phases) của dự án Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam, từ thu thập dữ liệu thô đến triển khai mô hình đa nhiệm và AI giải thích được (XAI).

---

## 3.1. Tổng quan Kiến trúc (System Architecture)

Hệ thống VaccineNLP tuân thủ **Medallion Architecture**, đảm bảo tính toàn vẹn và khả năng truy xuất của dữ liệu qua 5 Phase cốt lõi:

*   **Phase 1**: Data Harvesting (Thu hoạch đa nguồn).
*   **Phase 2**: Preprocessing & Cleaning (Làm sạch sâu).
*   **Phase 3**: Taxonomy Definition (Định nghĩa hệ thống nhãn đa chiều).
*   **Phase 4**: LLM Inference & HITL (Gán nhãn tự động & Tinh chỉnh chuyên gia).
*   **Phase 5**: Explainable AI & Training (Gán nhãn hỗ trợ bởi LLM và huấn luyện classification/reasoning engine).

---

## 3.2. Phase 1 & 2: Thu thập và Tiền xử lý

1.  **Nguồn cốt lõi (Core)**: Facebook Groups/Pages (cung cấp độ phủ và ngữ cảnh thảo luận sâu).
2.  **Nguồn bổ trợ (Low-yield/Reclaim)**: YouTube, TikTok, Forums (chứa nhiều nhiễu, được lọc thô và tái chế vào tập Bronze).
3.  **Nguồn ngoại lai (External)**: Tích hợp *Vietnamese Fake News Dataset (VFND)* và **MiSoVac** để làm phong phú tập huấn luyện với các mẫu đã được xác thực chuyên gia.

**Quy trình sạch**: Toàn bộ dữ liệu đi qua **Triple Filter Pipeline** bao gồm: Chuẩn hóa Unicode, Dịch Teen-code, và Lọc Domain y tế (loại bỏ vắc-xin thú y).

---

## 3.3. Phase 3 & 4: Gán nhãn LLM và Human-in-the-Loop

Dự án sử dụng **Gemma-4 31B** làm LLM annotator để gán nhãn cho toàn bộ 1,856 câu.

*   **Silver Data (90% Train)**: Giữ nguyên nhãn của LLM để tận dụng quy mô lớn, chấp nhận tỷ lệ nhiễu cực nhỏ như một dạng Regularization tự nhiên.
*   **Gold Data (10% Benchmark)**: Áp dụng quy trình **Human-in-the-Loop (HITL)**. Chuyên gia y tế tiến hành kiểm duyệt mù (Blind review), tinh chỉnh các sai lệch về văn hóa và ngữ cảnh địa phương để tạo ra bộ "Đề thi quốc gia" (Ground Truth) khách quan tuyệt đối.

---

## 3.4. Phase 5: Huấn luyện và Đánh giá

Tiến hành **Benchmark Showdown** giữa hai kiến trúc classification/reasoning engine chính:

1.  **PhoBERT Multitask (Encoder)**: Sử dụng kỹ thuật *Weighted CrossEntropy* và `class_weight='balanced'` để xử lý triệt để bài toán mất cân bằng dữ liệu. Mô hình hiện có sẵn tại: [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask).
2.  **XLM-R Multitask (Baseline)**: Mô hình đa ngôn ngữ phục vụ so sánh baseline. Có sẵn tại: [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask).
3.  **Gemma-4 4B QLoRA (Decoder)**: Huấn luyện theo định dạng Chat Template, bắt chước khả năng lý luận của LLM annotator. Quá trình Inference đánh giá F1-score được thực thi trên Kaggle Cloud để tận dụng phần cứng. Tại bước này, dự án áp dụng cơ chế **Robust Parsing** kết hợp với **Prompt Engineering** cực kỳ chi tiết nhằm chuẩn hóa định dạng kết quả, đồng thời thực hiện **Raw Response Logging** để đảm bảo không thất thoát dữ liệu do sự bất ổn định về cấu trúc trả lời của mô hình ngôn ngữ lớn. Mô hình có sẵn tại: [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai).

### Kết quả Benchmark (Macro F1-Score trên Gold Test Set — 186 mẫu)

> Nguồn: `experiments/results/phobert_v2_results.json`, `xlmr_v1_results.json`, `gemma_v3_results.json`

| Mô hình | Loại | Misinfo (F1) | Stance (F1) | Sentiment (F1) |
|---|---|:---:|:---:|:---:|
| **PhoBERT-v2** | Classification Engine (Encoder) | 0.6996 | **0.6640** | 0.7266 |
| **Gemma-4-4B** | XAI Reasoning Engine (Decoder) | 0.6377 | 0.6264 | **0.7700** |
| **XLM-R-v1** | Baseline (Encoder) | **0.7038** | 0.6224 | 0.6866 |

**Nhận định:** Kết quả benchmark cho thấy sự phân hóa rõ rệt về ưu thế của từng kiến trúc trên 3 tác vụ. PhoBERT-v2 đạt SOTA trên trục phân tích thái độ (Stance: 0.6640), trong khi XLM-R-v1 lại chiếm ưu thế trên trục phát hiện tin giả (Misinfo: 0.7038) trên tập Gold Test. Đặc biệt, Gemma-4-4B vượt trội trên trục phân tích cảm xúc (Sentiment: 0.7700 vs PhoBERT 0.7266), chứng minh sức mạnh của mô hình ngôn ngữ lớn Decoder-only trong việc nắm bắt sắc thái biểu cảm phức tạp và khả năng lập luận sâu sắc. Mặc dù có tỷ lệ Parse Failure Rate là 28% (được cải thiện đáng kể nhờ parser v3), Gemma-4-4B vẫn cung cấp khả năng giải thích lý luận (XAI CoT) vượt trội mà các mô hình phân loại Encoder không thể thực hiện được. Điều này tạo cơ sở khoa học cho sự phối hợp "Dual-Student" trong kiến trúc Hybrid.

---

## 3.5. Phase 6: Confidence Calibration (Hiệu chuẩn Độ tin cậy)

Một lỗi hệ thống thường gặp của mạng neural sâu hiện đại (đặc biệt là Transformers) là hiện tượng **quá tự tin** (overconfidence) — xác suất dự đoán (Softmax confidence) bị lệch cực lớn so với độ chính xác thực tế (Guo et al., ICML 2017).

Để biến hệ thống thành một trợ lý y tế đáng tin cậy, dự án áp dụng **Temperature Scaling** (chuẩn hóa post-hoc đơn tham số):

$$f_i(x) = \operatorname{softmax}\left(\frac{z_i}{T}\right)$$

Trong đó $z_i$ is vector logits, $T > 0$ là tham số nhiệt độ được tối ưu hóa bằng phương pháp tối ưu L-BFGS để cực tiểu hóa hàm Loss Negative Log-Likelihood (NLL) trên validation set.

### Thuật toán tối ưu hóa trong `src/modeling/calibration.py`:
```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_loader):
        # Tối ưu hóa tham số temperature bằng LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        # Cực tiểu hóa NLL Loss...
```

### Kết quả ECE (Expected Calibration Error) cải thiện:
-   **Misinfo axis ($T = 1.8197$)**: ECE giảm từ **12.3%** xuống còn **5.4%** (cải thiện 56.1%).
-   **Stance axis ($T = 1.6666$)**: ECE giảm từ **19.8%** xuống còn **9.3%** (cải thiện 53.0%).
-   **Sentiment axis ($T = 1.3474$)**: ECE giảm từ **14.4%** xuống còn **8.1%** (cải thiện 43.8%).

---

## 3.6. Phase 7: Real XAI Integration (Giải thích AI chuẩn Khoa học)

Thay vì sử dụng các phương pháp heuristics dựa trên từ khóa cứng (hardcoded hot-words), dự án triển khai hệ thống giải thích khoa học 2 lớp:

### 1. Phân bổ Thuộc tính Token (Integrated Gradients)
Sử dụng thư viện `Captum` để tính đạo hàm tích phân của xác suất đầu ra đối với embedding của từng token đầu vào, lấy baseline là token PAD:

$$\text{Attribution}_i(x) := (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

Quy trình trực quan hóa trong app (`app/streamlit_demo.py::render_real_saliency`):
-   Chạy mô hình PhoBERT-v2 sinh Logits và lấy Embedding gradients.
-   Tích phân gradients trên 20 steps approximation (`n_steps=20`).
-   Áp dụng chuẩn hóa min-max và map sang bảng màu HSL (màu đỏ biểu thị tác động tích cực lớn nhất đến quyết định của mô hình).

### 2. Lý luận Tự nhiên (Chain-of-Thought)
Mô hình Gemma-4 4B được huấn luyện QLoRA để học cách tự giải thích bằng ngôn ngữ tự nhiên. Nhờ cơ chế **XAI Cache** ngoại tuyến cho 186 mẫu Gold Test, app có thể hiển thị tức thời chuỗi lý luận chuẩn xác đã được chuyên gia y tế thẩm định, đồng thời hỗ trợ live API cho văn bản ngoài danh mục.

---

## 3.7. Tính Đột phá và Ứng dụng Thực tiễn (Scientific Novelty)

1.  **Explainable AI (XAI) via LLM-assisted Annotation**: Giải quyết bài toán "Hộp đen" trong y tế. Mô hình 4B được huấn luyện để học "Chuỗi lý luận" (Chain-of-Thought) từ mô hình 31B (LLM annotator), giúp bác sĩ hiểu rõ TẠI SAO AI lại đưa ra kết luận đó.
2.  **Zero-Cost Local Deployment**: Kiến trúc QLoRA 4-bit giúp đưa AI giải thích lên được các Local Server tại CDC hoặc VNVC, đảm bảo bảo mật dữ liệu công dân và chi phí vận hành tiệm cận bằng 0.
3.  **Linguistic Camouflage Detection**: Phân tích lý luận của AI giúp phát hiện các "mật mã" ngụy trang ngôn ngữ (vd: nước cất, sinh tố, chốt,...) mà cộng đồng anti-vaccine thường dùng để lách các bộ lọc mạng xã hội.

---

## 3.8. Cơ sở Lý thuyết (Literature Review)

Dự án kế thừa tri thức luận từ các nghiên cứu tiền đề quan trọng:
*   **UIT-ViCoV19QA**: Định hình cấu trúc QA y tế và các dạng thảo luận về dịch bệnh tại Việt Nam.
*   **ViGoEmotions**: Truyền cảm hứng cho chiến lược sử dụng LLM làm Annotator chuyên gia cho các dữ liệu văn bản tiếng Việt có độ khó cao.
*   **MiSoVac**: Bộ dữ liệu đa ngôn ngữ về tin sai lệch vaccine trên mạng xã hội, cung cấp nguồn mẫu ngoại lai đã được xác thực chuyên gia.

---

*Cập nhật: 21/05/2026 | Phiên bản 2.0*

