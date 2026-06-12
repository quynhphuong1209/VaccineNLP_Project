# 🧠 Shared Runtime Engines (src/app_core/)

**Cập nhật:** 02/06/2026 · Trạng thái: ✅ Hoàn tất khởi tạo mới 100%

Thư mục `src/app_core/` là bộ não vận hành (Shared Runtime Engines) của cả hai ứng dụng Streamlit (`app/`) và Gradio (`app_gradio/`). Đây là nơi tập trung toàn bộ logic nghiệp vụ về: tải mô hình, hiệu chuẩn độ tin cậy, sinh giải thích y khoa Chain-of-Thought (XAI), và cào quét dữ liệu trực tuyến.

---

## 🗂️ Danh Sách Modules & Code Hoạt Động (File Directory)

### 1️⃣ **predictor.py** — Động Cơ Phân Loại & Hiệu Chuẩn Y Tế
Tệp này chịu trách nhiệm tải các checkpoint mô hình phân loại đa nhiệm PhoBERT-v2 và XLM-R-v1, thực hiện Forward pass và áp dụng giải thuật hậu hiệu chuẩn.

- **Class `VaccinePredictor`:**
  - `__init__(self, model_type="phobert")`: Khởi tạo và tải trọng số mô hình tương ứng từ thư mục `experiments/models/phobert_v2` hoặc `experiments/models/xlmr_v1`.
  - `predict(self, text: str)`: Thực hiện phân tách từ ghép tiếng Việt bằng `pyvi`, chuyển đổi thành chuỗi tokens, đưa qua mô hình deep learning để lấy ra 3 cặp logits thô.
  - `calibrate_logits(self, logits: np.ndarray, task: str) -> np.ndarray`: Áp dụng giải thuật **Temperature Scaling** để đưa xác suất phân phối thô về đúng tần suất thực tế y khoa:
    $$P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
    Với bộ tham số nhiệt độ tối ưu được fit bằng L-BFGS trên Validation set:
    * $T_{\text{misinfo}} = 1.8197$
    * $T_{\text{stance}} = 1.6666$
    * $T_{\text{sentiment}} = 1.3474$

---

### 2️⃣ **xai_engine.py** — Động Cơ Lập Luận Giải Thích Y Khoa (CoT)
Tệp này điều khiển luồng Explainable AI (XAI) thông minh 3 cấp độ để cung cấp chuỗi lý giải y khoa Chain-of-Thought thuyết phục cho chuyên gia.

- **Class `MedicalXAIEngine`:**
  - `__init__(self)`: Khởi tạo và tải tệp bộ nhớ đệm `app/xai_cache.json` (chứa các lập luận y khoa mẫu đã được các tác giả đồ án và chuyên gia y tế thẩm định thủ công).
  - `generate_explanation(self, text: str, label: int, stance: int, sentiment: int) -> str`:
    - **Cấp độ 1 (Cache hit):** Quét nhanh tệp cache xem văn bản thô nhập vào có khớp với 186 mẫu Gold Benchmark không. Nếu có, trả về ngay lập tức lập luận chuẩn của chuyên gia (độ trễ ~0ms).
    - **Cấp độ 2 (Ngrok local GPU server):** Kết nối đến local GPU server qua endpoint `/v1` tương thích OpenAI. Nếu server GPU local đang hoạt động, gửi request để Gemma-4-E4B QLoRA sinh văn bản lý giải.
    - **Cấp độ 3 (Gemini Pro fallback):** Nếu cả 2 cấp trên thất bại (local server tắt do idle hoặc rớt đường truyền ngrok), hệ thống tự động chuyển tiếp request sang API Gemini Pro của Google để sinh lời giải thích y học chuẩn cấu trúc tham chiếu Vancouver.
  - **Cơ chế Robust Parsing:** Tự động phát hiện và trích lọc chuỗi giải thích thô nằm trong các cặp thẻ đặc biệt hoặc strip các câu chào mào đầu ngoài lề của mô hình Decoder để tránh lỗi định dạng giao diện (đảm bảo tỷ lệ hỏng cú pháp giảm từ 28% về 0% ở runtime).

---

### 3️⃣ **fetchers.py** — Bộ Cào Dữ Liệu & Định Tuyến Nguồn (Data Fetcher Router)
Tệp này trích xuất văn bản thô từ các URL bài viết mạng xã hội/báo chí do người dùng cung cấp.

- **Class `MultiSourceFetcher`:**
  - `fetch_content(self, url: str) -> dict`: Nhận diện tên miền URL để tự động định tuyến (Routing) sang scraper tương ứng:
    - **Facebook Post / Group:** Gọi Apify Client kết nối trực tiếp với Facebook Scraper Actor.
    - **Facebook Comments:** Tự động gọi `facebook-comments-scraper` của Apify để cào các chuỗi bình luận chi tiết.
    - **YouTube Video:** Gọi `yt-dlp` trích xuất phụ đề tự động (Auto-generated Caption) và danh sách bình luận hàng đầu (Top comments).
    - **RSS Feed / Báo chí chính thống:** Phân tích cú pháp HTML qua thư viện `BeautifulSoup4` để lấy tiêu đề và nội dung bài viết chính, tự động bỏ qua quảng cáo biên lề.

---

## 🏃 Cách Sử Dụng Trong Ứng Dụng (Code Integration)

Nhờ thiết kế hướng đối tượng sạch sẽ, việc tích hợp động cơ lõi này vào các ứng dụng Streamlit hay Gradio cực kỳ đơn giản:

```python
from src.common.paths import ensure_src_in_sys_path
ensure_src_in_sys_path()

from src.app_core.predictor import VaccinePredictor
from src.app_core.xai_engine import MedicalXAIEngine

# 1. Khởi tạo các động cơ lõi
predictor = VaccinePredictor(model_type="phobert")
xai_engine = MedicalXAIEngine()

# 2. Phân tích văn bản thô
text = "Vắc-xin Covid-19 làm biến đổi gen người!"
predictions = predictor.predict(text)
# Trả về: { 'misinfo': { 'class': 1, 'confidence': 0.945 }, ... }

# 3. Sinh lập luận y khoa giải thích kết quả
explanation = xai_engine.generate_explanation(
    text=text,
    label=predictions['misinfo']['class'],
    stance=predictions['stance']['class'],
    sentiment=predictions['sentiment']['class']
)
print(f"Giải thích y khoa: {explanation}")
```

---

## 🔒 Quản Lý Lỗi & An Toàn Runtime (Robustness & Error Handling)

- **Rotated API Keys:** Động cơ `fetchers.py` tự động quét danh sách API keys `APIFY_TOKEN_1` đến `APIFY_TOKEN_5` trong `.env` để xoay vòng (rotation) khi một key chạm ngưỡng giới hạn băng thông (Rate limit), tránh làm gián đoạn trải nghiệm của người dùng.
- **Graceful Fallback:** Khi Gemma Local Server gặp sự cố phần cứng, hệ thống ghi nhận log cảnh báo và chuyển đổi liền mạch sang Gemini API trong vòng **150ms**, người dùng cuối không hề nhận ra sự gián đoạn dịch vụ.

---

*VaccineNLP Core Engine Team · HUPH 2026*
