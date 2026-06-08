# HƯỚNG DẪN VẬN HÀNH HỆ THỐNG VACCINENLP_WEB

Tài liệu này hướng dẫn chi tiết cách thiết lập, chạy thử và vận hành hệ thống **VaccineNLP_Web** sau khi nâng cấp toàn bộ giao diện (Frontend React từ repo của Phương) và tích hợp các công nghệ backend nâng cao (Streaming EventStream, cờ nhất quán 3 trạng thái, fallback thô).

---

## 📐 Kiến Trúc Tổng Quan Hệ Thống

Hệ thống được đóng gói thành một cụm **Docker Compose** gồm 4 services chính:
1. **`db` (PostgreSQL):** Cơ sở dữ liệu lưu lịch sử phân tích và cache kết quả lý luận XAI.
2. **`api_service` (FastAPI - Port 8000):** Tải mô hình PhoBERT-v2 gốc, xử lý tách từ bằng `underthesea`, phân loại nhanh 3 trục, tính toán cờ nhất quán và làm proxy stream SSE.
3. **`xai_service` (FastAPI - Port 8001):** Cầu nối giao tiếp với LM Studio trên máy HOST qua giao thức stream để sinh Chain-of-Thought (CoT).
4. **`frontend` (Vite/React - Port 5173):** Giao diện người dùng tích hợp toàn bộ tính năng phân tích, radar chart, Captum heatmap, TTS và các tab so sánh/nâng cao.

```
[ Trình duyệt (5173) ] <───(SSE Stream)───> [ api_service (8000) ]
                                                   │
                                            (Cập nhật nhãn nhanh)
                                                   │
                                                   ▼
                                            [ xai_service (8001) ]
                                                   │
                                         (LM Studio trên Windows Host)
```

---

## ⚙️ Hướng Dẫn Cấu Hình Môi Trường (.env)

Tạo file `.env` tại thư mục gốc `VaccineNLP_Clean_V1/.env` (và bản sao trong `VaccineNLP_Web/.env`) với cấu hình dưới đây:

```ini
# --- CƠ SỞ DỮ LIỆU POSTGRES ---
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=vaccinenlp

# --- CẤU HÌNH LM STUDIO (Windows Host) ---
# Địa chỉ LM Studio trên máy chủ vật lý Windows (Docker truy cập qua Gateway host.docker.internal)
LMSTUDIO_URL=http://host.docker.internal:1234/v1
LMSTUDIO_MODEL=gemma-4-e4b-vaccine-xai-merged

# Cờ bật Safe Mode bằng GGUF local trong container
# - Để giá trị 0: Container nhẹ, không nạp model trong container, chỉ gọi LM Studio (Tối ưu RAM).
# - Để giá trị 1: Nếu LM Studio rớt, container tự động nạp file GGUF local để sinh giải thích (Tốn RAM).
LOCAL_FALLBACK=0

# Token bảo mật LM Studio (nếu có thiết lập)
LM_API_TOKEN=<YOUR_LM_STUDIO_TOKEN>
```

---

## 📂 Chuẩn Bị File Trọng Số Mô Hình (Weights)

Tạo thư mục `models` nằm tại `VaccineNLP_Clean_V1/VaccineNLP_Web/models/` và đặt các tệp trọng số sau vào đó:
1. **`phobert_multitask.pt`:** Trọng số PhoBERT-v2 đã huấn luyện (đây là tệp classifier 3 đầu ra song song).
2. **`gemma-4-4b-Q4_K_M.gguf`** *(Chỉ cần nếu bật `LOCAL_FALLBACK=1`)*: Trọng số mô hình Gemma GGUF để chạy offline trong Docker.

---

## 💻 Thiết Lập LM Studio Trên Windows

Để hệ thống sinh lý luận giải thích Chain-of-Thought (CoT) với tốc độ cao bằng GPU trên máy vật lý, hãy cấu hình LM Studio như sau:
1. Mở phần mềm **LM Studio** trên Windows.
2. Tìm và nạp mô hình đã fine-tune: `gemma-4-e4b-vaccine-xai-merged`.
3. Chuyển sang biểu tượng **Local Server** (hình sóng anten ở thanh công cụ bên trái).
4. Thiết lập Port là `1234` và nhấn nút **Start Server**.
5. Đảm bảo cổng `1234` có thể truy cập được từ bên ngoài hoặc từ localhost.

---

## 🚀 Khởi Chạy Hệ Thống Bằng Docker

Mở terminal tại thư mục `VaccineNLP_Clean_V1/VaccineNLP_Web` và thực hiện các lệnh:

### 1. Build đóng gói các dịch vụ:
```bash
docker compose build
```

### 2. Khởi chạy toàn bộ hệ thống ở chế độ nền (detached):
```bash
docker compose up -d
```

### 3. Kiểm tra trạng thái các container:
```bash
docker compose ps
```
Cả 4 service (`db`, `api_service`, `xai_service`, `frontend`) phải hiển thị trạng thái `Up`.

### 4. Xem log vận hành thời gian thực:
```bash
docker compose logs -f
```

---

## 🧪 Các Tính Năng Độc Quyền Được Tích Hợp

Khi vận hành giao diện tại địa chỉ `http://localhost:5173/`, Phương sẽ được trải nghiệm các tính năng nâng cấp quan trọng sau:

### 1. Phân Tích Nhãn Nhanh & SSE Stream (2 Nhịp)
* Không cần chờ đợi Gemma sinh lý luận chậm chạp. Ngay khi nhấn gửi, **PhoBERT-v2** trả ngay kết quả 3 nhãn trong 15ms.
* Ngay sau đó, giao diện tự động kết nối luồng **EventStream (Server-Sent Events)** hiển thị quá trình Gemma lý giải từng từ một trực tiếp trên UI kèm spinner "Đang truyền tải..." rất mượt mà.

### 2. Cờ Nhất Quán 3 Trạng Thái (Consistency Badge)
Badge được tính toán tự động dựa trên giả thuyết H1 và H3 của luận văn:
* **Hợp lệ (Emerald green):** Mọi tổ hợp logic chuẩn.
* **Tổ hợp bất thường (Amber orange):** Phát hiện trạng thái trái ngược lý thuyết (ví dụ: Phản đối nhưng cảm xúc Tích cực, hoặc Tin giả nhưng lập trường Ủng hộ/Trung lập).
* **Nguy cơ cao — nên rà soát (Red):** Nhóm đối tượng cực đoan chống vắc xin (Phản đối + Tiêu cực).

### 3. Fallback Nội Dung Thô (Parser Fallback)
* Nếu Gemma sinh lý luận quá dài và bị cắt ngang (chạm ngưỡng `MAX_TOKENS`), bộ phân tích sẽ bị lỗi định dạng (`parse_ok: false`).
* Thay vì lỗi trắng khung hoặc crash giao diện, UI của chúng ta sẽ hiển thị một thông báo màu cam cảnh báo định dạng chưa tối ưu, nhưng vẫn hiển thị toàn bộ nội dung lý luận thô (`raw_output`) để đảm bảo không mất thông tin.

### 4. Radar SVG & Phân Phối Đầy Đủ
* Bản đồ Radar vẽ đa giác tự động thể hiện phân bố 3 trục.
* Phân phối xác suất đầy đủ thể hiện xác suất của từng lớp nhãn con (ví dụ: Stance gồm cả Favor, Against, Neutral với các thanh tỷ lệ %).

### 5. Token Attribution (Captum IG Heatmap)
* Tab này tô màu nền đỏ với độ đậm nhạt khác nhau cho các từ khóa có trọng số đóng góp cao vào nhãn Tin giả (ví dụ: *vô sinh, biến đổi gen, chuột bạch, độc hại...*).

### 6. Các Công Cụ Nâng Cao (Advanced)
* **Batch Mode:** Kéo thả tệp văn bản bình luận `.txt` hoặc bảng bình luận `.csv` để xử lý hàng loạt lên tới 30 dòng cùng lúc, hỗ trợ tải kết quả dạng file `.csv`.
* **Model Compare:** Trực quan hóa so sánh trực tiếp độ tin cậy của PhoBERT-v2 (Multi-task tiếng Việt) đối chiếu với mô hình XLM-R-v1 (Baseline đa ngôn ngữ).
