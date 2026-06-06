# Hướng dẫn Khởi chạy VaccineNLP Web trên Local (Windows)

Hiện tại, hệ thống VaccineNLP Web được cấu trúc thành **4 microservices**:
1. **db**: Cơ sở dữ liệu PostgreSQL 15.
2. **api_service**: FastAPI API Gateway chính dùng mô hình PhoBERT-v2.
3. **xai_service**: FastAPI XAI Engine giải thích dùng Gemma-4B GGUF.
4. **frontend**: React + Vite + TailwindCSS v4.

Để khởi chạy dự án trên máy tính cá nhân (local) của bạn dưới hệ điều hành Windows, có 2 cách chính dưới đây:

---

## 🐳 CÁCH 1: Sử dụng Docker & Docker Desktop (Khuyên Dùng)

Vì môi trường local của bạn chưa cài đặt Node.js (npm), PostgreSQL và các thư viện C++ cần thiết cho mô hình AI, **sử dụng Docker là cách nhanh và ít lỗi nhất**. Docker sẽ tự động đóng gói và chạy tất cả các dịch vụ này trong các containers độc lập mà không cần bạn cài đặt thủ công.

### Các bước thực hiện:

#### Bước 1: Cài đặt và bật Docker Desktop
1. Tải bộ cài đặt Docker Desktop cho Windows tại: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Tiến hành cài đặt (nên chọn sử dụng WSL 2 backend nếu trình cài đặt gợi ý).
3. **Quan trọng**: Hãy mở ứng dụng Docker Desktop lên và đảm bảo biểu tượng góc dưới hiển thị trạng thái màu xanh lá (**Docker Engine is running**).

#### Bước 2: Tạo thư mục chứa trọng số mô hình (`models`)
Mở PowerShell tại thư mục `VaccineNLP_Web` và chạy:
```powershell
mkdir models
```
Tải 2 file trọng số mô hình thật từ HuggingFace và sao chép chúng vào thư mục `models/` vừa tạo:
* **gemma-4-4b-Q4_K_M.gguf**: [Tải tại đây (hung2903/gemma-4-E4B-vaccine-xai-merged)](https://huggingface.co/hung2903/gemma-4-E4B-vaccine-xai-merged/resolve/main/gemma-4-e4b-it.Q4_K_M.gguf)
* **phobert_multitask.pt**: [Tải tại đây (hung2903/phobert-vaccine-multitask)](https://huggingface.co/hung2903/phobert-vaccine-multitask/resolve/main/best_model.pt)

*Lưu ý đặt đúng tên file gốc để code tìm thấy:*
- `models/gemma-4-4b-Q4_K_M.gguf` (hoặc đổi tên file tải về thành đúng như vậy)
- `models/phobert_multitask.pt`

#### Bước 3: Tạo file cấu hình môi trường `.env`
Sao chép cấu hình mẫu sang cấu hình thực tế:
```powershell
copy .env.example .env
```
*(Bạn có thể chỉnh sửa mật khẩu Postgres trong file `.env` nếu muốn, mặc định đã cấu hình sẵn chạy được ngay).*

#### Bước 4: Khởi chạy bằng Docker Compose
Trong thư mục `VaccineNLP_Web`, chạy lệnh sau để tự động tải, build ảnh (images) và khởi chạy toàn bộ 4 services ngầm:
```powershell
docker compose up --build -d
```
Chờ vài phút để Docker tải các package và khởi tạo database. Kiểm tra xem các container đã chạy thành công chưa bằng:
```powershell
docker compose ps
```

#### Bước 5: Truy cập ứng dụng
* **Giao diện Web (Frontend):** [http://localhost:5173](http://localhost:5173)
* **API Gateway (api_service):** [http://localhost:8000](http://localhost:8000)
* **XAI Engine (xai_service):** [http://localhost:8001](http://localhost:8001)

Muốn dừng hệ thống, bạn chạy:
```powershell
docker compose down
```

---

## 💻 CÁCH 2: Khởi chạy Trực tiếp trên Windows Host (Không dùng Docker)

Nếu không muốn cài Docker Desktop, bạn phải cài đặt thủ công tất cả các runtime dependencies lên hệ thống của mình.

### Yêu cầu tiên quyết:
1. **Python**: Bạn đã cài sẵn `Python 3.10.16`.
2. **Node.js & npm** (để chạy frontend): Tải và cài đặt bản LTS từ [https://nodejs.org/](https://nodejs.org/). Sau khi cài, mở lại terminal và chạy `node -v` và `npm -v` để kiểm tra.
3. **PostgreSQL**: Tải và cài đặt PostgreSQL cho Windows tại [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/). 
   - Trong quá trình cài đặt, ghi nhớ mật khẩu cho tài khoản `postgres` (ví dụ: `yourpassword`).
   - Cài xong, dùng công cụ **pgAdmin** hoặc dòng lệnh để tạo một database mới có tên là `vaccinenlp_db`.

---

### Các bước khởi chạy từng thành phần:

#### 1. Tạo thư mục chứa models
Tạo thư mục `models/` tại thư mục gốc của `VaccineNLP_Web` và tải/sao chép 2 file mô hình (`gemma-4-4b-Q4_K_M.gguf` và `phobert_multitask.pt`) vào đây giống như ở cách 1.

#### 2. Khởi chạy `api_service`
Mở một cửa sổ PowerShell mới và di chuyển vào `VaccineNLP_Web\api_service`:
```powershell
cd c:\Users\dinhl\Downloads\VaccineNLP_Project-main\VaccineNLP_Web\api_service

# Tạo môi trường ảo python độc lập
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# Cài đặt PyTorch CPU và các thư viện cần thiết
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Thiết lập biến môi trường để kết nối Database và XAI Service
$env:DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/vaccinenlp_db"
$env:XAI_SERVICE_URL="http://127.0.0.1:8001"
$env:PHOBERT_PATH="../models/phobert_multitask.pt"

# Chạy ứng dụng API
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### 3. Khởi chạy `xai_service`
Mở một cửa sổ PowerShell khác và di chuyển vào `VaccineNLP_Web\xai_service`:
```powershell
cd c:\Users\dinhl\Downloads\VaccineNLP_Project-main\VaccineNLP_Web\xai_service

# Tạo môi trường ảo python độc lập
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Cài đặt llama-cpp-python (CPU version) để chạy Gemma GGUF
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Thiết lập biến môi trường chỉ đường dẫn model
$env:GGUF_MODEL_PATH="../models/gemma-4-4b-Q4_K_M.gguf"
$env:LLM_THREADS="4"

# Chạy ứng dụng XAI Engine
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

#### 4. Khởi chạy `frontend`
Mở một cửa sổ PowerShell khác và di chuyển vào `VaccineNLP_Web\frontend`:
```powershell
cd c:\Users\dinhl\Downloads\VaccineNLP_Project-main\VaccineNLP_Web\frontend

# Cài đặt các thư viện Node.js
npm install

# Thiết lập biến môi trường API Gateway cho frontend kết nối local
$env:VITE_API_URL="http://127.0.0.1:8000"

# Chạy frontend ở chế độ dev mode
npm run dev
```

Bây giờ bạn truy cập ứng dụng frontend qua đường dẫn [http://localhost:5173](http://localhost:5173) trong trình duyệt web.
