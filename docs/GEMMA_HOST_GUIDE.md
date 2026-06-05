# 🌐 Hướng dẫn Tự Host Gemma-4 E4B trên Kaggle & Tạo Ngrok Tunnel (PATH B1)

Tài liệu này hướng dẫn chi tiết quy trình chạy thực nghiệm **PATH B1** để tự host (self-host) mô hình Gemma-4 4B trên máy chủ Kaggle và mở một kênh truyền bảo mật HTTPS thông qua **Ngrok Tunnel**. Phương pháp này giúp tạo ra một điểm cuối (endpoint) API hoàn toàn miễn phí, độc lập và có tốc độ xử lý nhanh để thay thế cho HuggingFace Inference API serverless.

---

## 🎯 Kiến Trúc Hệ Thống

```
┌────────────────────────┐
│  HuggingFace (Gradio)  │
│         app.py         │
└───────────┬────────────┘
            │
            │ HTTPS POST predict
            ▼
┌────────────────────────┐
│      Ngrok Tunnel      │  ← Cung cấp địa chỉ URL HTTPS công khai
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│    Kaggle Notebook     │
│   FastAPI Server (8000)│  ← Phục vụ API suy luận
│   Gemma-4 E4B Model    │  ← Unsloth + QLoRA
└────────────────────────┘
```

---

## ⚠️ Thông Số Yêu Cầu

| Thành phần | Yêu cầu thiết lập |
|---|---|
| **Mô hình nền & Adapter** | `unsloth/gemma-4-E4B-it` + QLoRA adapter |
| **Tài khoản Ngrok** | Đăng ký miễn phí tại [ngrok.com](https://ngrok.com/) |
| **Kaggle GPU hours** | ~9 giờ GPU miễn phí mỗi tuần |
| **ngrok Token** | Lấy từ [Dashboard ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) |
| **Bảo mật** | Thêm secrets `HF_TOKEN` và `NGROK_TOKEN` vào Kaggle |

---

## 🚀 Các Bước Triển Khai Trên Kaggle Notebook

### Bước 1: Khởi tạo Notebook
1. Tạo một Notebook mới tại [Kaggle Code Console](https://www.kaggle.com/code).
2. Tại bảng điều khiển bên phải (Settings):
   - Đặt **Accelerator** thành **GPU T4 x2** (hoặc P100).
   - Bật **Internet** sang trạng thái **On**.
3. Thêm 2 khoá bảo mật **Secret**:
   - `HF_TOKEN`: Chứa token truy cập HuggingFace.
   - `NGROK_TOKEN`: Chứa token xác thực của ngrok (`NGROK_API_TOKEN`).

### Bước 2: Thực thi các Cell Code mã nguồn

#### Cell 1: Cài đặt Thư viện Dependencies
```python
%%capture
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
!pip install -q fastapi uvicorn pyngrok nest-asyncio
```

#### Cell 2: Tải các Token Bảo mật (Secrets)
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
NGROK_TOKEN = secrets.get_secret("NGROK_TOKEN")

print("✅ Đã tải các secrets thành công")
```

#### Cell 3: Tải Mô hình Gemma-4 E4B & Adapter bằng Unsloth (4-bit nf4)
```python
import torch
from unsloth import FastModel

LORA_ADAPTER = "hung2903/gemma-4-E4B-unsloth-vaccine-xai"
MAX_SEQ_LENGTH = 2048

print("⏳ Đang tải Gemma-4 E4B và LoRA adapter...")
model, tokenizer = FastModel.from_pretrained(
    model_name=LORA_ADAPTER,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    token=HF_TOKEN,
)
print(f"✅ Đã tải thành công. Dung lượng VRAM đã chiếm: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Chạy thử dummy inference để khởi động GPU
dummy = tokenizer("test", return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = model.generate(**dummy, max_new_tokens=10)
print("✅ Mô hình đã được khởi động và sẵn sàng")
```

#### Cell 4: Định nghĩa Hàm Suy luận tiếng Việt (Inference)
```python
def generate_reasoning(text: str, max_tokens: int = 350) -> str:
    """Sinh chuỗi văn bản giải thích bằng tiếng Việt cho nội dung vaccine."""
    prompt = (
        f"Bạn là Trí tuệ Nhân tạo có khả năng giải thích (Explainable AI) "
        f"trong lĩnh vực Y tế Công cộng. Hãy phân tích văn bản sau đây về "
        f"chủ đề vắc-xin, đưa ra lý luận chi tiết HOÀN TOÀN bằng tiếng Việt "
        f"về tính xác thực, thái độ và cảm xúc. Tuyệt đối không dùng tiếng Anh."
        f"\\n\\nVăn bản: {text[:1000]}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.2,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()
    
    response = response.replace("<end_of_turn>", "").replace("<|turn>", "").strip()
    if not response.startswith("Lý luận"):
        response = "Lý luận: " + response
    
    return response

# Chạy kiểm tra nhanh chất lượng suy luận
test_result = generate_reasoning("Vắc-xin COVID gây vô sinh ở phụ nữ trẻ.")
print("=== KẾT QUẢ KIỂM THỬ ===")
print(test_result)
```

#### Cell 5: Thiết lập FastAPI Server & Mở ngrok Tunnel
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok, conf
import nest_asyncio
import uvicorn
import threading
import time

conf.get_default().auth_token = NGROK_TOKEN

app = FastAPI(title="VaccineNLP Gemma-4 Inference Server")

class InferenceRequest(BaseModel):
    text: str
    max_tokens: int = 350

class InferenceResponse(BaseModel):
    reasoning: str
    status: str = "success"
    error: str = ""

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    try:
        if not req.text or not req.text.strip():
            return InferenceResponse(reasoning="", status="error", error="Empty text")
        
        reasoning = generate_reasoning(req.text, req.max_tokens)
        return InferenceResponse(reasoning=reasoning)
    except Exception as e:
        return InferenceResponse(reasoning="", status="error", error=str(e))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Gemma-4 E4B (VaccineNLP fine-tuned)",
        "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }

@app.get("/")
def root():
    return {"app": "VaccineNLP Gemma-4 Self-host", "version": "1.0"}

nest_asyncio.apply()

# Khởi tạo ngrok tunnel công khai
print("🌐 Đang khởi tạo kết nối ngrok tunnel...")
public_url = ngrok.connect(8000, "http").public_url
print(f"\\n{'='*60}")
print(f"🔗 PUBLIC URL: {public_url}")
print(f"🔗 PREDICT ENDPOINT: {public_url}/predict")
print(f"🔗 HEALTH: {public_url}/health")
print(f"{'='*60}\\n")
print("▲ SAO CHÉP ĐƯỜNG DẪN PREDICT ENDPOINT TRÊN — sẽ dùng để cập nhật cấu hình cho Gradio App")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)
print("✅ Server đã hoạt động ổn định. Vui lòng giữ cell này chạy liên tục trong suốt tiến trình demo!")
```

#### Cell 6: Thử nghiệm truy cập API từ chính Notebook
```python
import requests

test_url = f"{public_url}/predict"
response = requests.post(
    test_url,
    json={"text": "Vắc-xin COVID gây vô sinh ở phụ nữ trẻ và biến đổi gen ở trẻ em."},
    timeout=60,
)

print(f"Trạng thái HTTP: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"\\n=== REASONING ===")
    print(data["reasoning"])
else:
    print(f"❌ Lỗi: {response.text}")
```

#### Cell 7: Giữ kết nối ngầm (Keep-Alive)
```python
# Giữ cell này chạy liên tục để tránh Kaggle tắt notebook khi nhàn rỗi (idle)
import time
import requests

print("🟢 Server is alive. Waiting for HF Spaces requests...")
print(f"🔗 Endpoint: {public_url}/predict")
print(f"📊 Health check: {public_url}/health")
print()

while True:
    time.sleep(60)
    try:
        h = requests.get(f"{public_url}/health", timeout=10).json()
        print(f"[{time.strftime('%H:%M:%S')}] Alive | GPU: {h['gpu_mem_gb']} GB")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Health check failed: {e}")
```

---

## 🔧 Kết Nối Endpoint URL Với Gradio App (HuggingFace Spaces)

Sau khi Notebook đã chạy và sinh ra URL ngrok công khai (ví dụ: `https://7f3e-34-126-89-12.ngrok-free.app`):

### Bước 1: Sao chép URL
Copy phần Predict endpoint URL: `https://7f3e-34-126-89-12.ngrok-free.app/predict`

### Bước 2: Khai báo Secrets trên HuggingFace Spaces
1. Truy cập trang cấu hình của Space: `https://huggingface.co/spaces/hung2903/vaccinenlp-demo/settings`
2. Tại mục **Repository secrets**, chọn **New secret**:
   - **Name:** `GEMMA_ENDPOINT_URL`
   - **Value:** `https://7f3e-34-126-89-12.ngrok-free.app/predict`
3. Click **Save**.

### Bước 3: Khởi động lại Space
Tại Settings của Space, chọn **Factory Rebuild** để Gradio App tải cấu hình URL mới. Kể từ lúc này, mọi yêu cầu lý luận giải thích từ tab **Phân tích** sẽ được định tuyến tốc độ cao thẳng về API server chạy trên Kaggle GPU của bạn!
