# 🔬 Hướng dẫn Hợp nhất QLoRA Adapter & Tải Gemma-4 E4B Merged lên HuggingFace Hub

Tài liệu này hướng dẫn chi tiết quy trình chạy thực nghiệm **PATH A1** trên môi trường Kaggle để hợp nhất (merge) trọng số LoRA adapter của mô hình Gemma-4 4B vào mô hình nền (base model) và đẩy phiên bản 16-bit đã hợp nhất lên HuggingFace Hub. Quy trình này giúp tối ưu hóa hiệu năng phục vụ và cho phép sử dụng **HuggingFace Inference API** làm giải pháp layer XAI serverless tốc độ cao.

---

## 🚨 Thông số Thực nghiệm

| Thành phần | Đường dẫn định danh (ID) / Giá trị |
|---|---|
| **LoRA Adapter (Fine-tuned)** | `hung2903/gemma-4-E4B-unsloth-vaccine-xai` |
| **Base Model (Mô hình nền)** | `unsloth/gemma-4-E4B-it` |
| **Output Repo (Mô hình merged)** | `hung2903/gemma-4-E4B-vaccine-xai-merged` |
| **Môi trường huấn luyện** | Kaggle Notebook (Accelerator: **GPU T4 x2** hoặc P100) |
| **Thời gian thực hiện** | Merge: ~10 phút · Upload (8GB): ~30-60 phút |
| **Quyền truy cập** | Yêu cầu `HF_TOKEN` có quyền **Write** |

---

## 🚀 Các Bước Triển Khai Trên Kaggle Notebook

### Bước 1: Khởi tạo Notebook
1. Truy cập [Kaggle Code Console](https://www.kaggle.com/code) và tạo một Notebook mới.
2. Tại bảng điều khiển bên phải (Settings):
   - Đặt **Accelerator** thành **GPU T4 x2**.
   - Bật **Internet** sang trạng thái **On**.
3. Thêm khoá bảo mật **Secret** có tên `HF_TOKEN` chứa token có quyền ghi (write permission) của HuggingFace.

### Bước 2: Thực thi các Cell Code mã nguồn

#### Cell 1: Cài đặt Thư viện Unsloth & Dependencies
```python
%%capture
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
!pip install -q huggingface_hub
```

#### Cell 2: Kiểm tra Tài nguyên Phần cứng & Đăng nhập HuggingFace
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"GPU count: {torch.cuda.device_count()}")

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("✅ Đăng nhập HuggingFace Hub thành công!")
```

#### Cell 3: Khai báo Cấu hình Biến số
```python
LORA_ADAPTER = "hung2903/gemma-4-E4B-unsloth-vaccine-xai"
UNSLOTH_BASE = "unsloth/gemma-4-E4B-it"
MERGED_REPO = "hung2903/gemma-4-E4B-vaccine-xai-merged"
MAX_SEQ_LENGTH = 2048
SAVE_DIR = "/kaggle/working/merged_16bit"
```

#### Cell 4: Tải mô hình thông qua Unsloth (Lazy load & Auto-detect base)
```python
from unsloth import FastModel

print(f"Đang tải adapter: {LORA_ADAPTER}")
print(f"Base model tương ứng: {UNSLOTH_BASE}")
print("⏳ Tiến trình đang tải ~50MB adapter và ~5GB mô hình nền...")

model, tokenizer = FastModel.from_pretrained(
    model_name=LORA_ADAPTER,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    token=HF_TOKEN,
)

print(f"\n✅ Đã tải mô hình thành công!")
print(f"Model class: {type(model).__name__}")
print(f"Dung lượng VRAM đã chiếm dụng: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

#### Cell 5: Kiểm tra nhanh chất lượng văn bản giải thích trước khi Merge
```python
test_prompt = """Bạn là Trí tuệ Nhân tạo có khả năng giải thích (Explainable AI) trong lĩnh vực Y tế Công cộng. Hãy phân tích văn bản sau đây về chủ đề vắc-xin, đưa ra lý luận chi tiết HOÀN TOÀN bằng tiếng Việt về tính xác thực, thái độ và cảm xúc. Tuyệt đối không dùng tiếng Anh.

Văn bản: Vắc-xin COVID gây vô sinh ở phụ nữ trẻ và biến đổi gen ở trẻ em."""

messages = [{"role": "user", "content": test_prompt}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

print("Đang sinh thử câu trả lời...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        use_cache=True,
    )
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("\n=== KẾT QUẢ TEST ĐẦU RA ===")
print(response)
```
> [!IMPORTANT]
> **Yêu cầu đầu ra:** Văn bản sinh ra phải là tiếng Việt mạch lạc, phân tích chính xác các trục Misinfo (Tin giả), Stance (Phản đối) và Sentiment (Tiêu cực). Nếu đầu ra bị lỗi định dạng hoặc tiếng Anh, hãy **DỪNG LẠI** để kiểm tra prompt format và không thực hiện tiến trình merge.

#### Cell 6: Thực hiện Hợp nhất mô hình (Merge) & Lưu định dạng Float16 (16-bit)
```python
print(f"Đang tiến hành hợp nhất LoRA weights vào Base Model và lưu dưới dạng 16-bit tại {SAVE_DIR}...")
print("⏳ Tiến trình này sẽ chạy trong khoảng 5-10 phút...")

model.save_pretrained_merged(
    SAVE_DIR,
    tokenizer,
    save_method="merged_16bit",
)

print(f"\n✅ Đã lưu thành công mô hình đã hợp nhất (Float16)!")

# Kiểm tra dung lượng thư mục làm việc
import subprocess
size_output = subprocess.check_output(["du", "-sh", SAVE_DIR]).decode()
print(f"Tổng dung lượng mô hình: {size_output.strip()}")
# Kết quả mong đợi: ~8GB
```

#### Cell 7: Tạo Repository trên HuggingFace Hub
```python
from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)

create_repo(
    MERGED_REPO,
    token=HF_TOKEN,
    repo_type="model",
    private=False,
    exist_ok=True,
)
print(f"✅ Repository đã sẵn sàng tại: https://huggingface.co/{MERGED_REPO}")
```

#### Cell 8: Tải thư mục mô hình lên HuggingFace Hub
```python
print(f"Đang tải thư mục {SAVE_DIR} lên repository {MERGED_REPO}...")
print("⏰ Tiến trình tải lên ~8GB dữ liệu thông qua mạng Kaggle mất khoảng 30-60 phút...")

api.upload_folder(
    folder_path=SAVE_DIR,
    repo_id=MERGED_REPO,
    repo_type="model",
    commit_message="Upload Gemma-4 E4B merged with VaccineNLP QLoRA adapter (16-bit)",
)

print(f"\n✅ Tiến trình tải lên hoàn tất!")
print(f"🔗 Đường dẫn mô hình: https://huggingface.co/{MERGED_REPO}")
```

#### Cell 9: Tạo và Cập nhật Model Card (README.md)
```python
MODEL_CARD = """---
license: gemma
base_model: unsloth/gemma-4-E4B-it
tags:
  - vietnamese
  - vaccine
  - misinformation
  - public-health
  - explainable-ai
  - qlora-merged
language:
  - vi
pipeline_tag: text-generation
inference: true
---

# VaccineNLP — Gemma-4 E4B Reasoning Engine (Merged)

Merged version of [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai) QLoRA adapter merged into base [unsloth/gemma-4-E4B-it](https://huggingface.co/unsloth/gemma-4-E4B-it).

## Purpose

**XAI Reasoning Engine** for VaccineNLP system — Chain-of-Thought explanations for vaccine misinformation detection in Vietnamese.

## Performance (Gold Test Set, n=186)

| Metric | Score |
|---|:---:|
| Macro F1 Misinfo | 0.6377 |
| Macro F1 Stance | 0.6264 |
| **Macro F1 Sentiment** | **0.7700** 🥇 |
| Parse Success Rate | 72.0% |

## Usage

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="hung2903/gemma-4-E4B-vaccine-xai-merged", token="YOUR_HF_TOKEN")

prompt = '''Bạn là chuyên gia y tế công cộng phân tích nội dung về vắc-xin.
Văn bản: Vắc-xin COVID gây vô sinh.
Lý luận:'''

response = client.text_generation(prompt, max_new_tokens=300, temperature=0.3)
print(response)
```

## Citation

```bibtex
@thesis{vaccinenlp2026,
  title={Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine},
  author={Kim Mạnh Hưng and Đinh Lê Quỳnh Phương},
  school={Trường Đại học Y tế Công cộng (HUPH)},
  year={2026},
}
```
"""

from io import BytesIO

api.upload_file(
    path_or_fileobj=BytesIO(MODEL_CARD.encode("utf-8")),
    path_in_repo="README.md",
    repo_id=MERGED_REPO,
    repo_type="model",
)
print("✅ Đã cập nhật Model Card lên HF Hub")
```

#### Cell 10: Kiểm định hoạt động của HuggingFace Inference API
```python
import requests
import time

print("⏳ Đang đợi 5 phút để HuggingFace thực hiện index mô hình mới...")
time.sleep(300)

API_URL = f"https://api-inference.huggingface.co/models/{MERGED_REPO}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

test_prompt = """<start_of_turn>user
Bạn là chuyên gia y tế công cộng. Phân tích bằng tiếng Việt.

Văn bản: Vắc-xin COVID gây vô sinh.

Lý luận:<end_of_turn>
<start_of_turn>model
"""

response = requests.post(
    API_URL,
    headers=headers,
    json={
        "inputs": test_prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.3, "return_full_text": False},
        "options": {"wait_for_model": True},
    },
    timeout=120,
)

print(f"Trạng thái HTTP: {response.status_code}")
if response.status_code == 200:
    print(f"✅ Kết quả sinh từ API: {response.json()}")
    print("\n🎉 HuggingFace Inference API hoạt động hoàn hảo! Thực nghiệm PATH A1 thành công!")
elif response.status_code == 503:
    print("⏳ Mô hình đang được nạp vào GPU của HF, vui lòng đợi 30 giây và thử lại...")
elif response.status_code == 404:
    print("❌ Mô hình chưa được index hoàn chỉnh, vui lòng đợi thêm 10-20 phút")
else:
    print(f"⚠️ Lỗi phát sinh từ API: {response.text[:500]}")
    print("\n→ Đề xuất: Kích hoạt PATH B (tự host trên Kaggle + Ngrok) làm giải pháp dự phòng")
```

---

## 📊 Kịch Bản Kiểm Thử & Định Hướng
Sau khi thực hiện xong, nếu Cell 10 trả về trạng thái **200 OK**:
- **PATH A1** hoạt động thành công.
- Ứng dụng web Gradio sẽ tự động truy vấn mô hình mới thông qua HF Inference API một cách mượt mà và serverless.
- Nếu gặp lỗi **400 hoặc 500** kéo dài (do mô hình 8GB quá lớn vượt quá giới hạn tài nguyên Serverless miễn phí của HuggingFace): Hãy lập tức kích hoạt **PATH B** (sử dụng Kaggle host offline kết hợp kênh truyền bảo mật Ngrok) để làm giải pháp backup hoạt động ngoại tuyến.
