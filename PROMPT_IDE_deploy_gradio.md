# 🚀 PROMPT IDE: Deploy VaccineNLP Gradio App lên HuggingFace Spaces

## NHIỆM VỤ

Deploy ứng dụng Gradio mới (port từ Streamlit) lên HuggingFace Spaces tier CPU Basic 16GB free. Tất cả files đã được Claude chuẩn bị sẵn trong `/mnt/user-data/outputs/`.

**Repo HF Spaces:** `kimmnhhng/vaccinenlp-demo`
**Local repo:** `D:\VaccineNLP_Clean_V1`
**Deadline:** Hôm nay phải có app chạy được, ngày mai test, ngày kia thêm features A+B.

---

## CONTEXT QUAN TRỌNG

### Tại sao migrate?
- App Streamlit hiện tại trên Streamlit Cloud bị sập khi load PhoBERT (chỉ 1GB RAM)
- HF Spaces CPU Basic free = 16GB RAM, đủ cho 2 models + Captum + cache

### Architecture mới
- **Gradio 4.44.1** thay cho Streamlit
- **3-Layer XAI fallback:** Cache → HF Inference API → Captum IG
- Giữ nguyên tất cả features Streamlit (Multi-source fetcher, AI Voice, Batch mode, etc.)
- Bỏ live animation và theme toggle live (Gradio limitation)

### Files Claude đã chuẩn bị

Trong `/mnt/user-data/outputs/`:
1. **`gradio_app.py`** (~50KB, 1259 lines) — Main app
2. **`requirements.txt`** — Dependencies tối ưu HF Spaces
3. **`README.md`** — HF Spaces metadata YAML header
4. **`notebook_xlmr_calibration.md`** — Notebook bổ sung T params cho XLM-R
5. **`notebook_merge_lora_unsloth.md`** — Notebook merge LoRA approach Unsloth

---

## BƯỚC 1: COPY FILES VỀ LOCAL REPO

```bash
cd D:\VaccineNLP_Clean_V1

# Tạo branch mới cho migration
git checkout -b feat/gradio-migration

# Tạo thư mục app_gradio để không xung đột với app/ Streamlit cũ
mkdir app_gradio
cd app_gradio

# Copy files từ Claude outputs
# (Bạn cần download files từ Claude chat về máy trước)
copy "C:\Users\<your_user>\Downloads\gradio_app.py" .\
copy "C:\Users\<your_user>\Downloads\requirements.txt" .\
copy "C:\Users\<your_user>\Downloads\README.md" .\

# Tạo thư mục data/ với 3 files
mkdir data
```

### Copy files quan trọng từ app/ Streamlit cũ sang app_gradio/data/

```bash
# Copy xai_cache.json (cache 6 mẫu HARD_CACHE)
copy ..\app\xai_cache.json .\data\

# Copy temperature_params.json (chỉ có PhoBERT, sẽ bổ sung XLM-R sau)
copy ..\app\temperature_params.json .\data\

# Tạo benchmark_results.json từ experiments/results/
python -c "
import json
from pathlib import Path

results_dir = Path('../experiments/results')
files = {
    'phobert': 'phobert_v2_results.json',
    'xlmr': 'xlmr_v1_results.json',
    'gemma': 'gemma_v3_results.json',
}

bench = {}
for key, fname in files.items():
    fpath = results_dir / fname
    if fpath.exists():
        with open(fpath) as f:
            data = json.load(f)
        bench[key] = {
            'name': key,
            'misinfo': data.get('misinfo', {}).get('macro_f1', 0.0),
            'stance': data.get('stance', {}).get('macro_f1', 0.0),
            'sentiment': data.get('sentiment', {}).get('macro_f1', 0.0),
            'per_class_misinfo': data.get('misinfo', {}).get('per_class', []),
            'per_class_stance': data.get('stance', {}).get('per_class', []),
            'per_class_sentiment': data.get('sentiment', {}).get('per_class', []),
            'support_misinfo': data.get('misinfo', {}).get('support', []),
            'support_stance': data.get('stance', {}).get('support', []),
            'support_sentiment': data.get('sentiment', {}).get('support', []),
        }

with open('./data/benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(bench, f, indent=2, ensure_ascii=False)
print('✅ Created benchmark_results.json')
"
```

---

## BƯỚC 2: TEST LOCAL TRƯỚC KHI DEPLOY

```bash
cd D:\VaccineNLP_Clean_V1\app_gradio

# Tạo virtual environment (đỡ conflict với Streamlit cũ)
python -m venv .venv_gradio
.venv_gradio\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Chạy app local
python gradio_app.py
```

**Mở browser:** http://localhost:7860

### Checklist kiểm tra local:

- [ ] App khởi động được, không có ERROR trong terminal
- [ ] 6 tabs hiển thị đầy đủ (Phân tích · Benchmark · Đánh giá · Tài liệu · Phương pháp · Đề cương)
- [ ] Chọn 1 sample từ dropdown → text tự động điền vào textbox
- [ ] Nhấn "Phân tích" → có summary + radar chart + reasoning + saliency HTML
- [ ] AI Voice button có audio file
- [ ] Bench tab có chart và leaderboard table
- [ ] Sankey chart hiển thị ở tab Đánh giá

### Nếu gặp lỗi phổ biến:

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: underthesea` | `pip install underthesea` |
| `ModuleNotFoundError: captum` | `pip install captum` |
| `OOM khi load PhoBERT` | RAM máy thấp, deploy thẳng lên HF Spaces |
| `gradio app stuck loading` | Đợi lần đầu lâu (5-10 phút) — đang download model từ HF |

---

## BƯỚC 3: CREATE HF SPACES REPO

### 3a. Tạo Space mới trên HF

1. Truy cập https://huggingface.co/new-space
2. Settings:
   - Owner: `kimmnhhng`
   - Space name: `vaccinenlp-demo`
   - License: MIT
   - SDK: **Gradio**
   - Space hardware: **CPU basic** (free)
   - Visibility: Public
3. Click "Create Space"

### 3b. Add Secrets

Vào **Space Settings → Repository secrets**, add:

```
HF_TOKEN = hf_xxxxx (Token với Read permission là đủ cho Inference API)
APIFY_TOKEN_1 = apify_api_xxxxx
APIFY_TOKEN_2 = apify_api_xxxxx (optional)
APIFY_TOKEN_3 = apify_api_xxxxx (optional)
APIFY_TOKEN_4 = apify_api_xxxxx (optional)
APIFY_TOKEN_5 = apify_api_xxxxx (optional)
```

Lấy từ:
- HF Token: https://huggingface.co/settings/tokens
- Apify Tokens: Lấy từ Streamlit Secrets cũ trong `.streamlit/secrets.toml`

---

## BƯỚC 4: PUSH CODE LÊN HF SPACE

### 4a. Setup HF git CLI

```bash
# Clone HF Space repo
cd D:\
git clone https://huggingface.co/spaces/kimmnhhng/vaccinenlp-demo hf_vaccinenlp_space
cd hf_vaccinenlp_space

# Cấu hình HF credentials (nếu chưa)
huggingface-cli login
# Nhập HF Token với write permission
```

### 4b. Copy files từ local repo sang HF Space repo

```bash
# Từ app_gradio sang HF Space repo
xcopy /E /Y D:\VaccineNLP_Clean_V1\app_gradio\* D:\hf_vaccinenlp_space\
```

### 4c. Verify file structure

```
hf_vaccinenlp_space/
├── README.md                    # HF Spaces metadata
├── gradio_app.py                # Main app (~50KB)
├── requirements.txt
└── data/
    ├── xai_cache.json
    ├── temperature_params.json
    └── benchmark_results.json
```

### 4d. Commit và push

```bash
cd D:\hf_vaccinenlp_space

git lfs install
git add .
git commit -m "feat: initial Gradio app deployment with 6 tabs + 3-layer XAI"
git push
```

---

## BƯỚC 5: MONITOR BUILD

1. Vào https://huggingface.co/spaces/kimmnhhng/vaccinenlp-demo
2. Tab **"Logs"** xem build progress
3. Build mất khoảng **5-15 phút** (tùy network):
   - Pull dependencies (~3-5 phút)
   - Download models lần đầu (~5-10 phút)
   - Start app (~30-60 giây)

### Checklist build thành công:
- [ ] Logs hiện `Running on http://0.0.0.0:7860`
- [ ] Status badge chuyển sang xanh "Running"
- [ ] Truy cập URL public, app load được

---

## BƯỚC 6: SMOKE TEST TRÊN PRODUCTION

Sau khi build xong, test các features chính:

### Test 1: Phân tích đơn lẻ
```
Input: "Vắc-xin COVID gây vô sinh ở phụ nữ"
Expected: Misinfo=Tin giả, Stance=Phản đối, Sentiment=Tiêu cực
Reasoning từ Cache hoặc HF API
Captum heatmap hiển thị
```

### Test 2: Multi-source Fetcher (Tier 1 News)
```
URL: https://vnexpress.net/hon-15-7-trieu-tre-em-da-duoc-tiem-chung-mo-rong-4740150.html
Expected: Content được fetch (~2-5s), badge "📰 Báo điện tử Tier 1"
```

### Test 3: Batch mode
```
Input 3 dòng:
- Vắc-xin gây vô sinh
- Tiêm vaccine phòng bệnh tốt
- Bộ Y tế khuyến cáo tiêm đủ liều

Expected: Bảng kết quả với 3 hàng, mỗi hàng có nhãn + confidence
```

### Test 4: Compare PhoBERT vs XLM-R
```
Input: "Vắc-xin COVID gây vô sinh"
Expected: Bảng so sánh, có thể khác nhau ở 1-2 trục
```

### Test 5: AI Voice
```
Sau khi Phân tích, click play audio
Expected: Phát tiếng Việt mạch lạc qua gTTS
```

---

## BƯỚC 7: SYNC VỚI GITHUB ORIGINAL REPO

```bash
cd D:\VaccineNLP_Clean_V1

# Đảm bảo app_gradio/ đã được commit
git add app_gradio/
git commit -m "feat: add Gradio app port for HF Spaces deployment"

# Push lên feature branch
git push origin feat/gradio-migration

# Tạo PR vào main (optional, hoặc merge thẳng)
```

---

## BƯỚC 8: BƯỚC TIẾP THEO (Ngày mai)

### A. Bổ sung Temperature Params cho XLM-R

1. Mở Kaggle notebook mới
2. Copy nội dung `notebook_xlmr_calibration.md` vào notebook (chia thành cells)
3. Run all cells → có file `xlmr_temperature_params.json`
4. Merge vào `app_gradio/data/temperature_params.json`
5. Commit + push lên HF Space → app auto-rebuild

### B. Merge LoRA + Upload Gemma Merged Model

1. Mở Kaggle notebook mới với **GPU T4 x2**
2. Copy nội dung `notebook_merge_lora_unsloth.md` vào notebook
3. Run all cells → 2 merged models trên HF Hub:
   - `hung2903/gemma-4-E4B-vaccine-xai-merged` (16-bit, cho HF Inference API)
   - `hung2903/gemma-4-E4B-vaccine-xai-merged-4bit` (4-bit, cho self-host)
4. Cập nhật `gradio_app.py` line ~52: `"xai_models"` thêm merged model
5. Commit + push lên HF Space

---

## 🆘 TROUBLESHOOTING DEPLOY

### "Build failed: ModuleNotFoundError"
→ Kiểm tra `requirements.txt`, đảm bảo có module thiếu

### "Build failed: torch incompatible"
→ Set `torch==2.1.2` cụ thể trong requirements.txt

### "OOM during model loading"
→ HF Spaces CPU Basic chỉ có 16GB; nếu vẫn OOM, cần check:
   1. Có load cả PhoBERT lẫn XLM-R cùng lúc không? → Lazy load đã ok
   2. Có Captum hold reference không? → Check `_CACHE` dict

### "Inference API timeout"
→ Layer 2 (Gemma API) chậm → Fallback xuống Layer 3 (Captum IG) tự động

### "App slow on first load"
→ Bình thường, cold start mất 30-60s sau khi sleep

---

## ✅ DELIVERABLES CUỐI

Sau khi xong các bước trên, bạn sẽ có:

1. ✅ **Public Gradio app** tại `https://kimmnhhng-vaccinenlp-demo.hf.space`
2. ✅ **6 tabs đầy đủ** (Phân tích · Benchmark · Đánh giá · Tài liệu · Phương pháp · Đề cương)
3. ✅ **3-layer XAI** (Cache + HF API + Captum) hoạt động
4. ✅ **Multi-source Fetcher** với Apify tokens
5. ✅ **GitHub sync** với branch `feat/gradio-migration`

---

## 📞 BÁO CÁO LẠI CLAUDE

Sau khi hoàn tất, gửi cho Claude:
1. URL public của HF Space
2. Screenshot Logs build thành công
3. Bất kỳ lỗi runtime nào trong process

Claude sẽ:
1. Verify deploy thành công
2. Soạn prompt IDE cho thêm Features A (Compare 3 models) + B (Session history)
3. Cập nhật Phụ lục A trong luận văn (Streamlit → Gradio)
