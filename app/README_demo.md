# VaccineNLP · XAI Demo Dashboard

## Chạy nhanh (1 lệnh)

```bash
# Từ thư mục gốc dự án
streamlit run app/streamlit_demo.py
```

## Yêu cầu

```bash
pip install -r app/requirements_demo.txt
```

## Cấu trúc

| File | Mô tả |
|---|---|
| `streamlit_demo.py` | App chính |
| `xai_cache.json` | Cache lý luận Gemma-4 (186 mẫu) |
| `requirements_demo.txt` | Dependencies |

## Tính năng

- **PhoBERT Multitask Classifier**: Phân loại 3 chiều (Misinfo · Stance · Sentiment)
- **Gemma-4 XAI Reasoning**: Hiển thị lý luận AI bằng tiếng Việt cho 186 mẫu benchmark
- **3 mẫu demo**: Tin giả, từ lóng MXH, tin chính xác
- **Confidence scores**: Softmax probabilities cho từng lớp

## Lưu ý

- Model checkpoint: `experiments/models/phobert_multitask_v2/best_model.pt`
- Không load Gemma-4 trực tiếp (chỉ dùng cached reasoning)

---
HUPH · MSSV 2211090016 · 2026
