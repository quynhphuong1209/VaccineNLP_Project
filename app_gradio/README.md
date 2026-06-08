---
title: VaccineNLP Demo
emoji: 🦠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
short_description: Phát hiện thông tin sai lệch về vaccine tiếng Việt
models:
  - hung2903/phobert-vaccine-multitask
  - hung2903/xlmr-vaccine-multitask
  - hung2903/gemma-4-E4B-vaccine-xai-merged
---

# 🦠 VaccineNLP — Phát hiện Tin giả & Phân tích Thái độ Vaccine

**Đồ án tốt nghiệp HUPH 2026** · Kim Mạnh Hưng (2211090016) · Đinh Lê Quỳnh Phương (2211090031) · GVHD: TS. Trần Lâm Quân

## Kiến trúc Dual-Student Hybrid
| Thành phần | Mô hình | Vai trò |
|---|---|---|
| Động cơ Phân loại | PhoBERT-v2 (multi-task, 3 heads) | Misinfo · Stance · Sentiment |
| Động cơ Giải thích | Gemma-4 E4B-it (QLoRA) | Chain-of-Thought + cờ bất đồng thuận |
| Baseline | XLM-RoBERTa-v1 | So sánh đa ngôn ngữ |

## Benchmark — Macro F1, Gold Test Set (n=186)
| Mô hình | Misinfo | Stance | Sentiment | Trung bình |
|---|:---:|:---:|:---:|:---:|
| PhoBERT-v2 | 0.6996 | **0.6640** | 0.7266 | **0.6967** |
| Gemma-4 4B | 0.6377 | 0.6264 | **0.7700** | 0.6780 |
| XLM-R-v1 | **0.7038** | 0.6224 | 0.6866 | 0.6709 |

## Secrets (HF Spaces → Settings → Repository secrets)
```
HF_TOKEN=hf_...            # nếu repo model ở chế độ private (model hiện public → có thể bỏ)
APIFY_TOKEN_1=apify_...    # (tùy chọn) thu thập URL Facebook/TikTok/Threads
GEMMA_ENDPOINT_URL=...     # (tùy chọn) trỏ tới Kaggle+ngrok đang chạy để bật Gemma "live"
```

## File cần có trong Space
`app.py` · `requirements.txt` · `README.md` · `huph_logo.png` · thư mục `data/` (`xai_cache.json`, `benchmark_results.json`, `temperature_params.json`).

> ⚠️ **XAI trên HF Spaces:** không có LM Studio (localhost:1234) trên Space, nên Gemma "live" không chạy. Hệ thống dùng **cache (`data/xai_cache.json`)** cho các mẫu mẫu + **fallback template** cho văn bản mới. Để demo hiện lý giải Gemma thật + bảng bất đồng thuận, **phải kèm `data/xai_cache.json`** (chứa reasoning Gold Test Set).

*Powered by Gradio · HuggingFace Spaces (CPU Basic 16GB)*
