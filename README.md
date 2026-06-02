---
title: VaccineNLP Demo
emoji: 🦠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
short_description: Phát hiện thông tin sai lệch về vaccine tiếng Việt
tags:
  - vietnamese
  - misinformation
  - vaccine
  - public-health
  - phobert
  - gemma
  - xai
models:
  - hung2903/phobert-vaccine-multitask
  - hung2903/xlmr-vaccine-multitask
  - hung2903/gemma-4-E4B-unsloth-vaccine-xai
---

# 🦠 VaccineNLP — Phát hiện Thông tin Sai lệch Vaccine

**Đồ án tốt nghiệp HUPH 2026**
- Kim Mạnh Hưng · MSSV: 2211090016
- Đinh Lê Quỳnh Phương · MSSV: 2211090031
- GVHD: TS. Trần Lâm Quân

## 🏛️ Giới thiệu

VaccineNLP là hệ thống xử lý ngôn ngữ tự nhiên ứng dụng **Kiến trúc Dual-Student Hybrid** để phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam.

## 🏗️ Kiến trúc

| Thành phần | Mô hình | Vai trò |
|---|---|---|
| **Động cơ Phân loại** | PhoBERT-v2 (Multi-task, 3 heads) | Misinfo · Stance · Sentiment |
| **Động cơ Giải thích** | Gemma-4 E4B-it (QLoRA fine-tuned) | Chain-of-Thought Reasoning |
| **Baseline** | XLM-RoBERTa-v1 | So sánh đa ngôn ngữ |

## ⚙️ Tính năng

1. **Phân tích đa nhiệm** — Dự đoán đồng thời 3 trục
2. **Độ tin cậy kép** — Raw + Calibrated (Temperature Scaling)
3. **XAI 3 lớp** — Cache → HF Inference API → Captum IG
4. **Multi-source Fetcher** — News + YouTube + Facebook/TikTok/Threads (Apify)
5. **So sánh mô hình** — PhoBERT-v2 vs XLM-R-v1
6. **Batch mode** — Phân tích nhiều mẫu cùng lúc
7. **AI Voice** — Giọng đọc Tiếng Việt qua gTTS
9. **XAI h?u x? l?** ? ch?ng l?p, badge b?t ??ng, strip markdown cho TTS, v? n?p l?i file .xlsx/.csv ?? c?o v?o Batch
8. **Session history** — 10 lượt phân tích gần nhất

## 🔑 Setup HuggingFace Spaces Secrets

Vào **Settings → Repository secrets** và thêm:

```
HF_TOKEN=hf_...           # Inference API token cho Gemma reasoning
APIFY_TOKEN_1=apify_...   # Apify token cho Facebook/TikTok/Threads
APIFY_TOKEN_2=apify_...   # Token rotation (optional)
APIFY_TOKEN_3=apify_...
APIFY_TOKEN_4=apify_...
APIFY_TOKEN_5=apify_...
```

## 📊 Benchmark (Macro F1 trên Gold Test Set, n=186)

| Mô hình | Misinfo | Stance | Sentiment | Trung bình |
|---|:---:|:---:|:---:|:---:|
| **PhoBERT-v2** | 0.6996 | **0.6640** | **0.7266** | **0.6967** 🥇 |
| **XLM-R-v1** | **0.7038** | 0.6224 | 0.6866 | 0.6709 🥈 |
| **Gemma-4 4B** | 0.6377 | 0.6264 | **0.7700** | 0.6780 🥉 |

## 🔗 Tài nguyên

- **GitHub:** https://github.com/hwngkm/VaccineNLP-Thesis
- **Kaggle Dataset:** https://www.kaggle.com/datasets/inhlqunhphng/vaccinenlp-clean-data
- **HF Models:** [PhoBERT](https://huggingface.co/hung2903/phobert-vaccine-multitask) · [XLM-R](https://huggingface.co/hung2903/xlmr-vaccine-multitask) · [Gemma XAI](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai)

## 📜 License

MIT — Free for research and educational use.

---

*Powered by Gradio · Deployed on HuggingFace Spaces (CPU Basic 16GB)*
