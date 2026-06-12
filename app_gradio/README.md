---
title: HUPH NLP Demo
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
short_description: Phân tích văn bản y tế công cộng tiếng Việt — đồ án HUPH
---

# HUPH NLP Demo

**Đồ án tốt nghiệp HUPH 2026** · Kim Mạnh Hưng · Đinh Lê Quỳnh Phương · GVHD: TS. Trần Lâm Quân

Ứng dụng phân tích văn bản y tế công cộng tiếng Việt: phân loại đa nhiệm 3 trục
(PhoBERT-v2 / XLM-R) kèm giải thích (XAI) bằng mô hình ngôn ngữ và Captum Integrated Gradients.

## Lưu ý triển khai
- Dữ liệu nghiên cứu nhạy cảm (mẫu văn bản, cache giải thích) được lưu ở **kho dữ liệu
  PRIVATE, gated** và nạp lúc chạy qua `HF_TOKEN`; **repo công khai không chứa nội dung thô**.
- XAI: cache (kho private) → mô hình đám mây dự phòng → template. Có thể bật suy luận bằng
  mô hình cục bộ qua tunnel (secret `ENABLE_REMOTE_LLM=1` + `LM_STUDIO_BRIDGE_URL`).

## Secrets (HF Spaces → Settings)
```
HF_TOKEN=hf_...            # đọc kho dữ liệu private + tải model
GEMINI_API_KEY=...         # (+ _2.._5) XAI đám mây dự phòng
# Tùy chọn bật suy luận mô hình cục bộ qua tunnel:
# ENABLE_REMOTE_LLM=1
# LM_STUDIO_BRIDGE_URL=https://<tunnel>/v1
# LM_API_TOKEN=...   LM_STUDIO_MODEL=...
```

*Powered by Gradio · Hugging Face Spaces (CPU Basic)*
