# 🏗️ VaccineNLP Project Architecture (v3.1)

## 📊 Overview
Dự án VaccineNLP tập trung vào việc thu thập, xử lý và phân tích dữ liệu về vaccine trên mạng xã hội bằng các mô hình ngôn ngữ tiên tiến (PhoBERT, XLM-R, Gemma).

## 🏗️ Technical Stack
- **Languages:** Python 3.9+, PowerShell
- **Models:** PhoBERT (Multitask), Gemma-4B (QLoRA), XLM-RoBERTa
- **Data Pipeline:** Apify (Social Media), Facebook Graph API
- **NLP Tools:** pyvi, underthesea, transformers
- **Tracking:** MLflow (Experiments)

## 🔄 Core Pipeline Flow
1. **Collection:** Thu thập từ FB, TikTok, YouTube qua `src/data_pipeline/collection/`
2. **Preprocessing:** Làm sạch và gán nhãn qua `src/data_pipeline/preprocessing/`
3. **Modeling:** Huấn luyện multitask qua `src/modeling/`
4. **Inference:** Triển khai server dự đoán qua `src/modeling/llm_inference_engine.py`
5. **Dashboard:** Giao diện người dùng trực quan tại [vaccine-nlp-project.streamlit.app](https://vaccine-nlp-project.streamlit.app/)

---
*Cập nhật lần cuối: May 2026*
