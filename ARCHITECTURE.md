# 🏗️ VaccineNLP Project Architecture (v3.1)

## 📊 Overview
Dự án VaccineNLP tập trung vào việc thu thập, xử lý và phân tích dữ liệu về vaccine trên mạng xã hội bằng các mô hình ngôn ngữ tiên tiến (PhoBERT, XLM-R, Gemma).

## 🏗️ Technical Stack
- **Languages:** Python 3.9+, Streamlit, PowerShell
- **Models (Dual-Student Architecture):** 
  - **PhoBERT-v2 (Discriminator):** High-speed Multitask Classifier (Taxonomy v3 shape: 2-3-3).
  - **Gemma-4B (Reasoning Engine):** QLoRA fine-tuned for Explainable AI (XAI) and Chain-of-Thought generation.
  - **XLM-RoBERTa (Baseline):** Multilingual checkpoint for cross-architecture comparison.
- **Data Pipeline:** Apify (Social Media), Medallion Architecture (Bronze -> Silver -> Gold).
- **NLP Tools:** pyvi, underthesea, transformers
- **Tracking:** MLflow (Experiments)

## 🔄 Core Pipeline Flow
1. **Collection:** Thu thập dữ liệu từ FB, TikTok theo Tier-based (Tier A, B, C) qua `src/data_pipeline/collection/`.
2. **Preprocessing:** Làm sạch và gán nhãn tự động chuẩn Taxonomy v3 (Misinfo, Stance, Sentiment) qua `src/data_pipeline/preprocessing/`.
3. **Modeling:** Huấn luyện multitask model tốc độ cao qua `src/modeling/`.
4. **XAI Integration:** Sinh lý do giải thích y khoa tự động bằng LLM qua `src/modeling/llm_inference_engine.py`.
5. **Dashboard:** Giao diện trực quan tích hợp Saliency Map và AI Voice tại [vaccine-nlp-project.streamlit.app](https://vaccine-nlp-project.streamlit.app/).

---
*Cập nhật lần cuối: May 2026*
