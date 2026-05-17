# Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam
*(Applying NLP for Vaccine Misinformation Detection and Community Attitude Analysis in Vietnamese Digital Environments)*

**Người thực hiện:**
- Kim Mạnh Hưng - 2211090016
- Đinh Lê Quỳnh Phương - 2211090031

**Người hướng dẫn:** TS. Trần Lâm Quân

[![Model: Gemma-4-4B](https://img.shields.io/badge/Student_Model-Gemma--4--4B-blue)](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai)
[![Model: PhoBERT-Multitask](https://img.shields.io/badge/Small_Model-PhoBERT--Multitask-orange)](https://huggingface.co/hung2903/phobert-vaccine-multitask)
[![Model: XLMR-Multitask](https://img.shields.io/badge/Small_Model-XLMR--Multitask-red)](https://huggingface.co/hung2903/xlmr-vaccine-multitask)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)

## 📌 Tổng Quan (Overview)
**VaccineNLP** là một hệ thống AI tiên phong được thiết kế để phân tích đa chiều thông tin về vắc-xin trên mạng xã hội Việt Nam. Dự án giải quyết bài toán phân loại tin giả thông qua chiến lược **Knowledge Distillation** (Chưng cất tri thức) và cung cấp khả năng giải thích (**Explainable AI - XAI**) bằng tiếng Việt.

## 🚀 Điểm Đột Phá Khoa Học (Scientific Novelty)
1. **Explainable AI via CoT Distillation**: Ép mô hình 4B sinh ra chuỗi lý luận (Chain-of-Thought) chuyên sâu như các mô hình 30B+.
2. **Zero-Cost Local Deployment**: Tối ưu hóa qua QLoRA (4-bit quantization) giúp mô hình chạy mượt mành trên phần cứng dân dụng.
3. **Linguistic Camouflage Detection**: Nhận diện "mật ngữ" của cộng đồng anti-vaccine (vd: *nước cất, sinh tố, chốt*).

## 📊 Kết Quả Đánh Giá (Final Benchmarks)

Dưới đây là bảng so sánh hiệu năng (Macro F1-score) giữa các kiến trúc trên tập **Benchmark Test Set (Gold Data)**:

| Mô hình | Loại | Misinfo (F1) | Stance (F1) | Sentiment (F1) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PhoBERT-v2** | Student (Encoder) | 0.4547 | **0.6608** | **0.7325** | **SOTA (Student)** |
| **XLM-R-v1** | Baseline (Encoder) | **0.4572** | 0.6247 | 0.6918 | Baseline |
| **Gemma-4-4B** | Teacher (Decoder) | 0.4400 | 0.6200 | 0.6600 | Explainable AI |

> [!TIP]
> **PhoBERT-v2** hiện là mô hình có hiệu năng cao nhất cho các tác vụ phân loại tiếng Việt trong dự án này, minh chứng cho hiệu quả của việc chưng cất tri thức (Knowledge Distillation) vào các mô hình Encoder chuyên biệt.

**Nhận định:** Trong khi Gemma-4 (Teacher) cung cấp khả năng giải thích (XAI) vượt trội, các mô hình Student (PhoBERT) lại cho thấy sự ổn định và chính xác cao hơn trong việc gán nhãn phân loại nhờ cấu trúc Encoder hai chiều mạnh mẽ.

## 📂 Cấu Trúc Thư Mục
* `datasets/`: Quản lý dữ liệu phân lớp nghiêm ngặt.
* `notebooks/`: Các kịch bản huấn luyện tối ưu (PhoBERT & Gemma-4).
* `src/`: Mã nguồn xử lý logic và Pipeline.
* `experiments/`: Lưu trữ kết quả F1, phân tích lỗi (Error Analysis) và XAI Cache.

---
*© 2026 VaccineNLP Project Team. Khóa luận tốt nghiệp. Last Updated: 23/04/2026*
