# 🏛️ Kiến trúc Pipeline Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam (Medallion Architecture)

Tài liệu này đặc tả luồng dữ liệu và cấu trúc dự án dựa trên khung tham chiếu Medallion, đảm bảo tính khoa học và minh bạch cho nghiên cứu.

---

## 1. Kiến trúc Medallion (Data Tiers)

Dữ liệu trong VaccineNLP được phân lớp theo độ tin chuẩn:

### 🥉 Tầng Bronze (Dữ liệu Thô)
*   **Đường dẫn**: `datasets/01_raw/`
*   **Nguồn**: Thu thập từ Facebook (Core), YouTube/TikTok (Low-yield/Reclaim). Đây là dữ liệu nguyên bản chưa qua xử lý.

### 🥈 Tầng Silver (Dữ liệu Làm sạch & Gán nhãn)
*   **Silver-Unlabeled** (`datasets/02_processed/`): Dữ liệu đã qua bộ lọc 3 lớp (Unicode, Teen-code, Domain filter).
*   **Silver-Labeled** (`datasets/04_silver_labels/`): Dữ liệu được gán nhãn tự động bởi **Teacher Model (31B)**. Mỗi record bao gồm nhãn 3 trục và chuỗi lý luận (CoT Reasoning) để phục vụ việc chưng cất tri thức.

### 🥇 Tầng Gold (Dữ liệu Vàng & Benchmark)
*   **Đường dẫn**: `datasets/03_processed/`
*   **Nội dung**: Chứa tập **Benchmark Test Set** (10%) đã được chuyên gia y tế tinh chỉnh và xác thực (Human-validated). Đây là tập dữ liệu chuẩn mực (Ground Truth) để đánh giá mọi mô hình trong Phase 5.

---

## 2. Cấu trúc Thư mục Dự án

```mermaid
graph Dir
    ROOT["VaccineNLP_Clean_V1/"]
    ROOT --> DAT["datasets/ (Medallion Data)"]
    ROOT --> SRC["src/"]
    SRC --> DP["data_pipeline/ (Archive Proof of Work)"]
    SRC --> MOD["modeling/ (Inference & Training)"]
    SRC --> COM["common/ (paths.py)"]
    ROOT --> EXP["experiments/ (MLflow & Models)"]
    ROOT --> NTB["notebooks/ (PhoBERT & Gemma QLoRA)"]
    ROOT --> DOC["docs/ (Hồ sơ học thuật)"]
```

---

## 3. Kho lưu trữ Proof of Work (src/data_pipeline/)

Để đảm bảo môi trường huấn luyện của Phase 5 "vô trùng" và không bị xung đột thư viện, toàn bộ các kịch bản thu thập (Apify) và tiền xử lý dữ liệu ở giai đoạn đầu đã được di chuyển vào `src/data_pipeline/`. 

Việc duy trì kho lưu trữ này đóng vai trò là **Proof of Work** minh chứng cho năng lực kỹ thuật trong việc xây dựng hệ thống thu thập dữ liệu tự động và quy trình làm sạch dữ liệu lớn trước khi đưa vào đào tạo AI.

---

## 4. Hạ tầng Cloud Inference (Kaggle)

Trong Phase 5, để giải quyết các hạn chế về phần cứng cục bộ (như lỗi Out-Of-Memory) khi chạy đánh giá mô hình Gemma-4 4B, dự án sử dụng **Kaggle Cloud (GPU T4 x2)** làm backend Inference. Hạ tầng này được đồng bộ với mã nguồn thông qua `kaggle kernels push` và quản lý tiến trình bằng file checkpoint `jsonl` nhằm đảm bảo khả năng phục hồi (fault-tolerant) khi quá trình chạy bị gián đoạn.

---
## 5. Kho lưu trữ Mô hình (Hugging Face Hub)

Các mô hình của dự án được lưu trữ công khai tại Hugging Face để cộng đồng có thể tái sử dụng:

| Mô hình | Loại | Link |
| :--- | :--- | :--- |
| **Gemma-4-XAI** | Student (Decoder) | [hung2903/gemma-4-E4B-unsloth-vaccine-xai](https://huggingface.co/hung2903/gemma-4-E4B-unsloth-vaccine-xai) |
| **PhoBERT-Multitask** | Student (Encoder) | [hung2903/phobert-vaccine-multitask](https://huggingface.co/hung2903/phobert-vaccine-multitask) |
| **XLMR-Multitask** | Baseline | [hung2903/xlmr-vaccine-multitask](https://huggingface.co/hung2903/xlmr-vaccine-multitask) |

---
*Cập nhật: 22/04/2026 | Phiên bản 3.0*
