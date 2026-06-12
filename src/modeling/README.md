# 🤖 Động Cơ Huấn Luyện & Đánh Giá Học Sâu (src/modeling/)

**Cập nhật:** 02/06/2026 · Phiên bản: 2.0 · Trạng thái: ✅ Đồng bộ thực tế 100%

Thư mục `src/modeling/` lưu trữ toàn bộ các lớp đối tượng và module huấn luyện, đánh giá mô hình học sâu của dự án VaccineNLP. Nó chứa các giải thuật cốt lõi giúp các mô hình Transformers (PhoBERT-v2, XLM-R-v1, Gemma-4-E4B QLoRA) học tri thức và đưa ra dự báo.

---

## 🏗️ Kiến Trúc Mô Hình Học Sâu Thực Tế (Model Architectures)

### 1. Động cơ Phân loại Đa nhiệm PhoBERT-v2 (Classification Engine)
Khác biệt hoàn toàn với các phiên bản cũ, PhoBERT-v2 là một mô hình **Encoder-only** thuần túy chuyên biệt cho phân loại tiếng Việt đa nhiệm:

```
                  Văn bản Tiếng Việt thô
                            │
                            ▼ (Tách từ pyvi)
                  Từ ghép đã được phân tách
                            │
                            ▼
               [PhoBERT-v2 Base Shared Encoder]
                            │
                            ▼
              Vector đại diện ẩn [CLS] (768-dim)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     [Misinfo Head]   [Stance Head]   [Sentiment Head]
      (Linear Layer)  (Linear Layer)   (Linear Layer)
            │               │               │
      2 nhãn đầu ra   3 nhãn đầu ra   3 nhãn đầu ra
```

### 2. Động cơ Giải thích Gemma-4-E4B-it QLoRA (XAI Engine)
Mô hình ngôn ngữ lớn **Decoder-only** kích thước 4B được lượng tử hóa 4-bit và huấn luyện LoRA để đảm nhận vai trò sinh lập luận y khoa định tính Chain-of-Thought (CoT).

---

## 🗂️ Chi Tiết Các Modules & Giải Thuật (File Directory)

### 1️⃣ **phobert_multitask_trainer.py** — Trình Huấn Luyện Mạng Đa Nhiệm
Chứa định nghĩa kiến trúc mạng đa nhiệm và vòng lặp huấn luyện có tích hợp **MLflow**:

- **Class `VaccineMultitaskModel`:**
  - `__init__(self, num_misinfo=3, num_stance=3, num_sentiment=3)`: Tải mô hình nền `vinai/phobert-base-v2`, khai báo 3 lớp tuyến tính `head_misinfo`, `head_stance`, `head_sentiment` tương ứng với số lượng lớp nhãn.
  - `forward(self, input_ids, attention_mask)`: Trích xuất trạng thái ẩn của token đặc biệt `[CLS]`, đưa qua dropout $0.1$ và đồng thời trả về logits của 3 trục nhiệm vụ.
- **Class `MultitaskTrainer`:**
  - Cấu hình hàm loss weighted tự động cho từng đầu ra:
    $$\text{Loss}_{\text{total}} = \text{Loss}_{\text{misinfo}} + \text{Loss}_{\text{stance}} + \text{Loss}_{\text{sentiment}}$$
  - Thiết lập log metrics tự động theo từng Epoch lên MLflow để giám sát hiện tượng quá khớp (Overfitting).

---

### 2️⃣ **dataset_loader.py** — Trình Tải & Xử Lý Dữ Liệu PyTorch
- **Class `VaccineDataset`:** Chuyển đổi dữ liệu JSONL thô thành định dạng `torch.utils.data.Dataset`. Tự động gọi tokenizer để tạo `input_ids` và `attention_mask` có đệm (padding) và cắt ngắn (truncation) về độ dài tối đa 256 tokens.
- **Class `DatasetLoader`:** Cung cấp hàm `get_dataloaders()` thực hiện chia tập dữ liệu (Train/Val/Test) theo phương pháp phân tầng (Stratified Split) dựa trên phân phối nhãn Misinfo.

---

### 3️⃣ **llm_inference_engine.py** — Động Cơ Chạy Gemma QLoRA local
- **Class `LLMInferenceEngine`:**
  - Tải mô hình base `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` với cấu hình lượng tử hóa BitsAndBytes NF4.
  - Tải và áp các trọng số tinh chỉnh LoRA adapter từ thư mục `experiments/models/gemma_qlora_xai`.
  - Hàm `predict()` định cấu hình tham số sinh văn bản lý giải: `temperature=0.7`, `top_p=0.9`, `do_sample=True`.

---

### 4️⃣ **inference.py** — Cầu Nối Hợp Nhất (Unified API)
- **Class `VaccineNLPInference`:**
  - Hợp nhất PhoBERT-v2 và Gemma-4 thành một hệ thống **Dual-Student Hybrid** duy nhất.
  - Khi có yêu cầu phân tích, nó sẽ chuyển văn bản cho PhoBERT-v2 để có nhãn phân loại cực nhanh và chính xác cao, sau đó chuyển nhãn này kết hợp ngữ cảnh sang Gemma để sinh lập luận y học bổ trợ, trả về một kết quả cấu trúc JSON thống nhất.

---

### 5️⃣ **error_analysis.py** — Bộ Phân Tích Lỗi Định Tính
- Cung cấp các hàm quét chéo các nhãn dự đoán sai của mô hình so với Ground Truth (Gold Test Set) để tìm ra False Positive (FP), False Negative (FN), phân nhóm lỗi y khoa và kết xuất Confusion Matrix normalize.

---

## 🏃 Lệnh Huấn Luyện Tái Tạo (Training Commands)

Để tái huấn luyện mô hình PhoBERT-v2 từ đầu và ghi nhận logs vào MLflow cục bộ:

```bash
# 1. Kích hoạt venv
.venv\Scripts\activate

# 2. Khởi động script huấn luyện đa nhiệm
python scripts/train_phobert_multitask.py --epochs 5 --batch_size 32 --lr 5e-5
```

---

## 📊 Kết Quả Đánh Giá Hiệu Năng Trên Gold Test Set (n=186)

- **PhoBERT-v2:** F1-Macro đạt `0.6967` (SOTA phân loại nhanh).
- **Gemma-4-E4B QLoRA:** F1-Macro đạt `0.6780` (SOTA phân tích sắc thái cảm xúc Sentiment đạt `0.7700`).

---

*VaccineNLP Deep Learning Group · HUPH 2026*
