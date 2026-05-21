# 🧪 Experiments Module (experiments/)

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Active

## Mục Đích

Lưu trữ các model checkpoints, kết quả đánh giá, và artifacts từ các lần huấn luyện. Quản lý toàn bộ quy trình từ training đến evaluation.

---

## 🗂️ Cấu Trúc Thư Mục

```
experiments/
├── models/                      # 🤖 Lưu trữ model checkpoints
│   ├── phobert_multitask_v1/   # PhoBERT multitask (110M params)
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── README.md
│   │
│   ├── xlmr_multitask_v1/      # XLM-RoBERTa multitask
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   └── README.md
│   │
│   └── gemma4_qlora_v1/        # Gemma-4 4B + QLoRA
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       ├── config.json
│       └── README.md
│
└── results/                     # 📊 Lưu trữ kết quả đánh giá
    ├── benchmark_results/      # Kết quả trên Gold Test Set
    │   ├── phobert_v2_results.json
    │   ├── xlmr_v1_results.json
    │   ├── gemma4_4b_results.json
    │   └── comparison_metrics.json
    │
    ├── training_logs/          # Logs từ quá trình huấn luyện
    │   ├── phobert_training.log
    │   ├── xlmr_training.log
    │   └── gemma4_training.log
    │
    └── analysis/               # Phân tích lỗi, EDA
        ├── error_analysis.json
        ├── confusion_matrices.json
        └── class_imbalance_report.json
```

---

## 🤖 **models/** - Model Checkpoints & Artifacts

### Mục Đích
Lưu trữ các trained model checkpoints để sử dụng lại, inference, và versioning.

### Cấu Trúc File Mô Hình

**Đối với Encoder Models (PhoBERT, XLM-R):**
```
model_dir/
├── config.json              # Model configuration (vocab size, hidden_dim, etc.)
├── pytorch_model.bin        # Model weights (binary format)
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer config
├── special_tokens_map.json  # Special tokens mapping
└── README.md               # Model card & usage guide
```

**Đối với QLoRA Models (Gemma-4 4B):**
```
model_dir/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin        # LoRA adapter weights
├── config.json              # Base model config
├── special_tokens_map.json  # Special tokens mapping
└── README.md               # Model card
```

### Tải & Sử Dụng Models

**Load PhoBERT Model:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    'experiments/models/phobert_multitask_v1'
)
model = AutoModel.from_pretrained(
    'experiments/models/phobert_multitask_v1'
)

# Inference
inputs = tokenizer("Xin chào", return_tensors="pt")
outputs = model(**inputs)
```

**Load QLoRA Model (Gemma-4):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Gemma-4-4b',
    quantization_config=bnb_config,
    device_map='auto'
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    'experiments/models/gemma4_qlora_v1'
)
```

### Model Versioning
Mỗi model được đánh version theo pattern: `{model_name}_{version}`
- `v1` - Phiên bản đầu tiên
- `v2` - Cải thiện với thêm dữ liệu hoặc hyperparameter
- `v3` - Tối ưu hóa hiệu năng

---

## 📊 **results/** - Evaluation Results & Analysis

### **benchmark_results/** - Kết Quả Trên Gold Test Set

Chứa kết quả đánh giá chi tiết trên tập Gold Benchmark (xác nhận chất lượng cao).

**Format JSON:**
```json
{
  "metadata": {
    "model_name": "phobert_v2",
    "test_date": "2026-05-21",
    "dataset_size": 186,
    "test_set_version": "gold_benchmark_v3"
  },
  "overall_metrics": {
    "accuracy": 0.7312,
    "precision_macro": 0.7105,
    "recall_macro": 0.6922,
    "f1_macro": 0.6967
  },
  "per_task": {
    "misinfo_detection": {
      "accuracy": 0.8226,
      "f1": 0.6996,
      "class_breakdown": {
        "misinfo": {"f1": 0.5091, "support": 28},
        "correct": {"f1": 0.8901, "support": 158}
      }
    },
    "stance_analysis": {
      "accuracy": 0.6989,
      "f1": 0.6640,
      "class_breakdown": {
        "support": {"f1": 0.6122, "support": 54},
        "against": {"f1": 0.6596, "support": 48},
        "neutral": {"f1": 0.7202, "support": 84}
      }
    },
    "sentiment_classification": {
      "accuracy": 0.7581,
      "f1": 0.7266,
      "class_breakdown": {
        "positive": {"f1": 0.7467, "support": 80},
        "negative": {"f1": 0.6947, "support": 60},
        "neutral": {"f1": 0.7385, "support": 46}
      }
    }
  }
}
```

### **training_logs/** - Logs Từ Quá Trình Huấn Luyện

Chứa logs chi tiết về:
- Epoch-wise metrics (loss, accuracy, validation score)
- Learning rate scheduling
- GPU memory usage
- Training time
- Warnings & errors (nếu có)

**Format:**
```
[2026-05-20 10:15:32] Starting training...
[2026-05-20 10:15:35] GPU 0: 24GB VRAM available
[2026-05-20 10:16:00] Epoch 1/3
[2026-05-20 10:16:30] Batch 1/100 - Loss: 2.3456, LR: 5e-5
[2026-05-20 10:17:00] Batch 2/100 - Loss: 2.1234, LR: 5e-5
...
[2026-05-20 11:45:00] Epoch 1 finished - Val Loss: 1.8765, Val Acc: 0.72
```

### **analysis/** - Error Analysis & Detailed Metrics

**error_analysis.json:**
```json
{
  "total_samples": 240,
  "correct_predictions": 170,
  "incorrect_predictions": 70,
  "error_rate": 0.2917,
  "error_breakdown": {
    "misclassified_labels": 45,
    "contradicting_predictions": 15,
    "edge_cases": 10
  },
  "most_common_mistakes": [
    {
      "true_label": "misinfo",
      "predicted_label": "correct",
      "count": 18,
      "examples": ["text1", "text2", ...]
    }
  ]
}
```

**confusion_matrices.json:**
```json
{
  "misinfo_detection": {
    "true_negatives": 150,
    "false_positives": 8,
    "false_negatives": 8,
    "true_positives": 74
  },
  "stance_analysis": {
    "matrix": [
      [45, 5, 4],
      [8, 35, 5],
      [2, 3, 79]
    ],
    "class_labels": ["support", "against", "neutral"]
  }
}
```

---

## 🔧 Hướng Dẫn Sử Dụng

### Lưu Model Sau Training
```python
from pathlib import Path

model_dir = Path('experiments/models/phobert_multitask_v1')
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save evaluation results
import json
with open(model_dir / 'eval_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)
```

### So Sánh Models
```bash
# Tạo file so sánh từ các kết quả
python scripts/compare_models.py \
    --results_dir experiments/results/benchmark_results/ \
    --output experiments/results/comparison_metrics.json
```

### Track Experiment History
Mỗi lần training, tạo log directory với timestamp:
```
experiments/results/training_logs/
├── phobert_training_20260520_101532.log
├── xlmr_training_20260520_102045.log
└── gemma4_training_20260520_111200.log
```

---

## 📈 Benchmark Summary

| Mô hình | Misinfo F1 | Stance F1 | Sentiment F1 | Status |
|---|:---:|:---:|:---:|:---|
| **PhoBERT-v2** | 0.6996 | **0.6640** | 0.7266 | ✅ SOTA (Classification Engine) |
| **Gemma-4-4B** | 0.6377 | 0.6264 | **0.7700** | ✅ SOTA (Sentiment / XAI) |
| **XLM-R-v1** | **0.7038** | 0.6224 | 0.6866 | 📊 Baseline |

---

*Cập nhật: 21/05/2026 | Phiên bản 1.1 | Status: ✅ Complete & Production-Ready*
