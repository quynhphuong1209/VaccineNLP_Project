# 🧪 Thư Mục Thực Nghiệm (experiments/) - v1.0

**Cập nhật:** April 23, 2026 | Trạng thái: ✅ Active

## Mục Đích
Lưu trữ kết quả thực nghiệm, mô hình đã huấn luyện, metrics, logs, và tracking toàn bộ quá trình thử nghiệm.

Được tích hợp với **MLflow** để tracking tập trung tất cả experiments.

---

## 🏗️ Cấu Trúc Thư Mục

```
experiments/
├── models/                      # 🤖 Mô hình đã huấn luyện
│   ├── phobert_multitask_v1/   # PhoBERT multitask training
│   ├── phobert_multitask_v2/
│   ├── gemma4_qlora_v1/        # Gemma-4 QLoRA fine-tuning
│   └── gemma4_qlora_v2/
│
├── mlruns/                      # 📊 MLflow experiment tracking
│   ├── 0/
│   │   └── experiment_metadata.yaml
│   ├── 1/
│   │   └── [runs]/
│   │       ├── params.yaml
│   │       ├── metrics.json
│   │       └── artifacts/
│   └── ...
│
├── logs/                        # 📝 Training logs
│   ├── phobert_multitask_v1.log
│   ├── gemma4_qlora_v1.log
│   └── ...
│
└── results/                     # 📊 Kết quả thực nghiệm (optional)
    ├── evaluation_results.json
    └── comparison_table.csv
```

---

## 🤖 **models/** - Model Artifacts

### Mục Đích
Lưu trữ model weights, checkpoints, config, tokenizers

### PhoBERT Multitask Training
```
phobert_multitask_v1/
├── checkpoint-100/              # Checkpoint tại step 100
│   ├── pytorch_model.bin       # Model weights
│   ├── training_args.bin       # Training arguments
│   ├── optimizer.pt            # Optimizer state
│   └── scheduler.pt            # LR scheduler state
├── checkpoint-500/
├── best_model/                 # Best checkpoint (best val loss)
│   ├── pytorch_model.bin
│   ├── config.json             # Model config
│   └── tokenizer_config.json
├── final_model/                # Final checkpoint
├── training_results.json       # Metrics per epoch
├── eval_results.json           # Final evaluation
└── model_card.md               # Model documentation
```

### Gemma-4 QLoRA Fine-tuning
```
gemma4_qlora_v1/
├── adapter_model.bin           # LoRA adapter weights
├── adapter_config.json         # LoRA config
├── base_model.bin              # Base model (optional, usually remote)
├── training_args.bin
├── training_config.yaml        # QLoRA config
├── peft_config.json            # PEFT config
├── special_tokens_map.json
├── tokenizer_config.json
├── training_results.json
└── inference_config.yaml       # Inference setup
```

### Sử Dụng Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load PhoBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    'experiments/models/phobert_multitask_v1/best_model'
)
tokenizer = AutoTokenizer.from_pretrained(
    'experiments/models/phobert_multitask_v1/best_model'
)

# Load Gemma-4 with LoRA
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Gemma-4-4b')
model = PeftModel.from_pretrained(
    base_model,
    'experiments/models/gemma4_qlora_v1'
)
```

---

## 📊 **mlruns/** - MLflow Tracking

### Mục Đích
Centralized experiment tracking với MLflow

### Cấu Trúc
```
mlruns/
├── 0/                           # Default experiment
│   └── experiment.unk
├── 1/                           # Experiment 1: PhoBERT
│   ├── abc123def456.../ (run_id)
│   │   ├── params.yaml
│   │   │   ```yaml
│   │   │   learning_rate: 5e-5
│   │   │   batch_size: 32
│   │   │   num_epochs: 3
│   │   │   ```
│   │   ├── metrics/
│   │   │   ├── training_loss.json
│   │   │   ├── validation_loss.json
│   │   │   ├── accuracy.json
│   │   │   └── f1_score.json
│   │   ├── artifacts/
│   │   │   ├── pytorch_model.bin
│   │   │   ├── config.json
│   │   │   └── training_results.csv
│   │   └── meta.yaml
│   │       ```yaml
│   │       name: phobert_baseline
│   │       status: FINISHED
│   │       start_time: 1234567890
│   │       end_time: 1234567900
│   │       ```
│   └── xyz789.../ (another run)
│
└── 2/                           # Experiment 2: Gemma-4
    └── run1/
        ├── params.yaml
        ├── metrics/
        └── artifacts/
```

### MLflow UI
```bash
# Khởi động MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Truy cập: http://localhost:5000/
```

### MLflow API (Python)
```python
import mlflow

# Set experiment
mlflow.set_experiment("PhoBERT Multitask")

# Start run
with mlflow.start_run(run_name="baseline_v1"):
    # Log params
    mlflow.log_params({
        "learning_rate": 5e-5,
        "batch_size": 32,
        "epochs": 3
    })
    
    # Log metrics (per epoch)
    mlflow.log_metric("train_loss", 0.25, step=1)
    mlflow.log_metric("val_loss", 0.28, step=1)
    
    # Log artifacts
    mlflow.log_artifact("pytorch_model.bin")
    mlflow.log_artifact("training_results.json")
```

---

## 📝 **logs/** - Training Logs

### Mục Đích
Chi tiết logs từ training runs

### Format
```
[2024-03-15 10:30:45] INFO: Starting training for phobert_multitask_v1
[2024-03-15 10:31:02] DEBUG: Loaded 6800 training samples
[2024-03-15 10:31:05] DEBUG: Loaded 850 validation samples
[2024-03-15 10:35:10] INFO: Epoch 1/3 - Loss: 0.485 (train), 0.512 (val)
[2024-03-15 10:39:20] INFO: Epoch 2/3 - Loss: 0.235 (train), 0.312 (val)
[2024-03-15 10:43:30] INFO: Epoch 3/3 - Loss: 0.165 (train), 0.298 (val)
[2024-03-15 10:45:00] INFO: Training completed. Best model saved.
```

### Sử Dụng Logs
```bash
# Xem logs real-time
tail -f experiments/logs/phobert_multitask_v1.log

# Tìm errors
grep ERROR experiments/logs/*.log

# Lấy metrics
grep "Loss:" experiments/logs/phobert_multitask_v1.log
```

---

## 📊 **results/** - Experiment Results (Optional)

### Mục Đích
Tóm tắt kết quả experiments để so sánh

### Files

**evaluation_results.json**
```json
{
  "phobert_multitask_v1": {
    "accuracy": 0.875,
    "precision": 0.88,
    "recall": 0.87,
    "f1_score": 0.875,
    "roc_auc": 0.92,
    "confusion_matrix": [[...], [...], [...]]
  },
  "gemma4_qlora_v1": {
    "accuracy": 0.92,
    "precision": 0.93,
    "recall": 0.91,
    "f1_score": 0.92,
    "roc_auc": 0.95
  }
}
```

**comparison_table.csv**
```csv
Model,Accuracy,Precision,Recall,F1,ROC-AUC,Params,Train Time
PhoBERT v1,0.875,0.88,0.87,0.875,0.92,110M,45m
PhoBERT v2,0.89,0.892,0.888,0.89,0.93,110M,50m
Gemma-4 v1,0.92,0.93,0.91,0.92,0.95,4B,120m
Gemma-4 v2,0.93,0.94,0.92,0.93,0.96,4B,125m
```

---

## 🎯 Experiment Workflow

### 1️⃣ Training Phase
```python
# experiments/models/phobert_multitask_v1/
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    notebooks/01_phobert_multitask_training.ipynb
```

### 2️⃣ MLflow Tracking
```python
# Automatically tracked:
# - Hyperparameters
# - Metrics (loss, accuracy, etc.)
# - Model artifacts
# - Training config
```

### 3️⃣ Evaluation & Results
```bash
# View results
mlflow ui

# Compare models
mlflow.search_runs()
```

### 4️⃣ Archive & Document
```bash
# Archive old experiments
tar -czf experiments/archive/phobert_v1_old.tar.gz experiments/models/phobert_multitask_v1/

# Document in model_card.md
```

---

## 📈 Best Practices

1. **Naming Convention**
   - `phobert_multitask_v1`, `phobert_multitask_v2` (version incremental)
   - `gemma4_qlora_baseline`, `gemma4_qlora_tuned` (variant names)

2. **Checkpoint Frequency**
   - Save every N steps (e.g., 100 steps)
   - Save best model (best validation loss)
   - Save final model

3. **MLflow Logging**
   - Log all hyperparameters
   - Log metrics at regular intervals
   - Log final evaluation results
   - Log model artifacts

4. **Cleanup Policy**
   - Keep only meaningful checkpoints
   - Archive old experiments after 6 months
   - Remove duplicate/failed runs

5. **Documentation**
   - `model_card.md` cho mỗi model
   - Training notes trong MLflow
   - Architecture details trong README

---

## 🔍 Comparing Experiments

```python
import mlflow
import pandas as pd

# Get all runs
runs = mlflow.search_runs(experiment_ids=[1, 2])

# Filter & compare
comparison = runs[['run_id', 'params.learning_rate', 
                   'metrics.accuracy', 'metrics.f1_score']]

# Plot
comparison.plot(x='params.learning_rate', y=['metrics.accuracy', 'metrics.f1_score'])
```

---

## 📌 Typical Experiment Structure

```
Experiment: "PhoBERT Multitask v1"
├── Run 1: baseline (lr=5e-5, epochs=3)
│   ├── Accuracy: 0.875
│   ├── F1: 0.875
│   └── Time: 45m
│
├── Run 2: with_data_augmentation (lr=5e-5, epochs=3)
│   ├── Accuracy: 0.89
│   ├── F1: 0.89
│   └── Time: 50m
│
└── Run 3: with_scheduler (lr=5e-5→1e-5, epochs=3)
    ├── Accuracy: 0.88
    ├── F1: 0.88
    └── Time: 48m
```

---

**📅 Cập nhật:** April 2026  
**🔬 Tracking:** MLflow integrated  
**📊 Models:** PhoBERT, Gemma-4 QLoRA
