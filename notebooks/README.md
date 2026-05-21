# 📔 Thư Mục Notebooks (notebooks/) - v1.2

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Complete

## 🎯 Mục Đích

Lưu trữ các Jupyter Notebooks tương tác để:
- 🔍 Exploratory Data Analysis (EDA) - Data profiling
- 🧠 Model training & hyperparameter tuning - PhoBERT & Gemma-4
- 📊 Visualization & evaluation - Metrics & results analysis
- 🧪 Experiments & prototyping - Rapid development
- 🔮 Inference & evaluation - Model predictions & explanations

**Trạng thái:** 4 main notebooks + variations

---

## 📋 Danh Sách Notebooks (May 21, 2026 - Final Version)

### 1️⃣ **01_phobert_multitask_training.ipynb**

**Mục Đích:** Huấn luyện PhoBERT cho Multitask Learning

**Kiến Trúc:**
```
PhoBERT Base (110M params)
    ↓
[Shared Encoder]
    ↓
    ├─→ [Head 1] Classification (Misinformation Detection)
    │   └─→ Binary/Multi-class Output
    │
    └─→ [Head 2] CoT Generation (Chain-of-Thought)
        └─→ Text Generation Output
```

**Nội Dung Cells:**

1. **Setup & Imports**
   - Import libraries (transformers, torch, sklearn)
   - GPU detection & setup
   - Config parameters

2. **Data Loading**
   ```python
   # Load từ datasets/03_processed/
   from src.modeling.dataset_loader import DatasetLoader
   
   loader = DatasetLoader('datasets/03_processed/reclaimed_master_pool_vn_clean.json')
   train_dl, val_dl, test_dl = loader.get_dataloaders(
       batch_size=32,
       train_ratio=0.8,
       val_ratio=0.1
   )
   ```

3. **EDA (Exploratory Data Analysis)**
   - Label distribution
   - Text length distribution
   - Class imbalance analysis
   - Data quality checks

4. **Tokenization**
   ```python
   from transformers import AutoTokenizer
   
   tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
   # Tokenize texts
   ```

5. **Model Architecture**
   ```python
   # Custom multitask head
   class PhoBERTMultitask(nn.Module):
       def __init__(self):
           super().__init__()
           self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
           self.clf_head = ClassificationHead()    # Task 1
           self.cot_head = CoTGenerationHead()     # Task 2
   ```

6. **Training Loop**
   - Forward pass (both tasks)
   - Loss calculation (combined loss)
   - Backward pass
   - Optimizer step
   - Validation per epoch

7. **Evaluation**
   - Classification metrics (accuracy, F1, ROC-AUC)
   - CoT quality metrics
   - Confusion matrix
   - Error analysis

8. **Visualization**
   - Training curves (loss, accuracy)
   - Confusion matrices
   - Sample predictions

9. **Save & Export**
   - Save to `experiments/models/phobert_multitask_v1/`
   - Save tokenizer & config
   - MLflow logging

**Hyperparameters:**
```python
learning_rate = 5e-5
batch_size = 32
num_epochs = 3
warmup_steps = 500
weight_decay = 0.01
```

**Output:**
```
experiments/models/phobert_multitask_v1/
├── best_model/
├── final_model/
├── training_results.json
└── eval_results.json
```

---

### 2️⃣ **02_gemma4_4b_qlora_training.ipynb**

**Mục Đích:** Fine-tune Gemma-4 4B với QLoRA (Knowledge Distillation)

**Context:**
```
Teacher Model: PhoBERT 110M
    ↓
Generate CoT Explanations
    ↓
Student Model: Gemma-4 4B (QLoRA)
    ↓
Learn to mimic teacher's reasoning
```

**QLoRA Setup:**
- **Base Model:** meta-llama/Gemma-4-4b (4B parameters)
- **Quantization:** 4-bit (NF4)
- **LoRA Rank:** r=64
- **LoRA Alpha:** 16
- **Target Modules:** q_proj, v_proj

**Nội Dung Cells:**

1. **Setup & QLoRA Config**
   ```python
   from peft import LoraConfig, get_peft_model
   from bitsandbytes.nn import Linear4bit
   
   qlora_config = LoraConfig(
       r=64,
       lora_alpha=16,
       target_modules=['q_proj', 'v_proj'],
       lora_dropout=0.05,
       task_type="CAUSAL_LM"
   )
   ```

2. **Load Teacher Predictions**
   ```python
   # Load PhoBERT outputs (labels + CoT from teacher)
   teacher_outputs = load_predictions(
       'experiments/models/phobert_multitask_v1/predictions.json'
   )
   ```

3. **Prepare Training Data**
   ```python
   # Format: (text, expected_cot_from_teacher)
   # Instruction tuning format
   train_data = [
       {
           "instruction": "Phân tích bài viết sau",
           "input": "...",
           "output": "Đây là thông tin sai vì ..."
       }
   ]
   ```

4. **Model Loading**
   ```python
   from transformers import AutoModelForCausalLM
   
   model = AutoModelForCausalLM.from_pretrained(
       'meta-llama/Gemma-4-4b',
       quantization_config=bnb_config,
       device_map='auto'
   )
   
   model = get_peft_model(model, qlora_config)
   ```

5. **Training Configuration**
   ```python
   training_args = TrainingArguments(
       output_dir='experiments/models/gemma4_qlora_v1/',
       per_device_train_batch_size=4,  # Due to 4-bit quant
       gradient_accumulation_steps=4,
       num_train_epochs=1,
       learning_rate=2e-4,
       warmup_ratio=0.1,
       save_strategy='epoch',
       logging_steps=100
   )
   ```

6. **Training Loop**
   ```python
   trainer = SFTTrainer(
       model=model,
       train_dataset=train_dataset,
       args=training_args,
       peft_config=qlora_config
   )
   trainer.train()
   ```

7. **Evaluation**
   - BLEU score (CoT generation)
   - ROUGE score (text similarity)
   - Human evaluation (optional)
   - Inference speed

8. **Inference & Testing**
   ```python
   # Run inference
   prompt = "Bài viết này: ..."
   output = model.generate(
       inputs,
       max_length=256,
       temperature=0.7,
       top_p=0.9
   )
   ```

9. **Save & Export**
   - Save LoRA adapter
   - Save tokenizer
   - Create inference config

**Output:**
```
experiments/models/gemma4_qlora_v1/
├── adapter_model.bin
├── adapter_config.json
├── tokenizer_config.json
└── training_results.json
```

---

### 3️⃣ **03_gemma4_inference_eval.ipynb**

**Mục Đích:** Inference & Evaluation cho Gemma-4 Model

**Kiến Trúc:**
```
Load Trained Model
    ↓
Batch Inference
    ↓
Calculate Metrics
    ↓
Analyze Results
    ↓
Generate Report
```

**Nội Dung Cells:**

1. **Setup & Imports**
   - Load transformers, PEFT
   - GPU setup
   - Config parameters

2. **Load Model**
   ```python
   from src.modeling.llm_inference_engine import LLMInferenceEngine
   
   engine = LLMInferenceEngine('experiments/models/gemma4_qlora_v1')
   ```

3. **Load Test Data**
   - Load từ datasets/03_processed/
   - Prepare batches
   - Split test set

4. **Run Inference**
   ```python
   # Batch inference
   results = engine.batch_predict(test_texts, batch_size=8)
   ```

5. **Calculate Metrics**
   - BLEU score (text generation quality)
   - ROUGE score (CoT explanation quality)
   - Perplexity
   - Response time

6. **Visualization**
   - Plot metrics
   - Confusion matrices
   - Error distribution
   - Performance summary

7. **Generate Analysis Report**
   - Task-wise performance
   - Per-class metrics
   - Failure case analysis
   - Suggestions for improvement

8. **Save Results**
   - Save predictions (predictions.json)
   - Save metrics (metrics.json)
   - Generate HTML report
   - MLflow logging

**Output:**
```json
{
  "misinformation": {"precision": 0.78, "recall": 0.72, "f1": 0.75},
  "stance": {"precision": 0.68, "recall": 0.65, "f1": 0.66},
  "sentiment": {"precision": 0.72, "recall": 0.70, "f1": 0.71},
  "overall_accuracy": 0.72,
  "avg_response_time": 0.32
}
```

---

### 4️⃣ **vaccine-nlp-gemma-eval.ipynb** (Multiple Versions)

**Mục Đích:** Evaluation & Comparison cho Gemma Models

**Versions Available:**
- `vaccine-nlp-gemma-eval.ipynb` - Main evaluation notebook (latest)
- `vaccine-nlp-eval-v5.ipynb` - Version 5 evaluation (archived)
- Additional variations for different experiment scenarios

**Status (April 23, 2026):**
- ✅ Multiple versions maintained
- ✅ Comprehensive evaluation metrics
- ✅ Ready for comparative analysis

**Nội Dung:**

1. **Load Models**
   - Gemma-4 4B with QLoRA adapters
   - PhoBERT (for comparison baseline)
   - Alternative model versions

2. **Run Inference**
   ```python
   # Batch inference with timing
   start_time = time.time()
   results = engine.batch_predict(test_texts, batch_size=16)
   inference_time = time.time() - start_time
   ```

3. **Evaluate Results**
   ```python
   # Multi-task metrics
   metrics = {
       'misinformation': evaluate_classification(predictions, labels),
       'stance': evaluate_stance(predictions, labels),
       'sentiment': evaluate_sentiment(predictions, labels),
       'cot_quality': evaluate_cot_explanation(cot_text),
       'inference_speed': inference_time
   }
   ```

4. **Compare Models**
   - PhoBERT vs Gemma-4 performance
   - Speed vs accuracy trade-off
   - Parameter efficiency analysis
   - Deployment suitability

5. **Detailed Analysis**
   - Per-class performance breakdown
   - Error analysis & patterns
   - Confidence distribution
   - Edge case identification

6. **Visualize**
   - ROC curves (per task)
   - Confusion matrices (multi-class)
   - Performance comparison bars
   - Training/inference speed graphs

7. **Generate Report**
   ```python
   # Create comprehensive report
   report = {
       'timestamp': datetime.now(),
       'model_versions': model_info,
       'test_set_stats': test_stats,
       'results': metrics,
       'recommendations': improvement_suggestions
   }
   ```

8. **Export Results**
   - CSV report (metrics.csv)
   - JSON results (results.json)
   - PNG plots/images
   - Markdown summary

---

## 📊 Full Notebook Summary Table

| # | Notebook | Mục Đích | Loại | Phiên Bản | Status |
|---|----------|---------|------|----------|--------|
| 1️⃣ | 01_phobert_multitask_training.ipynb | PhoBERT training (multitask) | Training | v1.0 | ✅ Active |
| 2️⃣ | 02_gemma4_4b_qlora_training.ipynb | Gemma-4 QLoRA fine-tuning | Training | v1.0 | ✅ Active |
| 3️⃣ | 03_gemma4_inference_eval.ipynb | Gemma-4 inference & evaluation | Evaluation | v1.0 | ✅ Active (NEW) |
| 4️⃣ | vaccine-nlp-gemma-eval.ipynb | Multi-model comparison | Evaluation | v5+ | ✅ Maintained |

**Total:** 8 files (4 unique types + variations)

---

## 🚀 Chạy Notebooks

### Local Jupyter
```bash
# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
```

### VS Code Jupyter Extension
```
1. Open notebook file
2. Select kernel (Python environment)
3. Run cells individually (Shift+Enter)
4. Or run all cells (Ctrl+Shift+Enter)
```

### Command Line Execution
```bash
# Convert notebook to script
jupyter nbconvert --to python notebooks/01_phobert_multitask_training.ipynb
python notebooks/01_phobert_multitask_training.py

# Or use nbconvert with execution
jupyter nbconvert --to notebook --execute notebooks/01_phobert_multitask_training.ipynb
```

---

## 📊 Notebook Best Practices

### 1. Cell Organization
```
Setup & Imports       (Config, paths, libraries)
    ↓
Data Loading          (Load from datasets/)
    ↓
EDA                   (Explore, visualize)
    ↓
Model Architecture    (Define model)
    ↓
Training/Inference    (Run experiments)
    ↓
Evaluation            (Metrics, analysis)
    ↓
Visualization         (Plots, reports)
    ↓
Save & Export         (Models, results)
```

### 2. Documentation
- Markdown cells với descriptive instructions
- Code comments cho complex logic
- Docstrings cho custom functions
- Section headers clearly marked

### 3. Reproducibility
```python
# Set random seeds for reproducibility
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 4. Performance Monitoring
- Track GPU memory usage
- Monitor training time per epoch
- Log validation metrics
- Save best checkpoints

### 5. Output Management
- Clear cell outputs before committing
- Save important results to files
- Use MLflow for experiment tracking
- Generate final reports

---

## 🔄 Typical Notebook Workflow

```
1. SETUP
   - Import libraries
   - Configure paths
   - Check GPU
   
2. DATA
   - Load datasets
   - Check data shape
   - Explore samples
   
3. ANALYSIS
   - Distribution plots
   - Class imbalance
   - Quality checks
   
4. PREPROCESSING
   - Tokenization
   - Padding/Truncation
   - DataLoader creation
   
5. MODEL
   - Define architecture
   - Initialize weights
   - Set loss/optimizer
   
6. TRAINING
   - Training loop
   - Validation per epoch
   - Save best model
   
7. EVALUATION
   - Calculate metrics
   - Generate predictions
   - Error analysis
   
8. RESULTS
   - Visualization
   - Report generation
   - Final export
```

---

## 💾 Saving Models & Results

```python
# Save trained model
model.save_pretrained('experiments/models/phobert_multitask_v1/')
tokenizer.save_pretrained('experiments/models/phobert_multitask_v1/')

# Save evaluation results
import json
with open('experiments/models/phobert_multitask_v1/eval_results.json', 'w') as f:
    json.dump({
        'accuracy': accuracy,
        'f1_score': f1,
        'metrics': classification_report_dict
    }, f, indent=2)

# MLflow tracking
import mlflow
mlflow.log_params(hyperparameters)
mlflow.log_metrics(metrics)
mlflow.log_artifact('experiments/models/phobert_multitask_v1/')
```

---

## 🔐 Security & Best Practices

| ❌ DON'T | ✅ DO |
|----------|------|
| Hardcode API keys | Use `.env` file |
| Commit model files | Use `.gitignore` |
| Use absolute paths | Use relative paths |
| Keep large outputs | Clear before commit |
| Skip documentation | Add markdown cells |

---

## 📋 Kernel Management

```
File: notebooks/kernel-metadata.json  - Active kernel config
File: notebooks/kernel-metadata-bak.json - Backup
File: notebooks/kernel-metadata-test.json - Test config
```

Các metadata files này track kernel information cho notebook reproduction.

---

**📅 Updated:** April 22, 2026 (v1.1)  
**🔬 Framework:** PyTorch + Transformers (HuggingFace)  
**📊 Integration:** MLflow tracking + Unsloth optimization  
**🖥️ GPU Support:** CUDA 12.1, bitsandbytes 4-bit quantization  
**🐍 Python:** 3.10+
