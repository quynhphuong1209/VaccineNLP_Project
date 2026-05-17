# 💻 Thư Mục Mã Nguồn (src/)

## Mục Đích
Lưu trữ toàn bộ mã nguồn chính của dự án, tổ chức thành các modules con:
- `common/` - Utilities chung
- `data_pipeline/` - Proof of Work: Data Collection & Preprocessing
- `modeling/` - Model Training & Inference
- `preprocessing/` - Additional preprocessing (duplicate location)

---

## 🗂️ Cấu Trúc Tổng Quan

```
src/
├── __init__.py                  # Đánh dấu package
│
├── 📂 common/                   # 🔧 Shared utilities
│   ├── __init__.py
│   ├── paths.py                # Path management
│   ├── versioning_manager.py   # Version tracking
│   └── README.md               # Common utilities docs
│
├── 📂 data_pipeline/            # 🔄 Data collection & preprocessing
│   ├── README.md
│   ├── collection/             # Data sources
│   └── preprocessing/          # Data cleaning
│
├── 📂 modeling/                 # 🤖 Model training & inference
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── phobert_multitask_trainer.py
│   ├── llm_inference_engine.py
│   ├── inference.py
│   └── README.md
│
└── 📂 preprocessing/            # 🧹 Additional preprocessing
    ├── __init__.py
    ├── pipeline.py
    ├── text_cleaner_v2.py
    ├── vn_tokenizer.py
    ├── ontology_mapper.py
    ├── preprocess_external_data.py
    └── README.md
```

---

## 🔧 **common/** - Shared Utilities

### Mục Đích
Cung cấp utilities chung dùng bởi tất cả modules

### **paths.py** - Centralized Path Management

**Chức Năng:** Định nghĩa tất cả đường dẫn dự án

```python
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'datasets'
MODEL_DIR = ROOT_DIR / 'experiments' / 'models'
LOG_DIR = ROOT_DIR / 'experiments' / 'logs'
CONFIG_DIR = ROOT_DIR / 'configs'

# Data layers
BRONZE_DIR = DATA_DIR / '01_raw'
SILVER_INTERIM_DIR = DATA_DIR / '02_interim'
SILVER_PROCESSED_DIR = DATA_DIR / '02_processed'
GOLD_DIR = DATA_DIR / '03_processed'
SILVER_LABELS_DIR = DATA_DIR / '04_silver_labels'
TEMP_DIR = DATA_DIR / 'temp'

# Specific files
MASTER_DATASET = GOLD_DIR / 'reclaimed_master_pool_vn_clean.json'
CORPUS_UNLABELED = SILVER_PROCESSED_DIR / 'corpus_1856_unlabeled.json'
```

**Sử Dụng:**
```python
from src.common.paths import MASTER_DATASET, MODEL_DIR

# Load data
with open(MASTER_DATASET) as f:
    data = json.load(f)

# Save model
model.save_pretrained(MODEL_DIR / 'my_model')
```

### **versioning_manager.py** - Version Tracking

**Chức Năng:** Quản lý versioning cho datasets & models

```python
class VersioningManager:
    def create_version(self, data_path, version_tag="v1"):
        """Create versioned copy of dataset"""
        versioned_path = f"{data_path}_{version_tag}.json"
        copy_file(data_path, versioned_path)
        return versioned_path
    
    def get_latest_version(self, data_pattern):
        """Get latest version of dataset matching pattern"""
        pass
    
    def check_compatibility(self, model_version, data_version):
        """Check if model & data versions are compatible"""
        pass
```

**Sử Dụng:**
```python
vm = VersioningManager()

# Track dataset version
dataset_v1 = vm.create_version('datasets/03_processed/data.json', 'v1.0')

# Check compatibility
compatible = vm.check_compatibility(
    model_version='phobert_multitask_v1',
    data_version='v1.0'
)
```

---

## 🔄 **data_pipeline/** - Data Collection & Preprocessing

### Subfolders:
- `collection/` - Data sources (Apify, Facebook, etc.)
- `preprocessing/` - Text cleaning, filtering, tokenization

*Chi tiết xem: data_pipeline/README.md*

---

## 🤖 **modeling/** - Model Training & Inference

### Modules

### **dataset_loader.py** - Data Loading

**Chức Năng:** Load datasets, create PyTorch DataLoaders

```python
class DatasetLoader:
    def __init__(self, data_path, tokenizer=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
    
    def get_dataloaders(self, batch_size=32, train_ratio=0.8):
        """
        Load data & create train/val/test loaders
        
        Returns:
            train_loader, val_loader, test_loader
        """
        pass
    
    def get_statistics(self):
        """Get dataset statistics (size, class distribution, etc.)"""
        pass
```

**Sử Dụng:**
```python
loader = DatasetLoader('datasets/03_processed/master_data.json')
train_dl, val_dl, test_dl = loader.get_dataloaders(batch_size=32)

for batch in train_dl:
    texts, labels = batch
    # Training step
```

---

### **phobert_multitask_trainer.py** - PhoBERT Training

**Chức Năng:** Train PhoBERT với multitask learning

```python
class PhoBERTMultitaskTrainer:
    def __init__(self, model_name='vinai/phobert-base'):
        self.model = self._build_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _build_model(self, model_name):
        """
        PhoBERT + 2 heads:
        - Classification head
        - CoT generation head
        """
        pass
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=5e-5):
        """Train model with combined loss"""
        pass
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        pass
```

**Sử Dụng:**
```python
trainer = PhoBERTMultitaskTrainer()
trainer.train(train_loader, val_loader, epochs=3)
trainer.save('experiments/models/phobert_multitask_v1/')
```

---

### **llm_inference_engine.py** - Gemma-4 Inference Server

**Chức Năng:** Inference server cho Gemma-4 4B (QLoRA)

```python
class LLMInferenceEngine:
    def __init__(self, model_path, quantization=True):
        self.model_path = model_path
        self.model = self._load_model(quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def _load_model(self, quantization=True):
        """Load Gemma-4 with optional 4-bit quantization"""
        pass
    
    def predict(self, text, max_length=256, temperature=0.7):
        """Generate CoT explanation"""
        pass
    
    def batch_predict(self, texts, batch_size=8):
        """Batch inference"""
        pass
```

**Server Implementation (Flask/FastAPI):**
```python
from flask import Flask, request

app = Flask(__name__)
engine = LLMInferenceEngine('experiments/models/gemma4_qlora_v1')

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    text = data['text']
    result = engine.predict(text)
    return {'result': result}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### **inference.py** - Unified Inference API

**Chức Năng:** High-level API cho PhoBERT + Gemma inference

```python
class VaccineNLPInference:
    def __init__(self, phobert_path, gemma_path):
        self.phobert = PhoBERTModel(phobert_path)
        self.gemma = LLMInferenceEngine(gemma_path)
    
    def predict(self, text):
        """
        Predict label + generate CoT explanation
        
        Returns:
            {
                'text': input text,
                'label': 'misinformation' | 'safe' | 'misleading',
                'confidence': 0.92,
                'cot_explanation': 'Đây là thông tin sai vì...'
            }
        """
        # Step 1: PhoBERT classification
        label, confidence = self.phobert.classify(text)
        
        # Step 2: Gemma CoT generation
        cot = self.gemma.predict(f"Explain why: {text}")
        
        return {
            'text': text,
            'label': label,
            'confidence': confidence,
            'cot_explanation': cot
        }
```

**Sử Dụng:**
```python
inference = VaccineNLPInference(
    phobert_path='experiments/models/phobert_multitask_v1/',
    gemma_path='experiments/models/gemma4_qlora_v1/'
)

result = inference.predict("Nội dung bài viết về vaccine...")
print(f"Label: {result['label']}")
print(f"Explanation: {result['cot_explanation']}")
```

---

## 🧹 **preprocessing/** - Additional Preprocessing

### **pipeline.py** - Preprocessing Orchestrator

**Flow:**
```
Raw Text
  ↓
[Clean] text_cleaner_v2.py
  ↓
[Filter] language_filter.py
  ↓
[Tokenize] vn_tokenizer.py
  ↓
[Map] ontology_mapper.py
  ↓
Cleaned Text + Metadata
```

### **text_cleaner_v2.py** - Text Cleaning

```python
class TextCleaner:
    def clean(self, text):
        """
        Remove HTML, normalize unicode, remove special chars
        """
        text = self._remove_html(text)
        text = self._normalize_unicode(text)
        text = self._remove_extra_whitespace(text)
        return text
```

### **vn_tokenizer.py** - Vietnamese Tokenization

```python
class VietnameseTokenizer:
    def tokenize(self, text):
        """Segment text into words using Vietnamese rules"""
        # Using: pyvi, underthesea
        pass
    
    def get_pos_tags(self, tokens):
        """Get part-of-speech tags"""
        pass
```

### **ontology_mapper.py** - Taxonomy Mapping

```python
class OntologyMapper:
    def map_to_category(self, text):
        """Map text to vaccine taxonomy categories"""
        # Load taxonomy from configs/taxonomy.json
        # Match against categories
        pass
```

---

## 📚 Sử Dụng Modules

### Import Patterns

```python
# Import từ common
from src.common.paths import MASTER_DATASET, MODEL_DIR

# Import từ modeling
from src.modeling.inference import VaccineNLPInference
from src.modeling.dataset_loader import DatasetLoader

# Import từ preprocessing
from src.preprocessing.pipeline import PreprocessingPipeline
```

### Complete Example

```python
import json
from src.common.paths import MASTER_DATASET
from src.modeling.inference import VaccineNLPInference

# Load inference
inference = VaccineNLPInference(
    phobert_path='experiments/models/phobert_multitask_v1/',
    gemma_path='experiments/models/gemma4_qlora_v1/'
)

# Load data
with open(MASTER_DATASET) as f:
    dataset = json.load(f)

# Inference on all samples
results = []
for sample in dataset['data']:
    result = inference.predict(sample['text'])
    results.append(result)

# Save results
output_path = 'datasets/04_silver_labels/predictions.jsonl'
with open(output_path, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
```

---

## ✅ Best Practices

1. **Imports** - Use `from src.common.paths` cho tất cả path references
2. **Naming** - Descriptive names cho classes, functions, variables
3. **Docstrings** - Document mục đích, parameters, returns
4. **Error Handling** - Try-except, logging errors
5. **Type Hints** - Add type annotations khi có thể
6. **Modularity** - Single responsibility per module

---

## 📖 Documentation

- Mỗi module có docstrings
- `README.md` trong subfolders
- Type hints for better IDE support
- Examples trong code comments

---

**📅 Updated:** April 2026  
**🏗️ Architecture:** Modular, reusable components  
**🔗 Dependencies:** PyTorch, Transformers, PEFT
