# 🤖 Modeling Module (src/modeling/) - v1.2

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Active & Optimized
Cung cấp các công cụ cho Model Training, Inference, và Evaluation.

Bao gồm 2 mô hình chính:
1. **PhoBERT** - Classification + CoT generation (Multitask)
2. **Gemma-4 4B** - QLoRA fine-tuning (Knowledge Distillation)

---

## 🏗️ Model Architecture

### PhoBERT Multitask Architecture

```
Input Text
    ↓
[PhoBERT Base 110M params]
    ↓
Shared Encoder (768-dim hidden state)
    ↓
    ├──→ [Classification Head] → Misinfo Detection
    │     Binary/Multi-class Output
    │
    └──→ [CoT Generation Head] → Chain-of-Thought
          Sequence-to-Sequence Output
```

### Gemma-4 QLoRA Architecture

```
Base Model (Gemma-4 4B)
    ├─ Quantized to 4-bit (NF4)
    └─ LoRA Adapters
        ├─ Rank: 64
        ├─ Alpha: 16
        └─ Target Modules: q_proj, v_proj
            ↓
Input Text
    ↓
[Quantized Base + LoRA Adapters]
    ↓
Chain-of-Thought Explanation Output
```

---

## 📚 **dataset_loader.py** - Data Loading

**Chức Năng:** Load datasets, create PyTorch DataLoaders, handle batching

```python
class DatasetLoader:
    """
    Load and manage datasets for training
    
    Supports:
    - JSON/JSONL formats
    - Train/val/test splits
    - Batch processing
    - Data augmentation
    """
    
    def __init__(self, data_path, tokenizer=None, max_length=512):
        """
        Initialize loader
        
        Args:
            data_path: Path to dataset JSON
            tokenizer: Tokenizer (default: AutoTokenizer)
            max_length: Max sequence length
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
    
    def _load_data(self):
        """Load from JSON/JSONL"""
        pass
    
    def get_dataloaders(self, batch_size=32, train_ratio=0.8, 
                       val_ratio=0.1, shuffle=True):
        """
        Create train/val/test DataLoaders
        
        Returns:
            train_loader, val_loader, test_loader
        """
        pass
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'total_samples': len(self.data),
            'avg_length': np.mean([len(d['text'].split()) for d in self.data]),
            'label_distribution': self._get_label_dist(),
            'splits': {
                'train': int(len(self.data) * 0.8),
                'val': int(len(self.data) * 0.1),
                'test': int(len(self.data) * 0.1)
            }
        }
    
    def get_class_weights(self):
        """Get class weights for imbalanced data"""
        pass
```

**Usage:**
```python
from src.modeling.dataset_loader import DatasetLoader

loader = DatasetLoader('datasets/03_processed/master_data.json')
train_dl, val_dl, test_dl = loader.get_dataloaders(batch_size=32)

for texts, labels in train_dl:
    # Training step
    outputs = model(texts)
    loss = criterion(outputs, labels)
```

---

## 🧠 **phobert_multitask_trainer.py** - PhoBERT Training

**Chức Năng:** Train PhoBERT cho Multitask Learning

```python
class PhoBERTMultitask(nn.Module):
    """PhoBERT with 2 task heads"""
    
    def __init__(self, model_name='vinai/phobert-base', num_classes=3):
        super().__init__()
        
        # Shared encoder
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Task 1: Classification
        self.clf_head = nn.Linear(768, num_classes)
        
        # Task 2: CoT Generation (decoder)
        self.cot_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass for both tasks"""
        # Encode
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0][:, 0]  # [CLS] token
        
        # Drop
        sequence_output = self.dropout(sequence_output)
        
        # Task 1: Classification
        clf_logits = self.clf_head(sequence_output)
        
        # Task 2: CoT generation
        cot_logits = self.cot_head(sequence_output)
        
        return {
            'clf_logits': clf_logits,
            'cot_logits': cot_logits
        }


class PhoBERTMultitaskTrainer:
    """Train PhoBERT multitask model"""
    
    def __init__(self, model_name='vinai/phobert-base', 
                 num_classes=3, device='cuda'):
        self.device = device
        self.model = PhoBERTMultitask(model_name, num_classes).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(self, train_loader, val_loader, epochs=3, 
              learning_rate=5e-5, clf_weight=0.7, cot_weight=0.3):
        """
        Train model with combined loss
        
        Loss = clf_weight * classification_loss + cot_weight * cot_loss
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        clf_criterion = nn.CrossEntropyLoss()
        cot_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Training loop
            total_loss = 0
            for batch in train_loader:
                texts, clf_labels, cot_labels = batch
                
                # Forward
                outputs = self.model(texts)
                
                # Loss
                clf_loss = clf_criterion(outputs['clf_logits'], clf_labels)
                cot_loss = cot_criterion(outputs['cot_logits'], cot_labels)
                loss = clf_weight * clf_loss + cot_weight * cot_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            val_loss = self.evaluate(val_loader, clf_criterion, cot_criterion)
            
            print(f"Epoch {epoch+1}/{epochs} - Train: {total_loss:.4f}, Val: {val_loss:.4f}")
    
    def evaluate(self, test_loader, clf_criterion, cot_criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                texts, labels = batch
                outputs = self.model(texts)
                
                # Loss
                loss = clf_criterion(outputs['clf_logits'], labels)
                total_loss += loss.item()
                
                # Predictions
                preds = outputs['clf_logits'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return total_loss / len(test_loader)
    
    def save(self, output_dir):
        """Save model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
```

**Training Script:**
```python
from src.modeling.phobert_multitask_trainer import PhoBERTMultitaskTrainer
from src.modeling.dataset_loader import DatasetLoader

# Load data
loader = DatasetLoader('datasets/03_processed/master_data.json')
train_dl, val_dl, test_dl = loader.get_dataloaders()

# Train
trainer = PhoBERTMultitaskTrainer()
trainer.train(train_dl, val_dl, epochs=3, learning_rate=5e-5)

# Evaluate
trainer.evaluate(test_dl, ...)

# Save
trainer.save('experiments/models/phobert_multitask_v1/')
```

---

## 🚀 **llm_inference_engine.py** - Gemma-4 Inference

**Chức Năng:** Inference server cho Gemma-4 4B QLoRA

```python
class LLMInferenceEngine:
    """Gemma-4 inference with CoT generation"""
    
    def __init__(self, model_path, quantization=True, device='cuda'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to Gemma model + LoRA adapter
            quantization: Use 4-bit quantization
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.model = self._load_model(model_path, quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def _load_model(self, model_path, quantization):
        """Load Gemma-4 with LoRA adapter"""
        if quantization:
            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Gemma-4-4b',
                quantization_config=bnb_config,
                device_map='auto'
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Gemma-4-4b'
            )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        return model
    
    def predict(self, text, max_length=256, temperature=0.7, 
                top_p=0.9):
        """
        Generate CoT explanation
        
        Args:
            text: Input text
            max_length: Max output length
            temperature: Sampling temperature
            top_p: Nucleus sampling p
            
        Returns:
            Generated CoT explanation
        """
        # Prompt engineering
        prompt = f"""Phân tích bài viết về vaccine sau:

Bài viết: {text}

Giải thích chi tiết tại sao bài viết này là thông tin sai/an toàn:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Giải thích chi tiết")[1].strip()
    
    def batch_predict(self, texts, batch_size=8):
        """Batch inference"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.predict(text)
                results.append(result)
        return results
```

**Flask Server:**
```python
from flask import Flask, request, jsonify
from src.modeling.llm_inference_engine import LLMInferenceEngine

app = Flask(__name__)
engine = LLMInferenceEngine('experiments/models/gemma4_qlora_v1')

@app.route('/inference', methods=['POST'])
def inference():
    """
    Inference endpoint
    
    Request:
    {
        "text": "Nội dung bài viết",
        "max_length": 256,
        "temperature": 0.7
    }
    
    Response:
    {
        "text": "Nội dung bài viết",
        "cot_explanation": "Lý do tại sao..."
    }
    """
    data = request.json
    text = data.get('text')
    max_length = data.get('max_length', 256)
    
    try:
        cot = engine.predict(text, max_length=max_length)
        return jsonify({
            'text': text,
            'cot_explanation': cot,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/batch_inference', methods=['POST'])
def batch_inference():
    """Batch inference endpoint"""
    data = request.json
    texts = data.get('texts', [])
    
    results = engine.batch_predict(texts)
    return jsonify({
        'results': results,
        'count': len(results)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 🔮 **inference.py** - Unified API

**Chức Năng:** High-level API combining PhoBERT + Gemma

```python
class VaccineNLPInference:
    """Complete inference pipeline"""
    
    def __init__(self, phobert_path, gemma_path):
        """Load both models"""
        self.phobert = PhoBERTClassifier(phobert_path)
        self.gemma = LLMInferenceEngine(gemma_path)
    
    def predict(self, text):
        """
        Full inference: classification + explanation
        
        Returns:
        {
            'text': input text,
            'label': 'misinformation' | 'safe' | 'misleading',
            'confidence': 0.92,
            'cot_explanation': 'Đây là thông tin sai vì...'
        }
        """
        # Step 1: PhoBERT classification
        label_logits = self.phobert.classify(text)
        label = label_logits.argmax().item()
        confidence = label_logits.softmax(dim=0).max().item()
        
        # Step 2: Gemma CoT generation
        cot = self.gemma.predict(text)
        
        return {
            'text': text,
            'label': ['safe', 'misinformation', 'misleading'][label],
            'confidence': float(confidence),
            'cot_explanation': cot
        }
```

---

## 📊 Training Workflow

```
1. Load data
   ↓
2. Initialize model
   ↓
3. Setup optimizer
   ↓
4. Train loop (epochs)
   ├─ Forward pass
   ├─ Compute loss
   ├─ Backward pass
   ├─ Optimizer step
   └─ Validate
   ↓
5. Evaluate on test set
   ↓
6. Save model
   ↓
7. Log metrics to MLflow
```

---

## ✅ Best Practices

1. **Use DataLoader for efficiency**
2. **Track metrics with MLflow**
3. **Save best model only** (not every epoch)
4. **Use mixed precision** (for faster training)
5. **Validate regularly** (to catch overfitting)

---

**📅 Updated:** May 21, 2026 | Phiên bản 1.2
**🤖 Models:** PhoBERT 110M, Gemma-4 4B  
**🔧 Frameworks:** PyTorch, Transformers, PEFT, Bitsandbytes
