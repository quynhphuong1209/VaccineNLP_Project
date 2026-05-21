import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
import mlflow
import os
from pathlib import Path
from src.common.paths import MODELS_DIR, EXPERIMENTS_DIR, ensure_src_in_sys_path
ensure_src_in_sys_path()

class VaccineMultitaskModel(nn.Module):
    """
    Multitask model with shared PhoBERT encoder and task-specific heads.
    """
    def __init__(self, model_name='vinai/phobert-base-v2', num_misinfo=3, num_stance=3, num_sentiment=3):
        super(VaccineMultitaskModel, self).__init__()
        # Note: Using class-based numbers from taxonomy.json
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        self.head_misinfo = nn.Linear(hidden_size, num_misinfo)
        self.head_stance = nn.Linear(hidden_size, num_stance)
        self.head_sentiment = nn.Linear(hidden_size, num_sentiment)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        return (
            self.head_misinfo(pooled_output),
            self.head_stance(pooled_output),
            self.head_sentiment(pooled_output)
        )

class MultitaskTrainer:
    """
    Trainer with MLflow instrumentation and Weighted Loss support.
    """
    def __init__(self, model, device, weights_dict=None, experiment_name="VaccineNLP_FineTune"):
        self.model = model.to(device)
        self.device = device
        self._setup_mlflow(experiment_name)
        
        # CrossEntropyLoss with weights for imbalanced classes
        self.criterion_misinfo = nn.CrossEntropyLoss(
            weight=weights_dict.get('misinfo').to(device) if weights_dict else None,
            ignore_index=-100
        )
        self.criterion_stance = nn.CrossEntropyLoss(
            weight=weights_dict.get('stance').to(device) if weights_dict else None,
            ignore_index=-100
        )
        self.criterion_sentiment = nn.CrossEntropyLoss(
            weight=weights_dict.get('sentiment').to(device) if weights_dict else None,
            ignore_index=-100
        )

    def _setup_mlflow(self, experiment_name):
        """Configure MLflow tracking in the EXPERIMENTS_DIR."""
        mlflow.set_tracking_uri(EXPERIMENTS_DIR.as_uri())
        mlflow.set_experiment(experiment_name)

    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch and log results to MLflow."""
        self.model.train()
        total_loss = 0
        
        with mlflow.start_run(run_name=f"Epoch_{epoch}", nested=True):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                optimizer.zero_grad()
                m_logits, st_logits, se_logits = self.model(input_ids, attention_mask)
                
                # Loss calculation
                l_m = self.criterion_misinfo(m_logits, batch['misinfo'].to(self.device))
                l_st = self.criterion_stance(st_logits, batch['stance'].to(self.device))
                l_se = self.criterion_sentiment(se_logits, batch['sentiment'].to(self.device))
                
                batch_loss = torch.nan_to_num(l_m) + torch.nan_to_num(l_st) + torch.nan_to_num(l_se)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                
            avg_loss = total_loss / len(dataloader)
            
            # Post-epoch logging (Architect Directive: Log per Epoch only)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            return avg_loss

    def validate(self, dataloader, epoch):
        """Run validation and log metrics."""
        self.model.eval()
        # Mock evaluation logic for now
        val_loss = 0.5 # Placeholder
        f1_macro = 0.7 # Placeholder
        
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("macro_f1", f1_macro, step=epoch)
        return val_loss, f1_macro

    def save_checkpoint(self, experiment_name, epoch):
        """Save model checkpoint to MODELS_DIR."""
        save_path = MODELS_DIR / experiment_name / f"checkpoint_epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save state dict
        torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        # Save config
        self.model.config.save_pretrained(save_path)
        print(f"💾 Model checkpoint saved to: {save_path}")
        return save_path

def compute_class_weights(labels_list):
    """Compute weights for Class Imbalance (Master Architecture: Weighted Loss)."""
    unique, counts = np.unique(labels_list, return_counts=True)
    total = len(labels_list)
    weights = total / (len(unique) * counts)
    return torch.tensor(weights, dtype=torch.float)
