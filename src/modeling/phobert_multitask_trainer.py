import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import numpy as np
from src.common import paths

class VaccineMultitaskModel(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", n_misinfo=3, n_stance=4, n_sentiment=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.head_misinfo = nn.Linear(hidden, n_misinfo)
        self.head_stance = nn.Linear(hidden, n_stance)
        self.head_sentiment = nn.Linear(hidden, n_sentiment)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # PhoBERT uses pooler_output for the [CLS] representation
        pooled = self.dropout(out.pooler_output)

        return {
            "misinfo": self.head_misinfo(pooled),
            "stance": self.head_stance(pooled),
            "sentiment": self.head_sentiment(pooled)
        }

class PhoBertMultitaskTrainer:
    """Trainer module for PhoBERT Multitask Learning (Misinfo, Stance, Sentiment)"""
    
    def __init__(self, model_checkpoint="vinai/phobert-base", lr=2e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint = model_checkpoint
        self.model = VaccineMultitaskModel(model_name=model_checkpoint).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.save_dir = paths.MODEL_DIR / "phobert-multitask-v2"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 Model initialized from: {model_checkpoint}")
        print(f"🚀 Running on: {self.device}")

    def compute_loss(self, logits, batch, losses_fn):
        l1 = losses_fn['misinfo'](logits['misinfo'], batch['misinfo'].to(self.device))
        l2 = losses_fn['stance'](logits['stance'], batch['stance'].to(self.device))
        l3 = losses_fn['sentiment'](logits['sentiment'], batch['sentiment'].to(self.device))
        # Weighted sum of losses
        return 0.5 * l1 + 0.3 * l2 + 0.2 * l3

    def train_epoch(self, loader, losses_fn):
        self.model.train()
        epoch_losses = []
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)

            logits = self.model(ids, mask)
            total_loss = self.compute_loss(logits, batch, losses_fn)
            
            total_loss.backward()
            self.optimizer.step()
            epoch_losses.append(total_loss.item())
        return np.mean(epoch_losses)

    def train(self, train_loader, val_loader, losses_fn, epochs=5):
        print(f"🔥 Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_loader, losses_fn)
            print(f"📅 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            self.evaluate(val_loader, prefix=f"Validation Epoch {epoch+1}")
            
            # Save checkpoint
            torch.save(self.model.state_dict(), self.save_dir / "pytorch_model.bin")
            
        print(f"💾 Training complete. Model saved at: {self.save_dir}")

    def evaluate(self, loader, prefix="Evaluation"):
        self.model.eval()
        all_preds = {'m': [], 'st': [], 'se': []}
        all_labels = {'m': [], 'st': [], 'se': []}

        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                logits = self.model(ids, mask)

                all_preds['m'].extend(logits['misinfo'].argmax(dim=1).cpu().tolist())
                all_preds['st'].extend(logits['stance'].argmax(dim=1).cpu().tolist())
                all_preds['se'].extend(logits['sentiment'].argmax(dim=1).cpu().tolist())

                all_labels['m'].extend(batch['misinfo'].tolist())
                all_labels['st'].extend(batch['stance'].tolist())
                all_labels['se'].extend(batch['sentiment'].tolist())

        print(f"\n--- {prefix} Results ---")
        print(f"📊 Misinformation:\n{classification_report(all_labels['m'], all_preds['m'], digits=4)}")
        print(f"📊 Stance:\n{classification_report(all_labels['st'], all_preds['st'], digits=4)}")
        print(f"📊 Sentiment:\n{classification_report(all_labels['se'], all_preds['se'], digits=4)}")
        
        return {
            "misinfo_report": classification_report(all_labels['m'], all_preds['m'], output_dict=True),
            "stance_report": classification_report(all_labels['st'], all_preds['st'], output_dict=True),
            "sentiment_report": classification_report(all_labels['se'], all_preds['se'], output_dict=True),
        }
