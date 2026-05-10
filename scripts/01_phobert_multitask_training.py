# -*- coding: utf-8 -*-
"""
💉 01. PhoBERT Multitask Training - VaccineNLP
Modularized version for local and cloud execution.
"""

import os
import sys

# Thêm thư mục gốc của dự án vào sys.path để nhận diện module 'src'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# Centralized Path Management
from src.common import paths
from src.modeling.phobert_multitask_trainer import PhoBertMultitaskTrainer
from src.modeling.dataset_loader import VaccineDataset, DataLoader

# ==============================
# ⚙️ CONFIGURATION
# ==============================
MODEL_CHECKPOINT = "vinai/phobert-base"
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-5

# Paths from central path manager
TRAIN_DATA_PATH = paths.SILVER_LABELS_DIR / "train_set_final.jsonl"
BENCHMARK_DATA_PATH = paths.GOLD_DATA_DIR / "benchmark_test_set.jsonl"
MODEL_SAVE_DIR = paths.MODEL_DIR / "phobert-multitask-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_weights(dataset, task_key, n_classes):
    labels = [item[task_key] for item in dataset.data]
    present_classes = np.unique(labels)
    weights_val = compute_class_weight('balanced', classes=present_classes, y=labels)
    
    full_weights = torch.ones(n_classes).to(DEVICE)
    for idx, cls_id in enumerate(present_classes):
        if cls_id < n_classes:
            full_weights[int(cls_id)] = weights_val[idx]
    return nn.CrossEntropyLoss(weight=full_weights.to(torch.float32))

def main():
    # 1. Ensure directories exist
    paths.ensure_dirs()
    
    # 2. Initialize Tokenizer
    print(f"📡 Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 3. Load Datasets
    print(f"📂 Loading training data: {TRAIN_DATA_PATH}")
    full_train_ds = VaccineDataset(str(TRAIN_DATA_PATH), tokenizer)
    
    print(f"📂 Loading benchmark data: {BENCHMARK_DATA_PATH}")
    benchmark_ds = VaccineDataset(str(BENCHMARK_DATA_PATH), tokenizer)
    
    # 4. Train/Val Split (10% for validation)
    train_indices, val_indices = train_test_split(
        range(len(full_train_ds)), test_size=0.1, random_state=42
    )
    
    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)
    
    # 5. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    benchmark_loader = DataLoader(benchmark_ds, batch_size=BATCH_SIZE)
    
    print(f"📊 Samples: Train={len(train_ds)}, Val={len(val_ds)}, Benchmark={len(benchmark_ds)}")

    # 6. Setup Loss Functions with Class Weights
    # We use the full dataset to compute weights for stability
    losses_fn = {
        'misinfo': get_class_weights(full_train_ds, 'misinfo', 3),
        'stance': get_class_weights(full_train_ds, 'stance', 4),
        'sentiment': get_class_weights(full_train_ds, 'sentiment', 3)
    }
    
    # 7. Initialize Trainer
    trainer = PhoBertMultitaskTrainer(model_checkpoint=MODEL_CHECKPOINT, lr=LR)
    
    # 8. Run Training
    trainer.train(train_loader, val_loader, losses_fn, epochs=EPOCHS)
    
    # 9. Final Evaluation on Benchmark
    print("\n🏁 FINAL EVALUATION ON GOLD BENCHMARK SET")
    trainer.evaluate(benchmark_loader, prefix="Gold Benchmark")

if __name__ == "__main__":
    main()