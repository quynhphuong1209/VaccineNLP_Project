import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
import json
from pathlib import Path
from src.common.paths import DATA_SILVER_DIR, DATA_GOLD_DIR, ensure_src_in_sys_path
ensure_src_in_sys_path()

class VaccineDataset(Dataset):
    """
    Dataset class for VaccineNLP Multitask Learning.
    Handles mapping of text and metadata into PhoBERT tokens.
    """
    def __init__(self, texts, misinfo_labels, stance_labels, sentiment_labels, tokenizer, max_length=256):
        self.texts = [self._preprocess(t) for t in texts]
        self.labels = {
            'misinfo': torch.tensor(misinfo_labels, dtype=torch.long),
            'stance': torch.tensor(stance_labels, dtype=torch.long),
            'sentiment': torch.tensor(sentiment_labels, dtype=torch.long)
        }
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _preprocess(self, text):
        """
        Thực hiện word-segmentation (underthesea) để chuẩn hóa cho PhoBERT.
        Ví dụ: 'tiêm chủng' -> 'tiêm_chủng'
        """
        if not isinstance(text, str):
            return ""
        # PhoBERT yêu cầu dấu gạch dưới thay vì dấu cách để biểu thị từ ghép
        return word_tokenize(text, format="text")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'misinfo': self.labels['misinfo'][idx],
            'stance': self.labels['stance'][idx],
            'sentiment': self.labels['sentiment'][idx]
        }

def create_dataloaders(csv_path=None, batch_size=16, model_name='vinai/phobert-base-v2', max_length=256, is_warming=False):
    """
    Thực hiện Stratified Split (70/15/15) và trả về DataLoaders.
    is_warming: Nếu True, giữ nguyên nhãn -100 cho các task thiếu.
    """
    # Default to Silver Data if no path provided
    if csv_path is None:
        csv_path = DATA_SILVER_DIR / "annotated_v6.jsonl"
    
    # Load data based on extension
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dữ liệu không tồn tại tại: {csv_path}")

    if path_obj.suffix == '.jsonl':
        df = pd.read_json(csv_path, lines=True)
    else:
        df = pd.read_csv(csv_path)
    
    # Fill missing values ONLY if it's not a warming dataset
    # Warming datasets already have -100 for missing labels
    if not is_warming:
        df = df.fillna({
            'label_misinfo': 0, 
            'label_stance': 1, # Default Neutral
            'label_sentiment': 1  # Default Neutral
        })
    else:
        # Trong data ngoại vi, nhãn rỗng thường là -100
        df = df.fillna(-100)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Stratified Split (based on misinfo class)
    # 70% Train, 30% Temp
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['label_misinfo'], random_state=42
    )
    
    # 50% Temp = 15% Total for Dev and Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label_misinfo'], random_state=42
    )

    print(f"Dataset Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create Dataset objects
    train_ds = VaccineDataset(
        train_df['text'].values, 
        train_df['label_misinfo'].values,
        train_df['label_stance'].values,
        train_df['label_sentiment'].values,
        tokenizer, max_length
    )
    
    val_ds = VaccineDataset(
        val_df['text'].values, 
        val_df['label_misinfo'].values,
        val_df['label_stance'].values,
        val_df['label_sentiment'].values,
        tokenizer, max_length
    )
    
    test_ds = VaccineDataset(
        test_df['text'].values, 
        test_df['label_misinfo'].values,
        test_df['label_stance'].values,
        test_df['label_sentiment'].values,
        tokenizer, max_length
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer
