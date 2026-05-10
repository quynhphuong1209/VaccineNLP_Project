import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pyvi import ViTokenizer

class VaccineDataset(Dataset):
    """Dataset class for VaccineNLP Multitask Training (supports JSON and JSONL)"""
    
    def __init__(self, data_path, tokenizer, max_len=128, use_pyvi=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_pyvi = use_pyvi
        self.data = self._load_data(data_path)

    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            else:
                data = json.load(f)
        
        # Filter only annotated data if needed or prepare fields
        processed_data = []
        for item in data:
            # Get text (prefer cleaned or processed)
            text = item.get('text_cleaned', item.get('text_processed', item.get('text', '')))
            
            # Get labels (standardized IDs)
            # Default to neutral/unknown labels if missing
            ids = item.get('standardized_ids', [0, 3, 1]) 
            
            processed_data.append({
                "text": text,
                "misinfo": ids[0] if len(ids) > 0 else 0,
                "stance": ids[1] if len(ids) > 1 else 3,
                "sentiment": ids[2] if len(ids) > 2 else 1
            })
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        content = self.data[item]
        text = str(content['text'])
        
        # PhoBERT requires word segmentation
        if self.use_pyvi:
            text = ViTokenizer.tokenize(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'misinfo': torch.tensor(content['misinfo'], dtype=torch.long),
            'stance': torch.tensor(content['stance'], dtype=torch.long),
            'sentiment': torch.tensor(content['sentiment'], dtype=torch.long)
        }

def create_data_loader(data_path, tokenizer, batch_size=16, shuffle=True):
    ds = VaccineDataset(data_path, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
