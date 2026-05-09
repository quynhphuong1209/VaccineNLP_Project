import json
import torch
from torch.utils.data import Dataset, DataLoader

class VaccineDataset(Dataset):
    """Dataset class cho việc huấn luyện mô hình đa nhiệm"""
    
    def __init__(self, data_path, tokenizer, max_len=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        content = self.data[item]
        text = str(content['text_processed'])
        
        # Encoding văn bản
        encoding = self.tokenizer.encode_plus(
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
            'misinfo_label': torch.tensor(content.get('misinfo_id', 0), dtype=torch.long),
            'stance_label': torch.tensor(content.get('stance_id', 2), dtype=torch.long),
            'sentiment_label': torch.tensor(content.get('sentiment_id', 2), dtype=torch.long)
        }

def create_data_loader(data_path, tokenizer, batch_size=16):
    ds = VaccineDataset(data_path, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
