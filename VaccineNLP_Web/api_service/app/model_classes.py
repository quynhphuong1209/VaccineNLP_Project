import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class VaccineMultitaskModel(nn.Module):
    """
    Multitask model with shared PhoBERT encoder and task-specific heads.
    """
    def __init__(self, model_name='vinai/phobert-base-v2', num_misinfo=2, num_stance=3, num_sentiment=3):
        super(VaccineMultitaskModel, self).__init__()
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

# Alias for user-configured class name
PhoBERTMultitaskClassifier = VaccineMultitaskModel
