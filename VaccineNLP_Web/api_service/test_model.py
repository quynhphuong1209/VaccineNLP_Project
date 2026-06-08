import torch
import os
import json
from transformers import AutoTokenizer
from app.model_classes import PhoBERTMultitaskClassifier
from underthesea import word_tokenize

MODEL_PATH = "../models/phobert_multitask.pt"

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
model = PhoBERTMultitaskClassifier(num_misinfo=2, num_stance=3, num_sentiment=3)

state = torch.load(MODEL_PATH, map_location="cpu")
new_state = {}
for k, v in state.items():
    k_new = k.replace("heads.misinfo.", "head_misinfo.")\
             .replace("heads.stance.", "head_stance.")\
             .replace("heads.sentiment.", "head_sentiment.")
    new_state[k_new] = v
model.load_state_dict(new_state)
model.eval()

texts = [
    "Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.",
    "Canh bao vaccine vo sinh"
]

results = []
for text in texts:
    segmented_text = word_tokenize(text, format="text")
    enc = tokenizer(segmented_text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    
    probs_misinfo = torch.softmax(out[0], dim=-1)[0]
    probs_stance = torch.softmax(out[1], dim=-1)[0]
    probs_sentiment = torch.softmax(out[2], dim=-1)[0]
    
    results.append({
        "text": text,
        "segmented_text": segmented_text,
        "misinfo_probs": probs_misinfo.tolist(),
        "stance_probs": probs_stance.tolist(),
        "sentiment_probs": probs_sentiment.tolist()
    })

with open("test_output_tokenized.txt", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
