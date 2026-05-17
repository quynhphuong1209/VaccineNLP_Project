import json
import csv
import re
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
TEST_DATA_PATH = ROOT_DIR / "datasets" / "03_processed" / "benchmark_test_set.jsonl"
INFERENCE_PATH = ROOT_DIR / "experiments" / "results" / "gemma_inference_progress_v13.jsonl"
OUTPUT_CSV_PATH = ROOT_DIR / "experiments" / "results" / "error_analysis_gemma.csv"

# Label Mappings for Stance and Sentiment
STANCE_MAP = {0: "Support", 1: "Oppose", 2: "Neutral", 3: "Unknown"}
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Lưu ý: Gemma-4 v13 chỉ sinh ra nhãn phân loại, không có bước lý luận riêng.
# gemma_reasoning = raw_response (toàn bộ output của model, bao gồm nhãn).

# 1. Load benchmark dataset để lấy text gốc
test_data = {}
if TEST_DATA_PATH.exists():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                item_id = row.get('id')
                text = row.get('text_cleaned', row.get('text', ''))
                if item_id:
                    test_data[item_id] = {'text': text}
            except:
                continue

# 2. Process inference results and filter False Negatives
error_cases = []

if INFERENCE_PATH.exists():
    with open(INFERENCE_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                row = json.loads(line)
                item_id = row.get('id', idx)
                
                # Match ID
                if item_id not in test_data and isinstance(item_id, int):
                    keys = list(test_data.keys())
                    if idx < len(keys):
                        item_id = keys[idx]
                
                # gemma_reasoning = toàn bộ raw_response (Gemma v13 không sinh reasoning riêng)
                raw_response = row.get('raw_response', '')
                
                true_st = row.get('true_st')
                pred_st = row.get('pred_st')
                true_se = row.get('true_se')
                pred_se = row.get('pred_se')
                
                # False Negatives Condition
                is_fn_stance = (true_st == 1 and pred_st == 2)
                is_fn_sentiment = (true_se == 0 and pred_se == 1)
                
                if is_fn_stance or is_fn_sentiment:
                    text = test_data.get(item_id, {}).get('text', '')
                    
                    error_cases.append({
                        'id': item_id,
                        'text': text,
                        'true_stance': STANCE_MAP.get(true_st, true_st),
                        'pred_stance': STANCE_MAP.get(pred_st, pred_st),
                        'true_sentiment': SENTIMENT_MAP.get(true_se, true_se),
                        'pred_sentiment': SENTIMENT_MAP.get(pred_se, pred_se),
                        'gemma_reasoning': raw_response
                    })
            except:
                continue

# 3. Export to CSV (Xóa file cũ, tạo file mới)
if OUTPUT_CSV_PATH.exists():
    OUTPUT_CSV_PATH.unlink()

with open(OUTPUT_CSV_PATH, 'w', encoding='utf-8-sig', newline='') as csvfile:
    fieldnames = ['id', 'text', 'true_stance', 'pred_stance', 'true_sentiment', 'pred_sentiment', 'gemma_reasoning']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in error_cases:
        writer.writerow(row)

print(f"Error analysis done. Found {len(error_cases)} false negative cases.")
print(f"gemma_reasoning = raw_response (toàn bộ output Gemma-4 v13).")
print(f"Results saved to {OUTPUT_CSV_PATH}")
