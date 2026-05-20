# VaccineNLP — Benchmark Report
**Date:** 2026-05-20  
**Gold Test Set:** 186 samples (Human-validated)  
**Taxonomy v3:** Misinfo(2 cls) | Stance(3 cls) | Sentiment(3 cls)

## Macro F1 Summary

| Mô hình | Misinfo | Stance | Sentiment | Trung bình |
|---------|---------|--------|-----------|------------|
| XLM-R-v1 | 0.5823 | 0.4217 | 0.1842 | 0.3961 |
| PhoBERT-v2 | 0.7079 | 0.7107 | 0.7260 | 0.7149 |
| Gemma-4 4B (XAI) | 0.6925 | 0.5818 | 0.7196 | 0.6646 |

## Per-Class F1

### Misinfo

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tin giả | 0.4274 | 0.5085 | 0.5135 | 28 |
| Chính xác | 0.7373 | 0.9073 | 0.8714 | 158 |

### Stance

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Ủng hộ | 0.0000 | 0.6476 | 0.4068 | 54 |
| Phản đối | 0.5950 | 0.6869 | 0.6458 | 48 |
| Trung lập | 0.6701 | 0.7976 | 0.6929 | 84 |

### Sentiment

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tiêu cực | 0.5525 | 0.7808 | 0.7609 | 71 |
| Trung tính | 0.0000 | 0.8026 | 0.7934 | 75 |
| Tích cực | 0.0000 | 0.5946 | 0.6047 | 40 |
