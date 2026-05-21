# VaccineNLP — Benchmark Report
**Date:** 2026-05-17  
**Gold Test Set:** 186 samples (Human-validated)  
**Taxonomy v3:** Misinfo(2 cls) | Stance(3 cls) | Sentiment(3 cls)

## Macro F1 Summary

| Mô hình | Misinfo | Stance | Sentiment | Trung bình |
|---------|---------|--------|-----------|------------|
| XLM-R-v1 | 0.6632 | 0.5618 | 0.6394 | 0.6215 |
| PhoBERT-v2 | 0.6886 | 0.6383 | 0.7289 | 0.6853 |
| Gemma-4 4B (XAI) | 0.3588 | 0.2862 | 0.2883 | 0.3111 |

## Per-Class F1

### Misinfo

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tin giả | 0.4478 | 0.4933 | 0.3429 | 28 |
| Chính xác | 0.8785 | 0.8839 | 0.3747 | 158 |

### Stance

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Ủng hộ | 0.4706 | 0.5817 | 0.5660 | 54 |
| Phản đối | 0.5660 | 0.6316 | 0.3721 | 48 |
| Trung lập | 0.6489 | 0.6923 | 0.5106 | 84 |

### Sentiment

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tiêu cực | 0.6993 | 0.7606 | 0.5623 | 71 |
| Trung tính | 0.6618 | 0.7671 | 0.1579 | 75 |
| Tích cực | 0.5571 | 0.6600 | 0.0247 | 40 |
