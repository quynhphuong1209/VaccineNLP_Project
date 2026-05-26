# VaccineNLP — Benchmark Report
**Date:** 2026-05-21  
**Gold Test Set:** 186 samples (Human-validated)  
**Taxonomy v3:** Misinfo(2 cls) | Stance(3 cls) | Sentiment(3 cls)

## Macro F1 Summary

| Mô hình | Misinfo | Stance | Sentiment | Trung bình |
|---------|---------|--------|-----------|------------|
| XLM-R-v1 | 0.7038 | 0.6224 | 0.6866 | 0.6709 |
| PhoBERT-v2 | 0.6996 | 0.6640 | 0.7266 | 0.6967 |
| Gemma-4 4B (XAI) | 0.6377 | 0.6264 | 0.7700 | 0.6780 |

## Per-Class F1

### Misinfo

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tin giả | 0.5079 | 0.5075 | 0.4444 | 28 |
| Chính xác | 0.8997 | 0.8918 | 0.8309 | 158 |

### Stance

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Ủng hộ | 0.5495 | 0.5934 | 0.4528 | 54 |
| Phản đối | 0.6387 | 0.6612 | 0.6905 | 48 |
| Trung lập | 0.6790 | 0.7375 | 0.7360 | 84 |

### Sentiment

| Nhãn | XLM-R-v1 | PhoBERT-v2 | Gemma-4 4B (XAI) | Support |
|------| ------- | ------- | ------- | ------- |
| Tiêu cực | 0.7682 | 0.8000 | 0.8039 | 71 |
| Trung tính | 0.7162 | 0.7917 | 0.8034 | 75 |
| Tích cực | 0.5753 | 0.5882 | 0.7027 | 40 |
