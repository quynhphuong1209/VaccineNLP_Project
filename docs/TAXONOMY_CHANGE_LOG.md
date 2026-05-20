# TAXONOMY CHANGE LOG
**Dự án:** VaccineNLP-Thesis  
**Date:** 2026-05-13  
**Version:** v2 → v3  
**Author:** Antigravity Sub-Agent

---

## Lý do thay đổi

Sau kiểm toán dữ liệu (Data Audit), phát hiện 2 class không sử dụng trong corpus:

| Class | Trục | Lý do loại bỏ |
|---|---|---|
| `0` "Không liên quan" (Misinfo) | Misinfo | **0 mẫu** trong train và test. Pipeline keyword-filter đã loại bỏ hoàn toàn nội dung không liên quan từ bước thu thập dữ liệu. |
| `3` "Không rõ" (Stance) | Stance | **0 mẫu** trong train và test. Gemma 31B labeler không thực sự sinh ra nhãn này trong toàn bộ corpus 1663+ records. |

---

## Chi tiết thay đổi

### Trục Misinfo: 3 classes → 2 classes
| Nhãn cũ | Index cũ | Index mới | Nhãn mới |
|---|---|---|---|
| Không liên quan | 0 | — | **Loại bỏ** |
| Tin giả / Sai lệch | 1 | **0** | Tin giả |
| Chính xác | 2 | **1** | Chính xác |

### Trục Stance: 4 classes → 3 classes
| Nhãn | Index cũ | Index mới | Ghi chú |
|---|---|---|---|
| Ủng hộ | 0 | **0** | Không đổi |
| Phản đối | 1 | **1** | Không đổi |
| Trung lập | 2 | **2** | Không đổi |
| Không rõ | 3 | — | **Loại bỏ** |

### Trục Sentiment: Không thay đổi
- 3 classes: 0=Tiêu cực, 1=Trung tính, 2=Tích cực

---

## N_CLASSES mới (Taxonomy v3)
```python
N_CLASSES = {'misinfo': 2, 'stance': 3, 'sentiment': 3}
```

---

## Files dữ liệu bị ảnh hưởng (tạo mới, KHÔNG ghi đè gốc)

| File gốc (v2) | File mới (v3) | Records | Thay đổi |
|---|---|---|---|
| `datasets/05_model_ready/train_v2_seg_deduped.jsonl` | `datasets/05_model_ready/train_v2_seg_v3.jsonl` | 1663 | Remap misinfo: {1→0, 2→1} |
| `datasets/05_model_ready/test_v2_seg.jsonl` | `datasets/05_model_ready/test_v2_seg_v3.jsonl` | 186 | Remap misinfo: {1→0, 2→1} |
| `datasets/04_silver_labels/train_set_final_v2.jsonl` | `datasets/04_silver_labels/train_set_final_v3.jsonl` | 1663 | Remap misinfo: {1→0, 2→1} |
| `datasets/03_processed/benchmark_test_set.jsonl` | `datasets/03_processed/benchmark_test_set_v3.jsonl` | 186 | Remap misinfo: {1→0, 2→1} |

---

## Files notebooks đã cập nhật

| Notebook | Thay đổi |
|---|---|
| `notebook_test/vaccinenlp-phobert-v2-multitask-classifier.ipynb` | N_CLASSES, LABEL_NAMES, TRAIN_PATH→_v3, TEST_PATH→_v3, f1_score+labels= |
| `notebook_test/vaccinenlp-xlm-r-v1-multitask-classifi.ipynb` | N_CLASSES, LABEL_NAMES, TRAIN_PATH→_v3, TEST_PATH→_v3, f1_score+labels= |
| `notebooks/03_vaccinenlp-gemma-4-qlora-multitask.ipynb` | MISINFO_MAP, HC_MISINFO, REV maps, format_prompt, TRAIN_PATH→_v3, TEST_PATH→_v3 |

---

## Label Distribution (v3)

### Train set (1663 records)
| Task | 0 | 1 | 2 |
|---|---|---|---|
| Misinfo (Tin giả / Chính xác) | 304 | 1359 | — |
| Stance (Ủng hộ / Phản đối / Trung lập) | 475 | 500 | 688 |
| Sentiment (Tiêu cực / Trung tính / Tích cực) | 697 | 636 | 330 |

### Test set (186 records)
| Task | 0 | 1 | 2 |
|---|---|---|---|
| Misinfo | 28 | 158 | — |
| Stance | 54 | 48 | 84 |
| Sentiment | 71 | 75 | 40 |

---

## Verification: Tất cả PASS ✅

- **Phase 1 (Data Remap):** 4/4 files remap PASS, 16/16 assertions PASS
- **Phase 2 (PhoBERT):** N_CLASSES, paths, f1_score PASS
- **Phase 3 (XLM-R):** N_CLASSES, paths, f1_score PASS
- **Phase 4 (Gemma):** Maps consistency PASS, REV maps PASS
- **Phase 5 (Final):** 4/4 files PASS
