# 🌳 Public Repository Directory Tree
**Last Updated:** May 20, 2026  
**Security level:** Public (Sanitized - only tracked files are included)

---

## 📂 Visual Tree Structure

```text
VaccineNLP-Thesis (Public Repo)/
│
├── 📄 .env.template
├── 📄 .gitignore
├── 📄 README.md
├── 📂 app/
│   ├── 📂 data_fetchers/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 apify_fetcher.py
│   │   ├── 📄 news_fetcher.py
│   │   ├── 📄 router.py
│   │   ├── 📄 text_cleaner.py
│   │   └── 📄 youtube_fetcher.py
│   ├── 📄 README.md
│   ├── 📄 README_demo.md
│   ├── 📄 requirements_demo.txt
│   ├── 📄 streamlit_demo.py
│   └── 📄 xai_cache.json
├── 📂 configs/
│   ├── 📄 README.md
│   ├── 📄 class_weights_v2.json
│   ├── 📄 facebook.json
│   ├── 📄 seeds.json
│   └── 📄 taxonomy.json
├── 📂 datasets/
│   ├── 📂 03_processed/
│   │   ├── 📄 benchmark_review_HITL.xlsx
│   │   ├── 📄 benchmark_test_set.jsonl
│   │   ├── 📄 benchmark_test_set_v3.jsonl
│   │   └── 📄 reclaimed_master_pool_vn_clean.json
│   ├── 📂 05_model_ready/
│   │   ├── 📄 test_v2_seg_v3.jsonl
│   │   ├── 📄 train_v2_seg_deduped.jsonl
│   │   └── 📄 train_v2_seg_v3.jsonl
│   ├── 📄 README.md
│   └── 📄 __init__.py
├── 📂 docs/
│   ├── 📄 01_PIPELINE_ARCHITECTURE.md
│   ├── 📄 02_DATASET_CARD.md
│   ├── 📄 03_METHODOLOGY.md
│   ├── 📄 04_FUTURE_WORKS_XAI.md
│   ├── 📄 DEPLOYMENT_GUIDE.md
│   ├── 📄 DOCUMENTATION_INDEX.md
│   ├── 📄 FINAL_TECHNICAL_REPORT.md
│   ├── 📄 FOLDER_STRUCTURE.md
│   ├── 📄 README.md
│   └── 📄 TAXONOMY_CHANGE_LOG.md
├── 📂 experiments/
│   └── 📂 results/
│       ├── 📄 benchmark_report.md
│       ├── 📂 figures/
│       │   ├── 📄 confusion_matrices.png
│       │   ├── 📄 gemma_confusion_matrix.png
│       │   ├── 📄 macro_f1_comparison.png
│       │   ├── 📄 per_class_f1.png
│       │   ├── 📄 training_curves.png
│       │   └── 📄 xlmr_training_curves.png
│       ├── 📄 gemma_inference_results_v3.jsonl
│       ├── 📄 gemma_v3_results.json
│       ├── 📄 phobert_v2_results.json
│       └── 📄 xlmr_v1_results.json
├── 📄 huph_logo.png
├── 📂 notebooks/
│   ├── 📄 01_vaccinenlp-phobert-v2-multitask.ipynb
│   ├── 📄 02_vaccinenlp-xlm-r-v1-multitask-classifier.ipynb
│   ├── 📄 03A_vaccinenlp-gemma-4-training.ipynb
│   ├── 📄 03B_vaccinenlp-gemma-4-inference.ipynb
│   ├── 📄 04_vaccinenlp-model-benchmark-report.ipynb
│   └── 📄 README.md
├── 📄 requirements.txt
├── 📂 scripts/
│   ├── 📄 vaccinenlp_gemma_4_qlora_multitask.py
│   ├── 📄 vaccinenlp_model_benchmark_report.py
│   ├── 📄 vaccinenlp_phobert_v2_multitask_classifier.py
│   └── 📄 vaccinenlp_xlm_r_v1_multitask_classifier.py
└── 📂 src/
    ├── 📄 README.md
    ├── 📄 __init__.py
    ├── 📂 common/
    │   ├── 📄 README.md
    │   ├── 📄 __init__.py
    │   ├── 📄 paths.py
    │   └── 📄 versioning_manager.py
    ├── 📄 data_collection
    ├── 📂 data_pipeline/
    │   ├── 📄 README.md
    │   ├── 📂 collection/
    │   │   ├── 📂 actor_configs/
    │   │   │   ├── 📄 config_facebook_pages.json
    │   │   │   ├── 📄 config_tiktok_accounts.json
    │   │   │   ├── 📄 config_web_sources.json
    │   │   │   ├── 📄 config_youtube_channels.json
    │   │   │   ├── 📄 rss_feeds.json
    │   │   │   ├── 📄 threads.json
    │   │   │   ├── 📄 tiktok.json
    │   │   │   └── 📄 youtube.json
    │   │   ├── 📄 apify_social_collector_v2.py
    │   │   ├── 📄 facebook_page_collector.py
    │   │   └── 📄 master_collector_v2.py
    │   └── 📂 preprocessing/
    │       ├── 📄 language_filter.py
    │       ├── 📄 pipeline.py
    │       ├── 📄 text_cleaner_v2.py
    │       └── 📄 vn_tokenizer.py
    ├── 📂 modeling/
    │   ├── 📄 README.md
    │   ├── 📄 __init__.py
    │   ├── 📄 dataset_loader.py
    │   ├── 📄 error_analysis.py
    │   ├── 📄 inference.py
    │   ├── 📄 llm_inference_engine.py
    │   └── 📄 phobert_multitask_trainer.py
    └── 📂 preprocessing/
        ├── 📄 README.md
        ├── 📄 __init__.py
        ├── 📄 ontology_mapper.py
        ├── 📄 pipeline.py
        ├── 📄 preprocess_external_data.py
        ├── 📄 text_cleaner_v2.py
        └── 📄 vn_tokenizer.py
```

---
> **Lưu ý bảo mật:** Cây thư mục này được tự động tạo dựa trên các tệp tin được theo dõi bởi Git (`git ls-files`). 
> Toàn bộ các tài nguyên cục bộ, khóa bảo mật, tệp cấu hình IDE cá nhân, và các tệp huấn luyện mô hình nặng đã được loại bỏ thông qua `.gitignore` để đảm bảo an toàn thông tin.
