# 💉 VaccineNLP: Dual-Student Hybrid Architecture for Vietnamese Vaccine Misinformation

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/quynhphuong1209/VaccineNLP_Project)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vaccine-nlp-project.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

**VaccineNLP** is a production-grade research framework designed to combat vaccine-related misinformation in the Vietnamese digital landscape. Utilizing a **Dual-Student Hybrid Architecture**, the system balances high-reasoning explainability (via LLMs like Gemma-4) with high-efficiency classification (via encoders like PhoBERT).

### Key Features:
- **Multi-task Learning**: Simultaneous detection of Misinformation, Stance, and Sentiment.
- **Explainable AI (XAI)**: Generates reasoning chains for every classification using fine-tuned QLoRA models.
- **Medallion Data Architecture**: Robust pipeline from raw social media crawls to gold-standard benchmark sets.
- **Vietnamese Optimized**: Specifically tuned for the nuances of the Vietnamese language using `vinai/phobert-base-v2`.
- **Interactive Academic Dashboard**: Visualizes operational throughput (samples/sec), statistical accuracy (F1), and correlation flows.

### 🌐 Live Demo
Experience the dashboard live: **[vaccine-nlp-project.streamlit.app](https://vaccine-nlp-project.streamlit.app/)**

---

## 📂 Project Structure

The project follows a modular research-ready structure:

```text
VaccineNLP_Project/
├── app/                # Streamlit dashboard & interactive XAI interface (6 tabs)
├── configs/            # JSON configurations (class weights, taxonomy, seeds)
├── datasets/           # Medallion data pipeline (Bronze -> Silver -> Gold n=186)
├── docs/               # Technical reports and pipeline architecture diagrams
├── experiments/        # Model evaluation results (F1 stats, confusion matrices)
├── notebooks/          # Colab/Kaggle research training notebooks
├── scripts/            # Automation, downloading, and pipeline orchestration scripts
└── src/                # Core library (common utils, data pipeline, modeling MTL/XAI)
```

For more details on the directory structure, please read the [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) file.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/quynhphuong1209/VaccineNLP_Project.git
cd VaccineNLP_Project

# Install dependencies
pip install -r requirements.txt
```

### 2. Path Configuration
The project uses a centralized path manager. All paths are managed in `src/common/paths.py`. It automatically detects if you are running on **Kaggle** or **Local** environment.

### 3. Running the Dashboard
To launch the Streamlit interactive dashboard locally:
```bash
streamlit run app/streamlit_demo.py
```

### 4. Running the Pipeline
To run the full end-to-end pipeline (Collection -> Training -> Evaluation):
```bash
python scripts/unify_pipeline.py
```

---

## 🧠 Core Models

| Model | Task | Checkpoint |
|---|---|---|
| **PhoBERT-v2** | Multi-task Classifier | [`quynhphuong1209/phobert-multitask`](https://huggingface.co/quynhphuong1209/phobert-multitask) |
| **XLM-R-v1** | Multi-lingual Baseline | [`quynhphuong1209/xlmr-multitask`](https://huggingface.co/quynhphuong1209/xlmr-multitask) |
| **Gemma-4 4B** | XAI Reasoning Engine | [`quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai`](https://huggingface.co/quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai) |

---

## 📊 Results & Scientific Deep-dive

The models are evaluated on a **Human-in-the-Loop Gold Benchmark Set (n=186)**. 
- **Statistical accuracy**: PhoBERT-v2 achieves a macro average F1-Score of **0.6853** (Misinfo F1: 0.6886, Stance F1: 0.6383, Sentiment F1: 0.7289).
- **Operational efficiency**: PhoBERT-v2 reaches an inference throughput of **120.5 samples/second**, making it highly suitable for real-time social media scanning.
- **Explainable AI (XAI)**: Gemma-4 4B excels in producing high-quality reasoning chains for medical mislabeling, making it an essential backend consultant tool for HUPH experts.

Detailed evaluation charts, including interactive LaTeX calculations, correlation flows (Sankey), and confusion heatmaps are available in the **📈 ĐÁNH GIÁ CHUYÊN SÂU** tab of the dashboard.

---

## 🤝 Authors & Acknowledgments

- **Kim Mạnh Hưng** (2211090016)
- **Đinh Lê Quỳnh Phương** (2211090031)
- **Advisor**: TS. Trần Lâm Quân
- **Institution**: Hanoi University of Public Health (HUPH)

*Special thanks to the open-source community for providing PhoBERT and Unsloth.*
