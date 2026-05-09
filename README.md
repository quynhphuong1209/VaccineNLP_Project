# 💉 VaccineNLP: Dual-Student Hybrid Architecture for Vietnamese Vaccine Misinformation

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/quynhphuong1209/VaccineNLP_Project)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

**VaccineNLP** is a production-grade research framework designed to combat vaccine-related misinformation in the Vietnamese digital landscape. Utilizing a **Dual-Student Hybrid Architecture**, the system balances high-reasoning explainability (via LLMs like Gemma-4) with high-efficiency classification (via encoders like PhoBERT).

### Key Features:
- **Multi-task Learning**: Simultaneous detection of Misinformation, Stance, and Sentiment.
- **Explainable AI (XAI)**: Generates reasoning chains for every classification using fine-tuned QLoRA models.
- **Medallion Data Architecture**: Robust pipeline from raw social media crawls to gold-standard benchmark sets.
- **Vietnamese Optimized**: Specifically tuned for the nuances of the Vietnamese language using `vinai/phobert-base-v2`.

---

## 📂 Project Structure

The project follows a modular research-ready structure:

```text
VaccineNLP_Project/
├── app/                # Streamlit/FastAPI demo applications
├── configs/            # JSON configurations (taxonomy, seeds, weights)
├── datasets/           # Medallion data (Raw -> Processed -> Gold)
├── docs/               # Technical reports and architecture diagrams
├── experiments/        # Model checkpoints and evaluation results
├── notebooks/          # Colab/Kaggle research notebooks
├── scripts/            # Automation and training scripts
├── src/                # Core library (data pipeline, modeling, common)
└── requirements.txt    # Project dependencies
```

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

### 3. Running the Pipeline
To run the full end-to-end pipeline (Collection -> Training -> Evaluation):
```bash
python scripts/unify_pipeline.py
```

---

## 🧠 Core Models

| Model | Task | Checkpoint |
|---|---|---|
| **PhoBERT-v2** | Multi-task Classifier | `quynhphuong1209/phobert-multitask` |
| **Gemma-4-4B** | XAI Reasoning Engine | `quynhphuong1209/gemma-4-E4B-unsloth-vaccine-xai` |

---

## 📊 Results

The models are evaluated on a **Human-in-the-Loop Gold Benchmark Set (n=186)**. Detailed results can be found in `experiments/results/`.

---

## 🤝 Authors & Acknowledgments

- **Kim Mạnh Hưng** (2211090016)
- **Đinh Lê Quỳnh Phương** (2211090031)
- **Advisor**: TS. Trần Lâm Quân
- **Institution**: Hanoi University of Public Health (HUPH)

*Special thanks to the open-source community for providing PhoBERT and Unsloth.*
