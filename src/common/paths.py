import os
from pathlib import Path
import sys

# Project root (VaccineNLP_Clean_V1)
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# Configs & Docs
CONFIGS_DIR = ROOT_DIR / "configs"
DOCS_DIR = ROOT_DIR / "docs"

# Datasets (Medallion Architecture)
DATA_RAW_DIR = ROOT_DIR / "datasets" / "01_raw"                 # Bronze Data
DATA_UNLABELED_DIR = ROOT_DIR / "datasets" / "02_processed"     # Silver Data (Unlabeled)
DATA_GOLD_DIR = ROOT_DIR / "datasets" / "03_processed"          # Gold Data (Manual Labels)
DATA_SILVER_DIR = ROOT_DIR / "datasets" / "04_silver_labels"    # Silver Data (LLM Labels)
DATA_MODEL_READY_DIR = ROOT_DIR / "datasets" / "05_model_ready" # Model-Ready Data (Segmented + Remapped)
DATA_TEMP_DIR = ROOT_DIR / "datasets" / "temp"                  # Temp processing

# Experiments (MLflow & trained models)
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Ensure core directories exist
for d in [DATA_RAW_DIR, DATA_UNLABELED_DIR, DATA_GOLD_DIR, DATA_SILVER_DIR, DATA_MODEL_READY_DIR, DATA_TEMP_DIR, EXPERIMENTS_DIR, MODELS_DIR, RESULTS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def ensure_src_in_sys_path():
    """Add the project's 'src' directory to sys.path if not already present."""
    src_path = str(ROOT_DIR / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print(f"VaccineNLP Path Registry Synced:")
    print(f"  ROOT       : {ROOT_DIR}")
    print(f"  UNLABELED  : {DATA_UNLABELED_DIR}")
    print(f"  SILVER_LLM : {DATA_SILVER_DIR}")
