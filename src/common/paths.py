import os
from pathlib import Path

# =============================================================================
# VaccineNLP Centralized Path Management
# =============================================================================

# Đường dẫn gốc của dự án (Gốc là thư mục VaccineNLP_ĐỒ_ÁN)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Thư mục Cấu hình & Tài liệu ---
CONFIG_DIR = BASE_DIR / "configs"
DOCS_DIR = BASE_DIR / "docs"
APP_DIR = BASE_DIR / "app"

# --- Thư mục Dữ liệu (Medallion Architecture) ---
DATA_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATA_DIR / "01_raw"          # BRONZE
INTERIM_DATA_DIR = DATA_DIR / "02_interim"  # SILVER (WIP)
PROCESSED_DATA_DIR = DATA_DIR / "02_processed" # SILVER (Cleaned, Unlabeled)
GOLD_DATA_DIR = DATA_DIR / "03_processed"      # GOLD (Validated/Benchmark)
SILVER_LABELS_DIR = DATA_DIR / "04_silver_labels" # SILVER (Auto-labeled)
MODEL_READY_DIR = DATA_DIR / "05_model_ready"    # Final training ready
TEMP_DATA_DIR = DATA_DIR / "temp"

# --- Thư mục Thực nghiệm (Experiments) ---
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODEL_DIR = EXPERIMENTS_DIR / "models"
RESULT_DIR = EXPERIMENTS_DIR / "results"

# --- Thư mục Mã nguồn & Scripts ---
NOTEBOOK_DIR = BASE_DIR / "notebooks"
SCRIPT_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "src"
SCRATCH_DIR = BASE_DIR / "scratch"

# --- Helper Functions ---
def get_model_path(model_name: str) -> Path:
    """Trả về đường dẫn đến một model cụ thể trong experiments/models/"""
    path = MODEL_DIR / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_result_path(result_name: str) -> Path:
    """Trả về đường dẫn đến một file/thư mục kết quả cụ thể"""
    return RESULT_DIR / result_name

def ensure_dirs():
    """Tạo tất cả các thư mục cần thiết nếu chưa tồn tại"""
    directories = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, 
        GOLD_DATA_DIR, SILVER_LABELS_DIR, MODEL_READY_DIR,
        TEMP_DATA_DIR, MODEL_DIR, RESULT_DIR, SCRATCH_DIR,
        CONFIG_DIR, APP_DIR
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Chạy thử để kiểm tra
    print(f"🚀 VaccineNLP Project Base: {BASE_DIR}")
    print(f"📊 Raw Data: {RAW_DATA_DIR}")
    print(f"🤖 Models: {MODEL_DIR}")
    
    # ensure_dirs() # Uncomment to create structure
