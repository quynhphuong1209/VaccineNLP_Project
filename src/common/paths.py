import os
from pathlib import Path

# Đường dẫn gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Các thư mục cấu hình và tài liệu
CONFIG_DIR = BASE_DIR / "configs"
DOCS_DIR = BASE_DIR / "docs"

# Thư mục dữ liệu (Medallion Architecture)
DATA_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERIM_DATA_DIR = DATA_DIR / "02_interim"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

# Thư mục thực nghiệm (Experiments)
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODEL_DIR = EXPERIMENTS_DIR / "models"
RESULT_DIR = EXPERIMENTS_DIR / "results"

# Thư mục phát triển
NOTEBOOK_DIR = BASE_DIR / "notebooks"
SCRIPT_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "src"

def get_model_path(model_name: str) -> Path:
    """Trả về đường dẫn đến một model cụ thể"""
    return MODEL_DIR / model_name

def get_result_path(result_name: str) -> Path:
    """Trả về đường dẫn đến một file kết quả cụ thể"""
    return RESULT_DIR / result_name

if __name__ == "__main__":
    # In ra để kiểm tra
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Raw Data Directory: {RAW_DATA_DIR}")
