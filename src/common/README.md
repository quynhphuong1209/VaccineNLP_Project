# 🔧 Shared Utilities (src/common/) - v1.1

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Maintained

## Mục Đích
Cung cấp utilities chung và configuration được sử dụng bởi tất cả các modules khác trong dự án.

---

## 📋 Danh Sách Modules

### **paths.py** - Centralized Path Management

**Chức Năng:** Định nghĩa tất cả đường dẫn dự án ở một nơi

**Nội Dung:**
```python
from pathlib import Path

# === ROOT DIRECTORIES ===
ROOT_DIR = Path(__file__).parent.parent.parent  # d:\VaccineNLP_Clean_V1\
DATA_DIR = ROOT_DIR / 'datasets'
SRC_DIR = ROOT_DIR / 'src'
MODELS_DIR = ROOT_DIR / 'experiments' / 'models'
LOG_DIR = ROOT_DIR / 'experiments' / 'logs'
CONFIG_DIR = ROOT_DIR / 'configs'
DOCS_DIR = ROOT_DIR / 'docs'
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'

# === DATA LAYERS (Medallion Architecture) ===
BRONZE_DIR = DATA_DIR / '01_raw'               # Dữ liệu thô
SILVER_INTERIM_DIR = DATA_DIR / '02_interim'   # Dữ liệu đang xử lý
SILVER_PROCESSED_DIR = DATA_DIR / '02_processed'  # Dữ liệu đã làm sạch
GOLD_DIR = DATA_DIR / '03_processed'           # Dữ liệu đã xác nhận
SILVER_LABELS_DIR = DATA_DIR / '04_silver_labels'  # Dữ liệu auto-labeled
TEMP_DIR = DATA_DIR / 'temp'                   # Dữ liệu tạm

# === SPECIFIC FILES ===
# Gold standard datasets
MASTER_DATASET = GOLD_DIR / 'reclaimed_master_pool_vn_clean.json'
CORPUS_UNLABELED = SILVER_PROCESSED_DIR / 'corpus_1856_unlabeled.json'
ANNOTATED_LABELS = SILVER_LABELS_DIR / 'annotated_v6.jsonl'

# Configuration files
FACEBOOK_CONFIG = CONFIG_DIR / 'facebook.json'
SEEDS_CONFIG = CONFIG_DIR / 'seeds.json'
TAXONOMY_CONFIG = CONFIG_DIR / 'taxonomy.json'

# Model checkpoints
PHOBERT_MODEL_DIR = MODELS_DIR / 'phobert_multitask_v1'
GEMMA_MODEL_DIR = MODELS_DIR / 'gemma4_qlora_v1'

# === UTILITIES ===
def ensure_dirs(*dirs):
    """Ensure directories exist"""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Create all directories on import
ensure_dirs(
    BRONZE_DIR, SILVER_INTERIM_DIR, SILVER_PROCESSED_DIR,
    GOLD_DIR, SILVER_LABELS_DIR, TEMP_DIR,
    MODELS_DIR, LOG_DIR
)
```

**Sử Dụng:**
```python
from src.common.paths import MASTER_DATASET, MODELS_DIR, LOG_DIR

# Load data
with open(MASTER_DATASET) as f:
    data = json.load(f)

# Save logs
with open(LOG_DIR / 'training.log', 'w') as f:
    f.write("Training started...")

# Save model
model.save_pretrained(MODELS_DIR / 'my_model_v1')
```

**Lợi Ích:**
- ✅ Không hardcode paths
- ✅ Dễ thay đổi khi move files
- ✅ Consistent across modules
- ✅ Tự động tạo directories

---

### **versioning_manager.py** - Version Tracking & Compatibility

**Chức Năng:** Quản lý versioning cho datasets, models, và tracking compatibility

**API:**
```python
class VersioningManager:
    """Manage versions of datasets and models"""
    
    def create_version(self, source_path, version_tag="v1.0"):
        """
        Create versioned copy of dataset/model
        
        Args:
            source_path: Path to original file
            version_tag: Version tag (e.g., "v1.0", "v2.1")
            
        Returns:
            Path to versioned file
            
        Example:
            vm = VersioningManager()
            v1_path = vm.create_version('datasets/data.json', 'v1.0')
            # Creates: datasets/data_v1.0.json
        """
        pass
    
    def get_latest_version(self, base_path, pattern="v*"):
        """
        Get latest version of dataset matching pattern
        
        Args:
            base_path: Base path without version
            pattern: Version pattern (default: "v*")
            
        Returns:
            Path to latest version
            
        Example:
            latest = vm.get_latest_version('datasets/data')
            # Returns: datasets/data_v3.2.json (if latest)
        """
        pass
    
    def get_all_versions(self, base_path):
        """Get all versions of a dataset"""
        pass
    
    def check_compatibility(self, model_version, data_version):
        """
        Check if model and data versions are compatible
        
        Returns:
            bool: True if compatible, False otherwise
        """
        pass
    
    def get_version_metadata(self, file_path):
        """
        Get metadata about a versioned file
        
        Returns:
            {
                'path': file_path,
                'version': 'v1.0',
                'created_date': '2024-03-15',
                'size': 1024000,
                'format': 'json',
                'checksum': 'abc123'
            }
        """
        pass
    
    def create_version_snapshot(self, paths, snapshot_name):
        """
        Create snapshot of multiple files with version tracking
        
        Useful for: Reproducibility, comparing experiment versions
        """
        pass
```

**Sử Dụng:**
```python
from src.common.versioning_manager import VersioningManager

vm = VersioningManager()

# Create versioned dataset
vm.create_version('datasets/03_processed/data.json', 'v1.0')

# Check compatibility before loading
compatible = vm.check_compatibility(
    model_version='phobert_multitask_v1',
    data_version='v1.0'
)

if compatible:
    print("✅ Model & data versions compatible")
else:
    print("❌ Compatibility issue - upgrade required")

# Get latest version
latest = vm.get_latest_version('datasets/data')
print(f"Latest version: {latest}")
```

**Versioning Scheme:**
```
Semantic Versioning: MAJOR.MINOR.PATCH
- v1.0.0 - Initial release
- v1.1.0 - Minor update (backward compatible)
- v2.0.0 - Major update (breaking changes)

Files:
- data_v1.0.0.json
- model_phobert_v1.2.3/
```

---

## 🔐 Configuration Management

### Environment Variables (.env)

```bash
# Data paths (optional - overrides defaults)
DATA_ROOT=d:\VaccineNLP_Clean_V1\datasets
MODEL_ROOT=d:\VaccineNLP_Clean_V1\experiments\models

# API credentials
FACEBOOK_API_TOKEN=your_token
APIFY_API_TOKEN=your_token
VFND_API_KEY=your_key

# Model settings
MODEL_DEVICE=cuda  # or cpu
BATCH_SIZE=32
WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Usage

```python
import os
from dotenv import load_dotenv

load_dotenv()

FACEBOOK_TOKEN = os.getenv('FACEBOOK_API_TOKEN')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
```

---

## 📊 Utility Functions

### Available Utilities

```python
# Path utilities
from src.common.paths import (
    MASTER_DATASET,
    GOLD_DIR,
    MODELS_DIR,
    ensure_dirs,
    get_all_files,
    get_latest_file
)

# Versioning
from src.common.versioning_manager import VersioningManager

# Logging
from src.common.logger import get_logger

logger = get_logger(__name__)
logger.info("Starting process")
```

---

## 🎯 Best Practices

1. **Always import paths from src.common.paths**
   ```python
   # ✅ GOOD
   from src.common.paths import MASTER_DATASET
   
   # ❌ BAD
   path = 'datasets/03_processed/reclaimed_master_pool_vn_clean.json'
   ```

2. **Use versioning for important datasets**
   ```python
   # Create version before major processing
   vm = VersioningManager()
   vm.create_version(MASTER_DATASET, 'v1.0')
   ```

3. **Load environment early**
   ```python
   # In main script or __init__.py
   from dotenv import load_dotenv
   load_dotenv()
   ```

4. **Check paths exist before operations**
   ```python
   if MASTER_DATASET.exists():
       data = json.load(open(MASTER_DATASET))
   else:
       raise FileNotFoundError(f"{MASTER_DATASET} not found")
   ```

---

**📅 Updated:** April 2026  
**🔧 Framework:** Python standard library + pathlib  
**📊 Features:** Path management, versioning, configuration
