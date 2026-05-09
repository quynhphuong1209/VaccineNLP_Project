import json
from datetime import datetime
from src.common import paths

class VersioningManager:
    """Quản lý phiên bản dữ liệu và mô hình để đảm bảo tính nhất quán"""
    
    def __init__(self):
        self.metadata_path = paths.BASE_DIR / "version_metadata.json"

    def create_version(self, file_path, version_tag, description=""):
        metadata = {
            "version": version_tag,
            "timestamp": datetime.now().isoformat(),
            "file": str(file_path),
            "description": description
        }
        # Logic lưu metadata (giả định)
        print(f"📌 Đã tạo phiên bản {version_tag} cho {file_path}")
        return metadata

    def check_compatibility(self, model_version, data_version):
        """Kiểm tra xem mô hình và dữ liệu có khớp phiên bản không"""
        # Giả sử mọi thứ đều khớp trong bản demo
        return True
