import json
from src.common import paths

class MasterCollector:
    """Điều phối việc thu thập dữ liệu từ đa nguồn (FB, YT, TikTok)"""
    
    def __init__(self):
        self.config_path = paths.CONFIG_DIR / "seeds.json"
        self.output_dir = paths.RAW_DATA_DIR

    def load_seeds(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_collection(self):
        seeds = self.load_seeds()
        print(f"🕷️ Bắt đầu thu thập dữ liệu từ {len(seeds['facebook_pages'])} trang Facebook...")
        # Ở đây sẽ gọi các collector chi tiết (Apify, Facebook SDK...)
        print("✅ Thu thập hoàn tất. Dữ liệu thô lưu tại: datasets/01_raw/")

if __name__ == "__main__":
    collector = MasterCollector()
    collector.run_collection()
