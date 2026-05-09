import requests
import json
import time
from src.common import paths

class VFNDFetcher:
    """Tải dữ liệu từ Vietnam Fact-check Network Database với cơ chế Retry"""
    
    def __init__(self, api_endpoint="https://api.vfnd.vn/v1/factchecks"):
        self.api_endpoint = api_endpoint
        self.output_file = paths.RAW_DATA_DIR / "vfnd_data_raw.json"

    def fetch_with_retry(self, max_retries=3):
        for i in range(max_retries):
            try:
                print(f"🌐 Đang kết nối tới VFND (Lần thử {i+1})...")
                response = requests.get(self.api_endpoint, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"❌ Lỗi: {e}. Đang thử lại sau 5 giây...")
                time.sleep(5)
        return None

    def run(self):
        data = self.fetch_with_retry()
        if data:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Tải dữ liệu VFND thành công! Lưu tại: {self.output_file}")
        else:
            print("❌ Không thể tải dữ liệu từ VFND sau nhiều lần thử.")

if __name__ == "__main__":
    fetcher = VFNDFetcher()
    fetcher.run()
