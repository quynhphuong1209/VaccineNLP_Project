import json
import os
from src.common import paths

def extract_urls(input_json_name):
    input_path = paths.RAW_DATA_DIR / input_json_name
    output_path = paths.RAW_DATA_DIR / "target_urls_fb.txt"
    
    if not os.path.exists(input_path):
        print(f"⚠️ File {input_json_name} không tồn tại trong {paths.RAW_DATA_DIR}")
        return

    print(f"📍 Đang trích xuất URLs từ {input_json_name}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    urls = []
    for item in data:
        # Tùy vào định dạng của Apify, URL có thể nằm ở 'url' hoặc 'facebookUrl'
        url = item.get('url') or item.get('facebookUrl') or item.get('canonicalUrl')
        if url:
            urls.append(url)
    
    # Loại bỏ trùng lặp
    urls = list(set(urls))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(f"{url}\n")
            
    print(f"✅ Đã trích xuất thành công {len(urls)} URLs vào file {output_path}")

if __name__ == "__main__":
    # Thay 'apify_output.json' bằng tên file thực tế của bạn
    extract_urls("apify_output.json")
