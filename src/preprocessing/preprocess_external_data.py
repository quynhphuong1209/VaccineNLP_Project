import pandas as pd
import json
import os
import glob
from langdetect import detect_langs, DetectorFactory
from tqdm import tqdm

from src.common.paths import ROOT_DIR, DATA_RAW_DIR, DATA_UNLABELED_DIR

# Đảm bảo kết quả langdetect nhất quán
DetectorFactory.seed = 42

def detect_vietnamese(text, threshold=0.9):
    """
    Sử dụng langdetect để xác định ngôn ngữ.
    Ngưỡng 0.9 để đảm bảo lọc khắt khe theo yêu cầu của Architect.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False
    try:
        langs = detect_langs(text)
        for lang in langs:
            if lang.lang == 'vi' and lang.prob >= threshold:
                return True
    except:
        pass
    return False

def process_vfnd(base_path):
    """
    Xử lý Vietnamese Fake News Dataset.
    Format: JSON files in Real/Article_Contents and Fake/Article_Contents.
    """
    data = []
    
    # Các folder chứa dữ liệu
    folders = {
        'Real': 0,
        'Fake': 1
    }
    
    print("Processing VFND...")
    for label_str, label_val in folders.items():
        # Ensure path is a Path object
        search_pattern = str(Path(base_path) / label_str / 'Article_Contents' / '*.json')
        files = glob.glob(search_pattern)
        print(f"  - Found {len(files)} files in {label_str}")
        
        for f_path in tqdm(files, desc=f"Reading {label_str}"):
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                    
                title = item.get('title', '')
                desc = item.get('description', '')
                content = item.get('text', '')
                
                # Gộp Title + Description + Content để tối đa hóa ngữ cảnh
                full_text = f"{title}. {desc}. {content}".strip()
                full_text = full_text.replace('\n', ' ').replace('\r', ' ')
                
                if len(full_text) > 20:
                    data.append({
                        'text': full_text,
                        'label_misinfo': label_val,
                        'label_stance': -100,
                        'label_sentiment': -100,
                        'source': 'VFND'
                    })
            except Exception as e:
                # print(f"Error reading {f_path}: {e}")
                continue
                
    return data

def process_misovac(csv_path, threshold=0.9):
    """
    Xử lý MiSoVac dataset. Lọc nghiêm ngặt cho tiếng Việt.
    MiSoVac label: False = Misinfo (1), True = Fact (0)
    """
    print(f"Processing MiSoVac from {csv_path}...")
    df = pd.read_csv(csv_path)
    data = []
    
    # Đếm số lượng ban đầu
    initial_count = len(df)
    
    for _, row in tqdm(df.iterrows(), total=initial_count, desc="Filtering MiSoVac"):
        text = str(row['text'])
        # Architect rule: Strict Language Detection
        if detect_vietnamese(text, threshold):
            label_val = 1 if row['label'] == False else 0
            data.append({
                'text': text,
                'label_misinfo': label_val,
                'label_stance': -100,
                'label_sentiment': -100,
                'source': 'MiSoVac'
            })
            
    print(f"  - MiSoVac: Filtered {len(data)} Vietnamese samples from {initial_count} total.")
    return data

def main():
    # Paths (Đã đồng bộ với VaccineNLP_Clean_V1)
    # reference_data folder chứa các bộ dataset ngoài dùng để warmup mô hình
    ref_dir = ROOT_DIR / "reference_data"
    
    vfnd_base = ref_dir / "datasets" / "zenodo_6590948" / "VFND-VFND-vietnamese-fake-news-datasets-a52319a" / "Fake_Real_Dataset"
    misovac_path = ref_dir / "datasets" / "MiSoVac" / "MiSoVac.csv"
    output_path = DATA_UNLABELED_DIR / "external_warmup_data.csv"
    
    all_data = []
    
    # 1. Process VFND
    if vfnd_base.exists():
        all_data.extend(process_vfnd(vfnd_base))
    else:
        print(f"Warning: VFND path not found: {vfnd_base}")
        
    # 2. Process MiSoVac
    if misovac_path.exists():
        all_data.extend(process_misovac(misovac_path, threshold=0.9))
    else:
        print(f"Warning: MiSoVac path not found: {misovac_path}")
        
    # 3. Save Harmonized Data
    if all_data:
        combined_df = pd.DataFrame(all_data)
        
        # Ensure datasets directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Harmonization Complete!")
        print(f"Total samples for warming: {len(combined_df)}")
        print(f"Stats by source:\n{combined_df['source'].value_counts()}")
        print(f"Saved to: {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()

