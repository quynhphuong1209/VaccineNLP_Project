import sys
import os
from pathlib import Path

# Đảm bảo Python tìm thấy thư mục src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.collection.master_collector_v2 import MasterCollector
from src.modeling.phobert_multitask_trainer import PhoBertMultitaskTrainer

def run_e2e_pipeline():
    print("🌟 --- BẮT ĐẦU PIPELINE VACCINENLP --- 🌟")
    
    # 1. Thu thập dữ liệu
    collector = MasterCollector()
    collector.run_collection()
    
    # 2. Huấn luyện & Đánh giá (Giả lập)
    trainer = PhoBertMultitaskTrainer()
    results = trainer.evaluate(None)
    
    print("\n📈 KẾT QUẢ CUỐI CÙNG:")
    for metric, value in results.items():
        print(f" - {metric}: {value}")
    
    print("\n✅ Pipeline hoàn thành xuất sắc!")

if __name__ == "__main__":
    run_e2e_pipeline()
