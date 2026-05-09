# 🏗️ Pipeline Architecture

Chi tiết luồng dữ liệu của hệ thống VaccineNLP:

1. **Crawler Phase:** 
   - `scripts/apify_multi_router.py` điều phối việc lấy dữ liệu từ các nền tảng xã hội.
2. **Standardization Phase:**
   - Chuyển đổi dữ liệu thô về định dạng JSON chung.
3. **NLP Preprocessing:**
   - Xử lý ngôn ngữ tự nhiên (Cleaning, Tokenizing, Filtering) qua `src/data_pipeline/preprocessing/`.
4. **Storage:**
   - Lưu trữ dữ liệu qua các lớp Bronze, Silver, Gold trong `datasets/`.
