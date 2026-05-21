# 📱 VaccineNLP Dashboard (app/) - v1.4

**Cập nhật:** May 21, 2026 | Trạng thái: ✅ Active & Ready for Demo

**Mục đích:** Giao diện người dùng interactiv dành cho demo XAI capabilities của hệ thống VaccineNLP. Cho phép phân tích văn bản về vaccine với giải thích từ AI.

---

## 📋 Danh Sách Files

### 1. 📄 `streamlit_demo.py` - Main Application
**Chức Năng:** Ứng dụng web Streamlit chính (Entry point)

**Tính Năng Chính:**
- 🔍 **Real-time Text Classification**: Nhập văn bản bất kỳ và nhận phân loại đa chiều instant
  - Misinfo Detection (Phát hiện tin sai lệch)
  - Stance Analysis (Phân tích thái độ): Support/Against/Neutral
  - Sentiment Classification (Phân tích cảm xúc)

- 🧠 **Explainable AI (XAI)**: Hiển thị Chain-of-Thought reasoning từ Gemma-4
- 📊 **Model Comparison**: So sánh PhoBERT-v2 vs XLM-R vs Gemma-4
- 🎛️ **Sidebar Controls**: Chọn mô hình & settings

**Chạy Ứng Dụng:**
```bash
pip install -r app/requirements_demo.txt
streamlit run app/streamlit_demo.py
```

### 2. 💾 `xai_cache.json` - Reasoning Cache
Bộ nhớ đệm chứa Chain-of-Thought explanations từ Gemma-4 31B Teacher.
- Lưu trữ phản hồi reasoning chất lượng cao
- Format: text_hash → reasoning explanations
- Tăng tốc độ demo & đảm bảo độ chính xác

**Backup Files:**
- `xai_cache_backup_20260520_2317.json` - Previous version backup

### 3. 📦 `requirements_demo.txt` - Dependencies
Các Python packages cần cho Streamlit app

---

## 📊 Mô Hình Được Hỗ Trợ

| Mô Hình | Kích Thước | Hiệu Năng Misinfo F1 | Ghi Chú |
|---|---|---|---|
| **PhoBERT-v2** | 110M | **0.7079** ⭐ | Gợi ý sử dụng |
| **XLM-R-v1** | 370M | 0.5823 | Đa ngôn ngữ |
| **Gemma-4-4B** | 4B | 0.6925 | Với XAI |

---

## 🚀 Hướng Dẫn Nhanh

```bash
# 1. Cài đặt dependencies
pip install -r app/requirements_demo.txt

# 2. Chạy app
streamlit run app/streamlit_demo.py

# 3. Truy cập
http://localhost:8501
```

---

*Cập nhật: 21/05/2026 | Phiên bản 1.4*
