# 📱 VaccineNLP Dashboard (app/) - v1.3

**Cập nhật:** April 23, 2026 | Trạng thái: ✅ Active & Ready for Demo

**Ghi chú:** Ứng dụng Streamlit chính để demo XAI capabilities của hệ thống VaccineNLP

## 🚀 Thành Phần Chính

### 1. `streamlit_demo.py`
Ứng dụng Streamlit chính cung cấp 3 tính năng cốt lõi:
- **Real-time Classification**: Nhập văn bản và nhận kết quả phân tích đa chiều (Misinfo, Stance, Sentiment).
- **Explainable AI (XAI)**: Hiển thị chuỗi lý luận (Chain-of-Thought) giúp giải thích các dự đoán của AI.
- **Model Comparison (Benchmark)**: Trực quan hóa hiệu năng của PhoBERT-v2, XLM-R và Gemma-4 trên tập Gold Benchmark.

### 2. `xai_cache.json`
Bộ nhớ đệm chứa các phản hồi reasoning chất lượng cao từ Teacher Model (Gemma-4 31B), giúp tăng tốc độ demo và đảm bảo độ chính xác của giải thích.

## 🛠️ Hướng Dẫn Chạy
Để chạy dashboard cục bộ, bạn cần cài đặt các thư viện trong `requirements.txt` và chạy lệnh sau:

```bash
streamlit run app/streamlit_demo.py
```

## 📊 Mô Hình Hỗ Trợ
Ứng dụng cho phép chuyển đổi giữa các mô hình Encoder tại Sidebar:
- **PhoBERT-v2**: Tốt nhất cho tiếng Việt.
- **XLM-R-v1**: Baseline đa ngôn ngữ.

---
*Cập nhật: 23/04/2026 | Phiên bản 1.3*
