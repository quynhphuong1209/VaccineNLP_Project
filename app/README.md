# 📱 Thư Mục Ứng Dụng (Streamlit Dashboard)

Thư mục này chứa toàn bộ mã nguồn giao diện tương tác người dùng của dự án **VaccineNLP** được xây dựng trên nền tảng **Streamlit**.

## 📂 Danh sách tệp tin
*   [streamlit_demo.py](file:///c:/Users/dinhl/Downloads/VaccineNLP_ĐỒ_ÁN/app/streamlit_demo.py): Tệp chạy chính của ứng dụng, chịu trách nhiệm kết xuất giao diện và tích hợp suy luận mô hình.
*   `xai_cache.json`: Bộ nhớ đệm lưu trữ các giải thích y học (XAI) đã được mô hình Gemma-4 tạo ra trước đó nhằm tối ưu hóa tốc độ phản hồi cho người dùng.

---

## 🎨 Cấu trúc Giao diện Dashboard (6 Tab Chuyên Nghiệp)

Ứng dụng được thiết kế theo chuẩn luận văn khoa học với 6 phân hệ chính:

1.  **🔍 PHÂN TÍCH VĂN BẢN (Tab 1)**:
    *   Cho phép người dùng tự nhập văn bản tiếng Việt hoặc quét nhanh bài viết từ các URL báo chí/mạng xã hội gợi ý.
    *   Phân loại đồng thời 3 trục nhãn: **Misinformation (Tin giả)**, **Stance (Lập trường)**, và **Sentiment (Cảm xúc)** qua mô hình PhoBERT-v2.
    *   Tích hợp hệ thống giải thích y khoa chuyên sâu **Gemma-4 XAI Engine** lý giải lý do phân loại nhãn và đề xuất kịch bản phản hồi khủng hoảng truyền thông.
    *   Tích hợp giọng đọc AI đọc to nội dung giải thích và tính năng xuất báo cáo PDF/HTML chuyên nghiệp.
2.  **📊 BENCHMARK & BÁO CÁO KHOA HỌC (Tab 2 - Gộp mới)**:
    *   **📋 Báo cáo tĩnh (Sub-tab 1)**: Trình bày chi tiết bảng so sánh F1-Score của 3 mô hình (PhoBERT-v2, XLM-R-v1, Gemma-4), các chỉ số thay đổi động theo mô hình lựa chọn và biểu đồ Macro F1 dạng cột nhóm.
    *   **⚡ Đánh giá Live (Sub-tab 2)**: Giả lập quá trình chạy thực tế trên GPU, biểu đồ so sánh **Tốc độ xử lý (Throughput)** dạng mẫu/giây và **Khuyến nghị kiến trúc lai phối hợp triển khai thực tiễn**.
3.  **📈 ĐÁNH GIÁ CHUYÊN SƯU (Tab 3 - Nâng cấp mới)**:
    *   **Model Capability Radar**: Biểu đồ mạng nhện so sánh sự cân bằng 5 chiều của 3 kiến trúc mô hình.
    *   **Interactive Metric Calculator**: Bộ máy tính chỉ số thống kê động. Người dùng click chọn lớp nhãn cụ thể để hệ thống tính toán Precision, Recall, F1 qua LaTeX toán học cùng số liệu TP, FP, FN thực tế.
    *   **Sankey Flow Chart**: Biểu đồ dòng chảy tương quan Sắc thái Cảm xúc ➔ Lập trường của **186 mẫu kiểm thử thực tế**.
    *   **Sunburst & Confusion Heatmap**: Phân cấp nhãn lồng nhau của tập Gold Test Set và ma trận nhầm lẫn chéo của PhoBERT-v2.
4.  **📚 TÀI LIỆU & NOTEBOOKS (Tab 4)**: Cung cấp đầy đủ liên kết tới các Notebooks huấn luyện trên Kaggle, checkpoint mô hình trên HuggingFace và mã nguồn dự án trên GitHub của hai tác giả.
5.  **📜 PHƯƠNG PHÁP LUẬN (Tab 5)**: Sơ đồ khối kiến trúc hệ thống và quy trình thực nghiệm.
6.  **📑 ĐỀ CƯƠNG (Tab 6)**: Đề cương chi tiết mục lục đồ án tốt nghiệp cùng các giả thuyết khoa học.

---

## 🚀 Hướng dẫn khởi chạy ứng dụng tại Local

Bạn mở terminal tại thư mục gốc của dự án và chạy các lệnh sau:
```bash
# 1. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 2. Khởi chạy Streamlit
streamlit run app/streamlit_demo.py
```
Ứng dụng sẽ tự động được mở tại trình duyệt web ở địa chỉ mặc định `http://localhost:8501`.
