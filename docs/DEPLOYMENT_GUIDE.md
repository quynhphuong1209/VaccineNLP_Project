# HƯỚNG DẪN DEPLOY VACCINENLP APP LÊN STREAMLIT COMMUNITY CLOUD
# VaccineNLP | Updated: 21/05/2026

Tài liệu này hướng dẫn bạn từng bước đưa ứng dụng phân tích dữ liệu **VaccineNLP** từ máy cá nhân lên đám mây **Streamlit Community Cloud** hoàn toàn miễn phí, an toàn và chuyên nghiệp để phục vụ trình diễn trước Hội đồng bảo vệ luận văn.

---

## 📋 CÁC BƯỚC CHUẨN BỊ TRƯỚC KHI DEPLOY

1. **Mã nguồn đã sẵn sàng:** Toàn bộ code sạch đã được đẩy lên GitHub Repository của bạn:
   👉 [https://github.com/hwngkm/VaccineNLP-Thesis](https://github.com/hwngkm/VaccineNLP-Thesis) (nhánh `main`).
2. **Dependencies đã đồng bộ:** File `requirements.txt` ở thư mục gốc đã được tích hợp đầy đủ các thư viện thu thập dữ liệu và phân tích XAI (`captum`, `apify-client`, `yt-dlp`, `trafilatura`).

---

## 🚀 QUY TRÌNH DEPLOY 5 BƯỚC TRÊN STREAMLIT CLOUD

### Bước 1: Đăng nhập Streamlit Community Cloud
1. Truy cập trang web: [https://share.streamlit.io/](https://share.streamlit.io/)
2. Nhấp chọn **"Continue with GitHub"** để đăng nhập bằng tài khoản GitHub của bạn.

### Bước 2: Tạo ứng dụng mới (Create App)
1. Sau khi đăng nhập thành công vào Dashboard, nhấp nút **"Create app"** (nằm ở góc trên bên phải).
2. Chọn **"Yup, I have an app"** nếu được hỏi.

### Bước 3: Cấu hình thông tin Repository
Điền chính xác các thông tin cấu hình sau:
*   **Repository:** Chọn hoặc dán đường dẫn repo của bạn: `hwngkm/VaccineNLP-Thesis`
*   **Branch:** `main`
*   **Main file path:** `app/streamlit_demo.py`  *(⚠️ Lưu ý: Bắt buộc phải có tiền tố `app/` vì tệp demo nằm trong thư mục con)*
*   **App URL (Tùy chọn):** Bạn có thể tùy chỉnh tên miền mong muốn (ví dụ: `vaccinenlp-thesis`).

---

### Bước 4: Cấu hình Secrets bảo mật (Cực kỳ quan trọng) 🔒
Để hệ thống thu thập dữ liệu đa nguồn Apify hoạt động ổn định trên môi trường Cloud mà không bị lộ token, bạn cần cấu hình Secrets:

1. Tại giao diện cấu hình Deploy, nhấp vào nút **"Advanced settings..."** ở phía dưới.
2. Chọn mục **Secrets**.
3. Copy toàn bộ đoạn cấu hình dưới đây và dán vào ô nhập liệu:

```toml
# Danh sách 5 Apify API Tokens của bạn để xoay vòng tự động (Token Rotation)
APIFY_TOKENS = [
    "apify_api_XXXXXXXXXXXXXXX", # Thay bằng Token 1 của bạn
    "apify_api_YYYYYYYYYYYYXXX", # Thay bằng Token 2 của bạn
    "apify_api_ZZZZZZZZZZZZXXX", # Thay bằng Token 3 của bạn
    "apify_api_AAAAAAAAAAAAXXX", # Thay bằng Token 4 của bạn
    "apify_api_BBBBBBBBBBBBXXX"  # Thay bằng Token 5 của bạn
]

# (Tùy chọn) Hugging Face Token nếu mô hình PhoBERT/Gemma yêu cầu quyền riêng tư
# HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```
4. Thay thế các chuỗi `"apify_api_...` bằng các API Token thật của bạn (lấy trực tiếp từ file `.env` local của bạn).
5. Nhấp nút **Save** để lưu lại.

---

### Bước 5: Tiến hành Deploy!
1. Sau khi cấu hình xong xuôi, nhấp nút **"Deploy!"**.
2. Streamlit Cloud sẽ tiến hành:
   * Khởi tạo máy ảo Docker.
   * Tải mã nguồn từ GitHub.
   * Cài đặt các thư viện phụ thuộc từ `requirements.txt` (quá trình này mất khoảng 2-4 phút ở lần chạy đầu tiên).
   * Khởi chạy Web App.
3. Bạn có thể theo dõi tiến trình cài đặt thông qua hộp thoại **"Manage app"** (nhấp vào góc dưới bên phải màn hình).

---

## 🚨 MỘT SỐ LƯU Ý KHI CHẠY TRÊN CLOUD

> [!IMPORTANT]
> **Giới hạn tài nguyên RAM:** Máy ảo miễn phí của Streamlit Cloud được cấp **1GB RAM**.
> * Ứng dụng VaccineNLP chạy cực kỳ tối ưu vì phần suy luận Gemma được gọi qua **Inference API của HuggingFace** (không tốn RAM máy ảo).
> * Mô hình PhoBERT và Captum XAI chạy suy luận trực tiếp trên CPU của máy ảo và tiêu tốn khoảng **500MB RAM**, hoàn toàn nằm trong ngưỡng an toàn.
> * **Khuyến cáo:** Tránh chạy phân tích hàng loạt (Batch Mode) với số lượng mẫu quá lớn (> 50 dòng văn bản cùng một lúc) để tránh máy ảo bị khởi động lại do tràn bộ nhớ (Out of Memory).

---

### 🎉 HOÀN THÀNH!
Sau khi deploy thành công, Streamlit Cloud sẽ cung cấp cho bạn một đường dẫn URL công khai (dạng `https://xxxx.streamlit.app/`). Bạn có thể chia sẻ link này trực tiếp cho giảng viên hướng dẫn hoặc trình chiếu LIVE mượt mà tại buổi bảo vệ luận văn!
