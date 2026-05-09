# ⚙️ Thư Mục Cấu Hình (Configs)

**Mục đích:** Tập trung các file cấu hình JSON để điều khiển hành vi của các thành phần hệ thống.

## 📄 Các file chính:
- `facebook.json`: Cấu hình Facebook Pages (URLs, access tokens).
- `seeds.json`: Danh sách seed URLs ban đầu cho thu thập dữ liệu.
- `taxonomy.json`: Bộ phân loại chủ đề vaccine (ontology).

## 🚀 Cách sử dụng:
Các file này được sử dụng trực tiếp bởi các collector trong `src/data_pipeline/collection/`. Bạn chỉ cần chỉnh sửa nội dung JSON mà không cần can thiệp vào mã nguồn.
