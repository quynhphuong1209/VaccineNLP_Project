# ⚙️ Thư Mục Cấu Hình Hệ Thống (Configurations)

Thư mục này tập trung toàn bộ các tệp cấu hình JSON để điều khiển hành vi thu thập, tiền xử lý, gán nhãn và thiết lập trọng số mất mát (loss weights) trong quá trình huấn luyện các mô hình AI.

## 📄 Các tệp cấu hình chính

1.  **`taxonomy.json`**:
    *   *Mục đích*: Định nghĩa bộ phân loại cấu trúc chủ đề về vắc-xin tại Việt Nam (Ontology/Taxonomy).
    *   *Nội dung*: Bao gồm các chủ đề chính như: Sự an toàn (Safety), Hiệu quả (Efficacy), Chính sách tiêm chủng (Policy), Nghi ngờ vô căn cứ (Conspiracy), và các từ khóa tiếng Việt tương ứng làm bộ lọc mẫu.
2.  **`class_weights_v2.json`**:
    *   *Mục đích*: Lưu trữ trọng số lớp được tính toán tự động bằng thuật toán Inverse Class Frequency để xử lý sự mất cân bằng dữ liệu nghiêm trọng (class imbalance) trong tập huấn luyện đa nhiệm.
    *   *Ứng dụng*: Được nạp trực tiếp vào hàm mất mát Weighted Cross Entropy khi huấn luyện mô hình PhoBERT-v2 và XLM-R-v1.
3.  **`seeds.json`**:
    *   *Mục đích*: Danh sách các URL seed ban đầu (nhóm cộng đồng, trang tin y tế, diễn đàn công cộng) để định hướng cho bộ công cụ thu thập tin tự động.
4.  **`facebook.json`**:
    *   *Mục đích*: Chứa cấu hình kết nối các Fanpage, Group Facebook công khai để quét bài viết và các lượt bình luận thảo luận của người dân về chiến dịch vắc-xin.

---

## 🚀 Cách tùy biến cấu hình

Các tệp JSON này được thiết kế để tách biệt cấu trúc tham số ra khỏi mã nguồn logic. 
*   Khi muốn thay đổi từ khóa lọc tin hoặc thêm trang y tế cần quét, bạn chỉ cần sửa nội dung tương ứng trong `seeds.json` hoặc `taxonomy.json` mà không cần viết lại bất kỳ dòng code Python nào trong `src/data_pipeline/`.
*   Trọng số trong `class_weights_v2.json` sẽ tự động cập nhật khi bạn chạy script tính toán phân bổ dữ liệu huấn luyện mới.
