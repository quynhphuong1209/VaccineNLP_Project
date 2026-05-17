# 💻 Thư Mục Mã Nguồn Dự Án (Core Library - src)

Thư mục `src` là bộ xương sống logic của dự án **VaccineNLP**, được thiết kế theo dạng thư viện mô-đun (modular library) tái sử dụng cao, phục vụ cho luồng xử lý dữ liệu và mô hình hóa AI.

## 📂 Các mô-đun cốt lõi

1.  **`common/`**:
    *   *Nhiệm vụ*: Chứa các lớp tiện ích dùng chung như trình quản lý đường dẫn tự động (`paths.py`), cấu hình hạt giống ngẫu nhiên (seed manager), trình tải biến môi trường và logging hệ thống.
2.  **`data_pipeline/`**:
    *   *Nhiệm vụ*: Phân hệ xử lý dữ liệu Medallion:
        *   `collection/`: Trình thu thập dữ liệu tự động từ các nền tảng mạng xã hội qua API và bộ quét web.
        *   `preprocessing/`: Các lớp làm sạch văn bản tiếng Việt, chuẩn hóa bảng mã Unicode, tách từ y học (Word Segmentation), và trích xuất đặc trưng.
        *   `labelling/`: Công cụ gán nhãn tự động bằng heuristics hoặc hỗ trợ gán nhãn bán tự động qua mô hình.
3.  **`modeling/`**:
    *   *Nhiệm vụ*: Thiết kế kiến trúc và huấn luyện mô hình:
        *   `phobert_multitask/`: Định nghĩa lớp mô hình PhoBERT đa nhiệm, các hàm mất mát (weighted cross entropy loss) và trình tối ưu hóa AdamW.
        *   `gemma_xai/`: Kiến trúc tinh chỉnh Gemma-4 QLoRA qua Unsloth, kịch bản tạo prompt-template và xử lý suy luận chuỗi lập luận (Reasoning Engine).
        *   `evaluation/`: Bộ tính toán Precision, Recall, Macro F1, và xuất báo cáo kết quả thực nghiệm.

---

## 🏗️ Nguyên tắc thiết kế Modular

Tất cả các thành phần trong `src/` đều tuân thủ nguyên tắc thiết kế hướng đối tượng (OOP) và lập trình hàm sạch sẽ. Khi phát triển các tính năng mới cho ứng dụng hoặc thay đổi kiến trúc mô hình, các nhà phát triển chỉ cần import các mô-đun từ `src/` (ví dụ: `from src.modeling.phobert_multitask import ...`) để tái sử dụng tối đa mã nguồn, đảm bảo tính nhất quán cao của toàn bộ hệ thống.
