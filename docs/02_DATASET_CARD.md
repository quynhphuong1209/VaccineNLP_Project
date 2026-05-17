# 📊 Thẻ Dữ liệu (Dataset Card) - Ứng dụng Xử lý Ngôn ngữ Tự nhiên trong phát hiện thông tin sai lệch về vaccine và phân tích thái độ cộng đồng trên môi trường số tại Việt Nam

Tài liệu này đặc tả các thuộc tính, nguồn gốc và hệ thống phân loại của bộ dữ liệu phục vụ nghiên cứu.

## 1. Nguồn gốc Dữ liệu (Data Sources)

| Phân nhóm | Nguồn | Mô tả |
|-----------|-------|-------|
| **Core** | Facebook | Chất lượng dữ liệu cao nhất, bao quát các hội nhóm y tế và cộng đồng tự phát. |
| **Reclaim** | YouTube, TikTok, Forums | Độ nhiễu cao, hiệu suất thấp, dùng để mở rộng quy mô bổ trợ đa nền tảng. |
| **External** | VFND, MiSoVac | Dữ liệu kế thừa từ các bộ dữ liệu chuẩn học thuật đã được chuyên gia gán nhãn. |

## 2. Hệ thống Nhãn 3 Trục (Taxonomy)

Các giá trị ID dưới đây đã được chuẩn hóa để khớp với logic huấn luyện mô hình đa nhiệm (**Multi-task Training**):

### A. Misinformation (Trục Tin giả)
- `0` - **Không chắc chắn / Không liên quan**: Dành cho dữ liệu nhiễu, các bình luận không thể kiểm chứng được hoặc không liên quan trực tiếp đến vaccine.
- `1` - **Tin giả / Sai lệch**: Thông tin sai sự thật hoặc gây hiểu lầm dựa trên các tiêu chí y khoa.
- `2` - **Chính xác**: Thông tin đúng chuẩn y khoa và các tuyên bố chính thống.

### B. Stance (Trục Quan điểm)
- `0` - **Ủng hộ**: Thể hiện sự tin tưởng, thúc đẩy hoặc đã thực hiện tiêm chủng.
- `1` - **Phản đối**: Thể hiện sự bài trừ, do dự hoặc khuyên không tiêm.
- `2` - **Trung lập**: Chỉ đặt câu hỏi hoặc chia sẻ thông tin khách quan.
- `3` - **Không rõ / Fallback Bucket**: Nhãn dự phòng cho các trường hợp LLM không thể suy luận rõ ràng.

### C. Sentiment (Trục Sắc thái)
- `0` - **Tiêu cực**: Giận dữ, lo lắng, sợ hãi hoặc mỉa mai.
- `1` - **Trung tính / Trung lập**: Không mang sắc thái cảm xúc rõ rệt.
- `2` - **Tích cực**: Hy vọng, vui mừng, cảm ơn.

## 3. Cấu trúc Schema (JSONL) có tính năng XAI

Mỗi hàng dữ liệu trong tập Silver/Gold được thiết kế để phục vụ huấn luyện **Explainable AI**:

```json
{
  "id": "uuid-v4",
  "text_cleaned": "Nội dung đã tiền xử lý...",
  "llm_reasoning": "Chuỗi lý luận nội bộ (Chain-of-Thought) trích xuất từ Teacher Model 31B.",
  "llm_parsed_labels": {
    "Misinformation": "...",
    "Stance": "...",
    "Sentiment": "..."
  }
}
```

> [!IMPORTANT]
> Việc chuẩn hóa ID này là bắt buộc để đảm bảo sự đồng bộ giữa tập **Silver Labels** (do LLM gán) và tập **Gold Labels** (do chuyên gia HITL duyệt).
