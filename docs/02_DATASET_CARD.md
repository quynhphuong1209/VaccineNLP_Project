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

### A. Misinformation (Trục Tin giả - 2 lớp)
- `0` - **Tin giả**: Thông tin sai lệch, thiếu cơ sở khoa học, gây hoang mang hoặc bóp méo sự thật về vắc-xin.
- `1` - **Chính xác**: Thông tin đúng chuẩn y khoa, chia sẻ kiến thức khoa học hoặc các tuyên bố chính thống từ tổ chức y tế.

### B. Stance (Trục Quan điểm - 3 lớp)
- `0` - **Ủng hộ**: Thể hiện thái độ đồng tình, tin tưởng, khuyên tiêm hoặc đã tiêm vắc-xin.
- `1` - **Phản đối**: Thể hiện thái độ bài trừ, từ chối, do dự lo ngại tác dụng phụ hoặc khuyên không nên tiêm.
- `2` - **Trung lập**: Chia sẻ thông tin khách quan, đặt câu hỏi thắc mắc thuần túy hoặc không bày tỏ quan điểm rõ ràng.

### C. Sentiment (Trục Sắc thái - 3 lớp)
- `0` - **Tiêu cực**: Giận dữ, lo sợ, mỉa mai, hoang mang.
- `1` - **Trung tính**: Không mang sắc thái cảm xúc rõ rệt, khách quan.
- `2` - **Tích cực**: Hy vọng, biết ơn, vui mừng.

## 3. Cấu trúc Schema (JSONL) có tính năng XAI

Mỗi hàng dữ liệu trong tập Silver/Gold được thiết kế để phục vụ huấn luyện **Explainable AI**:

```json
{
  "id": "uuid-v4",
  "text_cleaned": "Nội dung đã tiền xử lý...",
  "llm_reasoning": "Chuỗi lý luận nội bộ (XAI Reasoning) trích xuất từ LLM annotator 31B.",
  "llm_parsed_labels": {
    "Misinformation": "...",
    "Stance": "...",
    "Sentiment": "..."
  }
}
```

> [!IMPORTANT]
> Việc chuẩn hóa ID này là bắt buộc để đảm bảo sự đồng bộ giữa tập **Silver Labels** (do LLM gán) và tập **Gold Labels** (do chuyên gia HITL duyệt).

## 4. Thống kê Bộ dữ liệu (Dataset Statistics)

### A. Phân chia Tập dữ liệu
| Tập | File | Số mẫu | Vai trò |
|-----|------|:------:|---------|
| **Train** | `datasets/05_model_ready/train_v2_seg_v3.jsonl` | ~1,572 | Huấn luyện (Silver Labels) |
| **Gold Test** | `datasets/03_processed/benchmark_test_set.jsonl` | 186 | Đánh giá chuẩn mực (Human-validated) |
| **HITL Review** | `datasets/03_processed/benchmark_review_HITL.xlsx` | 186 | Bản duyệt gốc của chuyên gia |

### B. Phân bố Nhãn trên Gold Test Set (186 mẫu)

**Misinformation (2 lớp):**
| Nhãn | Số mẫu |
|------|:------:|
| Chính xác | 158 |
| Tin giả | 28 |

**Stance (3 lớp):**
| Nhãn | Số mẫu |
|------|:------:|
| Trung lập | 84 |
| Ủng hộ | 54 |
| Phản đối | 48 |

**Sentiment (3 lớp):**
| Nhãn | Số mẫu |
|------|:------:|
| Trung tính | 75 |
| Tiêu cực | 71 |
| Tích cực | 40 |

> [!NOTE]
> Nguồn: Support counts trích từ `experiments/results/phobert_v2_results.json` khớp hoàn toàn với nhãn chuẩn hóa Taxonomy v3 trong thực tế huấn luyện mô hình.

---

*Cập nhật: 21/05/2026 | Phiên bản 2.0*

