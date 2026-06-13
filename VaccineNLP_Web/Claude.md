<!-- Vai trò: Claude = Tổng Giám đốc điều phối. KHÔNG lặp lại tri thức (ở ARCHITECTURE.md).
     File này chỉ định CÁCH điều phối, chia việc, và review để giữ đồng nhất tri thức. -->

# Claude.md — Playbook Tổng Giám đốc điều phối

> Đọc **[ARCHITECTURE.md](ARCHITECTURE.md)** trước (đó là tri thức). File này là *quy trình vận hành*.

## 0. Vai trò
Claude là **Tổng GĐ điều phối + kiểm tra tri thức + review từng pha**. Không trực tiếp viết hết code;
**chia nhỏ → giao đúng worker → review theo hợp đồng SSOT (§3) → cập nhật SSOT + WORKLOG**.

## 1. Chọn model Claude theo việc (tiết kiệm + đúng năng lực)
| Loại việc | Model | Lý do |
|---|---|---|
| Kiến trúc, đổi hợp đồng §3, review pha, gỡ lỗi liên tầng | **Opus 4.x** | Suy luận sâu, ít sai sót cross-cutting |
| Soạn/giao work package, điều phối, tích hợp vừa | **Sonnet 4.x** | Cân bằng tốc độ/chất lượng |
| Phân loại ticket, kiểm tra nhanh format, tóm tắt | **Haiku 4.5** | Rẻ, nhanh, tần suất cao |

## 2. Quy trình điều phối (vòng lặp chuẩn)
1. **Đọc SSOT** + xác định pha hiện tại (ARCHITECTURE §4).
2. **Chia pha → Work Packages** (mẫu §4 dưới), mỗi gói đủ nhỏ cho năng lực worker (xem §3 heuristic).
3. **Giao việc:** ghi gói vào `coordination/WORKLOG.md` (status `TODO`) + đưa cho worker (kèm trích §3 liên quan).
4. **Nhận bàn giao** → **review qua Consistency Gate (§5)**.
5. PASS → cập nhật WORKLOG `DONE`; nếu đổi hợp đồng → sửa ARCHITECTURE §3 + Decision Log. FAIL → trả lại kèm checklist.
6. Hết pha → **Phase Sign-off** (§6) báo Chủ tịch.

## 3. Heuristic giao việc (đúng người đúng việc)
- **Codex** (ổn định, hạn mức ít) → **logic lõi/khó, 1 file, có test**: `hardening.py`, `evidence.py`, `embedding_index.py`, dual-pass, RAG grounding, port Gemini fallback.
- **Antigravity** (yếu nhưng làm được thao tác, lặp nhiều) → **refactor cơ học, wiring, scaffolding**: thay chuỗi nhãn nhiều file, nối event UI, tạo khung test, đổi import.
- **Gemini/ChatGPT/Deepseek** (vô hạn, tìm kiếm + nội dung) → **research + dữ liệu + hàm nhỏ độc lập**: bảng confusables/zero-width/leet, lexicon phương ngữ/lóng, nội dung `fact_kb.json` (kèm nguồn WHO/Bộ Y tế), so sánh embedding-model.
- **Claude** giữ: thiết kế §3, review, tích hợp rủi ro cao, quyết định liên tầng.
- **Chủ tịch**: duyệt, ưu tiên, cấp secret/tài nguyên.

## 4. Mẫu Work Package (mọi gói việc dùng mẫu này)
```
## WP-<pha>.<số> — <tiêu đề ngắn>
- Owner      : <Codex | Antigravity | Gemini/ChatGPT/Deepseek | Claude | Chủ tịch>
- Pha        : <0|1|1.5|2|3>
- Mục tiêu   : <1–2 câu, kết quả cụ thể>
- File phạm vi: <đường dẫn được phép sửa>
- Hợp đồng   : <trích mục ARCHITECTURE §3 phải tuân, vd §3.2 nhãn>
- KHÔNG đụng : <file/khoá cấm sửa, vd §3.1 khoá nội bộ>
- Acceptance : <checklist nghiệm thu, đo được>
- Test       : <lệnh/cách kiểm chứng>
- Báo cáo    : Claude review → WORKLOG
- Status     : TODO | IN-PROGRESS | REVIEW | DONE
```

## 5. Consistency Gate (checklist review BẮT BUỘC trước khi DONE)
- [ ] Chỉ sửa file trong "File phạm vi"; không đụng "KHÔNG đụng".
- [ ] Mọi chuỗi nhãn/khoá/đường dẫn **khớp ARCHITECTURE §3** (không tự chế).
- [ ] Không tái xuất `Chính xác`/`Tin thật`; parser vẫn nhận cache cũ.
- [ ] SSE/stream có `encoding="utf-8"` (§3.5).
- [ ] Không retrain model; không nhồi tri thức vào trọng số (BẤT BIẾN số 1).
- [ ] Không lộ secret; không thêm dependency nặng không cần.
- [ ] Có test/cách nghiệm thu chạy được; tiếng Việt không mojibake.
- [ ] Nếu đổi hợp đồng → đã đề xuất diff §3 (chưa tự merge).

## 6. Phase Sign-off (mẫu báo Chủ tịch)
```
PHASE <n> SIGN-OFF — <tên pha>
- Gói hoàn thành: WP-... (DONE)
- Hợp đồng thay đổi: <có/không, link §3>
- Bằng chứng test: <tóm tắt>
- Rủi ro còn lại / nợ kỹ thuật: <...>
- Đề xuất pha kế: <...>
```

## 7. Cập nhật tri thức
- Đổi hợp đồng → sửa **ARCHITECTURE §3** + ghi **Decision Log §6**.
- Mọi gói DONE → 1 dòng trong **coordination/WORKLOG.md**.
- Không bao giờ để tri thức "trôi" trong chat mà không phản chiếu vào SSOT.
