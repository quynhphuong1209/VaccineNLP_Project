<!-- TEAM-PRIVATE (gitignored). Kế hoạch dùng thử + kiểm thử VaccineNLP_Web (agent tự động + user UAT). Cập nhật 2026-06-13. -->

# TEST PLAN — Dùng thử & Kiểm thử VaccineNLP_Web (Pha 0-3)

> Mục tiêu: agent **tự động kiểm thử** các tính năng mới + checklist **UAT cho user**. Cách chạy hệ thống: xem [AGENT-GUIDE-VaccineNLP_Web.md](AGENT-GUIDE-VaccineNLP_Web.md) §3. Hợp đồng field: ARCHITECTURE §3.4.

## 0. Điều kiện & 3 tầng kiểm thử
| Tầng | Cần gì | Chạy được ở đâu |
|---|---|---|
| **T1 — Tín hiệu backend** (hàm thuần) | `.venv` + `data/*.json` + index (tự build) | Agent chạy thẳng, KHÔNG cần LM Studio/UI |
| **T2 — API HTTP** | api_service :8000 chạy (cần checkpoint PhoBERT) | Sau khi `uvicorn` api_service |
| **T3 — Giải thích + UI (UAT)** | full stack + LM Studio/Gemini + frontend :5173 | User bấm trên giao diện |

> ⚠️ Lưu ý import xai_service (evidence/anomaly): thêm `sys.path=[p for p in sys.path if p not in ('','.')]` trước import để né xung đột thư mục `datasets/` cục bộ.

---

## 1. T1 — KIỂM THỬ TỰ ĐỘNG (agent · backend) — bộ smoke đủ 4 trụ cột
Chạy từ gốc repo `D:\VaccineNLP_Clean_V1`:

```python
# === Pha 1 (kháng nhiễu ký tự) + Pha 3 (ngữ nghĩa) — api_service ===
PYTHONUTF8=1 python -c "import sys; sys.path.insert(0,'VaccineNLP_Web/api_service'); \
from app.main import phobert_infer as F; \
o=F('v4cc1n3 nước biến gen gây vô sinh'); c=F('Tiêm vaccine rất tốt cho trẻ'); \
print('obf nhiễu:', o['obfuscation']['level'], '| coded:', [h['variant'] for h in o['coded_language']]); \
print('obf sạch:', c['obfuscation']['level'], '| coded:', c['coded_language']); \
assert o['obfuscation']['level']=='high' and o['coded_language']; \
assert c['obfuscation']['level']=='none' and c['coded_language']==[]; \
print('>>> T1.1 Pha1+Pha3-semantic PASS')"

# === Pha 2 (RAG evidence) + Pha 3 Tầng-4 (anomaly) — xai_service ===
PYTHONUTF8=1 python -c "import sys; sys.path=[p for p in sys.path if p not in ('','.')]; sys.path.insert(0,'VaccineNLP_Web/xai_service'); \
from app.evidence import retrieve_evidence as R; from app.anomaly import embedding_anomaly as A; \
e=R('vaccine covid làm hiếm muộn vô sinh',3); \
print('evidence TOP:', e[0]['id'], round(e[0]['score'],3), '| #src:', len(e[0]['sources'])); \
print('anomaly vaccine:', A('vaccine có gây vô sinh không')['is_anomalous'], '| lạc đề:', A('giá xăng hôm nay tăng mạnh')['is_anomalous']); \
assert e[0]['id']=='myth_003' and e[0]['fact']; \
assert A('vaccine có gây vô sinh không')['is_anomalous']==False; \
assert A('giá xăng hôm nay tăng mạnh')['is_anomalous']==True; \
print('>>> T1.2 Pha2+Pha3-anomaly PASS')"

# === Regression kháng nhiễu (không phá tiếng Việt sạch) ===
.venv\Scripts\python -m pytest VaccineNLP_Web/api_service/tests/test_hardening.py -q
```

**Kỳ vọng:** `obf nhiễu=high`, `coded=['nước biến gen']`, `obf sạch=none`; evidence TOP=`myth_003`; anomaly vaccine=`False`/lạc đề=`True`; pytest `9 passed`.

---

## 2. T2 — KIỂM THỬ QUA API (cần api_service chạy)
```powershell
# Phân tích nhanh — kiểm field Pha 0/1/3
curl -s -X POST http://127.0.0.1:8000/api/analyze -H "Content-Type: application/json" `
  -d '{\"text\":\"v4cc1n3 nước biến gen gây vô sinh\"}' | python -m json.tool
```
**Kỳ vọng JSON có:** `display_label` (không bao giờ là "Chính xác"), `disclaimer`, `consistency_flag` (có thể `evasion_suspected`), `obfuscation.level`, `coded_language` (mảng có phần tử). `/api/explain-stream` (cần XAI backend) → event `final` chứa `evidence[]` + `anomaly`.

---

## 3. T3 — UAT CHO USER (bấm trên giao diện :5173)
| # | Nhập văn bản | Thao tác | KỲ VỌNG thấy | Pass? |
|---|---|---|---|---|
| U1 | `Tiêm vaccine sởi rất an toàn và cần thiết cho trẻ` | Phân tích | Nhãn **"Không phát hiện dấu hiệu sai lệch"** (KHÔNG có chữ "Chính xác") + dòng disclaimer | ☐ |
| U2 | `v4cc1n3 g4y v0 s1nh` (chèn số) | Phân tích | Badge ⚠️ **"Nghi ngờ lách bộ lọc (nhiễu ký tự)"** | ☐ |
| U3 | `bọn nó tiêm nước biến gen vào người` | Phân tích | Badge ⚠️ **"Phát hiện uyển ngữ/ngôn ngữ ngụy trang"** | ☐ |
| U4 | `vaccine covid gây vô sinh hiếm muộn` | Phân tích → **Sinh giải thích** | Panel **"🔍 Đối chiếu bằng chứng"** hiện lầm tưởng↔đính chính + nguồn WHO/CDC | ☐ |
| U5 | `hôm nay trời đẹp tôi đi chợ mua rau` | Phân tích → Sinh giải thích | Chip ℹ️ **"Văn bản lệch xa diễn ngôn vaccine đã biết"** | ☐ |
| U6 | (1 câu mơ hồ, độ tin cậy thấp) | Phân tích | Nhãn vùng **"Cần kiểm chứng"** (khi score<60%) | ☐ |
| U7 | Bất kỳ | Tắt xai_service rồi phân tích | Phần PhoBERT **vẫn hiển thị** (Error Boundary — không trắng trang) | ☐ |
| U8 | Tiếng Việt có dấu | Sinh giải thích (stream) | Chữ tiếng Việt **không bị mojibake** (lỗi font) | ☐ |

---

## 4. Bộ input mẫu theo trụ cột (tái dùng cho mọi tầng)
| Trụ cột | Input | Tín hiệu kỳ vọng |
|---|---|---|
| Pha 0 liêm chính | "Tiêm vaccine rất tốt" | display_label = "Không phát hiện dấu hiệu sai lệch" |
| Pha 1 nhiễu ký tự | "v4cc1n3 g4y v0 s1nh" | obfuscation.level=high, evasion badge |
| Pha 3 uyển ngữ | "tiêm nước biến gen", "vx" | coded_language có phần tử / vx→vaccine |
| Pha 2 RAG | "vaccine gây tự kỷ" → myth_001; "5G lan truyền corona" → myth_006 | evidence TOP đúng id |
| Pha 3 anomaly | "giá xăng tăng" | anomaly.is_anomalous=True |

---

## 5. Tiêu chí PASS toàn hệ
- T1 (3 lệnh) PASS hết → lõi 4 trụ cột hoạt động (không cần LM Studio).
- T2 trả đủ field §3.4, không lộ "Chính xác".
- T3: U1-U8 đều ✓ → trải nghiệm người dùng đạt.
- **Bất biến:** không mojibake (U8), không trắng trang khi xai chết (U7), nhãn liêm chính (U1).

> Agent tự động: chạy T1 (đủ để smoke không cần stack đầy đủ) → báo Claude output + assert. UAT (T3): user/Chủ tịch/Phương bấm theo bảng, đánh dấu ☐.
