# Bàn giao thiết kế: VaccineNLP — Frontend Redesign

> Gói tài liệu cho lập trình viên triển khai giao diện mới của hệ thống **VaccineNLP** (phát hiện tin giả & phân tích thái độ về vaccine) bằng tiếng Việt.

---

## 1. Tổng quan

VaccineNLP là nền tảng phân tích đa nhiệm văn bản tiếng Việt về vaccine, kết luận đồng thời trên **3 trục nhãn** và sinh **lý giải có thể giải thích (XAI)**. Bản thiết kế này thay thế giao diện cũ (dark dashboard nhiều emoji) bằng một hệ giao diện **chuẩn mực, hiện đại, sang trọng** phù hợp đối tượng là **nhà nghiên cứu, cán bộ quản lý y tế công cộng** — theo góp ý của thành viên hội đồng.

Định hướng thị giác: nền sáng làm mặc định (có dark mode), màu nhấn **teal y tế**, typography **Be Vietnam Pro**, line-icon thay emoji, ưu tiên hàng đầu cho **kết luận Tin giả/Tin thật**.

---

## 2. Về các file trong gói này

Các file trong gói là **bản tham chiếu thiết kế dựng bằng HTML** — prototype thể hiện *diện mạo* và *hành vi mong muốn*, **KHÔNG phải code production để copy trực tiếp**.

Frontend thật của dự án là **React + TypeScript + Vite + TailwindCSS v4** (`frontend/src/App.tsx`). Nhiệm vụ là **dựng lại các màn hình trong bản thiết kế HTML này thành component React** theo đúng kiến trúc và thư viện sẵn có của codebase, rồi nối vào API thật (mô tả ở mục 7). Bản HTML đóng vai trò **đặc tả trực quan (visual spec)** để code theo.

---

## 3. Mức độ hoàn thiện: **Hi-fi (pixel-level)**

Đây là mockup hi-fi với màu, typography, spacing, tương tác đã chốt. Lập trình viên nên **tái dựng pixel-perfect** bằng Tailwind, ánh xạ các token ở mục 6 thành biến CSS / `tailwind.config`.

---

## 4. Các màn hình

Ứng dụng dùng layout **sidebar trái (256px) + topbar dính + vùng nội dung trung tâm (max-width 1320px)**. Điều hướng client-side giữa 5 màn hình.

### 4.1 Phân tích văn bản (`analyze`) — màn hình chính
- **Mục đích**: người dùng dán văn bản (hoặc URL) → nhận kết luận 3 trục + lý giải XAI.
- **Layout**: grid 2 cột — cột nhập `minmax(0, 420px)` (dính `top:96px`), cột kết quả `1fr`, gap 24px. Dưới 1040px xếp dọc 1 cột.
- **Cột nhập** (card): textarea (min-height 148px) → hàng chip "Bộ ví dụ mẫu" (pill bo tròn) → select Mô hình → accordion "Hoặc thu thập từ URL" → nút primary full-width "Tiến hành phân tích đa nhiệm".
- **Cột kết quả** (xếp dọc, gap 20px):
  1. **Verdict hero** — khối kết luận lớn nhất, ưu tiên cao nhất (xem mục 5.1).
  2. **Card "Kết quả phân loại nhãn"** — 3 axis card ngang nhau (mục 5.2) + 2 ô viz (radar, phân phối `phobert_probs`) + **chú giải cờ nhất quán 3 trạng thái** (mục 5.3).
  3. **Card XAI** — pill `xai_status`, nút "Mô phỏng luồng trực tiếp", 2 tab: **Chain-of-Thought** (các bước suy luận + bảng bất đồng thuận PhoBERT vs Gemma) và **Token Attribution** (heatmap Captum).

### 4.2 Công cụ nâng cao (`advanced`)
- 2 tab: **Hàng loạt** (upload .txt/.csv hoặc dán nhiều dòng phân tách `---`, bảng kết quả + viz phân bố) và **So sánh mô hình** (PhoBERT-v2 vs XLM-R-v1, thanh điểm 3 trục).
- ⚠️ **Chưa có endpoint backend** — xem mục 7.3.

### 4.3 Benchmark hiệu năng (`benchmark`)
- Bảng xếp hạng F1 ba mô hình (PhoBERT-v2, Gemma-4-4B, XLM-R-v1) trên 3 trục, n=186 gold set + 2 ô viz (radar, confusion matrix).
- ⚠️ **Chưa có endpoint backend** — xem mục 7.3.

### 4.4 Tài liệu hệ thống (`docs`)
- Sơ đồ pipeline 4 bước (Đầu vào → PhoBERT-v2 → Nhất quán → Gemma-4B), thẻ mô hình, mô tả dữ liệu & nhãn, tài nguyên/trích dẫn.

### 4.5 Phương pháp luận (`method`)
- 4 bước phương pháp (phân loại đa nhiệm, kiểm tra nhất quán, XAI, đánh giá & minh bạch), max-width 840px căn giữa.

---

## 5. Component chủ chốt (chi tiết)

### 5.1 Verdict hero
- Khối flex ngang, padding 24–26px, `border-radius: 18px`, có dải màu 4px bên trái + `box-shadow` nổi.
- **Biến thể theo kết luận**:
  - *Tin giả*: viền & dải đỏ (`--danger #d2453a`), nền gradient `--danger-50`, glyph khiên đỏ, tiêu đề `Tin giả` 30px/700 màu `--danger-2`.
  - *Tin thật*: biến thể teal (class `.ok`), glyph check.
- Nội dung: nhãn nhỏ "KẾT LUẬN ĐỐI SOÁT" (11px uppercase) → kết luận 30px → dòng meta "Độ tin cậy mô hình **91.4%**". Bên phải: pill trạng thái + mã phiên (`#1287 · PhoBERT-v2`).
- **Map dữ liệu**: kết luận = `misinfo_label`; % = `misinfo_score`; pill phải = `consistency_flag` (mục 5.3).

### 5.2 Axis card (×3)
- Padding 18px, `border-radius: 13px`, nền `--surface-2`, viền `--line`.
- Cấu trúc: caption uppercase (Tính xác thực / Lập trường / Cảm xúc) → giá trị nhãn 19px/700 (đỏ nếu xấu, teal nếu tốt, ink nếu trung tính) → **meter** (thanh 7px bo tròn, fill teal hoặc đỏ, width = score) → dòng `scoreline` (ngưỡng/các lớp ↔ % mono).
- **Map dữ liệu**: lần lượt `misinfo_*`, `stance_*`, `sentiment_*` (label + score). Width meter = `score * 100%`.

### 5.3 Cờ nhất quán (`consistency_flag`) — 3 trạng thái
Khớp đúng hàm `compute_consistency` ở backend (`api_service/app/main.py`):
| Giá trị | Hiển thị | Màu | Ý nghĩa |
|---|---|---|---|
| `plausible` | "hợp lệ" | teal (pill ok) | tổ hợp nhãn bình thường |
| `unusual` | "bất thường (nghi mô hình sai)" | vàng (pill warn) | tổ hợp hiếm theo H1/H3 (vd Against+Positive, hoặc Fake+Favor/Neutral) |
| `high_risk` | "nguy cơ cao — nên rà soát" | đỏ (pill danger) | profile chống-vaccine (Against+Negative) |
Verdict hero **phải** đổi pill phải theo đúng 3 giá trị này (hiện mockup minh hoạ `high_risk`).

### 5.4 Card XAI — luồng UX 2 nhịp / streaming
Đây là tính năng **đặc trưng nhất** của hệ thống, cần làm đúng:
- **Nhịp 1 (tức thời)**: ngay khi có response `type:"phobert"` từ `/api/analyze-stream` → render verdict + axis + cờ nhất quán; pill `xai_status: pending`.
- **Nhịp 2 (streaming)**: với mỗi event `type:"token"` → **append từng token** vào vùng `.streamwrap`, hiện con trỏ nhấp nháy (`.cursor`). Khi nhận `type:"final"` → ẩn vùng streaming, render các **bước CoT có cấu trúc** + **bảng bất đồng thuận** (từ `final.disagreement` & `final.gemma_labels`); pill chuyển `xai_status: done`.
- Mockup mô phỏng việc này bằng nút "Mô phỏng luồng trực tiếp" + hàm `streamCoT` (xem `<script>` cuối file HTML) — đây chỉ là **demo client**, bản thật phải đọc SSE.
- Tab **Token Attribution**: heatmap token — nền token tô đậm dần theo điểm đóng góp Captum (thấp `--surface-2` → cao `--danger`). Hiện mockup hardcode; bản thật cần endpoint attribution (chưa có — mục 7.3).

### 5.5 Bảng bất đồng thuận (disagreement)
- 4 cột: Trục | PhoBERT-v2 | Gemma-4-4B | Khớp (✓ / ≠). Hàng lệch tô nền `--warn-50` (class `.flag`).
- **Map dữ liệu**: PhoBERT = nhãn của 3 trục; Gemma = `final.gemma_labels`; cột Khớp = phủ định `final.disagreement[trục]` (true = lệch).

---

## 6. Design tokens (giá trị chính xác)

Khai báo dưới dạng CSS custom properties trong file; nên ánh xạ vào `tailwind.config.js` (theme.extend).

### Màu — Light (mặc định)
| Token | Hex/giá trị | Dùng cho |
|---|---|---|
| `--bg` | `#f4f6f6` | nền trang |
| `--bg-2` | `#edf0ef` | nền phụ/hover |
| `--surface` | `#ffffff` | card |
| `--surface-2` | `#fafbfb` | card lồng/input |
| `--ink` | `#15201e` | chữ chính |
| `--ink-2` | `#54625e` | chữ phụ |
| `--ink-3` | `#8a9692` | chữ mờ/caption |
| `--line` | `#e5e9e8` | viền |
| `--line-2` | `#eef1f0` | viền nhạt |
| `--teal` | `#0e9384` | nhấn chính |
| `--teal-strong` | `#0b6a60` | nhấn đậm/chữ teal |
| `--teal-50` | `#eaf6f3` | nền nhấn nhạt |
| `--teal-100` | `#d4ece7` | viền/nền nhấn |
| `--danger` | `#d2453a` | tin giả/cảnh báo |
| `--danger-2` | `#b23a31` | chữ tin giả |
| `--danger-50` | `#fbece9` | nền cảnh báo |
| `--warn` | `#b67d1c` | bất thường |
| `--warn-50` | `#f9f1de` | nền bất thường |

### Màu — Dark (`[data-theme="dark"]`)
| Token | giá trị |
|---|---|
| `--bg` `#0a1211` · `--bg-2` `#0d1816` · `--surface` `#101d1b` · `--surface-2` `#13201d` |
| `--ink` `#e9efed` · `--ink-2` `#9db0ab` · `--ink-3` `#687a75` |
| `--line` `rgba(255,255,255,.085)` · `--line-2` `rgba(255,255,255,.05)` |
| `--teal` `#2bcfba` · `--teal-strong` `#63e0cf` · `--teal-50` `rgba(43,207,186,.10)` · `--teal-100` `rgba(43,207,186,.18)` |
| `--danger` `#f0786c` · `--danger-2` `#f4938a` · `--danger-50` `rgba(240,120,108,.12)` |
| `--warn` `#e2b563` · `--warn-50` `rgba(226,181,99,.12)` |

### Typography
- **UI**: `'Be Vietnam Pro'` (weights 300/400/500/600/700), fallback `system-ui, sans-serif`. Body 15px / line-height 1.5.
- **Mono** (số liệu, tên trường, mã): `'JetBrains Mono'` 400/500/600, `font-variant-numeric: tabular-nums`.
- Thang cỡ chữ: caption 11–12px · body 13.5–15px · nhãn axis 19px · verdict 30px · h1 topbar 20px.
- `letter-spacing` tiêu đề âm nhẹ (−.2 đến −.6px); caption uppercase `+.5px`.

### Bo góc / bóng / spacing
- Radius: `--r-sm 9px` · `--r 13px` · `--r-lg 18px` · pill `99px`.
- Shadow: `--shadow-sm 0 1px 2px rgba(16,32,30,.05), 0 2px 6px rgba(16,32,30,.04)` · `--shadow-md 0 16px 40px -22px rgba(16,32,30,.28)` · `--shadow-lg ...`.
- Spacing card padding 22px; gap cột 24px; gap nội bộ 16–20px.
- Meter: cao 7px, bo `99px`, transition `width 1s cubic-bezier(.2,.8,.2,1)`.
- Animation vào màn hình: `rise .42s cubic-bezier(.2,.7,.3,1)` (opacity + translateY 10px). **Lưu ý**: dùng end-state làm base, animate *từ* trạng thái ẩn để print/reduced-motion vẫn thấy nội dung.

---

## 7. Hợp đồng API (từ `api_service/app/main.py`)

Base URL mặc định: `http://localhost:8000`. CORS chỉ mở cho `http://localhost:5173`.

### 7.1 Taxonomy v3 (nhãn)
| Trục | Nhãn |
|---|---|
| `misinfo` | `Fake`, `Real` |
| `stance` | `Favor`, `Against`, `Neutral` |
| `sentiment` | `Negative`, `Neutral`, `Positive` |

### 7.2 Endpoint đã có
- **`POST /api/analyze`** → body `{ text, source_url? }` → trả `AnalysisResponse`:
  `{ id, misinfo_label, misinfo_score, stance_label, stance_score, sentiment_label, sentiment_score, phobert_probs, consistency_flag, xai_status, xai_explanation }`. `phobert_probs` = softmax đầy đủ từng lớp, vd `{ misinfo:{Fake,Real}, stance:{Favor,Against,Neutral}, sentiment:{...} }` → dùng cho ô viz "Phân phối softmax".
- **`POST /api/analyze-stream`** (SSE, **luồng chính cho màn Phân tích**) → body `{ text, source_url? }` → stream sự kiện:
  - `type:"phobert"` (tức thời): `{ id, misinfo_label, misinfo_score, stance_*, sentiment_*, probs, consistency_flag, xai_status:"pending" }`
  - `type:"token"`: `{ content }` — từng mẩu CoT, append liên tục.
  - `type:"final"`: `{ ...gemma_labels, reasoning, disagreement }` (`disagreement[trục]` = bool lệch).
  - `type:"error"`: `{ message }`. Kết thúc bằng `data: [DONE]`.
- **`POST /api/explain-stream`** (SSE) → sinh XAI on-demand cho 1 văn bản (tái dùng nhãn PhoBERT đã cache).
- **`GET /api/analysis/{id}`** → lấy lại 1 bản ghi lịch sử.
- **`GET /health`** & **`GET /`** → trạng thái service + `model_loaded`.

### 7.3 ⚠️ Endpoint CHƯA có (cần bổ sung nếu triển khai)
Backend hiện **không** có endpoint cho:
- **Phân tích hàng loạt** (màn 4.2 tab Hàng loạt) — cần endpoint nhận nhiều dòng/file.
- **So sánh mô hình** (màn 4.2 tab So sánh) — cần endpoint chạy >1 model trên cùng input.
- **Benchmark** (màn 4.3) — cần endpoint trả F1/confusion matrix trên gold set.
- **Token attribution / Captum** (tab Token Attribution) — cần endpoint trả điểm đóng góp token.

→ Các màn/khối này hiện là **đề xuất tính năng**, dữ liệu trong mockup là minh hoạ. Trao đổi với team backend trước khi build, hoặc ẩn cho tới khi có API.

---

## 8. Trạng thái & quản lý state (gợi ý)

- `activeScreen`: `'analyze' | 'advanced' | 'benchmark' | 'docs' | 'method'`.
- `theme`: `'light' | 'dark'` (nên lưu `localStorage`, set `data-theme` trên `<html>`).
- Màn analyze: `inputText`, `sourceUrl`, `model`, `analysis` (kết quả phobert), `xaiStatus` (`idle|pending|done|failed`), `cotTokens` (chuỗi tích luỹ khi stream), `disagreement`.
- Luồng stream: mở `EventSource`/`fetch`+ReadableStream tới `/api/analyze-stream`, phân nhánh theo `type` (mục 7.2). Hiển thị con trỏ nhấp nháy trong lúc `pending`.
- Caching: backend cache theo `text_hash` (md5) — không cần xử lý phía client, nhưng response có thể trả ngay từ cache.

---

## 9. Assets & icon

- **Icon**: bộ line-icon SVG tự vẽ (sprite `<symbol>` trong `<head>` của file HTML), stroke-width 1.6–1.7, không dùng emoji. ID: `i-analyze, i-advanced, i-bench, i-docs, i-method, i-shield, i-check, i-download, i-volume, i-link, i-spark, i-arrow, i-send, i-sun, i-upload, i-scale, i-data, i-chevron`. Có thể thay bằng thư viện tương đương (Lucide…) trong codebase.
- **Font**: Google Fonts — Be Vietnam Pro + JetBrains Mono.
- Không dùng ảnh bitmap; các ô "viz" là placeholder — thay bằng thư viện chart thật (Recharts/Chart.js) khi tích hợp.

---

## 10. File trong gói

- `VaccineNLP Redesign.html` — bản thiết kế hi-fi đầy đủ 5 màn hình, light/dark, demo streaming. Mở trực tiếp trên trình duyệt để xem tương tác. Toàn bộ CSS token nằm trong `<style>` ở `<head>`; logic màn hình + streaming nằm trong `<script>` ở cuối `<body>`.

---

*Tài liệu này tự đủ — một lập trình viên không tham gia hội thoại vẫn có thể triển khai từ README này. Mọi giá trị màu/cỡ chữ lấy trực tiếp từ file thiết kế.*
