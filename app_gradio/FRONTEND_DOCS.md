# 🖥️ VaccineNLP — Tài liệu Front-End

> **File gốc:** `app_gradio/app.py` & `app_gradio/gradio_app.py`  
> **Framework:** [Gradio](https://gradio.app/) ≥ 4.x  
> **Deploy target:** HuggingFace Spaces (CPU Basic 16 GB)  
> **Version:** v2.0 (Gradio migration từ Streamlit)

---

## 📁 Cấu trúc thư mục

```
app_gradio/
├── app.py              # Bản chính — entry point khi chạy local
├── gradio_app.py       # Bản mirror — dùng cho HuggingFace Spaces deploy
├── xai_postprocess.py  # Hàm hậu xử lý XAI (render HTML, parse label…)
├── fetchers.py         # Multi-source fetcher (News / YouTube / Apify)
├── thread_parser.py    # Parser luồng hội thoại từ mạng xã hội
├── requirements.txt    # Python dependencies
└── data/               # JSON cache (XAI cache, benchmark results, temperature params)
```

> **Lưu ý:** `app.py` và `gradio_app.py` có nội dung gần như đồng nhất.  
> Mọi thay đổi CSS/layout cần áp dụng song song cho cả hai file.

---

## 🧱 Kiến trúc tổng thể

```
build_app()
│
├── gr.Blocks(css=CSS_STYLE, theme=gr.themes.Default())
│   │
│   ├── [JavaScript] — theme toggle, sidebar collapse
│   │
│   ├── Header HTML        (logo, badge HUPH, info)
│   ├── gr.Row #main-layout-row
│   │   ├── gr.Column #sidebar-col   ← Sidebar điều khiển (290px)
│   │   └── gr.Column #content-col  ← Nội dung chính (flex-grow)
│   │       ├── Hero Banner
│   │       └── gr.Tabs()           ← 5 Tab chính
│   │           ├── 🔍 PHÂN TÍCH VĂN BẢN
│   │           ├── 🔧 CÔNG CỤ NÂNG CAO  ← Batch Mode + So sánh
│   │           ├── 📊 BENCHMARK & BÁO CÁO KHOA HỌC
│   │           ├── 📚 TÀI LIỆU & NOTEBOOKS
│   │           └── 📜 PHƯƠNG PHÁP LUẬN
│   │
│   └── Footer HTML
│
└── Event handlers (Gradio `.change()`, `.click()`, `.submit()`)
```

> **Thay đổi v2.1:** Batch Mode và So sánh mô hình được tách khỏi Tab 1 vào tab riêng **🔧 CÔNG CỤ NÂNG CAO** để giữ Tab 1 gọn gàng. XAI section và biểu đồ ẩn đến khi phân tích xong.

---

## 🎨 Hệ thống CSS (`CSS_STYLE`)

Toàn bộ CSS được nhúng dưới dạng chuỗi Python (`CSS_STYLE = """..."""`) và truyền vào tham số `css=` của `gr.Blocks`.

### Design Tokens — CSS Custom Properties

| Token | Light Mode | Dark Mode |
|---|---|---|
| `--bg-color` | `#ffffff` | `#04091a` |
| `--bg-gradient` | trắng thuần | `linear-gradient(160deg, #04091a → #070e1a → #050d1f)` |
| `--text-color` | `#000000` | `#ccd6f6` |
| `--accent-color` | `#00b894` | `#00d4aa` |
| `--accent-bright` | `#00d4aa` | `#00ffcc` |
| `--card-bg` | `#ffffff` | `rgba(8,18,40,0.75)` |
| `--card-border` | `#e2e8f0` | `rgba(0,212,170,0.18)` |
| `--input-bg` | `#ffffff` | `rgba(10,20,45,0.85)` |
| `--sidebar-bg` | `#ffffff` | `linear-gradient(180deg, #050f1f → #04091a)` |
| `--shadow-color` | `rgba(0,0,0,0.06)` | `rgba(0,0,0,0.55)` |

### Theme Toggle — Light / Dark

Cơ chế: JavaScript thuần thêm/xoá class `.dark` trên `document.body` + `document.documentElement`.

```js
// Kích hoạt Dark Mode
document.body.classList.add('dark');
document.documentElement.classList.add('dark');
localStorage.setItem('vaccinenlp-theme', 'dark');
```

CSS áp dụng rule `:root.dark, body.dark, .dark { ... }` để override tất cả custom properties.

### Typography

```css
* { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
code, pre { font-family: 'Fira Code', 'JetBrains Mono', Consolas, monospace !important; }
```

Scale:
- `h1` → `2.3rem / 800`  
- `h2` → `1.75rem / 700`  
- `h3` → `1.38rem / 700`  
- body → `1.06rem / line-height 1.65`

### Keyframe Animations

| Tên | Mô tả |
|---|---|
| `shimmer` | Gradient quét từ trái sang phải (hero banner line, divider) |
| `pulse-glow` | Neon glow nhấp nháy (dark mode accents) |
| `float` | Lên xuống nhẹ 6px (hero emoji) |
| `fadeInUp` | Fade + slide từ dưới lên (result cards) |
| `dropdownFadeIn` | Dropdown mở mượt từ trên xuống |

---

## 📐 Layout

### Sidebar (`#sidebar-col`)

```css
#sidebar-col {
    width: 290px; min-width: 290px; max-width: 290px;
    position: sticky; top: 0; height: 100vh;
    overflow-y: auto; overflow-x: hidden;
    z-index: 9999;
}
```

**Trạng thái collapsed:**
```css
#sidebar-col.collapsed {
    width: 0px; padding: 0px; opacity: 0;
    transform: translateX(-290px);
    pointer-events: none;
}
```

**Nút toggle:** `#sidebar-toggle-btn` — fixed position, dùng SVG mask icon (chevron), xoay 180° khi collapsed.

**Mobile (≤768px):** Sidebar chuyển sang `position: fixed`, mặc định ẩn (`transform: translateX(-290px)`), hiện lên khi click toggle.

### Content Area (`#content-col`)

```css
#content-col {
    flex: 1 1 auto;
    padding-top: 50px; padding-left/right: 20px;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

Khi sidebar mở: `width: calc(100% - 290px)`  
Khi sidebar đóng: `width: 100%`

---

## 🧩 Các thành phần giao diện chính

### 1. Header

HTML tĩnh được render qua `gr.HTML(get_header_html())`. Chứa:
- Logo / tên dự án
- Badge HUPH + thông tin sinh viên
- Nút theme toggle (Light ☀️ / Dark 🌙) với `<input type="checkbox" id="theme-toggle-switch">`

### 2. Hero Banner (`#hero-banner`)

```python
gr.HTML(get_hero_html())
```

Bao gồm:
- `.hero-accent-line` — đường kẻ gradient shimmer phía trên
- `.hero-emojis` — icon nổi (animation float)
- `.hero-title` — tiêu đề responsive (`clamp(1.6rem, 3.5vw, 2.6rem)`)
- `.hero-divider` — thanh phân cách gradient
- `.hero-subtitle` — mô tả ngắn

### 3. Sidebar — Các điều khiển

| Thành phần | Gradio Component | Nội dung |
|---|---|---|
| Theme toggle | `gr.HTML` + JS | Light/Dark switch |
| Mẫu thử nghiệm | `gr.Dropdown` | Chọn nhóm mẫu (5 nhóm) |
| Loại văn bản | `gr.Dropdown` | Visible khi chọn nhóm cụ thể |
| Mô hình phân loại | `gr.Dropdown` | PhoBERT-v2 / XLM-R-v1 |
| Thông tin model | `gr.HTML` | Card info tự cập nhật |
| Xoá cache | `gr.Button` | Restart hệ thống |

### 4. Tab PHÂN TÍCH VĂN BẢN

Đây là tab chính, chứa toàn bộ luồng phân tích:

```
📝 Nhập văn bản
│  ├── gr.Textbox (multiline)
│  ├── Nút ⚡ Phân tích (primary)
│  ├── gr.Examples (4 mẫu nhanh)
│  └── Fetch từ URL (News/YouTube/Apify)
│
📊 Kết quả phân loại
│  └── gr.HTML (render_result_cards_html)
│      ├── Card: Tính xác thực (Tin giả / Tin thật)
│      ├── Card: Quan điểm (Ủng hộ/Phản đối/Trung lập)
│      └── Card: Cảm xúc (Tích cực/Tiêu cực/Trung tính)
│
🎯 Radar — độ tin cậy nhãn dự đoán
│  └── gr.Plot (Plotly radar chart)
│
📈 Phân phối xác suất đầy đủ (chuẩn hóa)
│  └── gr.Plot (Plotly bar chart)
│
🧠 Giải thích AI (XAI 3-Layer Engine)
   ├── gr.Tab: Chain-of-Thought Reasoning
   │   └── gr.HTML (reasoning HTML)
   └── gr.Tab: Token Attribution (Captum IG)
       ├── gr.Checkbox (opt-in Captum)
       └── gr.HTML (saliency map)
```

**Nhãn phân loại (`LABEL_MAPS`):**

```python
LABEL_MAPS = {
    "misinfo":   {0: "Tin giả",   1: "Tin thật"},
    "stance":    {0: "Ủng hộ",    1: "Phản đối", 2: "Trung lập"},
    "sentiment": {0: "Tiêu cực",  1: "Trung tính", 2: "Tích cực"},
}
```

### 5. Tab BENCHMARK & BÁO CÁO KHOA HỌC

Chứa các sub-tab:

| Sub-tab | Nội dung |
|---|---|
| 📋 BÁO CÁO BENCHMARK | Bảng so sánh Macro F1 (HTML tĩnh) |
| ⚡ ĐÁNH GIÁ LIVE | Radar chart, bar chart, per-class breakdown theo model |

Dữ liệu từ: `DATA_DIR/benchmark_results.json`

### 6. Tab TÀI LIỆU & NOTEBOOKS

HTML tĩnh (`RESOURCES_HTML`) — 2 cột:
- Kim Mạnh Hưng (Kaggle + HuggingFace + GitHub)
- Đinh Lê Quỳnh Phương (Kaggle + HuggingFace + GitHub)

### 7. Tab PHƯƠNG PHÁP LUẬN

HTML tĩnh (`METHODOLOGY_HTML`):
- Kiến trúc Dual-Student Hybrid (PhoBERT + Gemma)
- Pipeline xử lý (ASCII art)
- 3 nhiệm vụ chính
- Quy trình thực nghiệm

---

## 📊 Visualisation (Plotly)

Tất cả biểu đồ dùng **Plotly** và trả về qua `gr.Plot`.

| Biểu đồ | Hàm tạo | Loại |
|---|---|---|
| Radar confidence | `make_radar_chart()` | `go.Scatterpolar` |
| Probability bar | `make_prob_bar_chart()` | `go.Bar` |
| Benchmark radar | `make_benchmark_radar()` | `go.Scatterpolar` |
| Per-class bar | `make_per_class_chart()` | `go.Bar` + subplots |
| Confusion matrix | Plotly heatmap | `go.Heatmap` |

**CSS override cho Plotly:**
```css
.js-plotly-plot { background-color: transparent !important; }
.js-plotly-plot text { fill: var(--text-color) !important; }
.dark .js-plotly-plot text { fill: #ffffff !important; }
```

---

## 🎛️ Dropdown — Ghi chú CSS

Dropdown Gradio sử dụng class nội bộ (`svelte-*`), cần override bằng nhiều selector:

```css
/* Container */
.gradio-dropdown, .select-wrap { font-size: 0.72rem; }

/* Danh sách gợi ý */
.gradio-container .options .option,
ul.options > li,
.options li {
    padding: 3px 4px 3px 6px;
    font-size: 0.70rem;
    line-height: 1.2;
}

/* Hover */
.gradio-container .options .option:hover {
    background-color: rgba(0, 212, 170, 0.1);
    color: var(--accent-color);
    padding-left: 19px; /* slide right effect */
}

/* Selected */
.gradio-container .options .option.selected {
    background: linear-gradient(135deg, #00d4aa, #00b894);
    color: #ffffff;
    font-weight: 600;
}
```

---

## ⚙️ Event Handlers — Mapping

| Trigger | Handler | Outputs |
|---|---|---|
| `analyze_btn.click` | `handle_analyze()` | result cards, radar, bar chart, XAI |
| `sample_category.change` | `handle_category_change()` | `sample_detail` dropdown |
| `sample_detail.change` | `handle_sample_select()` | `text_input` |
| `model_choice.change` | `handle_model_change()` | `info_box` |
| `fetch_btn.click` | `handle_fetch()` | `text_input`, `fetch_status` |
| `clear_cache_btn.click` | `handle_clear_cache()` | `clear_cache_status` |
| `export_btn.click` | `handle_export()` | File download |
| `selected_model_view.change` | `update_benchmark_view()` | All benchmark plots |

---

## 🌐 JavaScript — Các chức năng

JavaScript được nhúng qua `gr.HTML("""<script>...</script>""")` hoặc tham số `js=` của `gr.Blocks`.

### Theme Toggle
```js
function applyTheme(theme) {
    if (theme === 'dark') {
        document.body.classList.add('dark');
        document.documentElement.classList.add('dark');
    } else {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('vaccinenlp-theme', theme);
}
// Khôi phục từ localStorage khi load trang
const savedTheme = localStorage.getItem('vaccinenlp-theme') || 'light';
applyTheme(savedTheme);
```

### Sidebar Toggle
```js
document.getElementById('sidebar-toggle-btn').addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar-col');
    const btn = document.getElementById('sidebar-toggle-btn');
    sidebar.classList.toggle('collapsed');
    btn.classList.toggle('sidebar-is-collapsed');
});
```

---

## 📱 Responsive Design

| Breakpoint | Thay đổi |
|---|---|
| `≤ 768px` | Sidebar `position: fixed`, ẩn mặc định; content `padding: 8px`; tab font `0.76rem` |
| `> 768px` | Sidebar sticky 290px; layout 2 cột flex |

---

## 🚀 Chạy local

```bash
# Cài dependencies
pip install -r app_gradio/requirements.txt

# Chạy app (hot-reload tự động)
python app_gradio/app.py
# → http://127.0.0.1:7860
```

**Biến môi trường cần thiết:**

| Biến | Mục đích |
|---|---|
| `HF_TOKEN` | HuggingFace Inference API |
| `GEMINI_API_KEY` | Gemini XAI fallback |
| `OPENROUTER_KEY` | OpenRouter fallback |
| `APIFY_TOKEN` | Thu thập dữ liệu MXH |
| `GEMMA_ENDPOINT_URL` | Kaggle+ngrok endpoint |
| `LM_STUDIO_URL` | LM Studio local server |

---

## 📝 Lưu ý bảo trì

1. **Sửa CSS:** Luôn cập nhật **cả hai** `app.py` và `gradio_app.py` — chúng không import lẫn nhau.
2. **Thêm tab mới:** Thêm vào `build_app()` dưới `with gr.Tabs():` và đăng ký event handler tương ứng.
3. **Nhãn phân loại:** Sửa `LABEL_MAPS` và cập nhật đồng thời tất cả hàm render HTML, biểu đồ, benchmark.
4. **Plotly dark mode:** Khi thêm biểu đồ, luôn truyền `paper_bgcolor="rgba(0,0,0,0)"` và `plot_bgcolor="rgba(0,0,0,0)"` để tương thích CSS.
5. **Z-index:** Sidebar = `9999`, Toggle button = `10001`, Dropdown = `999999`.

---

## 💡 Gợi ý cải thiện

---

## ⚠️ Vấn đề & Rủi ro bảo trì

### 1. Hai file app gần như đồng nhất

`app.py` và `gradio_app.py` chứa nội dung gần như giống hệt nhau. Mọi thay đổi CSS/layout phải áp dụng song song cho cả hai:

- Dễ sửa một file nhưng quên file còn lại → lệch layout giữa local và HF Spaces.
- Khó review vì cùng logic bị nhân đôi.
- **Giải pháp:** Tách `CSS_STYLE`, `LABEL_MAPS`, `LABEL_ICONS`, `LABEL_COLORS` vào `shared_styles.py` rồi import.

### 2. CSS nhúng trong Python khó mở rộng

`CSS_STYLE` là chuỗi Python ~2900 dòng. Nhược điểm:

- Không lint/format được như CSS thật.
- Khó tái sử dụng token giữa các component.
- Ngày càng dài → khó kiểm soát.

**Giải pháp:** Tách ra `styles.css`, đọc bằng `open("styles.css").read()`.

### 3. Phụ thuộc selector nội bộ Gradio

Dropdown override dùng `.svelte-*`, `.options`, `.select-wrap` — những class này **có thể đổi khi upgrade Gradio**.

- Sau mỗi `pip install gradio --upgrade`, kiểm tra DevTools để confirm selector còn đúng không.
- Xem bảng upgrade checklist bên dưới.

### 4. Z-index rất cao

| Phần tử | Z-index |
|---|---|
| Sidebar | `9999` |
| Toggle button | `10001` |
| Dropdown | `999999` |

Các giá trị này giải quyết va chạm trước mắt nhưng sẽ tạo vấn đề khi thêm modal, toast, tooltip. Nên chuẩn hóa bằng CSS variables (xem mục bên dưới).

### 5. Thiếu mô tả trạng thái async 2 nhịp

Flow hiện tại là blocking — kết quả phân loại và XAI trả về cùng lúc. Nếu backend được tách thành 2 phase (phân loại nhanh → XAI sau), cần cập nhật:
- Contract request/response.
- Polling hoặc WebSocket.
- Timeout/error state của XAI.
- Stale request protection.

### 6. Chưa có API contract

Nếu frontend được tách thành SPA, cần định nghĩa:
- Request shape: `{text, model, use_captum}`.
- Response shape: `{misinfo, stance, sentiment, probs, xai_reasoning}`.
- Error response: `{error_code, message}`.

---

### Nếu tiếp tục dùng Gradio


#### 1. Tách shared layout/CSS giữa `app.py` và `gradio_app.py`

Hiện tại cả hai file chứa toàn bộ CSS giống hệt nhau (~2900 dòng CSS inline). Khi có thay đổi phải cập nhật tay cả hai — dễ gây lệch version.

**Đề xuất:** Trích xuất CSS ra file riêng:
```python
# shared_styles.py
CSS_STYLE = """..."""
LABEL_MAPS = {...}
LABEL_ICONS = {...}
LABEL_COLORS = {...}
```
```python
# app.py và gradio_app.py
from shared_styles import CSS_STYLE, LABEL_MAPS, LABEL_ICONS, LABEL_COLORS
```

> ⚠️ Cần kiểm tra HuggingFace Spaces có nhận import relative trong cùng thư mục không (thường là có nếu `app.py` và `shared_styles.py` cùng cấp).

---

#### 2. Quy ước z-index

Hiện tại các z-index được viết rải rác trong CSS. Nên thống nhất bằng CSS variables:

```css
:root {
    --z-sidebar:        9999;
    --z-sidebar-toggle: 10001;
    --z-dropdown:       999999;
    --z-modal:          100000;
    --z-toast:          200000;
}
```

---

#### 3. Quy ước cập nhật CSS khi upgrade Gradio

Mỗi lần Gradio ra version mới, internal class names (`svelte-*`, `.options`, `.select-wrap`…) có thể thay đổi. Checklist cần làm khi upgrade:

| Bước | Hành động |
|---|---|
| 1 | Chạy app và mở DevTools → Inspect dropdown đang mở |
| 2 | So sánh class names thực tế với selectors trong `CSS_STYLE` |
| 3 | Nếu lệch → cập nhật selectors trong block `COMPACT DROPDOWNS` |
| 4 | Kiểm tra `DROPDOWN FIX` block (z-index, animation) |
| 5 | Kiểm tra sidebar có bị Gradio override không (`#sidebar-col`) |
| 6 | Chạy thử cả Light và Dark mode |

---

#### 4. State flow cho phân tích async

Luồng state hiện tại khi nhấn **Phân tích**:

```
user click → handle_analyze() [blocking]
    │
    ├── gr.Progress(0.1) "Đang tải mô hình..."
    ├── load_model()         → model loaded into _CACHE
    ├── gr.Progress(0.4) "Đang phân loại..."
    ├── classify(text)       → result dict
    ├── gr.Progress(0.7) "Đang giải thích AI..."
    ├── call_xai_engine()    → reasoning_md
    ├── gr.Progress(0.9) "Đang render..."
    └── return (summary_html, radar, prob_dist, ..., gr.update(visible=True))
                                                         ↑
                                              charts_row hiện ra
```

**Vấn đề hiện tại:** `handle_analyze` là blocking — nếu model chưa load, UI bị freeze. Khi triển khai trên CPU Spaces, nên cân nhắc:
- Dùng `gr.Request` + background thread nếu Gradio hỗ trợ.
- Hoặc pre-load model khi khởi động app (`app.load(fn=warmup)`).

---

#### 5. Error states cần xử lý rõ ràng

| Tình huống | Trạng thái hiện tại | Đề xuất |
|---|---|---|
| **Fetch URL thất bại** | Hiển thị `❌ msg` trong `fetch_status` HTML | ✅ Đã có |
| **Model load lỗi** | Exception → UI trắng | ⚠️ Cần wrap try/except, trả về thông báo lỗi trong `summary_out` |
| **XAI timeout** | Silently fallback | ⚠️ Nên thêm indicator "XAI không khả dụng" |
| **Cache clear lỗi** | Hiển thị trong `clear_cache_status` | ✅ Đã có |
| **Văn bản rỗng** | Không có guard rõ ràng | ⚠️ Nên kiểm tra `if not text.strip()` trước khi phân tích |
| **Batch quá 50 mẫu** | Silently truncate | ⚠️ Nên cảnh báo user |

---

#### 6. Checklist test responsive desktop / mobile

**Desktop (≥ 1280px):**
- [ ] Sidebar hiển thị đúng 290px, sticky
- [ ] Toggle button đúng vị trí (left: 258px khi mở, 16px khi đóng)
- [ ] Dropdown không bị cắt bởi sidebar
- [ ] Charts row hiện sau khi phân tích
- [ ] Dark/Light mode chuyển đổi không bị flash

**Tablet (768px – 1280px):**
- [ ] Sidebar không che content khi đang mở
- [ ] Tab text không bị overflow

**Mobile (< 768px):**
- [ ] Sidebar `position: fixed` — ẩn mặc định
- [ ] Toggle button ở `left: 16px`
- [ ] Content padding thu gọn (8px)
- [ ] Dropdown không bị crop ngoài viewport
- [ ] Plotly charts scale đúng (width: 100%)
- [ ] Hero title readable (`clamp(1.6rem, 3.5vw, 2.6rem)`)

---

### Nếu dùng tài liệu này để migration sang React / SPA

Dùng tài liệu này như **inventory tính năng**, không phải spec triển khai trực tiếp. Cần thêm phần mapping:

```md
## Migration Mapping: Gradio → React

| Gradio component | React component |
|---|---|
| Header HTML (get_header_html) | `<Header />` |
| Sidebar controls | `<Sidebar />` |
| gr.Tabs (5 tabs) | React Router hoặc `<Tabs />` |
| render_result_cards_html | `<ResultCards />` |
| LABEL_MAPS | TypeScript constants (`labels.ts`) |
| Plotly gr.Plot (radar) | `react-plotly.js <Plot />` |
| Plotly gr.Plot (bar) | `react-plotly.js <Plot />` |
| CSS custom properties | CSS variables hoặc Tailwind theme |
| gr.Markdown (reasoning_out) | `<ReactMarkdown />` |
| gr.HTML (saliency_out) | `dangerouslySetInnerHTML` hoặc iframe |
| gr.Dropdown | `<Select />` (react-select hoặc shadcn) |
| gr.Textbox (multiline) | `<Textarea />` |
| gr.File | `<FileUpload />` |
| gr.Examples | `<ExampleList />` |
| JS theme toggle | Zustand / Context theme state |
| JS sidebar collapse | Zustand / Context sidebar state |
| Gradio event handlers | API client methods (axios/fetch) |
| gr.State | Zustand store hoặc React useState |
| gr.Progress | React loading skeleton / toast |
```

**API contract cần định nghĩa khi tách backend:**

```typescript
// Request
interface AnalyzeRequest {
  text: string;
  model: "PhoBERT-v2" | "XLM-R-v1";
  use_captum: boolean;
}

// Response
interface AnalyzeResponse {
  misinfo: { pred: number; probs: number[] };
  stance:  { pred: number; probs: number[] };
  sentiment: { pred: number; probs: number[] };
  xai_reasoning: string;    // Markdown
  saliency_html?: string;   // HTML (optional, Captum)
  elapsed_ms: number;
}

// Error
interface ApiError {
  error_code: string;   // "MODEL_NOT_LOADED" | "TEXT_TOO_SHORT" | "XAI_TIMEOUT" | ...
  message: string;
}
```

---

## 📌 Kết luận

`FRONTEND_DOCS.md` là tài liệu tốt cho frontend Gradio hiện có, đặc biệt về layout, CSS, tabs và charts.

- **Nếu tiếp tục với Gradio:** Ưu tiên tách `shared_styles.py`, chuẩn hóa z-index, và thêm error state guards.
- **Nếu chuyển sang React:** Dùng tài liệu này như inventory tính năng, bổ sung API contract và migration mapping ở trên.
