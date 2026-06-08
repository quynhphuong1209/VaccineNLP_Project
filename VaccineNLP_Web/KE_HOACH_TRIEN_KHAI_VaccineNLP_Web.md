# KẾ HOẠCH TRIỂN KHAI ĐẦY ĐỦ — VaccineNLP_Web (Public)
> Bản này HỢP NHẤT và THAY THẾ `KE_HOACH_PUBLIC_DEPLOY.md` + `KE_HOACH_XLMR_IG.md`.
> Đối tượng thực thi: IDE Coding Agent. Thực hiện TUẦN TỰ theo phase; mỗi phase có bước KIỂM trước khi sang phase sau.

---

## 0. BỐI CẢNH & NGUYÊN TẮC

- **Mục tiêu:** đưa hệ thống lên Internet công khai; **mọi số liệu/kết quả hiển thị đều phải THẬT** (không bịa, không mô phỏng đội lốt thật).
- **Kiến trúc đã CHỐT (Phương án 4):** mặt tiền public = **PhoBERT-v2** (phân loại nhanh, luôn chạy) + **phân phối softmax thật + cờ nhất quán + token attribution**. Phần **Gemma XAI chạy ON-DEMAND** (người dùng bấm nút) bằng **GGUF trên CPU NGAY trong container** (không phụ thuộc LM Studio khi public).
- **Quy tắc số liệu:** chỉ dùng **Bảng 4.2** (mục 0.1). Mọi con số khác là vô hiệu, phải xoá.
- **Quy tắc an toàn:** không commit secret; verify trước khi tin (đặc biệt Phase 2).

### 0.1 SỐ LIỆU BẤT BIẾN (KHÔNG được sửa)

**Bảng 4.2 — Macro F1 (LIVE Kaggle 21/05/2026):**
| Model | Misinfo | Stance | Sentiment | Avg |
|---|---|---|---|---|
| PhoBERT-v2 | 0,6996 | 0,6640 | 0,7266 | **0,6967** |
| Gemma-4 4B | 0,6377 | 0,6264 | 0,7700 | 0,6780 |
| XLM-R | 0,7038 | 0,6224 | 0,6866 | 0,6709 |

**Taxonomy v3 (ID2LABEL — thứ tự stance CỐ Ý phi-bảng-chữ-cái):**
- misinfo: `{0: Fake, 1: Real}`
- stance: `{0: Favor, 1: Against, 2: Neutral}`
- sentiment: `{0: Negative, 1: Neutral, 2: Positive}`

**⛔ CÁC CON SỐ BỊ CẤM (corrupted — phải XOÁ khỏi mọi nơi):**
PhoBERT 0.7079/0.7107/0.7260; Gemma 0.6925/0.5818/0.7196; XLM-R 0.5823/0.4217/0.1842; "F1 Avg 71.5% / 39.6%"; Parse Failure 33.3%; bất kỳ số dạng 0.4547 (bản FINAL_TECHNICAL_REPORT cũ).

### 0.2 BẢN ĐỒ FILE CLAUDE ĐÃ GIAO → ĐƯỜNG DẪN ĐÍCH
| File đã giao | Đặt tại | Vai trò |
|---|---|---|
| `database.py` | `api_service/app/database.py` | Thêm cột JSONB `phobert_probs` |
| `main.py` | `api_service/app/main.py` | full-softmax + (sửa thêm mục 1.B) |
| `xai_service_main_PA4.py` | `xai_service/app/main.py` | GGUF-CPU tối ưu + stream + công tắc backend |

> Các snippet FRONTEND và 2 sửa nhỏ của `api_service` nằm INLINE trong tài liệu này (không có file rời).

---

# PHASE 0 — TOÀN VẸN DỮ LIỆU (BẮT BUỘC, làm trước tiên)

## 0.A — Backend full-softmax
1. Đặt `database.py` → `api_service/app/database.py` (đã thêm `phobert_probs: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)`).
2. Đặt `main.py` → `api_service/app/main.py` (đã: `phobert_infer` trả thêm `_probs`; lưu `phobert_probs`; đẩy `probs` qua SSE event `phobert`; `AnalysisResponse` có `phobert_probs`).
3. **Migration:** cột mới chỉ tự sinh trên DB MỚI (`create_all` không ALTER bảng cũ). Vì public dùng DB mới → OK. DB dev cũ: `docker compose down -v` (xoá volume) rồi up; hoặc `ALTER TABLE analysis_history ADD COLUMN phobert_probs JSONB;`.

## 0.B — Frontend: số benchmark THẬT
File `frontend/src/App.tsx`, thay object `benchmarkData`:
```javascript
const benchmarkData = {
  phobert: { name: "PhoBERT-v2 (Classification Engine)", misinfo: 0.6996, stance: 0.6640, sentiment: 0.7266, average: 0.6967 },
  gemma:   { name: "Gemma-4 4B (XAI Reasoning Engine)",  misinfo: 0.6377, stance: 0.6264, sentiment: 0.7700, average: 0.6780 },
  xlmr:    { name: "XLM-R-v1 (Baseline)",                 misinfo: 0.7038, stance: 0.6224, sentiment: 0.6866, average: 0.6709 },
}
```
- Trong tab So sánh: badge PhoBERT `F1 Avg: 71.5%` → `F1 Avg: 69.67%`.
- Số "Tốc độ suy luận" `120.5/85.2/1.8 mẫu/s`: nếu chưa đo thật → gắn nhãn "(ước lượng minh hoạ)".

## 0.C — Frontend: card "Phân phối xác suất đầy đủ" THẬT
1. Interface: thêm vào `AnalysisResult`:
```tsx
phobert_probs?: Record<string, Record<string, number>>
```
2. Handler SSE `event.type === 'phobert'`: thêm `phobert_probs: event.probs` vào object kết quả.
3. Thêm helper (cạnh `renderSVGCharts`):
```tsx
const PROB_COLORS: Record<string, Record<string,string>> = {
  misinfo:{Fake:'bg-red-500',Real:'bg-green-500'},
  stance:{Favor:'bg-green-500',Against:'bg-red-500',Neutral:'bg-blue-500'},
  sentiment:{Positive:'bg-green-500',Negative:'bg-red-500',Neutral:'bg-blue-500'} }
const PROB_TEXT: Record<string,string> = {Fake:'text-red-500',Real:'text-green-500',Favor:'text-green-500',Against:'text-red-500',Positive:'text-green-500',Negative:'text-red-500',Neutral:'text-blue-400'}
const AXIS_TITLE: Record<string,string> = {misinfo:'Trục Xác thực (Misinfo):',stance:'Trục Lập trường (Stance):',sentiment:'Trục Cảm xúc (Sentiment):'}
const renderProbAxis = (task:'misinfo'|'stance'|'sentiment', probs?:Record<string,number>) => {
  if (!probs) return null
  return (<div className="space-y-1" key={task}>
    <span className={`text-xs font-bold block ${isLightMode?'text-slate-850':'text-slate-400'}`}>{AXIS_TITLE[task]}</span>
    <div className="space-y-1.5">{Object.entries(probs).map(([label,p])=>(
      <div className="flex items-center text-xs font-mono" key={label}>
        <span className={`w-16 font-bold ${PROB_TEXT[label]||'text-slate-400'}`}>{label}:</span>
        <div className="flex-1 bg-slate-900/10 dark:bg-slate-900 rounded-lg h-4 overflow-hidden relative border border-slate-300 dark:border-transparent">
          <div className={`${PROB_COLORS[task][label]||'bg-slate-500'} h-4 rounded`} style={{width:`${(p*100).toFixed(1)}%`}} />
          <span className={`absolute inset-y-0 right-2 flex items-center font-bold text-[9px] ${isLightMode?'text-slate-900':'text-white'}`}>{(p*100).toFixed(1)}%</span>
        </div>
      </div>))}</div></div>)
}
```
4. Thay TOÀN BỘ `<div className="space-y-4">…3 khối bịa…</div>` trong "Probability Distribution Card" bằng:
```tsx
<div className="space-y-4">
  {res.phobert_probs ? (<>
    {renderProbAxis('misinfo', res.phobert_probs.misinfo)}
    {renderProbAxis('stance', res.phobert_probs.stance)}
    {renderProbAxis('sentiment', res.phobert_probs.sentiment)}
  </>) : <p className="text-xs text-slate-500 italic">Phân phối đầy đủ khả dụng cho phân tích mới.</p>}
</div>
```

## 0.D — Frontend: tab "Captum IG" → TRUNG THỰC
- Nhãn nút: `🔥 Token Attribution (Captum IG)` → `🔍 Từ khoá nổi bật (gợi ý)`. *(Sẽ đổi lại thành IG thật ở Phase 2.B.)*
- Mô tả: → `💡 Gợi ý dựa trên TỪ KHOÁ miền y tế — heuristic minh hoạ, KHÔNG phải attribution gradient từ mô hình.`
- `getSaliencyTokens()`: **bỏ mọi `Math.random()`**, dùng tất định:
```tsx
const getSaliencyTokens = () => {
  if (!result) return []
  const words = result.source_text.split(/\s+/)
  const predictedAsFake = result.misinfo_label === 'Fake'
  const fakeKeywords = ["vô","sinh","biến","đổi","gen","chuột","bạch","tập","đoàn","dược","độc","hại","thải"]
  const realKeywords = ["an","toàn","khỏe","mạnh","đầy","đủ","khuyến","cáo","chính","xác","phòng","bệnh"]
  const kws = predictedAsFake ? fakeKeywords : realKeywords
  return words.map(word => {
    const clean = word.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"")
    return { word, score: kws.some(kw => clean.includes(kw)) ? 0.7 : 0.08 }
  })
}
```

## 0.E — Frontend: tab So sánh → tạm gắn "MÔ PHỎNG" (đến khi có XLM-R thật ở 2.A)
- Badge XLM-R `F1 Avg: 39.6%` → text `⚠️ MÔ PHỎNG`.
- Thêm dòng cảnh báo dưới ô nhập So sánh: `⚠️ Cột XLM-R hiện là mô phỏng minh hoạ — chưa nạp mô hình XLM-R thật.`
- *(Phase 2.A sẽ thay bằng XLM-R thật và bỏ cảnh báo này.)*

## 0.F — Token: xoá rò rỉ
- Xoá dòng `LM_API_TOKEN=sk-lm-62qEdhxR:dC6TggrLf3m4ksLBKPgU` khỏi `HD_VAN_HANH_VACCINENLP.md` và mục token trong `README.md`.
- **Rotate** token mới trong LM Studio; chỉ giữ trong `.env`.
- Repo public → **purge khỏi git history**: `git filter-repo --replace-text` (hoặc BFG) rồi force-push.

## ✅ KIỂM PHASE 0
```bash
docker compose down -v && docker compose up --build
```
- Mở `localhost:5173`, phân tích 1 câu → card "Phân phối" hiện 2–3 thanh có số cộng ≈ 100% (THẬT, đổi theo input).
- Tab Benchmark hiện đúng số Bảng 4.2; tab So sánh badge PhoBERT 69.67%.
- Tab "Từ khoá nổi bật" tất định (chạy lại cùng câu → cùng kết quả).

---

# PHASE 1 — PHƯƠNG ÁN 4 + TĂNG TỐC GGUF-CPU (kiến trúc on-demand)

## 1.A — xai_service: thay bằng bản PA4
1. Đặt `xai_service_main_PA4.py` → `xai_service/app/main.py`.
2. `xai_service/requirements.txt` giữ `requests`; **thêm** (cho backend local): cài `llama-cpp-python` (Dockerfile hiện đã cài qua wheel CPU — giữ nguyên).
3. `docker-compose.yml` (service `xai_service`) — env:
   - **Public:** `XAI_BACKEND=local` (chạy GGUF-CPU trong container, on-demand).
   - **Local dev (có LM Studio):** `XAI_BACKEND=lmstudio`.
   - Bỏ `LOCAL_FALLBACK` (backend `local` tự nạp GGUF). Giữ `LMSTUDIO_URL/MODEL` để dùng cho dev. Có thể set `N_CTX=1024` để bớt RAM. `LLM_THREADS` = số nhân của host.
   - Giữ `extra_hosts: ["host.docker.internal:host-gateway"]` (cho dev lmstudio).
   - `mem_limit`: nếu `XAI_BACKEND=local` cần ≥ 4–6GB cho GGUF.

**Tăng tốc CPU đã đóng sẵn trong file** (không cần làm thêm): `flash_attn=True` (tự rớt nếu bản không hỗ trợ), `n_threads/n_threads_batch` theo nhân, `LlamaRAMCache` (tái dùng KV của SYSTEM_PROMPT), warm-up lúc khởi động, `_gen_lock` serialize, stream trực tiếp từ GGUF.
> Đòn bẩy tăng tốc LỚN NHẤT (để dành, tùy chọn): build `llama-cpp-python` từ nguồn với cờ CPU (AVX2/AVX-512/OpenBLAS) thay vì wheel dựng sẵn.

## 1.B — api_service: tách "phân tích" (idle) khỏi "giải thích" (on-demand)
File `api_service/app/main.py` (bản Claude đã giao), sửa 2 chỗ:

**(i) `/api/analyze`** — bỏ tự chạy XAI:
```python
        consistency_flag=compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"]),
        xai_status="idle",          # PA4: KHÔNG tự chạy Gemma
    )
    db.add(row); db.commit(); db.refresh(row)
    return row                       # XOÁ dòng bg.add_task(...) và tham số bg: BackgroundTasks
```

**(ii) THAY `/api/analyze-stream` bằng `/api/explain-stream`** (nút gọi; tái dùng nhãn PhoBERT đã cache):
```python
@app.post("/api/explain-stream")
def explain_stream(req: AnalyzeRequest, db: Session = Depends(get_db)):
    h = hashlib.md5(req.text.encode("utf-8")).hexdigest()
    row = db.scalar(select(AnalysisHistory).where(AnalysisHistory.text_hash == h))
    if row is None:                                  # phòng khi bấm explain trước analyze
        p = phobert_infer(req.text); labels = {k: v[0] for k, v in p.items() if k != "_probs"}
        row = AnalysisHistory(source_text=req.text, source_url=req.source_url, text_hash=h,
            misinfo_label=p["misinfo"][0], misinfo_score=p["misinfo"][1],
            stance_label=p["stance"][0], stance_score=p["stance"][1],
            sentiment_label=p["sentiment"][0], sentiment_score=p["sentiment"][1],
            phobert_probs=p["_probs"], xai_status="pending",
            consistency_flag=compute_consistency(labels["misinfo"], labels["stance"], labels["sentiment"]))
        db.add(row); db.commit(); db.refresh(row)
    else:
        labels = {"misinfo": row.misinfo_label, "stance": row.stance_label, "sentiment": row.sentiment_label}
        row.xai_status = "pending"; db.commit()
    row_id = row.id
    def gen():
        xai_url = os.environ.get("XAI_SERVICE_URL", "http://xai_service:8001")
        final_data = None
        try:
            with requests.post(f"{xai_url}/api/explain-stream",
                               json={"text": req.text, "predicted_labels": labels},
                               stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"): continue
                    payload = line[5:].strip()
                    if payload == "[DONE]": break
                    try: ev = json.loads(payload)
                    except json.JSONDecodeError: continue
                    if ev.get("type") == "token": yield _sse(ev)
                    elif ev.get("type") == "final": final_data = ev
                    elif ev.get("type") == "error": yield _sse(ev)
            dbx = SessionLocal()
            try:
                rr = dbx.get(AnalysisHistory, row_id)
                if final_data:
                    final_data["disagreement"] = {k: (labels.get(k) != final_data.get("gemma_labels", {}).get(k)) for k in labels}
                    if rr: rr.xai_explanation = final_data; rr.xai_status = "done"; dbx.commit()
                    yield _sse(final_data)
                else:
                    if rr: rr.xai_status = "failed"; dbx.commit()
                    yield _sse({"type": "error", "message": "no final block"})
            finally: dbx.close()
        except Exception as e:
            dbx = SessionLocal()
            try:
                rr = dbx.get(AnalysisHistory, row_id)
                if rr: rr.xai_status = "failed"; dbx.commit()
            finally: dbx.close()
            yield _sse({"type": "error", "message": f"XAI: {type(e).__name__}"})
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

## 1.C — frontend: tách 2 hành động + nút on-demand
```tsx
// "Phân tích" CHỈ chạy PhoBERT (nhanh)
const handleAnalyze = async (e?: React.FormEvent) => {
  if (e) e.preventDefault(); if (!text.trim()) return
  setLoading(true); setResult(null); stopVoice()
  try { const { data } = await axios.post<AnalysisResult>(`${API_URL}/api/analyze`, { text, source_url: sourceUrl || null }); setResult(data) }
  catch { alert("Lỗi khi phân tích văn bản.") } finally { setLoading(false) }
}
// Nút "Sinh giải thích (chậm)" -> stream Gemma on-demand
const [explaining, setExplaining] = useState(false)
const handleExplain = async () => {
  if (!result || !text.trim()) return
  setExplaining(true)
  setResult(p => p ? { ...p, xai_status: 'pending', xai_explanation: { parse_ok: true, reasoning: '' } } : p)
  try {
    const resp = await fetch(`${API_URL}/api/explain-stream`, { method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text, source_url: sourceUrl || null }) })
    if (!resp.body) throw new Error("no body")
    const reader = resp.body.getReader(), dec = new TextDecoder("utf-8"); let buf = ""
    while (true) {
      const { value, done } = await reader.read(); if (done) break
      buf += dec.decode(value, { stream: true }); const lines = buf.split("\n"); buf = lines.pop() || ""
      for (const line of lines) {
        if (!line.startsWith("data:")) continue; const s = line.slice(5).trim(); if (s === "[DONE]") continue
        try { const ev = JSON.parse(s)
          if (ev.type === 'token') setResult(p => p ? { ...p, xai_status:'pending', xai_explanation:{ ...(p.xai_explanation||{parse_ok:true}), parse_ok:true, reasoning:((p.xai_explanation?.reasoning)||"")+ev.content } } : p)
          else if (ev.type === 'final') setResult(p => p ? { ...p, xai_status:'done', xai_explanation:{ parse_ok:ev.parse_ok, reasoning:ev.reasoning, raw_output:ev.raw_output, disagreement:ev.disagreement, gemma_labels:ev.gemma_labels } } : p)
          else if (ev.type === 'error') setResult(p => p ? { ...p, xai_status:'failed' } : p)
        } catch {}
      }
    }
  } catch { setResult(p => p ? { ...p, xai_status:'failed' } : p) } finally { setExplaining(false) }
}
```
Nút (trong tab CoT, hiện khi `result.xai_status === 'idle'`):
```tsx
{result.xai_status === 'idle' && (
  <button onClick={handleExplain} disabled={explaining}
    className="w-full py-3 rounded-xl font-bold bg-purple-600/20 text-purple-300 border border-purple-700 hover:bg-purple-600/30 disabled:opacity-50">
    {explaining ? '🐢 Đang sinh giải thích…' : '🐢 Sinh giải thích chi tiết (Gemma, chậm ~30–90s)'}
  </button>
)}
```
> Dọn nợ: xoá `startPolling/stopPolling`, `pollingRef`, `polling`, và `useEffect` no-op gom import; xoá import lucide không dùng.

## ✅ KIỂM PHASE 1
- `XAI_BACKEND=local`, đặt `gemma-4-4b-Q4_K_M.gguf` vào `models/`.
- `docker compose up --build`; log xai_service báo "GGUF loaded" + "warm-up xong".
- Phân tích 1 câu → nhãn + phân phối hiện ngay (nhanh). Bấm nút → token CoT chảy ra (chậm, ~30–90s). Bấm lại câu khác → vẫn chạy (serialize, không sập).

---

# PHASE 2 — XLM-R THẬT + IG THẬT (cần VERIFY trước khi code)

## 2.PRE — 3 việc VERIFY (BẮT BUỘC; đoán sai sẽ cho kết quả "thật" nhưng SAI)
1. **Tiền xử lý XLM-R:** notebook `vaccinenlp-xlm-r-v1` có gọi `word_tokenize(..., format="text")` không? → quyết định `XLMR_USE_SEGMENTATION` (1 nếu segmented, 0 nếu text thô).
2. **Nhãn XLM-R:** eyeball vài câu theo `stance_id` (0=Ủng hộ, 1=Phản đối, 2=Trung lập) — xác nhận ID2LABEL dùng chung với PhoBERT.
3. **Đường dẫn embedding PhoBERT (cho IG):** sau khi nạp model chạy `print(type(model.encoder.embeddings.word_embeddings))` → xác nhận tồn tại.

## 2.A — XLM-R THẬT (tab So sánh)
`api_service/app/main.py` — thêm:
```python
XLMR_PATH = os.environ.get("XLMR_PATH", "/models/xlmr_multitask.pt")
XLMR_USE_SEGMENTATION = os.environ.get("XLMR_USE_SEGMENTATION", "0") == "1"  # ⚠️ set theo 2.PRE#1
xlmr = {}

def _load_xlmr():
    if "model" in xlmr: return
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = PhoBERTMultitaskClassifier(model_name="xlm-roberta-base", num_misinfo=2, num_stance=3, num_sentiment=3)
    state = torch.load(XLMR_PATH, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
    state = { k.replace("heads.misinfo.","head_misinfo.").replace("heads.stance.","head_stance.").replace("heads.sentiment.","head_sentiment."): v for k,v in state.items() }
    model.load_state_dict(state, strict=True); model.eval()
    xlmr["tok"], xlmr["model"] = tok, model

def _xlmr_clean(text: str) -> str:
    if XLMR_USE_SEGMENTATION: return prepare_text(text)
    from .preprocess import _clean; return _clean(text)

@torch.no_grad()
def xlmr_infer(text: str) -> dict:
    _load_xlmr()
    enc = xlmr["tok"](_xlmr_clean(text), return_tensors="pt", truncation=True, max_length=256)
    out = xlmr["model"](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = {"misinfo": out[0], "stance": out[1], "sentiment": out[2]}; res = {}
    for task in ("misinfo","stance","sentiment"):
        prob = torch.softmax(logits[task], dim=-1)[0]; idx = int(prob.argmax())
        res[task] = (ID2LABEL[task][idx], round(float(prob[idx]), 4))
    return res

@app.post("/api/compare")
def compare(req: AnalyzeRequest):
    p = phobert_infer(req.text); x = xlmr_infer(req.text)
    fmt = lambda r: {k: {"label": r[k][0], "score": r[k][1]} for k in ("misinfo","stance","sentiment")}
    return {"phobert": fmt(p), "xlmr": fmt(x)}
```
- **compose:** `XLMR_PATH=/models/xlmr_multitask.pt`, `XLMR_USE_SEGMENTATION=<2.PRE#1>`; tải `xlmr_multitask.pt` (HF `hung2903/xlmr-vaccine-multitask`) vào `models/`; `api_service` `mem_limit` +~1.5GB (XLM-R ~270M params, lazy-load).
- **frontend `handleCompare`:** gọi `POST /api/compare`, BỎ toàn bộ `Math.random`; badge XLM-R = `67.09%` (Bảng 4.2); **xoá cảnh báo "MÔ PHỎNG" ở 0.E** (giờ là so sánh thật).

## 2.B — INTEGRATED GRADIENTS THẬT (heatmap)
- `api_service/requirements.txt`: thêm `captum`.
- `api_service/app/main.py` — thêm:
```python
from captum.attr import LayerIntegratedGradients

def _ig_attribute(text: str, task: str = "misinfo", n_steps: int = 32):
    model, tok = phobert["model"], phobert["tok"]
    enc = tok(prepare_text(text), return_tensors="pt", truncation=True, max_length=256)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    task_idx = {"misinfo":0,"stance":1,"sentiment":2}[task]
    with torch.no_grad():
        pred = int(torch.softmax(model(input_ids=input_ids, attention_mask=attn)[task_idx], -1).argmax())
    def fwd(ids, mask): return model(input_ids=ids, attention_mask=mask)[task_idx]
    emb_layer = model.encoder.embeddings.word_embeddings          # ⚠️ 2.PRE#3
    lig = LayerIntegratedGradients(fwd, emb_layer)
    baseline = torch.full_like(input_ids, tok.pad_token_id)
    special = set(tok.all_special_ids)
    for i, t in enumerate(input_ids[0].tolist()):
        if t in special: baseline[0, i] = t
    attributions = lig.attribute(inputs=input_ids, baselines=baseline,
                                 additional_forward_args=(attn,), target=pred, n_steps=n_steps)
    attr = attributions.sum(dim=-1).squeeze(0); attr = attr / (attr.abs().max() + 1e-9)
    toks = tok.convert_ids_to_tokens(input_ids[0].tolist()); out = []
    for tk, a in zip(toks, attr.tolist()):
        if tk in tok.all_special_tokens: continue
        out.append({"token": tk.replace("@@","").replace("▁",""), "score": round(float(a), 4)})
    return {"task": task, "pred_class": pred, "tokens": out}

@app.post("/api/attribute")
def attribute(req: AnalyzeRequest):
    if "model" not in phobert: return {"tokens": [], "error": "model_not_loaded"}
    return _ig_attribute(req.text, task="misinfo")
```
- **frontend:** `getSaliencyTokens()` → gọi `POST /api/attribute` (on-demand, vài giây CPU), nhận `tokens:[{token,score}]`, tô nền theo `|score|`; đổi nhãn tab lại thành **"Token Attribution (Integrated Gradients)"**.

## ✅ KIỂM PHASE 2
- `/api/compare` câu "ủng hộ tiêm chủng" → 2 cột khớp kỳ vọng (XLM-R là model thật, không phải nhiễu PhoBERT).
- `/api/attribute` câu tin giả → token "vô sinh / biến đổi gen" sáng nhất.

---

# PHASE 3 — PRODUCTIONIZE & DEPLOY PUBLIC

## 3.A — Frontend bản production (bỏ dev server)
`frontend/Dockerfile` (multi-stage build → phục vụ tĩnh bằng Caddy):
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci || npm install
COPY . .
RUN npm run build
FROM caddy:2-alpine
COPY --from=build /app/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile
```
- Bỏ `watch.usePolling` trong `vite.config.ts` (chỉ cần cho dev).

## 3.B — Reverse proxy + URL tương đối + HTTPS
`Caddyfile` (đặt cạnh frontend Dockerfile):
```
your-domain.com {
    handle_path /api/* { reverse_proxy api_service:8000 }
    handle {
        root * /srv
        try_files {path} /index.html
        file_server
    }
}
```
- Frontend gọi API **tương đối**: set `VITE_API_URL=""` (hoặc bỏ env) → `${API_URL}/api/...` thành `/api/...` → Caddy proxy. **Xoá hardcode `localhost`.**
- Caddy tự cấp HTTPS (Let's Encrypt) khi có domain trỏ về. SSE đi qua Caddy mặc định không buffer.

## 3.C — Bảo mật & cấu hình public
- CORS: same-origin qua proxy ⇒ có thể bỏ; nếu giữ thì set đúng `https://your-domain.com` (bỏ `localhost:5173`).
- `.env`: `POSTGRES_PASSWORD` mạnh; **KHÔNG** publish cổng `5432` ra Internet (bỏ `ports: 5432:5432` của `db`).
- DB Postgres mới + volume + lịch backup.
- `mem_limit` theo RAM host (nhớ cộng GGUF nếu `XAI_BACKEND=local`).

## 3.D — Hạ tầng đề xuất
- **Đơn giản nhất:** 1 VPS (Hetzner CX22 ~€4 / DO ~$6, ≥8GB nếu chạy GGUF-CPU) → cài Docker → `docker compose up -d` → trỏ domain → Caddy tự HTTPS.
- **Hoặc tách:** frontend tĩnh (Cloudflare Pages/Netlify/Vercel) + `api_service` (Render/Railway/Fly) + Postgres managed (Neon/Supabase). Lưu ý free tier ngủ/giới hạn; XAI on-demand vẫn cần nơi chạy GGUF (VPS) hoặc serverless GPU.

## ✅ KIỂM PHASE 3
- Truy cập `https://your-domain.com` từ máy khác/điện thoại → phân tích chạy, HTTPS hợp lệ, không lỗi CORS, cổng 5432 không lộ.

---

## PHỤ LỤC — LỆNH TEST NHANH
```bash
# rebuild sạch + DB mới (sau Phase 0/1)
docker compose down -v && docker compose up --build

# xai_service (PA4 local) — health + thử explain stream
curl -s http://localhost:8001/                       # backend, local_gguf_loaded
curl -N -X POST http://localhost:8001/api/explain-stream -H "Content-Type: application/json" \
  -d "{\"text\":\"tiêm vaccine cúm bảo vệ thai phụ\",\"predicted_labels\":{\"misinfo\":\"Real\",\"stance\":\"Favor\",\"sentiment\":\"Positive\"}}"

# api_service — analyze (PhoBERT-only, có phobert_probs) rồi explain on-demand
curl -s -X POST http://localhost:8000/api/analyze -H "Content-Type: application/json" -d "{\"text\":\"...\"}"
curl -N -X POST http://localhost:8000/api/explain-stream -H "Content-Type: application/json" -d "{\"text\":\"...\"}"

# Phase 2 (sau verify)
curl -s -X POST http://localhost:8000/api/compare  -H "Content-Type: application/json" -d "{\"text\":\"...\"}"
curl -s -X POST http://localhost:8000/api/attribute -H "Content-Type: application/json" -d "{\"text\":\"...\"}"
```

## THỨ TỰ THỰC THI TÓM TẮT
Phase 0 (toàn vẹn) → KIỂM → Phase 1 (PA4 + tăng tốc) → KIỂM → **(verify 2.PRE)** → Phase 2 (XLM-R + IG thật) → KIỂM → Phase 3 (productionize + deploy) → KIỂM.
> Có thể public sớm sau Phase 0+1+3 (đã sạch & thật); Phase 2 nâng tab So sánh + heatmap lên "thật" hoàn toàn.
