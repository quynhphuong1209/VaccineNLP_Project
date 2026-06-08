"""
vnlp_ui.py — Lớp giao diện dùng chung cho gradio_app (đồng bộ với redesign React).

Cách dùng (ghép vào app.py Gradio hiện có):
    from vnlp_ui import THEME_CSS, ICON_SPRITE, render_result, render_cot, render_disagreement

    with gr.Blocks(css=THEME_CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(ICON_SPRITE)          # nạp sprite icon 1 lần
        ...
        out_result = gr.HTML()        # verdict + axes + consistency + distribution
        out_xai    = gr.HTML()        # CoT stream + bảng bất đồng thuận

    # khi có kết quả PhoBERT:
    out_result.value = render_result(result_dict)

    # streaming CoT (generator):  yield render_cot(accumulated, streaming=True)

Yêu cầu: đặt redesign.css cùng thư mục với file này (dùng chung 1 nguồn với frontend).
Mọi ánh xạ nhãn / màu khớp 1:1 với App.tsx để hai giao diện không lệch nhau.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# CSS: nạp redesign.css (nguồn chung) → đổi selector dark cho Gradio → + skin native
# ----------------------------------------------------------------------
def _base_css() -> str:
    path = os.path.join(_HERE, "redesign.css")
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
    except FileNotFoundError:
        # fallback tối thiểu nếu thiếu file (chỉ token, không đủ đẹp)
        css = ":root{--bg:#f4f6f6;--surface:#fff;--ink:#15201e;--teal:#0e9384;--danger:#d2453a;}"
    # Gradio bật dark bằng class .dark trên wrapper, không phải [data-theme]
    css = css.replace('[data-theme="dark"]', ".dark")
    return css

# Skin mỏng cho component GỐC của Gradio (textbox, button, tab...).
# Lưu ý: selector phụ thuộc phiên bản Gradio — chỉnh nếu lên/xuống version.
_NATIVE_SKIN = """
/* ===== nền + font toàn app ===== */
.gradio-container { background: var(--bg) !important; color: var(--ink); font-family: var(--ui); max-width: 100% !important; }
.gradio-container .prose, .gradio-container label, .gradio-container span { color: inherit; }
footer { display: none !important; }

/* ===== textbox ===== */
.gradio-container textarea, .gradio-container input[type=text] {
  background: var(--surface-2) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: var(--r) !important;
  font-family: var(--ui) !important; font-size: 14.5px !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
  border-color: var(--teal) !important; box-shadow: 0 0 0 3px var(--teal-50) !important;
}

/* ===== buttons ===== */
.gradio-container button.primary, .gradio-container .primary > button {
  background: var(--teal) !important; border-color: var(--teal) !important; color: #fff !important;
  border-radius: var(--r-sm) !important; font-weight: 600 !important;
}
.gradio-container button.secondary, .gradio-container .secondary > button {
  background: var(--surface) !important; color: var(--ink-2) !important;
  border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important;
}

/* ===== khối / panel của Gradio: gỡ bớt chrome để gr.HTML của ta tự trình bày ===== */
.gradio-container .block, .gradio-container .form { border: none !important; background: transparent !important; box-shadow: none !important; }

/* ===== tabs gốc Gradio ===== */
.gradio-container .tab-nav button { color: var(--ink-3) !important; border: none !important; font-weight: 600 !important; }
.gradio-container .tab-nav button.selected { color: var(--teal-strong) !important; border-bottom: 2px solid var(--teal) !important; }

/* dark: theo .dark của Gradio (token đã đổi ở _base_css) */
.dark .gradio-container { background: var(--bg) !important; }
"""

THEME_CSS = _base_css() + _NATIVE_SKIN

# ----------------------------------------------------------------------
# ICON SPRITE (đồng bộ App.tsx) — nạp 1 lần qua gr.HTML
# ----------------------------------------------------------------------
ICON_SPRITE = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <symbol id="i-analyze" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/><path d="M11 8v6M8 11h6"/></symbol>
  <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3 5 6v5c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6Z"/><path d="M12 8v4M12 15.5h.01"/></symbol>
  <symbol id="i-check" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5L16 9.5"/></symbol>
  <symbol id="i-spark" viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"/></symbol>
  <symbol id="i-scale" viewBox="0 0 24 24"><path d="M12 3v18M7 7h10M5 7l-2 5a3 3 0 0 0 6 0L7 7M17 7l-2 5a3 3 0 0 0 6 0l-2-5"/></symbol>
  <symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14m0 0-6-6m6 6-6 6"/></symbol>
</svg>
"""

def _icon(i, cls="icon sm"):
    return f'<svg class="{cls}"><use href="#{i}"/></svg>'

# ----------------------------------------------------------------------
# ÁNH XẠ NHÃN + MÀU (khớp App.tsx)
# ----------------------------------------------------------------------
VI = {
    "misinfo":   {"Fake": "Tin giả", "Real": "Tin thật"},
    "stance":    {"Favor": "Ủng hộ", "Against": "Phản đối", "Neutral": "Trung lập"},
    "sentiment": {"Positive": "Tích cực", "Negative": "Tiêu cực", "Neutral": "Trung tính"},
}
BAD  = {"Fake", "Against", "Negative"}
GOOD = {"Real", "Favor", "Positive"}
CONSISTENCY = {
    "plausible": ("ok", "Tổ hợp nhãn hợp lệ", "i-check"),
    "unusual":   ("warn", "Bất thường — nghi mô hình sai", "i-shield"),
    "high_risk": ("danger", "Nguy cơ cao — nên rà soát", "i-shield"),
}

def _val_cls(lbl):   return "bad" if lbl in BAD else "ok" if lbl in GOOD else "neu"
def _meter_cls(lbl): return "bad" if lbl in BAD else "teal" if lbl in GOOD else ""
def _bar_color(lbl): return "var(--danger)" if lbl in BAD else "var(--teal)" if lbl in GOOD else "var(--ink-3)"
def _vi(task, lbl):  return VI.get(task, {}).get(lbl, lbl)

# ----------------------------------------------------------------------
# RENDER: verdict + classification card (verdict, axes, consistency, distribution)
# ----------------------------------------------------------------------
def _axis(task_vi, task_key, label, score, scalehint):
    pct = f"{score*100:.1f}%"
    return f"""
    <div class="axis">
      <div class="cap">{task_vi}</div>
      <div class="val {_val_cls(label)}">{_vi(task_key, label)}</div>
      <div class="meter {_meter_cls(label)}"><i style="width:{score*100:.0f}%"></i></div>
      <div class="scoreline"><span>{scalehint}</span><span class="mono">{pct}</span></div>
    </div>"""

def _distribution(probs):
    if not probs:
        return '<div class="muted" style="font-size:12px">Không có dữ liệu phân phối.</div>'
    rows = ""
    for t in ("misinfo", "stance", "sentiment"):
        bars = ""
        for lbl, p in probs.get(t, {}).items():
            bars += f"""<div class="bar"><span class="nm">{_vi(t, lbl)}</span>
              <span class="track"><i style="width:{p*100:.1f}%;background:{_bar_color(lbl)}"></i></span>
              <span class="pct mono">{p*100:.1f}%</span></div>"""
        rows += f'<div class="grp"><div class="glabel">{t}</div>{bars}</div>'
    return f'<div class="distrib">{rows}</div>'

def _consistency_legend(flag):
    chips = ""
    for k, (pill, text, _ic) in CONSISTENCY.items():
        on = (k == flag)
        style = ('outline:2px solid color-mix(in srgb,var(--ink-3) 40%,transparent);outline-offset:1px'
                 if on else 'opacity:.55')
        chips += f'<span class="pill {pill}" style="{style}">{k} · {text}{" ◂ hiện tại" if on else ""}</span>'
    return f"""<div style="margin-top:16px;border-top:1px solid var(--line-2);padding-top:14px">
      <div class="muted" style="font-size:12px;margin-bottom:9px">Cờ nhất quán tam giác nhãn <span class="mono">(consistency_flag)</span> — đối chiếu Gold n=186:</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">{chips}</div></div>"""

def render_result(r: dict) -> str:
    """HTML cho verdict hero + card phân loại (axes + distribution + consistency)."""
    fake = r.get("misinfo_label") == "Fake"
    pill, text, ic = CONSISTENCY.get(r.get("consistency_flag", "plausible"), CONSISTENCY["plausible"])
    conf = f'{r.get("misinfo_score", 0)*100:.1f}%'
    verdict = f"""
    <div class="verdict {'' if fake else 'ok'}">
      <div class="glyph">{_icon('i-shield' if fake else 'i-check', 'icon lg')}</div>
      <div>
        <div class="vtitle">Kết luận đối soát</div>
        <div class="vmain">{'Tin giả' if fake else 'Tin thật'}</div>
        <div class="vmeta">Độ tin cậy mô hình <b class="mono">{conf}</b></div>
      </div>
      <div class="vside">
        <span class="pill {pill}">{_icon(ic)} {text}</span>
        <span class="muted mono" style="font-size:12px">Phiên #{r.get('id','—')} · PhoBERT-v2</span>
      </div>
    </div>"""
    axes = (_axis("Tính xác thực", "misinfo", r.get("misinfo_label",""), r.get("misinfo_score",0), "Ngưỡng 50%")
            + _axis("Lập trường", "stance", r.get("stance_label",""), r.get("stance_score",0), "Favor · Against · Neutral")
            + _axis("Cảm xúc", "sentiment", r.get("sentiment_label",""), r.get("sentiment_score",0), "Pos · Neg · Neutral"))
    card = f"""
    <div class="card card-pad" style="margin-top:20px">
      <div class="section-label">{_icon('i-analyze','icon sm ic')} Kết quả phân loại nhãn</div>
      <div class="axes">{axes}</div>
      <div style="border:1px solid var(--line);border-radius:var(--r);background:var(--surface-2);padding:16px;margin-top:16px">
        <div class="muted" style="font-size:11px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;margin-bottom:10px">Phân phối softmax · phobert_probs</div>
        {_distribution(r.get('phobert_probs'))}
      </div>
      {_consistency_legend(r.get('consistency_flag','plausible'))}
    </div>"""
    return f'<div class="stack">{verdict}{card}</div>'

# ----------------------------------------------------------------------
# RENDER: XAI (CoT streaming + bảng bất đồng thuận)
# ----------------------------------------------------------------------
def render_cot(reasoning: str, streaming: bool = True, parse_ok: bool = True, raw_output: str = "") -> str:
    """Vùng CoT. Khi stream: streaming=True (hiện con trỏ). Khi xong: streaming=False."""
    cursor = '<span class="cursor"></span>' if streaming else ""
    if (not parse_ok) and raw_output:
        body = (f'<span class="pill warn">{_icon("i-shield")} Định dạng chưa tối ưu — hiển thị thô</span>\n'
                + _esc(raw_output))
    else:
        body = _esc(reasoning or "") + cursor
    return f'<div class="streamwrap">{body}</div>'

def render_disagreement(r: dict, gemma_labels: dict, disagreement: dict) -> str:
    """Bảng PhoBERT vs Gemma. disagreement[key]=True nghĩa là lệch."""
    rows = ""
    for key, vi in (("misinfo", "Tính xác thực"), ("stance", "Lập trường"), ("sentiment", "Cảm xúc")):
        p = r.get(f"{key}_label", "")
        g = (gemma_labels or {}).get(key)
        diff = bool((disagreement or {}).get(key))
        rows += f"""<tr class="{'flag' if diff else ''}">
          <td>{vi}</td><td>{_vi(key, p)}</td><td>{_vi(key, g) if g else '—'}</td>
          <td style="text-align:center" class="{'no' if diff else 'yes'}">{'≠' if diff else '✓'}</td></tr>"""
    return f"""<div style="margin-top:20px">
      <div class="section-label" style="margin-bottom:10px">{_icon('i-scale','icon sm ic')} Bất đồng thuận nhãn — PhoBERT vs Gemma</div>
      <table class="dtable">
        <thead><tr><th>Trục</th><th>PhoBERT-v2</th><th>Gemma-4-4B</th><th style="text-align:center">Khớp</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>"""

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
