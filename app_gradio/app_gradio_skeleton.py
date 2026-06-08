"""
app_gradio_skeleton.py — KHUNG THAM CHIẾU (không phải app.py production).

Mục đích: minh hoạ cách ghép vnlp_ui.py vào Gradio + luồng streaming 2 nhịp.
Bạn KHÔNG chạy file này trực tiếp lên Space — hãy ghép các mảnh này vào app.py
hiện có (giữ nguyên phần nạp model, Context-Aware XAI, ngrok/Kaggle của bạn),
thay 3 chỗ `# >>> GHÉP HÀM THẬT` bằng hàm thật.

Đồng bộ hoàn toàn với frontend React (cùng redesign.css + cùng ánh xạ nhãn).
"""
import gradio as gr
from vnlp_ui import (
    THEME_CSS, ICON_SPRITE,
    render_result, render_cot, render_disagreement,
)

PRESETS = {
    "Tin giả cực đoan": "Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.",
    "Ủng hộ tiêm chủng": "Em cũng đang tiêm từng mũi 1 cho con, chậm mà đủ và an toàn cho con là được.",
    "Thông tin chuẩn": "Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng.",
}

# ======================================================================
# NHỊP 1 — phân tích nhanh (PhoBERT). Trả HTML verdict + axes + distribution.
# ======================================================================
def do_analyze(text: str):
    if not text.strip():
        return "", gr.update(value="", visible=False)
    # >>> GHÉP HÀM THẬT: gọi PhoBERT của bạn, trả về dict đúng schema dưới đây.
    #     phobert_probs là softmax đầy đủ; consistency_flag từ compute_consistency.
    result = phobert_predict(text)   # noqa: F821  (hàm sẵn có trong app.py của bạn)
    # result = {
    #   "id": 1287,
    #   "misinfo_label": "Fake", "misinfo_score": 0.914,
    #   "stance_label": "Against", "stance_score": 0.78,
    #   "sentiment_label": "Negative", "sentiment_score": 0.642,
    #   "phobert_probs": {"misinfo": {"Fake":0.914,"Real":0.086}, "stance": {...}, "sentiment": {...}},
    #   "consistency_flag": "high_risk",
    # }
    return render_result(result), gr.update(visible=True)

# ======================================================================
# NHỊP 2 — sinh giải thích (Gemma, STREAM). Generator: yield HTML mỗi token.
# ======================================================================
def do_explain(text: str):
    """Generator cho streaming=True của Gradio: mỗi yield cập nhật out_xai."""
    if not text.strip():
        yield ""
        return
    acc = ""
    # >>> GHÉP HÀM THẬT: token_iter là nguồn stream sẵn có của bạn
    #     (LM Studio/ngrok khi dev, hoặc GGUF-CPU khi public — tái dùng nhãn PhoBERT đã cache).
    #     Mỗi phần tử là một mẩu text (token/đoạn).
    for chunk in xai_token_stream(text):   # noqa: F821
        acc += chunk
        yield render_cot(acc, streaming=True)      # con trỏ nhấp nháy

    # khi stream xong: lấy nhãn Gemma + cờ bất đồng thuận đã parse
    # >>> GHÉP HÀM THẬT:
    final = xai_finalize(text, acc)   # noqa: F821
    # final = {"parse_ok": True, "reasoning": acc, "raw_output": "",
    #          "gemma_labels": {"misinfo":"Fake","stance":"Neutral","sentiment":"Negative"},
    #          "disagreement": {"misinfo":False,"stance":True,"sentiment":False},
    #          "phobert_result": result_dict_from_nhip_1}
    body = render_cot(final["reasoning"], streaming=False,
                      parse_ok=final["parse_ok"], raw_output=final.get("raw_output", ""))
    if final["parse_ok"] and final.get("gemma_labels"):
        body += render_disagreement(final["phobert_result"], final["gemma_labels"], final["disagreement"])
    yield body

# ======================================================================
# LAYOUT  (header bằng gr.HTML dùng class redesign; 2 cột input/kết quả)
# Lưu ý paradigm: KHÔNG dựng được sidebar dính + topbar y hệt React,
# nhưng tông màu/typography/khối kết quả khớp 100% nhờ chung CSS.
# ======================================================================
HEADER = """
<header class="topbar" style="border-radius:var(--r-lg);margin-bottom:18px">
  <div class="brand" style="padding:0">
    <div class="logo"><svg class="icon lg"><use href="#i-shield"/></svg></div>
    <div><div class="name">Vaccine<span>NLP</span></div>
    <div class="tag">Phát hiện tin giả · Phân tích thái độ</div></div>
  </div>
  <div class="spacer"></div>
  <span class="status-chip"><span class="pulse"></span> PhoBERT-v2 · trực tuyến</span>
</header>"""

with gr.Blocks(css=THEME_CSS, theme=gr.themes.Base(), title="VaccineNLP") as demo:
    gr.HTML(ICON_SPRITE)
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, elem_classes="card card-pad"):
            inp = gr.Textbox(label="Nội dung cần đối soát (tiếng Việt)", lines=6,
                             value=PRESETS["Tin giả cực đoan"],
                             placeholder="Dán bình luận, bài viết hoặc tin nhắn về vaccine…")
            preset = gr.Radio(list(PRESETS.keys()), label="Bộ ví dụ mẫu",
                              value="Tin giả cực đoan")
            preset.change(lambda k: PRESETS[k], preset, inp)
            analyze_btn = gr.Button("Tiến hành phân tích đa nhiệm", variant="primary")

        with gr.Column(scale=3):
            out_result = gr.HTML()
            with gr.Group(visible=False, elem_classes="card card-pad") as xai_card:
                gr.HTML('<div class="section-label"><svg class="icon sm ic"><use href="#i-spark"/></svg> '
                        'Giải thích của mô hình (XAI)</div>'
                        '<p class="muted" style="font-size:12px;margin:6px 0">Nhịp 1 · PhoBERT tức thời → '
                        'Nhịp 2 · Gemma-4B stream CoT theo token</p>')
                explain_btn = gr.Button("Sinh giải thích (Gemma, chậm)", variant="secondary")
                out_xai = gr.HTML()

    # nhịp 1: analyze → hiện card XAI
    analyze_btn.click(do_analyze, inp, [out_result, xai_card])
    # nhịp 2: explain → stream vào out_xai (Gradio tự cập nhật mỗi yield)
    explain_btn.click(do_explain, inp, out_xai)

if __name__ == "__main__":
    demo.launch()
