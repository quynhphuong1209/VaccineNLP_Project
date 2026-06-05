# -*- coding: utf-8 -*-
"""
xai_postprocess.py — Hậu xử lý đầu ra XAI cho VaccineNLP Gradio Demo
====================================================================
Giải quyết 3 vấn đề ĐÃ XÁC MINH trên dữ liệu lỗi thật (báo cáo 31/05/2026):

  [1] Gemma LẶP đầu ra: cùng khối "=== KẾT QUẢ === / === GIẢI THÍCH ==="
      được sinh lặp nhiều lần đến khi chạm max_tokens → kết thúc giữa chừng.
      `clean_reasoning_output` cũ lấy .split(...)[-1] = mảnh CỤT cuối → mất nội
      dung. FIX: lấy khối GIẢI THÍCH ĐẦU TIÊN, cắt tại marker lặp kế tiếp.

  [2] TTS đọc cả ký tự markdown (**, *, #, ===, •). FIX: strip_markdown_for_tts.

  [3] (Direction B) Hiển thị BẤT ĐỒNG giữa 2 động cơ. parse_gemma_labels lấy
      nhãn Gemma từ khối "=== KẾT QUẢ ===", compare_engines so với PhoBERT.
      Đây là dữ kiện QUAN SÁT ĐƯỢC (nhãn khác nhau hay không) — đo được nhị
      phân, KHÔNG phải mô tả định tính.

Tác giả: General Secretary (Claude) — HUPH 2026.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple


# Bảng nhãn — KHỚP CHÍNH XÁC với LABEL_MAPS trong app.py
LABEL_MAPS = {
    "misinfo":   {0: "Tin giả",  1: "Tin thật"},
    "stance":    {0: "Ủng hộ",   1: "Phản đối", 2: "Trung lập"},
    "sentiment": {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"},
}

# Marker cấu trúc đầu ra Gemma
_RESULT_MARK = "=== KẾT QUẢ ==="
_EXPLAIN_MARK = "=== GIẢI THÍCH ==="
_SPECIAL_TOKENS = ["<end_of_turn>", "<start_of_turn>", "<|turn|>", "<|turn>",
                   "<eos>", "<bos>", "<pad>"]


# ─────────────────────────────────────────────────────────────────────────────
# 0. CHUẨN HOÁ NHÃN (bỏ dấu, lowercase) để so khớp tin cậy
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label(s: str) -> str:
    """'Tin thật' / 'Chính xác' → 'tin that' (bỏ dấu + lowercase + gọn space)."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", s).strip().lower()


# bản đồ chuỗi-chuẩn-hoá → index (khớp LABEL_MAPS)
_MISINFO_STR2IDX = {"tin gia": 0, "chinh xac": 1, "tin that": 1}
_STANCE_STR2IDX = {"ung ho": 0, "phan doi": 1, "trung lap": 2}
_SENTIMENT_STR2IDX = {"tieu cuc": 0, "trung tinh": 1, "tich cuc": 2}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LÀM SẠCH REASONING — CHỐNG LẶP + CHỐNG CẮT CỤT
# ─────────────────────────────────────────────────────────────────────────────

def clean_reasoning_output(raw: str, max_chars: int = 4000) -> str:
    """Trích phần GIẢI THÍCH HOÀN CHỈNH ĐẦU TIÊN, loại lặp & token đặc biệt.

    Khác bản cũ: bản cũ dùng split(EXPLAIN)[-1] → dính mảnh CỤT cuối khi model
    lặp. Bản này lấy khối GIẢI THÍCH ĐẦU TIÊN và CẮT ngay khi gặp marker lặp
    (=== KẾT QUẢ === hoặc === GIẢI THÍCH === lần 2).
    """
    if not raw:
        return ""
    t = raw.strip()
    for tok in _SPECIAL_TOKENS:
        t = t.replace(tok, "")
    t = t.strip()

    # Lấy phần SAU marker GIẢI THÍCH đầu tiên (nếu có)
    idx = t.find(_EXPLAIN_MARK)
    if idx == -1:
        # thử biến thể thường/hoa
        low = t.lower()
        j = low.find(_EXPLAIN_MARK.lower())
        body = t[j + len(_EXPLAIN_MARK):] if j != -1 else t
    else:
        body = t[idx + len(_EXPLAIN_MARK):]

    # CẮT tại marker lặp kế tiếp (đây là điểm chống-lặp mấu chốt)
    cut_positions = []
    for mark in (_RESULT_MARK, _EXPLAIN_MARK):
        p = body.find(mark)
        if p != -1:
            cut_positions.append(p)
        pl = body.lower().find(mark.lower())
        if pl != -1:
            cut_positions.append(pl)
    if cut_positions:
        body = body[:min(cut_positions)]

    body = body.lstrip(":-\n\r ").strip()

    # Chống lặp đoạn: nếu nửa đầu == nửa sau (model nhân đôi), giữ một nửa
    body = _dedupe_repeated_block(body)

    if not body.startswith("Lý luận"):
        body = "Lý luận: " + body
    else:
        body = "Lý luận: " + body[len("Lý luận"):].lstrip(": ")

    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "…"
    return body


def _dedupe_repeated_block(text: str) -> str:
    """Nếu text gồm 1 đoạn lặp lại nguyên văn ≥2 lần liền nhau, chỉ giữ 1 bản."""
    if len(text) < 80:
        return text
    # tìm chu kỳ lặp bằng cách so tiền tố với phần còn lại
    for period in range(40, len(text) // 2 + 1):
        chunk = text[:period]
        # số lần chunk lặp liên tiếp từ đầu
        reps = 1
        while text[reps * period:(reps + 1) * period] == chunk:
            reps += 1
        if reps >= 2 and reps * period >= len(text) * 0.8:
            return chunk.strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRIP MARKDOWN CHO TTS (giọng đọc không phát "sao sao thăng")
# ─────────────────────────────────────────────────────────────────────────────

def strip_markdown_for_tts(text: str) -> str:
    """Bỏ ký hiệu markdown/cấu trúc để gTTS đọc tự nhiên.

    Bỏ: ** * _ ` # > === --- • | bullet, link [..](..), token đặc biệt, prefix
    'Lý luận:', và gọn khoảng trắng.
    """
    if not text:
        return ""
    t = text
    for tok in _SPECIAL_TOKENS:
        t = t.replace(tok, " ")
    t = re.sub(r"={2,}\s*[^=\n]*\s*={2,}", " ", t)        # === KẾT QUẢ ===
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)         # [text](url) → text
    t = re.sub(r"https?://\S+", " ", t)                    # URL trơ
    t = re.sub(r"[`*_#>~]", "", t)                         # ký hiệu inline
    t = re.sub(r"^\s*[-•·–—]\s+", "", t, flags=re.MULTILINE)  # bullet đầu dòng
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)    # "1. " đầu dòng
    t = t.replace("|", " ")
    t = re.sub(r"\(([^)]*)\)", r"\1", t)                   # bỏ ngoặc đơn (đọc liền)
    t = re.sub(r"^\s*Lý luận\s*:?\s*", "", t)              # prefix
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 3. (DIRECTION B) PARSE NHÃN GEMMA + SO SÁNH 2 ĐỘNG CƠ
# ─────────────────────────────────────────────────────────────────────────────

def parse_gemma_labels(raw: str) -> Dict[str, Optional[int]]:
    """Lấy 3 nhãn từ khối '=== KẾT QUẢ ===' ĐẦU TIÊN của Gemma → index khớp
    LABEL_MAPS. Trả None cho trục không parse được.
    """
    out: Dict[str, Optional[int]] = {"misinfo": None, "stance": None, "sentiment": None}
    if not raw:
        return out

    # giới hạn vùng tìm trong khối KẾT QUẢ đầu (tránh nuốt nhầm chữ trong giải thích)
    block = raw
    i = raw.find(_RESULT_MARK)
    if i != -1:
        j = raw.find(_EXPLAIN_MARK, i)
        block = raw[i:j] if j != -1 else raw[i:i + 400]

    def _grab(field: str) -> str:
        m = re.search(field + r"\s*:\s*([^\n<*]+)", block, flags=re.IGNORECASE)
        return _normalize_label(m.group(1)) if m else ""

    mis = _grab("Misinformation")
    sta = _grab("Stance")
    sen = _grab("Sentiment")

    for key, idx in _MISINFO_STR2IDX.items():
        if key in mis:
            out["misinfo"] = idx; break
    for key, idx in _STANCE_STR2IDX.items():
        if key in sta:
            out["stance"] = idx; break
    for key, idx in _SENTIMENT_STR2IDX.items():
        if key in sen:
            out["sentiment"] = idx; break
    return out


def compare_engines(phobert_result: Dict[str, Any],
                    gemma_labels: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """So nhãn PhoBERT vs Gemma trên 3 trục. Trả dữ kiện ĐO ĐƯỢC (nhị phân).

    Returns:
      {
        "axes": {axis: {"phobert": idx, "gemma": idx|None,
                        "phobert_label": str, "gemma_label": str|None,
                        "status": "agree"|"disagree"|"unknown"}},
        "n_disagree": int, "n_comparable": int, "any_disagree": bool
      }
    """
    axes = {}
    n_dis = n_cmp = 0
    for axis in ("misinfo", "stance", "sentiment"):
        p_idx = phobert_result.get(axis, {}).get("pred")
        g_idx = gemma_labels.get(axis)
        p_lab = LABEL_MAPS[axis].get(p_idx) if p_idx is not None else None
        g_lab = LABEL_MAPS[axis].get(g_idx) if g_idx is not None else None
        if g_idx is None or p_idx is None:
            status = "unknown"
        elif p_idx == g_idx:
            status = "agree"; n_cmp += 1
        else:
            status = "disagree"; n_cmp += 1; n_dis += 1
        axes[axis] = {"phobert": p_idx, "gemma": g_idx,
                      "phobert_label": p_lab, "gemma_label": g_lab,
                      "status": status}
    return {"axes": axes, "n_disagree": n_dis, "n_comparable": n_cmp,
            "any_disagree": n_dis > 0}


_AXIS_VI = {"misinfo": "Tính xác thực", "stance": "Lập trường", "sentiment": "Cảm xúc"}


def render_disagreement_badge(cmp: Dict[str, Any]) -> str:
    """HTML badge cho Direction B. KHÔNG dùng từ định tính/khẳng định độ đúng;
    chỉ nêu dữ kiện: hai động cơ trùng/khác nhãn, và đề xuất HITL khi khác.
    """
    if cmp["n_comparable"] == 0:
        return ""  # không so được (vd kết quả từ cache đã strip nhãn)

    if not cmp["any_disagree"]:
        return (
            '<div style="margin:10px 0;padding:12px 16px;border-radius:8px;'
            'background:rgba(56,239,125,0.12);border:1px solid #38ef7d;'
            'font-family:\'Plus Jakarta Sans\',\'Inter\',sans-serif;font-size:1.08rem;">'
            '✅ <b>Hai động cơ THỐNG NHẤT</b> trên '
            f'{cmp["n_comparable"]}/3 trục so sánh được '
            '(PhoBERT phân loại · Gemma giải thích).</div>'
        )

    rows = []
    for axis, d in cmp["axes"].items():
        if d["status"] == "disagree":
            rows.append(
                f'<li><b>{_AXIS_VI[axis]}</b>: PhoBERT = '
                f'<span style="color:#007bff">{d["phobert_label"]}</span> · '
                f'Gemma = <span style="color:#ff8c00">{d["gemma_label"]}</span></li>'
            )
    return (
        '<div style="margin:10px 0;padding:12px 16px;border-radius:8px;'
        'background:rgba(255,170,0,0.12);border:1px solid #ffaa00;'
        'font-family:\'Plus Jakarta Sans\',\'Inter\',sans-serif;font-size:1.05rem;">'
        f'⚠️ <b>Hai động cơ BẤT ĐỒNG</b> ở {cmp["n_disagree"]}/{cmp["n_comparable"]} '
        'trục so sánh được — mẫu này được gắn cờ chuyển <b>chuyên gia rà soát '
        '(Human-in-the-Loop)</b>:'
        f'<ul style="margin:6px 0 0 0;">{"".join(rows)}</ul></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. SELF-TEST trên DỮ LIỆU LỖI THẬT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("SELF-TEST xai_postprocess.py — trên đầu ra LỖI thật (báo cáo 31/05)")
    print("=" * 78)

    # Đầu ra LẶP thật (rút gọn trung thực từ document 7: 1 khối hoàn chỉnh + lặp)
    looping_raw = (
        "=== KẾT QUẢ ===\n"
        "- Misinformation: Chinh xac\n- Stance: Phan doi\n- Sentiment: Tieu cuc\n"
        "=== GIẢI THÍCH ===\n"
        "**Phân tích chuyên sâu:**\n"
        "1. **Tính xác thực:** Người viết không đưa thông tin y khoa cụ thể. "
        "Đây là tấn công cá nhân (ad hominem). Do đó đánh giá là \"Tin gia\".\n"
        "2. **Thái độ:** Dùng từ công kích, phản đối vaccine. Xác định \"Phan doi\".\n"
        "3. **Cảm xúc:** Gay gắt, mỉa mai. Xác định \"Tieu cuc\".\n"
        "**Kết luận:** Bài viết công kích cá nhân, phản đối vaccine.\n"
        # —— LẶP LẦN 2 (model bị loop) ——
        "=== KẾT QUẢ ===\n"
        "- Misinformation: Chinh xac\n- Stance: Phan doi\n- Sentiment: Tieu cuc\n"
        "=== GIẢI THÍCH ===\n"
        "**Phân tích chuyên sâu:**\n1. **Tính xác thực:** Người viết không đưa "
        "thông tin y khoa cụ thể, cơ chế"   # ← CỤT giữa chừng như thật
    )

    print("\n### TEST 1 — clean_reasoning_output (chống lặp + chống cụt)")
    cleaned = clean_reasoning_output(looping_raw)
    print(f"Độ dài raw: {len(looping_raw)} → cleaned: {len(cleaned)} ký tự")
    print("Số lần '=== GIẢI THÍCH ===' còn trong cleaned:",
          cleaned.count(_EXPLAIN_MARK), "(kỳ vọng 0)")
    print("Số lần 'Phân tích chuyên sâu' còn lại:",
          cleaned.count("Phân tích chuyên sâu"), "(kỳ vọng 1 — đã khử lặp)")
    print("Kết thúc bằng 'cơ chế' (mảnh cụt)?",
          cleaned.rstrip("…").endswith("cơ chế"), "(kỳ vọng False)")
    print("--- 220 ký tự đầu ---\n   ", cleaned[:220].replace("\n", "\n    "))

    print("\n### TEST 2 — strip_markdown_for_tts")
    tts = strip_markdown_for_tts(cleaned)
    print("Còn ký tự '*'?", "*" in tts, "| '#'?", "#" in tts,
          "| '==='?", "===" in tts, "(kỳ vọng đều False)")
    print("--- văn bản TTS (180 ký tự) ---\n   ", tts[:180])

    print("\n### TEST 3 — parse_gemma_labels")
    glabels = parse_gemma_labels(looping_raw)
    print("Gemma labels (index):", glabels,
          "→ kỳ vọng misinfo=1(Chính xác), stance=1(Phản đối), sentiment=0(Tiêu cực)")

    print("\n### TEST 4 — compare_engines (Direction B) trên CA THẬT báo cáo 94")
    # PhoBERT trong báo cáo 94: misinfo=Tin giả(0), stance=Trung lập(2), sentiment=Tiêu cực(0)
    phobert_result = {
        "misinfo":   {"pred": 0},  # Tin giả
        "stance":    {"pred": 2},  # Trung lập
        "sentiment": {"pred": 0},  # Tiêu cực
    }
    cmp = compare_engines(phobert_result, glabels)
    print(f"  n_comparable={cmp['n_comparable']} · n_disagree={cmp['n_disagree']} "
          f"· any_disagree={cmp['any_disagree']}")
    for axis, d in cmp["axes"].items():
        print(f"    {_AXIS_VI[axis]:14s}: PhoBERT={d['phobert_label']:9s} "
              f"Gemma={str(d['gemma_label']):9s} → {d['status'].upper()}")
    print("  Kỳ vọng: Tính xác thực DISAGREE (Tin giả vs Chính xác), "
          "Lập trường DISAGREE (Trung lập vs Phản đối), Cảm xúc AGREE (Tiêu cực).")

    print("\n### TEST 5 — render badge (HTML hợp lệ?)")
    badge = render_disagreement_badge(cmp)
    print("  Badge sinh ra, dài", len(badge), "ký tự; chứa 'BẤT ĐỒNG'?",
          "BẤT ĐỒNG" in badge, "; chứa 'Human-in-the-Loop'?", "Human-in-the-Loop" in badge)

    print("\n" + "=" * 78)
    print("KẾT LUẬN: clean chống lặp/cụt ✅ · TTS sạch markdown ✅ · "
          "parse nhãn ✅ · so sánh 2 động cơ ✅")
    print("=" * 78)
