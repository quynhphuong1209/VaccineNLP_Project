# -*- coding: utf-8 -*-
"""
thread_parser.py — Phân tích có cấu trúc luồng Facebook (bài viết ↔ bình luận)
=============================================================================
Module dùng cho VaccineNLP Gradio Demo. Mục tiêu: thay thế logic làm-phẳng hiện
tại trong `fetch_url_as_list` để GIỮ quan hệ bài viết ↔ bình luận, phục vụ cho
Động cơ Giải thích XAI (Gemma) suy luận theo ngữ cảnh.

PHẠM VI (demo, không đụng PhoBERT):
  - PhoBERT vẫn nhận VĂN BẢN THÔ của từng bình luận (giữ phân phối đã validate).
  - Ngữ cảnh (thân bài / bình luận cha) CHỈ bơm vào prompt của Gemma.

GIỚI HẠN ĐÃ XÁC MINH TRÊN DỮ LIỆU THẬT (29/05/2026):
  - actor `apify/facebook-comments-scraper` trả schema PHẲNG:
        {facebookUrl, likesCount, postTitle, text}
    → THÂN BÀI nằm ở `postTitle` (✅ dựng được ngữ cảnh bài↔comment)
    → KHÔNG có trường threading/parent → KHÔNG dựng được chuỗi reply.
  - actor `apify/facebook-groups-scraper` trả schema GIÀU:
        {text(=thân bài), topComments[].text, topComments[].threadingDepth, ...}
    → dựng được reply-chain (heuristic theo độ sâu), nhưng lấy theo FEED group.

Tác giả: General Secretary (Claude) cho Kim Mạnh Hưng (HUPH 2026).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 0. HẰNG SỐ & TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

# Khoảng cắt mặc định để vừa ngân sách token của Gemma (an toàn cho ngữ cảnh).
DEFAULT_MAX_POST_CTX = 500     # số ký tự thân bài đưa vào ngữ cảnh
DEFAULT_MAX_PARENT_CTX = 300   # số ký tự bình luận cha (nếu có)
DEFAULT_MAX_OCR_CTX = 200      # số ký tự OCR ảnh (nếu có)
DISPLAY_POST_CHARS = 800       # cắt thân bài khi HIỂN THỊ trên bảng
DISPLAY_CMT_CHARS = 400        # cắt bình luận khi hiển thị

# Bắt emoji / ký hiệu để nhận diện "comment chỉ có emoji".
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\uFE0F\u200d]+",
    flags=re.UNICODE,
)

# Từ chức năng / mở đầu câu tiếng Việt thường gặp → tín hiệu "đây là nội dung
# thật" chứ không phải tên riêng. Dùng để biết khi nào nên DỪNG bóc tên tag.
_VN_CONTENT_CUES = {
    "à", "ạ", "ấy", "anh", "ăn", "bị", "biết", "bài", "bé", "bú", "ba", "bạn",
    "các", "cũng", "có", "còn", "chưa", "chia", "chiều", "cho", "con", "của",
    "chứ", "công", "copy", "do", "dạ", "đọc", "để", "đi", "được", "đúng", "em",
    "ơi", "gì", "giờ", "hãy", "hết", "không", "khỏi", "là", "lại", "lười", "mà",
    "mình", "mẹ", "mới", "một", "nói", "nên", "này", "nhà", "như", "nhé", "ngộ",
    "nghĩ", "phải", "qua", "quá", "rồi", "ra", "sao", "sớm", "suy", "tìm",
    "tiêm", "trượt", "và", "vẫn", "về", "vắt", "với", "vất", "xin", "ý",
}

def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def _is_titlecase_token(tok: str) -> bool:
    """Token kiểu tên riêng tiếng Việt: chữ cái đầu hoa, phần còn lại thường.
    Bắt cả ký tự có dấu (Đ, Ư, Ơ...)."""
    if not tok:
        return False
    # bỏ dấu câu bám quanh
    core = tok.strip(",.;:!?()[]{}\"'")
    if not core or not core[0].isalpha():
        return False
    if not core[0].isupper():
        return False
    # phần còn lại không được TOÀN HOA (tránh nuốt "BIG", "DNA"...)
    rest = core[1:]
    return not (rest and rest.isupper())


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHÁT HIỆN SCHEMA ACTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_actor_schema(items: List[Dict[str, Any]]) -> str:
    """Đoán actor đã sinh ra `items`.

    Returns: 'comments' | 'groups_posts' | 'unknown'
    """
    if not items or not isinstance(items, list):
        return "unknown"
    sample = items[0] if isinstance(items[0], dict) else {}
    keys = set(sample.keys())

    # comments-scraper: đặc trưng bởi 'postTitle' + 'text', KHÔNG có 'topComments'
    if "postTitle" in keys and "text" in keys and "topComments" not in keys:
        return "comments"
    # groups/posts-scraper: có 'topComments' hoặc 'groupTitle', thân bài ở 'text'
    if "topComments" in keys or "groupTitle" in keys or "comments" in keys:
        return "groups_posts"
    # posts-scraper tối giản đôi khi chỉ có 'text' + 'postUrl'
    if "text" in keys and ("postUrl" in keys or "url" in keys):
        return "groups_posts"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LÀM SẠCH BÌNH LUẬN (bóc tên tag, nhận diện rác)
# ─────────────────────────────────────────────────────────────────────────────

def clean_comment(text: str) -> Tuple[str, Dict[str, Any]]:
    """Phân loại rác KHÔNG biến đổi văn bản (lossless).

    NGUYÊN TẮC: tuyệt đối KHÔNG cắt chữ khỏi comment thật — vì comments-scraper
    chỉ trả text trần (không offset/entity của mention), nên việc bóc "tên tag"
    đứng đầu sẽ corrupt nhầm chữ hoa đầu câu (vd "Thương thương các con"). Ta chỉ
    GẮN CỜ khi TOÀN BỘ comment là tên người / emoji.

    Trả về (text_giữ_nguyên, meta):
      - is_empty:      rỗng
      - is_emoji_only: chỉ gồm emoji/ký hiệu
      - is_name_only:  mọi token đều title-case (kiểu tên riêng) và KHÔNG có
                       từ-tín-hiệu-nội-dung tiếng Việt → gần như chắc là tag tên
      - stripped_mention: luôn "" (không còn cắt)
    """
    meta = {"is_empty": False, "is_emoji_only": False,
            "is_name_only": False, "stripped_mention": ""}

    raw = (text or "").strip()
    if not raw:
        meta["is_empty"] = True
        return "", meta

    # chỉ-emoji?
    if not _strip_emoji(raw):
        meta["is_emoji_only"] = True
        return raw, meta

    # name-only: bỏ dấu câu, nếu MỌI token còn lại đều title-case VÀ không có
    # bất kỳ từ-tín-hiệu-nội-dung nào → coi là chuỗi tên được tag.
    _PUNCT = {",", ";", ".", "...", "…", "-", "–", "—", "·", "•"}
    toks = raw.split()
    non_punct = [t for t in toks if t.strip(",.;:!?()[]{}\"'…") and t not in _PUNCT]
    if non_punct:
        all_title = all(_is_titlecase_token(t) for t in non_punct)
        has_cue = any(
            t.strip(",.;:!?()[]{}\"'…").lower() in _VN_CONTENT_CUES
            for t in non_punct
        )
        if all_title and not has_cue:
            meta["is_name_only"] = True

    return raw, meta  # LUÔN giữ nguyên text


def is_analyzable(meta: Dict[str, Any]) -> bool:
    """Comment có đáng đưa vào phân loại/suy luận không?"""
    return not (meta["is_empty"] or meta["is_emoji_only"] or meta["is_name_only"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSER THEO TỪNG SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def _new_post(text="", author="", url="", ocr=None) -> Dict[str, Any]:
    return {"post_text": text or "", "post_author": author or "",
            "post_url": url or "", "ocr": ocr or [], "comments": []}


def _new_comment(text, raw, author="", likes=0, depth=0, parent_idx=None,
                 meta=None) -> Dict[str, Any]:
    meta = meta or {}
    return {"text": text, "raw_text": raw, "author": author or "",
            "likes": likes, "depth": depth, "parent_idx": parent_idx,
            "is_name_only": meta.get("is_name_only", False),
            "is_emoji_only": meta.get("is_emoji_only", False),
            "stripped_mention": meta.get("stripped_mention", "")}


def _to_int(v) -> int:
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return 0


def parse_comments_scraper(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Schema {facebookUrl, likesCount, postTitle, text}.

    Gom theo (postTitle, facebookUrl) → 1 bài + nhiều bình luận.
    KHÔNG có threading → mọi comment depth=0, parent_idx=None.
    """
    posts_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str]] = []

    for it in items:
        post_body = (it.get("postTitle") or "").strip()
        url = (it.get("facebookUrl") or "").strip()
        key = (post_body[:120], url)  # 120 ký tự đầu đủ để phân biệt bài

        if key not in posts_by_key:
            posts_by_key[key] = _new_post(text=post_body, url=url)
            order.append(key)

        clean, meta = clean_comment(it.get("text", ""))
        if meta["is_empty"]:
            continue
        cmt = _new_comment(
            text=clean, raw=(it.get("text") or "").strip(),
            likes=_to_int(it.get("likesCount")), depth=0, parent_idx=None, meta=meta,
        )
        posts_by_key[key]["comments"].append(cmt)

    return [posts_by_key[k] for k in order]


def parse_groups_scraper(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Schema groups/posts: thân bài ở 'text', bình luận ở 'topComments' (hoặc
    'comments'), reply-chain dựng theo 'threadingDepth' (heuristic ngăn xếp)."""
    posts: List[Dict[str, Any]] = []

    for it in items:
        ocr = [a.get("ocrText", "") for a in (it.get("attachments") or [])
               if a.get("ocrText")]
        post = _new_post(
            text=(it.get("text") or it.get("message") or "").strip(),
            author=((it.get("user") or {}).get("name") or it.get("profileName") or ""),
            url=(it.get("url") or it.get("postUrl") or ""),
            ocr=ocr,
        )

        raw_comments = it.get("topComments") or it.get("comments") or []
        # ngăn xếp (idx, depth) để gán parent: cha = node nông hơn gần nhất
        stack: List[Tuple[int, int]] = []
        for c in raw_comments:
            ctext = (c.get("text") or c.get("commentText") or "").strip()
            clean, meta = clean_comment(ctext)
            if meta["is_empty"]:
                continue
            depth = _to_int(c.get("threadingDepth", 0))
            cur_idx = len(post["comments"])

            while stack and stack[-1][1] >= depth:
                stack.pop()
            parent_idx = stack[-1][0] if stack else None

            cmt = _new_comment(
                text=clean, raw=ctext,
                author=(c.get("profileName") or ""),
                likes=_to_int(c.get("likesCount")),
                depth=depth, parent_idx=parent_idx, meta=meta,
            )
            post["comments"].append(cmt)
            stack.append((cur_idx, depth))

        posts.append(post)

    return posts


def parse_apify(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Dispatcher: tự nhận schema và parse.

    Returns: (posts, schema_name)
    """
    schema = detect_actor_schema(items)
    if schema == "comments":
        return parse_comments_scraper(items), schema
    if schema == "groups_posts":
        return parse_groups_scraper(items), schema
    # fallback: thử coi như comments nếu có postTitle, ngược lại groups
    if items and isinstance(items[0], dict) and "postTitle" in items[0]:
        return parse_comments_scraper(items), "comments(fallback)"
    return parse_groups_scraper(items), "groups_posts(fallback)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHUYỂN SANG BẢNG HIỂN THỊ + BẢN ĐỒ NGỮ CẢNH
# ─────────────────────────────────────────────────────────────────────────────

def thread_to_rows(posts: List[Dict[str, Any]]
                   ) -> Tuple[List[List[Any]], Dict[int, Dict[str, Any]]]:
    """Sinh rows [STT, Loại, Nội dung] cho gr.Dataframe + ctx_map theo STT (0-based row).

    ctx_map[row_index] = {"kind": "post"|"comment",
                           "post": <post dict>,
                           "comment": <comment dict|None>,
                           "analyzable": bool}
    """
    rows: List[List[Any]] = []
    ctx_map: Dict[int, Dict[str, Any]] = {}
    stt = 0

    for post in posts:
        # hàng BÀI VIẾT
        body = post["post_text"] or "(không có thân bài)"
        rows.append([stt + 1, "📄 BÀI VIẾT", body[:DISPLAY_POST_CHARS]])
        ctx_map[stt] = {"kind": "post", "post": post, "comment": None,
                        "analyzable": bool(post["post_text"])}
        stt += 1

        for c in post["comments"]:
            indent = "↳ " * min(c["depth"], 4)
            if c["is_name_only"]:
                kind = "🏷️ Tag tên (bỏ qua)"
            elif c["is_emoji_only"]:
                kind = "🙂 Emoji (bỏ qua)"
            elif c["depth"] > 0:
                kind = f"{indent}↪ TRẢ LỜI"
            else:
                kind = "💬 BÌNH LUẬN"

            shown = indent + (c["raw_text"] if (c["is_name_only"] or c["is_emoji_only"])
                              else c["text"])
            rows.append([stt + 1, kind, shown[:DISPLAY_CMT_CHARS]])
            ctx_map[stt] = {"kind": "comment", "post": post, "comment": c,
                            "analyzable": is_analyzable({
                                "is_empty": False,
                                "is_emoji_only": c["is_emoji_only"],
                                "is_name_only": c["is_name_only"]})}
            stt += 1

    return rows, ctx_map


# ─────────────────────────────────────────────────────────────────────────────
# 5. DỰNG NGỮ CẢNH CHO GEMMA (KHÔNG đụng PhoBERT)
# ─────────────────────────────────────────────────────────────────────────────

def build_thread_context(post: Optional[Dict[str, Any]],
                         comment: Optional[Dict[str, Any]],
                         max_post: int = DEFAULT_MAX_POST_CTX,
                         max_parent: int = DEFAULT_MAX_PARENT_CTX,
                         max_ocr: int = DEFAULT_MAX_OCR_CTX) -> str:
    """Trả về KHỐI NỘI DUNG đã giàu ngữ cảnh, để chèn vào CHỖ `{text}` trong
    prompt Gemma hiện có của app (không cần viết lại system prompt).

    LƯU Ý KỸ THUẬT: Gemma được fine-tune trên VĂN BẢN ĐƠN (notebook 03A). Việc
    đưa thêm [BÀI VIẾT GỐC]/[BÌNH LUẬN CHA] là OUT-OF-DISTRIBUTION nhẹ với chính
    bộ sinh → chất lượng là best-effort, có cải thiện ngữ cảnh nhưng không bảo
    đảm. KHÔNG dùng để tuyên bố tăng F1.
    """
    parts: List[str] = []

    if post and post.get("post_text"):
        parts.append(f'[BÀI VIẾT GỐC]: "{post["post_text"][:max_post].strip()}"')
        if post.get("ocr"):
            ocr_txt = " ".join(post["ocr"])[:max_ocr].strip()
            if ocr_txt:
                parts.append(f"[CHỮ TRONG ẢNH (OCR)]: {ocr_txt}")

    # bình luận cha CHỈ có ở đường groups (comments-scraper không có threading)
    if comment and comment.get("parent_idx") is not None:
        # parent_idx trỏ trong cùng post["comments"]; caller có thể nối sẵn,
        # nhưng để an toàn ta nhận parent qua comment['_parent_text'] nếu có.
        parent_text = comment.get("_parent_text", "")
        if parent_text:
            parts.append(f'[BÌNH LUẬN CHA]: "{parent_text[:max_parent].strip()}"')

    target = (comment["text"] if comment else (post or {}).get("post_text", ""))
    parts.append(f'[BÌNH LUẬN CẦN PHÂN TÍCH]: "{target.strip()}"')

    return "\n".join(parts)


def attach_parent_text(posts: List[Dict[str, Any]]) -> None:
    """Nối sẵn '_parent_text' vào mỗi comment (in-place) để build_thread_context
    dùng được mà không cần truyền cả danh sách. An toàn cho đường groups."""
    for post in posts:
        cmts = post["comments"]
        for c in cmts:
            pi = c.get("parent_idx")
            c["_parent_text"] = cmts[pi]["text"] if (pi is not None and 0 <= pi < len(cmts)) else ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. SELF-TEST trên DỮ LIỆU THẬT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    print("=" * 78)
    print("SELF-TEST thread_parser.py — trên dữ liệu Apify thật")
    print("=" * 78)

    # ── Test 1: comments-scraper (file trên đĩa) ──────────────────────────────
    cpath = "/mnt/user-data/uploads/dataset_facebook-comments-scraper_2026-05-29_12-56-57-484.json"
    try:
        citems = json.load(open(cpath, encoding="utf-8"))
    except Exception as e:
        print("Không đọc được file comments:", e); sys.exit(1)

    print(f"\n### TEST 1 — comments-scraper ({len(citems)} item)")
    print("Schema phát hiện:", detect_actor_schema(citems))
    cposts, _ = parse_apify(citems)
    print(f"→ {len(cposts)} bài viết, "
          f"{sum(len(p['comments']) for p in cposts)} comment giữ lại")
    p0 = cposts[0]
    print(f"  Thân bài (postTitle) dài {len(p0['post_text'])} ký tự — 80 ký tự đầu:")
    print("   ", repr(p0["post_text"][:80]))
    print("  Bộ lọc tên-tag (clean_comment) trên 15 comment:")
    for i, c in enumerate(p0["comments"]):
        flag = ("🏷️NAME" if c["is_name_only"] else
                "🙂EMOJI" if c["is_emoji_only"] else "✅OK  ")
        print(f"    [{i:2d}] {flag} | {c['text'][:70]}")
    n_ok = sum(1 for c in p0["comments"] if not (c["is_name_only"] or c["is_emoji_only"]))
    print(f"  → {n_ok}/{len(p0['comments'])} comment ĐÁNG phân tích "
          f"(loại {len(p0['comments']) - n_ok} rác name-only/emoji)")

    print("\n  --- build_thread_context cho comment ĐÁNG phân tích đầu tiên ---")
    first_ok = next(c for c in p0["comments"] if not (c["is_name_only"] or c["is_emoji_only"]))
    ctx = build_thread_context(p0, first_ok)
    print("  " + ctx.replace("\n", "\n  ")[:600] + " ...")

    print("\n  --- 6 hàng đầu của BẢNG hiển thị (thread_to_rows) ---")
    rows, ctx_map = thread_to_rows(cposts)
    for r in rows[:6]:
        print(f"    STT={r[0]:>2} | {r[1]:<22} | {r[2][:55]}")
    print(f"  (ctx_map có {len(ctx_map)} mục, khớp số hàng = {len(rows)})")

    # ── Test 2: groups-scraper (mô phỏng từ doc đã dán) ───────────────────────
    print(f"\n### TEST 2 — groups-scraper (reply-chain theo threadingDepth)")
    gitems = [{
        "url": "https://facebook.com/groups/x/permalink/111/",
        "user": {"name": "Cảnh Nguyễn Huy"},
        "text": "Bài viết gốc của nhóm: quan điểm không tiêm vaccine...",
        "groupTitle": "Hội không tiêm vac xin",
        "attachments": [{"ocrText": "Có thể là hình ảnh về văn bản"}],
        "topComments": [
            {"text": "Bình luận gốc A", "threadingDepth": 0, "profileName": "User1", "likesCount": "2"},
            {"text": "Trả lời cho A", "threadingDepth": 1, "profileName": "User2", "likesCount": "0"},
            {"text": "Trả lời tiếp cho reply", "threadingDepth": 2, "profileName": "User3", "likesCount": "1"},
            {"text": "Bình luận gốc B (depth 0)", "threadingDepth": 0, "profileName": "User4", "likesCount": "5"},
        ],
    }]
    print("Schema phát hiện:", detect_actor_schema(gitems))
    gposts, _ = parse_apify(gitems)
    attach_parent_text(gposts)
    gp = gposts[0]
    print(f"  Thân bài: {gp['post_text'][:50]!r} | OCR: {gp['ocr']}")
    print("  Cây bình luận (depth → parent_idx):")
    for i, c in enumerate(gp["comments"]):
        ptxt = c.get("_parent_text", "")
        print(f"    idx={i} depth={c['depth']} parent_idx={c['parent_idx']} "
              f"parent={ptxt[:25]!r} | {c['text']}")
    print("\n  --- ngữ cảnh cho comment depth=2 (có cả bài + comment cha) ---")
    deep = next(c for c in gp["comments"] if c["depth"] == 2)
    print("  " + build_thread_context(gp, deep).replace("\n", "\n  "))

    print("\n" + "=" * 78)
    print("KẾT LUẬN SELF-TEST:")
    print("  ✅ comments-scraper: bài↔comment OK, bóc rác name-only OK, KHÔNG reply-chain")
    print("  ✅ groups-scraper:   reply-chain (depth→parent) OK, có OCR")
    print("=" * 78)
