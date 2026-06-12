# -*- coding: utf-8 -*-
"""Shared Gemma XAI prompt contract for Web and Gradio entrypoints."""

XAI_MAX_TOKENS = 2048
XAI_TEMPERATURE = 0.1
XAI_STOP = ["<end_of_turn>", "<|end_of_turn|>"]

XAI_SYSTEM_PROMPT = """Bạn là chuyên gia AI phân tích vấn đề y tế công cộng (Explainable AI - XAI).
Hãy suy luận và phân tích nội dung thông tin vaccine bằng TIẾNG VIỆT theo 3 chiều:
(1) Dấu hiệu sai lệch: chỉ chọn "Có dấu hiệu tin giả" hoặc "Không có dấu hiệu sai lệch".
(2) Thái độ với vaccine: chỉ chọn "Ủng hộ", "Phản đối" hoặc "Trung lập".
(3) Cảm xúc tổng thể: chỉ chọn "Tiêu cực", "Trung tính" hoặc "Tích cực".

Lưu ý bắt buộc:
- Phải sử dụng tiếng Việt; không dùng các nhãn tiếng Anh như Reasoning, Therefore, Misinformation, Stance, Sentiment trong câu trả lời.
- Cẩn trọng với phương ngữ địa phương, tiếng lóng, teen-code và từ viết tắt; nếu từ ngữ mơ hồ, hãy dựa vào ngữ cảnh và thái độ tổng thể.
- Nhãn PhoBERT chỉ là tham khảo để đối chiếu, không phải câu trả lời bắt buộc sao chép.
- Trả lời bắt đầu ngay bằng "=== KẾT QUẢ ==="; không thêm lời chào, không thêm token đặc biệt.
- Không thêm marker kết thúc như "=== HẾT GIẢI THÍCH ===".

Trả lời theo ĐÚNG cấu trúc:
=== KẾT QUẢ ===
- Dấu hiệu sai lệch: <Có dấu hiệu tin giả HOẶC Không có dấu hiệu sai lệch>
- Thái độ với vaccine: <Ủng hộ HOẶC Phản đối HOẶC Trung lập>
- Cảm xúc tổng thể: <Tiêu cực HOẶC Trung tính HOẶC Tích cực>
=== GIẢI THÍCH ===
<lý luận chi tiết bằng tiếng Việt, giải thích lần lượt 3 nhãn trên>"""

LABEL_VI = {
    "misinfo": {"Fake": "Có dấu hiệu tin giả", "Real": "Không phát hiện dấu hiệu sai lệch"},
    "stance": {"Favor": "Ủng hộ", "Against": "Phản đối", "Neutral": "Trung lập"},
    "sentiment": {"Positive": "Tích cực", "Negative": "Tiêu cực", "Neutral": "Trung tính"},
}


def predicted_labels_vi(predicted_labels: dict | None) -> dict:
    return {
        axis: LABEL_VI.get(axis, {}).get(value, value)
        for axis, value in (predicted_labels or {}).items()
    }


def build_xai_user_prompt(text: str, predicted_labels: dict | None = None) -> str:
    parts = [
        "Văn bản cần phân tích:",
        (text or "").strip(),
        "",
    ]

    predicted_vi = predicted_labels_vi(predicted_labels)
    if predicted_vi:
        parts.extend([
            "Nhãn PhoBERT tham khảo:",
            f"- Dấu hiệu sai lệch: {predicted_vi.get('misinfo', 'Không có')}",
            f"- Thái độ với vaccine: {predicted_vi.get('stance', 'Không có')}",
            f"- Cảm xúc tổng thể: {predicted_vi.get('sentiment', 'Không có')}",
            "",
        ])

    parts.extend([
        "Yêu cầu bắt buộc:",
        "- Chỉ trả lời bằng tiếng Việt.",
        "- Không dùng nhãn tiếng Anh như Reasoning, Therefore, Misinformation, Stance, Sentiment.",
        "- Cẩn trọng với phương ngữ địa phương, tiếng lóng, teen-code và từ viết tắt.",
        "- Trả lời bắt đầu ngay bằng === KẾT QUẢ ===, không thêm lời chào hay token đặc biệt.",
        "",
        "Trả lời theo đúng cấu trúc:",
        "=== KẾT QUẢ ===",
        "- Dấu hiệu sai lệch: <Có dấu hiệu tin giả HOẶC Không có dấu hiệu sai lệch>",
        "- Thái độ với vaccine: <Ủng hộ HOẶC Phản đối HOẶC Trung lập>",
        "- Cảm xúc tổng thể: <Tiêu cực HOẶC Trung tính HOẶC Tích cực>",
        "=== GIẢI THÍCH ===",
        "<lý luận chi tiết bằng tiếng Việt, giải thích lần lượt 3 nhãn trên>",
    ])
    return "\n".join(parts)


def build_xai_messages(text: str, predicted_labels: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": XAI_SYSTEM_PROMPT},
        {"role": "user", "content": build_xai_user_prompt(text, predicted_labels)},
    ]
