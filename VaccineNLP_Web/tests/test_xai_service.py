# -*- coding: utf-8 -*-
import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Force backend configuration for test duration before importing
os.environ["XAI_BACKEND"] = "lmstudio"
os.environ["LMSTUDIO_URL"] = "http://mock-lmstudio:1234/v1"

# Add xai_service path to import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../xai_service')))

# Clear cached app modules to avoid namespace collisions with api_service
for k in list(sys.modules.keys()):
    if k == 'app' or k.startswith('app.'):
        sys.modules.pop(k)

from app.main import parse_gemma_output, app, _build_messages

from fastapi.testclient import TestClient

class TestXaiServiceLogic(unittest.TestCase):
    def test_parse_gemma_output_standard(self):
        """Test parse_gemma_output with a standard, properly formatted Gemma response."""
        raw = "Vắc-xin là phát minh y học vĩ đại giúp bảo vệ cộng đồng khỏi dịch bệnh.\nKết quả: Chính Xác | Ủng Hộ | Tích Cực"
        out = parse_gemma_output(raw)
        
        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Real")
        self.assertEqual(out["gemma_labels"]["stance"], "Favor")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Positive")
        self.assertIn("phát minh y học vĩ đại", out["reasoning"])

    def test_parse_gemma_output_alternative_casing_and_names(self):
        """Test parsing of different Vietnamese labels and formats."""
        raw = "Kết luận: Tin Giả | Phản Đối | Tiêu Cực"
        out = parse_gemma_output(raw)
        
        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Fake")
        self.assertEqual(out["gemma_labels"]["stance"], "Against")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Negative")

    def test_parse_gemma_output_structured_result_block(self):
        """Test the notebook-trained block format used by LM Studio/Gemma."""
        raw = """=== KẾT QUẢ ===
- Tính xác thực: Tin giả
- Thái độ với vaccine: Phản đối
- Cảm xúc tổng thể: Tiêu cực
=== GIẢI THÍCH ===
Nội dung đưa ra các khẳng định không có bằng chứng về vaccine và thể hiện thái độ phản đối tiêm chủng.
=== HẾT GIẢI THÍCH ==="""
        out = parse_gemma_output(raw)

        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Fake")
        self.assertEqual(out["gemma_labels"]["stance"], "Against")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Negative")
        self.assertIn("khẳng định không có bằng chứng", out["reasoning"])
        self.assertNotIn("HẾT GIẢI THÍCH", out["reasoning"])

    def test_parse_gemma_output_current_raw_lmstudio_shape(self):
        """Parse the current bad LM Studio shape instead of falling back to raw UI."""
        raw = (
            "Reasoning: The user makes several highly specific and unsubstantiated claims regarding vaccines.\n"
            "Therefore: Misinformation [Tin giả], Stance [Phản đối], Sentiment [Tiêu cực].\n"
            "Kết quả: Misinformation | Phản đối | Tiêu cực<end_of_turn>"
        )
        out = parse_gemma_output(raw)

        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Fake")
        self.assertEqual(out["gemma_labels"]["stance"], "Against")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Negative")
        self.assertNotIn("<end_of_turn>", out["reasoning"])
        self.assertNotIn("Therefore:", out["reasoning"])

    def test_parse_gemma_output_tolerates_axis_and_label_typos(self):
        raw = """=== KẾT QUẢ ===
- Tính xác thực: Tin giả
- Thái độ với vaccine: Phản đối
- Cảm Thiếu Tổng Thể: Tiêu thích
=== GIẢI THÍCH ===
Nội dung phủ nhận vai trò vaccine và tạo sắc thái tiêu cực."""
        out = parse_gemma_output(raw)

        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Fake")
        self.assertEqual(out["gemma_labels"]["stance"], "Against")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Negative")

    def test_build_messages_uses_vietnamese_structured_prompt(self):
        messages = _build_messages(
            "Vắc xin COVID gây vô sinh.",
            {"misinfo": "Fake", "stance": "Against", "sentiment": "Negative"},
        )

        self.assertIn("=== KẾT QUẢ ===", messages[0]["content"])
        self.assertIn("=== GIẢI THÍCH ===", messages[0]["content"])
        self.assertIn("Tính xác thực: Tin giả", messages[1]["content"])
        self.assertIn("Thái độ với vaccine: Phản đối", messages[1]["content"])
        self.assertIn("Cảm xúc tổng thể: Tiêu cực", messages[1]["content"])
        self.assertNotIn("{'misinfo'", messages[1]["content"])

    def test_parse_gemma_output_with_thought_channel(self):
        """Test parsing of response containing CoT thought channel tags."""
        raw = "<|channel>thought\nUser thinks vaccines are bad but this is real\n<|channel>thought\nKết luận: Không Tin Giả | Trung Lập | Trung Tính"
        out = parse_gemma_output(raw)
        
        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["gemma_labels"]["misinfo"], "Real")
        self.assertEqual(out["gemma_labels"]["stance"], "Neutral")
        self.assertEqual(out["gemma_labels"]["sentiment"], "Neutral")

    def test_parse_gemma_output_malformed(self):
        """Test fallback behavior for malformed or empty outputs."""
        raw = "Bài viết này không đề cập đến vắc xin."
        out = parse_gemma_output(raw)
        
        self.assertFalse(out["parse_ok"])
        self.assertEqual(out["gemma_labels"], {})
        self.assertEqual(out["reasoning"], "")

class TestXaiServiceEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @patch('requests.get')
    def test_health_check_endpoint(self, mock_get):
        """Verify the health check endpoint returns 200 and reachable status."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"data": [{"id": "gemma-4-e4b-vaccine-xai-merged"}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["service"], "XAI Engine")
        self.assertEqual(data["backend"], "lmstudio")
        self.assertTrue(data["lm_studio_reachable"])
        self.assertEqual(data["effective_model"], "gemma-4-e4b-vaccine-xai-merged")
        self.assertIn("gemma-4-e4b-vaccine-xai-merged", data["available_models"])

    @patch('app.main._stream_backend')
    def test_explain_endpoint(self, mock_stream):
        """Verify /api/explain processes input and returns structured labels."""
        # Mock streaming backend to yield Gemma response
        mock_stream.return_value = [(
            "=== KẾT QUẢ ===\n"
            "- Tính xác thực: Chính xác\n"
            "- Thái độ với vaccine: Ủng hộ\n"
            "- Cảm xúc tổng thể: Tích cực\n"
            "=== GIẢI THÍCH ===\n"
            "Vắc-xin an toàn."
        )]

        payload = {
            "text": "Vắc xin an toàn",
            "predicted_labels": {"misinfo": "Real", "stance": "Favor", "sentiment": "Positive"}
        }
        resp = self.client.post("/api/explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["parse_ok"])
        self.assertEqual(data["gemma_labels"]["misinfo"], "Real")
        self.assertEqual(data["gemma_labels"]["stance"], "Favor")
        self.assertEqual(data["gemma_labels"]["sentiment"], "Positive")
        self.assertEqual(data["reasoning"], "Vắc-xin an toàn.")

    @patch('app.main._stream_backend')
    def test_explain_stream_endpoint(self, mock_stream):
        """Verify /api/explain-stream streams SSE events correctly."""
        mock_stream.return_value = [
            "=== KẾT QUẢ ===\n",
            "- Tính xác thực: Chính xác\n",
            "- Thái độ với vaccine: Ủng hộ\n",
            "- Cảm xúc tổng thể: Tích cực\n",
            "=== GIẢI THÍCH ===\n",
            "Vắc-xin an toàn.",
        ]

        payload = {
            "text": "Vắc xin an toàn",
            "predicted_labels": {"misinfo": "Real", "stance": "Favor", "sentiment": "Positive"}
        }
        with self.client.stream("POST", "/api/explain-stream", json=payload) as r:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.headers["Content-Type"].startswith("text/event-stream"))
            lines = [line for line in r.iter_lines() if line]

        events = []
        for line in lines:
            if line.startswith("data:"):
                event_str = line[5:].strip()
                if event_str == "[DONE]":
                    continue
                events.append(json.loads(event_str))

        self.assertTrue(len(events) >= 3)
        self.assertEqual(events[0]["type"], "token")
        self.assertEqual(events[0]["content"], "=== KẾT QUẢ ===\n")
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["parse_ok"])
        self.assertEqual(events[-1]["gemma_labels"]["misinfo"], "Real")
        self.assertEqual(events[-1]["reasoning"], "Vắc-xin an toàn.")

if __name__ == '__main__':
    unittest.main()
