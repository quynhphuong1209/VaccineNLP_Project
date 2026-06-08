# -*- coding: utf-8 -*-
import unittest
import sys
import os
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# Force SQLite for test duration before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# Patch transformers and torch before importing main app to avoid downloading/loading weights
import transformers
transformers.AutoTokenizer.from_pretrained = MagicMock()

import torch
torch.load = MagicMock()

# Add api_service path to import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_service')))

# Clear cached app modules to avoid namespace collisions with xai_service
for k in list(sys.modules.keys()):
    if k == 'app' or k.startswith('app.'):
        sys.modules.pop(k)

from fastapi.testclient import TestClient
from app.main import app, get_db
import app.main as api_main
from app.database import Base, engine, SessionLocal, AnalysisHistory

class TestApiCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create all tables on SQLite memory db
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    def setUp(self):
        # Clear tables before each test
        db = SessionLocal()
        try:
            db.query(AnalysisHistory).delete()
            db.commit()
        finally:
            db.close()

    def test_health_check(self):
        """Verify the health check endpoint returns 200."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "Ready")

    def test_analyze_endpoint_new_text(self):
        """Verify /api/analyze classifies text, returns probs, and sets xai_status=idle (on-demand)."""
        payload = {
            "text": "Vắc xin sởi rất an toàn cho trẻ nhỏ.",
            "source_url": "https://vnexpress.net/tiem-soi"
        }
        resp = self.client.post("/api/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        data = resp.json()
        self.assertIn("id", data)
        self.assertEqual(data["xai_status"], "idle")
        self.assertIsNotNone(data["phobert_probs"])
        self.assertEqual(data["phobert_probs"]["misinfo"]["Real"] + data["phobert_probs"]["misinfo"]["Fake"], 1.0)
        
        # Verify saved in Database
        db = SessionLocal()
        row = db.get(AnalysisHistory, data["id"])
        self.assertIsNotNone(row)
        self.assertEqual(row.xai_status, "idle")
        self.assertEqual(row.source_url, "https://vnexpress.net/tiem-soi")
        db.close()

    def test_analyze_endpoint_cache_reuse(self):
        """Verify subsequent /api/analyze request for identical text hash reuses database record."""
        payload = {"text": "Vắc xin ngừa cúm bảo vệ sức khỏe."}
        resp1 = self.client.post("/api/analyze", json=payload)
        id1 = resp1.json()["id"]
        
        resp2 = self.client.post("/api/analyze", json=payload)
        id2 = resp2.json()["id"]
        
        self.assertEqual(id1, id2)

    @patch('requests.post')
    def test_explain_stream_endpoint(self, mock_post):
        """Verify /api/explain-stream streams Gemma tokens and updates XAI status to done."""
        # 1. Run analyze first to establish record
        payload = {"text": "Trẻ em tiêm phòng vắc xin phòng sởi."}
        resp_anal = self.client.post("/api/analyze", json=payload)
        row_id = resp_anal.json()["id"]
        
        # 2. Mock XAI service SSE response generator
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/event-stream"}
        
        def mock_iter_lines(decode_unicode=True):
            # Stream tokens then final parsed block (escaped unicode values)
            yield "data: {\"type\": \"token\", \"content\": \"L\\\\u00fd \"}"
            yield "data: {\"type\": \"token\", \"content\": \"do: \"}"
            yield "data: {\"type\": \"token\", \"content\": \"v\\\\u1eafc xin an to\\\\u00e0n.\"}"
            yield "data: {\"type\": \"final\", \"parse_ok\": true, \"reasoning\": \"L\\u00fd do: v\\u1eafc xin an to\\u00e0n.\", \"gemma_labels\": {\"misinfo\": \"Real\", \"stance\": \"Favor\", \"sentiment\": \"Positive\"}, \"raw_output\": \"L\\u00fd do: v\\u1eafc xin an to\\u00e0n. K\\u1ebft qu\\u1ea3: Ch\\u00ednh X\\u00e1c | \\u1ee6ng H\\u1ed9 | T\\u00edch C\\u1ef1c\"}"
            yield "data: [DONE]"
            
        mock_response.iter_lines = mock_iter_lines
        mock_response.__enter__.return_value = mock_response
        mock_post.return_value = mock_response

        # 3. Request explanation stream
        with self.client.stream("POST", "/api/explain-stream", json=payload) as r:
            self.assertEqual(r.status_code, 200)
            lines = [line for line in r.iter_lines() if line]
            
        # Parse SSE events returned
        events = []
        for line in lines:
            if line.startswith("data:"):
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    continue
                events.append(json.loads(payload_str))
                
        self.assertTrue(len(events) >= 2)
        self.assertEqual(events[0]["type"], "token")
        self.assertEqual(events[-1]["type"], "final")
        
        # 4. Verify DB was updated to "done"
        db = SessionLocal()
        row = db.get(AnalysisHistory, row_id)
        self.assertIsNotNone(row)
        self.assertEqual(row.xai_status, "done")
        self.assertEqual(row.xai_explanation["gemma_labels"]["misinfo"], "Real")
        db.close()

    def test_attribute_endpoint_mock_fails_with_503(self):
        """Verify that /api/attribute returns 503 when running in MOCK mode (no model loaded)."""
        payload = {"text": "Vắc xin COVID gây vô sinh ở trẻ em."}
        resp = self.client.post("/api/attribute", json=payload)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("MOCK", resp.json()["detail"])

    def test_attribute_endpoint_success_with_mocked_saliency(self):
        """Verify that /api/attribute returns 200 with tokens list when _captum_saliency is mocked."""
        orig_saliency = getattr(api_main, '_captum_saliency', None)
        api_main._captum_saliency = MagicMock(return_value=(
            ["vắc", "xin", "Ġsởi", "Ġan", "Ġtoàn"],
            [0.1, 0.2, 0.9, 0.8, 0.5],
            1,  # pred_class
            "encoder.embeddings",
        ))
        # Temporarily inject 'model' into phobert dict to bypass MOCK check
        api_main.phobert["model"] = MagicMock()
        try:
            payload = {"text": "vắc xin sởi an toàn"}
            resp = self.client.post("/api/attribute", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["pred_class"], 1)
            self.assertEqual(data["pred_label"], "Real")
            self.assertEqual(data["embedding_layer"], "encoder.embeddings")
            self.assertTrue(len(data["tokens"]) > 0)
            self.assertEqual(data["tokens"][0]["token"], "vắc")
            self.assertEqual(data["tokens"][0]["score"], 0.1)
        finally:
            api_main.phobert.pop("model", None)
            if orig_saliency:
                api_main._captum_saliency = orig_saliency
            else:
                delattr(api_main, '_captum_saliency')

    def test_resolve_embedding_layer_prefers_encoder_embeddings(self):
        """Verify resolver reports the expected PhoBERT/RoBERTa encoder embeddings path."""
        layer = object()
        model = SimpleNamespace(encoder=SimpleNamespace(embeddings=layer))

        resolved, label = api_main._resolve_embedding_layer(model)

        self.assertIs(resolved, layer)
        self.assertEqual(label, "encoder.embeddings")

    def test_resolve_embedding_layer_uses_model_embeddings(self):
        """Verify resolver falls back to model.embeddings."""
        layer = object()
        model = SimpleNamespace(embeddings=layer)

        resolved, label = api_main._resolve_embedding_layer(model)

        self.assertIs(resolved, layer)
        self.assertEqual(label, "embeddings")

    def test_resolve_embedding_layer_uses_get_input_embeddings(self):
        """Verify resolver falls back to get_input_embeddings()."""
        layer = object()

        class ModelWithGetter:
            def get_input_embeddings(self):
                return layer

        resolved, label = api_main._resolve_embedding_layer(ModelWithGetter())

        self.assertIs(resolved, layer)
        self.assertEqual(label, "get_input_embeddings()")

    def test_resolve_embedding_layer_raises_when_missing(self):
        """Verify resolver fails clearly when no embedding path exists."""
        with self.assertRaisesRegex(RuntimeError, "Embedding layer not found"):
            api_main._resolve_embedding_layer(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
