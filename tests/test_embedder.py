import inspect

import numpy as np
import pytest
import torch

from text_change_detector.embedder import Embedder, SentenceTransformerEmbedder
from tests.helpers import MINILM


@pytest.fixture
def mini():
    try:
        embedder = SentenceTransformerEmbedder(model_name=MINILM, device="cpu", dtype=torch.float32)
    except Exception as exc:
        pytest.skip(f"could not load {MINILM}: {exc}")

    yield embedder

    embedder.close()


class TestEncode:
    def test_forwards_batch_size_and_normalize_flag(self):
        embedder = SentenceTransformerEmbedder(
            model_name=MINILM,
            device="cpu",
            dtype=torch.float32,
            batch_size=2
        )
        captured = {}
        original = embedder._model.encode

        def spy(sentences, **kwargs):
            captured.update(kwargs)

            return original(sentences, **kwargs)

        embedder._model.encode = spy

        try:
            embedder.encode(["a", "b", "c"], normalize_embeddings=True)
        finally:
            embedder.close()

        assert captured["batch_size"] == 2
        assert captured["normalize_embeddings"] is True

    def test_omits_batch_size_when_unset(self, mini):
        captured = {}
        original = mini._model.encode

        def spy(sentences, **kwargs):
            captured.update(kwargs)

            return original(sentences, **kwargs)

        mini._model.encode = spy

        mini.encode(["a"])

        assert "batch_size" not in captured
        assert captured["normalize_embeddings"] is True


class TestClose:
    def test_close_drops_model_and_is_idempotent(self, mini):
        mini.close()
        assert mini._model is None
        mini.close()
        assert mini._model is None


class TestProtocol:
    def test_encode_signature(self):
        params = inspect.signature(Embedder.encode).parameters
        assert "sentences" in params
        assert params["normalize_embeddings"].default is True


class TestRealDefaultModel:
    def test_qwen_encodes_and_separates_meaning(self, real_embedder):
        out = real_embedder.encode(
            [
                "The tenant pays the monthly rent.",
                "The tenant pays the monthly rent.",
                "The database replicates writes to standby nodes.",
            ]
        )
        assert out.shape[0] == 3
        assert out.shape[1] > 0
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-2)
        assert np.allclose(out[0], out[1], atol=1e-3)
        assert float(out[0] @ out[1]) > float(out[0] @ out[2])
