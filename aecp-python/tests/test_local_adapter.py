"""
Tests for LocalModelAdapter.

Validates that LocalModelAdapter correctly wraps pre-loaded model objects
and integrates with the AECP protocol for calibration and transfer.
"""

import pytest
import numpy as np
import math
import hashlib
from typing import List

from aecp.adapters.local import LocalModelAdapter
from aecp.adapters.mock import MockAdapter
from aecp.types import EmbeddingProvider
from aecp import AECP


# ── Fake models for testing (no sentence-transformers dependency) ────


class FakeSentenceTransformer:
    """Mimics a SentenceTransformer model with .encode() and
    .get_sentence_embedding_dimension().

    Always returns a 2-D numpy array when called with a list of texts
    (matching real SentenceTransformer behaviour when given a list).
    """

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions
        self.model_name_or_path = "fake-st-model"

    def _make_embedding(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        emb = []
        for i in range(self._dimensions):
            chunk = h[(i * 2) % 60: (i * 2) % 60 + 4]
            val = int(chunk, 16) / 65535.0 * 2 - 1
            val = math.sin(val * (i + 1))
            emb.append(val)
        norm = math.sqrt(sum(v * v for v in emb))
        if norm > 0:
            emb = [v / norm for v in emb]
        return emb

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        """Return deterministic pseudo-embeddings as a 2-D numpy array."""
        if isinstance(texts, str):
            texts = [texts]

        results = [self._make_embedding(t) for t in texts]
        return np.array(results)  # Always 2-D: [n_texts, dimensions]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimensions


class MinimalEncodeModel:
    """Model with only an encode method (no get_sentence_embedding_dimension).

    Does NOT accept normalize_embeddings / batch_size kwargs — this exercises
    the LocalModelAdapter's TypeError fallback path.
    """

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        np.random.seed(42)
        return np.random.randn(len(texts), 128)  # Always 2-D


class NoEncodeModel:
    """Model without an encode method — requires custom encode_fn."""

    def predict(self, texts):
        return [[0.1] * 64 for _ in texts]


# ── Initialization Tests ────────────────────────────────────────────


class TestLocalModelAdapterInit:

    def test_basic_init_with_fake_st(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        assert adapter.get_dimensions() == 384
        assert "local:" in adapter.get_model_id()

    def test_custom_model_id(self):
        model = FakeSentenceTransformer()
        adapter = LocalModelAdapter(model, model_id="my-custom-model")

        assert adapter.get_model_id() == "my-custom-model"

    def test_explicit_dimensions_override(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model, dimensions=512)

        assert adapter.get_dimensions() == 512

    def test_auto_detect_dimensions_from_method(self):
        model = FakeSentenceTransformer(dimensions=768)
        adapter = LocalModelAdapter(model)

        assert adapter.get_dimensions() == 768

    def test_auto_detect_dimensions_by_probing(self):
        """Model without get_sentence_embedding_dimension → probe via encode.
        MinimalEncodeModel doesn't accept normalize_embeddings, so the adapter
        falls back through its TypeError handler."""
        model = MinimalEncodeModel()
        adapter = LocalModelAdapter(model)

        # Should have probed and found 128 dimensions
        assert adapter.get_dimensions() == 128

    def test_no_encode_raises_type_error(self):
        model = NoEncodeModel()

        with pytest.raises(TypeError, match="encode"):
            LocalModelAdapter(model)

    def test_custom_encode_fn(self):
        model = NoEncodeModel()

        def custom_encode(m, texts, **kwargs):
            return np.array(m.predict(texts))

        adapter = LocalModelAdapter(
            model,
            encode_fn=custom_encode,
            dimensions=64,
            model_id="custom:no-encode",
        )

        assert adapter.get_dimensions() == 64
        assert adapter.get_model_id() == "custom:no-encode"

    def test_implements_embedding_provider(self):
        model = FakeSentenceTransformer()
        adapter = LocalModelAdapter(model)

        assert isinstance(adapter, EmbeddingProvider)


# ── Embedding Tests ─────────────────────────────────────────────────


class TestLocalModelAdapterEmbed:

    def test_embed_single(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        embedding = adapter.embed("hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_batch(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        embeddings = adapter.embed_batch(["hello", "world", "test"])

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_deterministic(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        emb1 = adapter.embed("hello world")
        emb2 = adapter.embed("hello world")

        assert emb1 == emb2

    def test_different_texts_different_embeddings(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        emb1 = adapter.embed("hello")
        emb2 = adapter.embed("world")

        assert emb1 != emb2

    def test_normalized(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)

        embedding = adapter.embed("hello world")
        norm = np.linalg.norm(embedding)

        assert norm == pytest.approx(1.0, rel=0.05)

    def test_custom_encode_fn_embed(self):
        model = NoEncodeModel()

        def custom_encode(m, texts, **kwargs):
            return np.array(m.predict(texts))

        adapter = LocalModelAdapter(
            model,
            encode_fn=custom_encode,
            dimensions=64,
        )

        embedding = adapter.embed("test")
        assert len(embedding) == 64

    def test_embed_empty_raises(self):
        model = FakeSentenceTransformer()
        adapter = LocalModelAdapter(model)

        with pytest.raises(ValueError):
            adapter.embed("")

    def test_embed_batch_empty_raises(self):
        model = FakeSentenceTransformer()
        adapter = LocalModelAdapter(model)

        with pytest.raises(ValueError):
            adapter.embed_batch([])

    def test_batch_size_param(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model, batch_size=2)

        # Should still work even with small batch_size
        embeddings = adapter.embed_batch(["a", "b", "c"])
        assert len(embeddings) == 3


# ── AECP Integration Tests ──────────────────────────────────────────


class TestLocalModelAdapterWithAECP:

    def test_aecp_embed(self):
        model = FakeSentenceTransformer(dimensions=384)
        adapter = LocalModelAdapter(model)
        agent = AECP(adapter, agent_id="local_agent")

        embedding = agent.embed("hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_calibration_between_local_adapters(self):
        model1 = FakeSentenceTransformer(dimensions=384)
        model2 = FakeSentenceTransformer(dimensions=768)

        agent1 = AECP(LocalModelAdapter(model1), agent_id="agent1")
        agent2 = AECP(LocalModelAdapter(model2), agent_id="agent2")

        vocab = ["hello", "world", "test", "example", "data"] * 4
        result = agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)

        assert result.success
        assert result.validation_similarity > 0

    def test_transfer_between_local_adapters(self):
        model1 = FakeSentenceTransformer(dimensions=384)
        model2 = FakeSentenceTransformer(dimensions=768)

        agent1 = AECP(LocalModelAdapter(model1), agent_id="agent1")
        agent2 = AECP(LocalModelAdapter(model2), agent_id="agent2")

        vocab = ["hello", "world", "test", "example", "data"] * 4
        agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)

        transfer = agent1.transfer_to(agent2.agent_id, "hello world")
        assert isinstance(transfer.embedding, np.ndarray)
        assert transfer.embedding.shape == (768,)
        assert transfer.source_agent == "agent1"
        assert transfer.target_agent == "agent2"

    def test_calibration_local_vs_mock(self):
        """LocalModelAdapter can calibrate with MockAdapter."""
        model = FakeSentenceTransformer(dimensions=384)
        local_adapter = LocalModelAdapter(model)
        mock_adapter = MockAdapter(dimensions=768)

        agent1 = AECP(local_adapter, agent_id="local")
        agent2 = AECP(mock_adapter, agent_id="mock")

        vocab = ["hello", "world", "test", "data", "model"] * 4
        result = agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)

        assert result.success

    def test_roundtrip_transfer(self):
        """Transfer A→B then B→A should roughly recover original.

        With fake embeddings and a small vocabulary the round-trip
        quality is limited, so we only check that the similarity is
        positive (i.e. the direction is preserved).
        """
        model1 = FakeSentenceTransformer(dimensions=384)
        model2 = FakeSentenceTransformer(dimensions=384)

        agent1 = AECP(LocalModelAdapter(model1), agent_id="a1")
        agent2 = AECP(LocalModelAdapter(model2), agent_id="a2")

        vocab = ["hello", "world", "test", "example", "data",
                 "machine", "learning", "neural", "network", "model"] * 2
        agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)

        original = agent1.embed("hello world")
        forward = agent1.transfer_embedding_to(agent2.agent_id, original)
        back = agent2.transfer_embedding_to(agent1.agent_id, forward.embedding)

        from aecp.matrix import cosine_similarity
        sim = cosine_similarity(original, back.embedding)
        # Round-trip should at least preserve direction (positive similarity)
        assert sim > 0.0
