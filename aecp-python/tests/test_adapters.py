"""
Tests for AECP adapters.
"""

import pytest
import numpy as np

from aecp.adapters import MockAdapter
from aecp.types import EmbeddingProvider


class TestMockAdapter:
    """Tests for MockAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = MockAdapter(dimensions=384)
        
        assert adapter.get_dimensions() == 384
        assert "mock" in adapter.get_model_id()
    
    def test_embed_single(self):
        """Test embedding a single text."""
        adapter = MockAdapter(dimensions=384)
        
        embedding = adapter.embed("hello world")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_embed_batch(self):
        """Test embedding multiple texts."""
        adapter = MockAdapter(dimensions=384)
        
        embeddings = adapter.embed_batch(["hello", "world", "test"])
        
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)
    
    def test_deterministic(self):
        """Same text should produce same embedding."""
        adapter = MockAdapter(dimensions=384, seed=42)
        
        emb1 = adapter.embed("hello world")
        emb2 = adapter.embed("hello world")
        
        assert emb1 == emb2
    
    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        adapter = MockAdapter(dimensions=384)
        
        emb1 = adapter.embed("hello")
        emb2 = adapter.embed("world")
        
        assert emb1 != emb2
    
    def test_normalized(self):
        """Embeddings should be approximately unit length."""
        adapter = MockAdapter(dimensions=384)
        
        embedding = adapter.embed("hello world")
        norm = np.linalg.norm(embedding)
        
        assert norm == pytest.approx(1.0, rel=0.01)
    
    def test_custom_dimensions(self):
        """Test custom dimensions."""
        adapter = MockAdapter(dimensions=768)
        
        embedding = adapter.embed("test")
        
        assert len(embedding) == 768
    
    def test_implements_provider_interface(self):
        """MockAdapter should implement EmbeddingProvider."""
        adapter = MockAdapter()
        
        assert isinstance(adapter, EmbeddingProvider)


class TestAdapterValidation:
    """Tests for adapter input validation."""
    
    def test_empty_text_raises(self):
        """Empty text should raise ValueError."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError):
            adapter.embed("")
    
    def test_whitespace_only_raises(self):
        """Whitespace-only text should raise ValueError."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError):
            adapter.embed("   ")
    
    def test_none_text_raises(self):
        """None text should raise ValueError."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError):
            adapter.embed(None)
    
    def test_empty_batch_raises(self):
        """Empty batch should raise ValueError."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError):
            adapter.embed_batch([])


class TestAdapterIntegration:
    """Integration tests for adapters with AECP."""
    
    def test_adapter_with_aecp(self):
        """Test using adapter with AECP class."""
        from aecp import AECP
        
        adapter = MockAdapter(dimensions=384)
        agent = AECP(adapter, agent_id="test_agent")
        
        embedding = agent.embed("hello world")
        
        assert embedding.shape == (384,)
    
    def test_two_agents_calibration(self):
        """Test calibration between two agents with different adapters."""
        from aecp import AECP
        
        adapter1 = MockAdapter(dimensions=384, model_name="model-a")
        adapter2 = MockAdapter(dimensions=768, model_name="model-b")
        
        agent1 = AECP(adapter1, agent_id="agent_1")
        agent2 = AECP(adapter2, agent_id="agent_2")
        
        vocab = ["hello", "world", "test", "example", "data"] * 4
        result = agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)
        
        assert result.success
    
    def test_transfer_after_calibration(self):
        """Test transfer works after calibration."""
        from aecp import AECP
        
        adapter1 = MockAdapter(dimensions=384)
        adapter2 = MockAdapter(dimensions=768)
        
        agent1 = AECP(adapter1, agent_id="agent_1")
        agent2 = AECP(adapter2, agent_id="agent_2")
        
        vocab = ["hello", "world", "test", "example", "data"] * 4
        agent1.calibrate_with(agent2, vocabulary=vocab, verbose=False)
        
        transfer = agent1.transfer_to(agent2.agent_id, "hello world")
        
        assert transfer.embedding.shape == (768,)
