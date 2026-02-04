"""
Tests for AECP high-level patterns.
"""

import pytest
import numpy as np

from aecp.adapters import MockAdapter
from aecp.patterns import CostOptimizer, PrivacyBridge, AgentHandoff


class TestCostOptimizer:
    """Tests for CostOptimizer pattern."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a cost optimizer with mock adapters."""
        cheap = MockAdapter(dimensions=384, model_name="cheap")
        expensive = MockAdapter(dimensions=768, model_name="expensive")
        return CostOptimizer(
            cheap_adapter=cheap,
            expensive_adapter=expensive,
        )
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.cheap_agent is not None
        assert optimizer.expensive_agent is not None
        assert not optimizer._calibrated
    
    def test_calibrate(self, optimizer):
        """Test calibration."""
        result = optimizer.calibrate(verbose=False)
        
        assert result.success
        assert optimizer._calibrated
    
    def test_embed_low_precision(self, optimizer):
        """Test embedding with low precision."""
        optimizer.calibrate(verbose=False)
        
        embedding = optimizer.embed("hello world", precision="low")
        
        assert embedding.shape == (384,)  # Cheap model dimensions
        assert optimizer._stats["cheap_calls"] == 1
        assert optimizer._stats["expensive_calls"] == 0
    
    def test_embed_high_precision(self, optimizer):
        """Test embedding with high precision."""
        optimizer.calibrate(verbose=False)
        
        # High precision uses expensive model, target_space="expensive" returns its dimensions
        embedding = optimizer.embed("hello world", precision="high", target_space="expensive")
        
        assert embedding.shape == (768,)  # Expensive model dimensions
        assert optimizer._stats["expensive_calls"] == 1
    
    def test_get_stats(self, optimizer):
        """Test getting statistics."""
        optimizer.calibrate(verbose=False)
        optimizer.embed("test", precision="low")
        optimizer.embed("test", precision="low")
        optimizer.embed("test", precision="high")
        
        stats = optimizer.get_stats()
        
        assert stats["cheap_calls"] == 2
        assert stats["expensive_calls"] == 1
        assert stats["savings_percentage"] > 0
    
    def test_reset_stats(self, optimizer):
        """Test resetting statistics."""
        optimizer.calibrate(verbose=False)
        optimizer.embed("test", precision="low")
        optimizer.reset_stats()
        
        stats = optimizer.get_stats()
        assert stats["cheap_calls"] == 0


class TestPrivacyBridge:
    """Tests for PrivacyBridge pattern."""
    
    @pytest.fixture
    def bridge(self):
        """Create a privacy bridge with mock adapters."""
        local = MockAdapter(dimensions=384, model_name="local")
        cloud = MockAdapter(dimensions=768, model_name="cloud")
        return PrivacyBridge(
            local_adapter=local,
            cloud_adapter=cloud,
        )
    
    def test_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.local_agent is not None
        assert bridge.cloud_agent is not None
        assert not bridge._calibrated
    
    def test_calibrate(self, bridge):
        """Test calibration."""
        result = bridge.calibrate(verbose=False)
        
        assert result.success
        assert bridge._calibrated
    
    def test_embed_local(self, bridge):
        """Test local embedding."""
        embedding = bridge.embed_local("sensitive data")
        
        assert embedding.shape == (384,)
        assert bridge._stats["local_embeddings"] == 1
    
    def test_transfer_to_cloud(self, bridge):
        """Test transferring to cloud space."""
        bridge.calibrate(verbose=False)
        
        local_emb = bridge.embed_local("sensitive data")
        cloud_emb = bridge.transfer_to_cloud(local_emb)
        
        assert cloud_emb.shape == (768,)
        assert bridge._stats["transfers_to_cloud"] == 1
    
    def test_transfer_without_calibration_raises(self, bridge):
        """Transfer without calibration should raise."""
        local_emb = bridge.embed_local("test")
        
        with pytest.raises(ValueError):
            bridge.transfer_to_cloud(local_emb)
    
    def test_embed_and_transfer(self, bridge):
        """Test combined embed and transfer."""
        bridge.calibrate(verbose=False)
        
        cloud_emb = bridge.embed_and_transfer("sensitive data")
        
        assert cloud_emb.shape == (768,)
        assert bridge._stats["local_embeddings"] == 1
        assert bridge._stats["transfers_to_cloud"] == 1


class TestAgentHandoff:
    """Tests for AgentHandoff pattern."""
    
    @pytest.fixture
    def handoff(self):
        """Create agent handoff with mock adapters."""
        return AgentHandoff({
            'code': MockAdapter(dimensions=384, model_name="code"),
            'general': MockAdapter(dimensions=512, model_name="general"),
            'multilingual': MockAdapter(dimensions=768, model_name="multilingual"),
        })
    
    def test_initialization(self, handoff):
        """Test handoff initialization."""
        assert len(handoff.agents) == 3
        assert handoff.default_agent == 'code'  # First agent
    
    def test_calibrate_pair(self, handoff):
        """Test calibrating a pair of agents."""
        result = handoff.calibrate('code', 'general', verbose=False)
        
        assert result.success
        assert handoff.is_calibrated('code', 'general')
        assert handoff.is_calibrated('general', 'code')
    
    def test_calibrate_all(self, handoff):
        """Test calibrating all agent pairs."""
        results = handoff.calibrate_all(verbose=False)
        
        assert len(results) == 3  # 3 unique pairs
        assert all(r.success for r in results.values())
    
    def test_start_task(self, handoff):
        """Test starting a task."""
        context = handoff.start("Debug this code", agent='code')
        
        assert context.content == "Debug this code"
        assert context.embedding.shape == (384,)
        assert 'code' in context.history
    
    def test_transfer_context(self, handoff):
        """Test transferring context between agents."""
        handoff.calibrate_all(verbose=False)
        
        context = handoff.start("Debug this code", agent='code')
        context = handoff.transfer(context, to_agent='general')
        
        assert context.embedding.shape == (512,)
        assert context.history == ['code', 'general']
    
    def test_transfer_without_calibration_raises(self, handoff):
        """Transfer without calibration should raise."""
        context = handoff.start("test", agent='code')
        
        with pytest.raises(ValueError):
            handoff.transfer(context, to_agent='general')
    
    def test_get_current_agent(self, handoff):
        """Test getting current agent."""
        handoff.calibrate_all(verbose=False)
        
        context = handoff.start("test", agent='code')
        assert handoff.get_current_agent(context) == 'code'
        
        context = handoff.transfer(context, to_agent='general')
        assert handoff.get_current_agent(context) == 'general'
    
    def test_get_stats(self, handoff):
        """Test getting statistics."""
        handoff.calibrate_all(verbose=False)
        handoff.start("test", agent='code')
        
        stats = handoff.get_stats()
        
        assert stats["starts"] == 1
        assert stats["contexts_created"] == 1
        assert 'code' in stats["agents"]
