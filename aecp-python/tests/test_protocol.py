"""
Tests for AECP protocol implementation.
"""

import pytest
import numpy as np

from aecp import AECP, ProtocolHandler, CalibrationResult
from aecp.types import AgentCapabilities, TransferMatrix, SemanticTransfer
from aecp.errors import AgentNotCalibratedError


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""
    
    def test_valid_capabilities(self):
        """Test creating valid capabilities."""
        caps = AgentCapabilities(
            agent_id="test_agent",
            embedding_model="test-model",
            dimensions=384,
        )
        assert caps.agent_id == "test_agent"
        assert caps.dimensions == 384
    
    def test_empty_agent_id_raises(self):
        """Empty agent_id should raise ValueError."""
        with pytest.raises(ValueError):
            AgentCapabilities(
                agent_id="",
                embedding_model="test",
                dimensions=384,
            )
    
    def test_negative_dimensions_raises(self):
        """Negative dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            AgentCapabilities(
                agent_id="test",
                embedding_model="test",
                dimensions=-1,
            )
    
    def test_invalid_threshold_raises(self):
        """Invalid quality threshold should raise ValueError."""
        with pytest.raises(ValueError):
            AgentCapabilities(
                agent_id="test",
                embedding_model="test",
                dimensions=384,
                min_quality_threshold=1.5,
            )


class TestProtocolHandler:
    """Tests for ProtocolHandler class."""
    
    def test_initialization(self, mock_adapter_384):
        """Test handler initialization."""
        handler = ProtocolHandler("test_agent", mock_adapter_384)
        
        assert handler.agent_id == "test_agent"
        assert handler.capabilities.dimensions == 384
    
    def test_invalid_agent_id_raises(self, mock_adapter_384):
        """Invalid agent_id should raise ValueError."""
        with pytest.raises(ValueError):
            ProtocolHandler("", mock_adapter_384)
    
    def test_send_handshake(self, agent_a):
        """Test handshake message creation."""
        handshake = agent_a.send_handshake()
        
        assert handshake["message_type"] == "handshake"
        assert handshake["agent_id"] == "agent_a"
        assert "embedding_model" in handshake
        assert "timestamp" in handshake
    
    def test_receive_valid_handshake(self, agent_a, agent_b):
        """Test receiving a valid handshake."""
        handshake = agent_b.send_handshake()
        success, error = agent_a.receive_handshake(handshake)
        
        assert success is True
        assert error is None
    
    def test_receive_invalid_handshake(self, agent_a):
        """Test receiving an invalid handshake."""
        handshake = {"message_type": "handshake"}  # Missing fields
        success, error = agent_a.receive_handshake(handshake)
        
        assert success is False
        assert error is not None
    
    def test_calibrate_success(self, agent_a, agent_b, sample_vocabulary):
        """Test successful calibration."""
        result = agent_a.calibrate(
            agent_b,
            vocabulary=sample_vocabulary,
            verbose=False,
        )
        
        assert result.success is True
        assert result.transfer_matrix is not None
        assert result.validation_similarity > 0
    
    def test_calibrate_stores_matrices(self, agent_a, agent_b, sample_vocabulary):
        """Test that calibration stores matrices in both handlers."""
        agent_a.calibrate(agent_b, vocabulary=sample_vocabulary, verbose=False)
        
        key_ab = f"{agent_a.agent_id}_{agent_b.agent_id}"
        key_ba = f"{agent_b.agent_id}_{agent_a.agent_id}"
        
        assert key_ab in agent_a.transfer_matrices
        assert key_ba in agent_b.transfer_matrices


class TestAECP:
    """Tests for AECP main class."""
    
    def test_auto_agent_id(self, mock_adapter_384):
        """Test automatic agent_id generation."""
        agent = AECP(mock_adapter_384)
        
        assert agent.agent_id.startswith("agent_")
    
    def test_custom_agent_id(self, mock_adapter_384):
        """Test custom agent_id."""
        agent = AECP(mock_adapter_384, agent_id="custom_id")
        
        assert agent.agent_id == "custom_id"
    
    def test_embed(self, agent_a):
        """Test embedding generation."""
        embedding = agent_a.embed("hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    def test_calibrate_with(self, agent_a, agent_b, sample_vocabulary):
        """Test calibrate_with convenience method."""
        result = agent_a.calibrate_with(
            agent_b,
            vocabulary=sample_vocabulary,
            verbose=False,
        )
        
        assert result.success is True


class TestSemanticTransfer:
    """Tests for semantic transfer functionality."""
    
    def test_transfer_to(self, calibrated_agents):
        """Test transferring text to another agent."""
        agent_a, agent_b = calibrated_agents
        
        transfer = agent_a.transfer_to(agent_b.agent_id, "hello world")
        
        assert isinstance(transfer, SemanticTransfer)
        assert transfer.source_agent == agent_a.agent_id
        assert transfer.target_agent == agent_b.agent_id
        assert transfer.embedding.shape == (768,)  # agent_b dimensions
    
    def test_transfer_embedding_to(self, calibrated_agents):
        """Test transferring pre-computed embedding."""
        agent_a, agent_b = calibrated_agents
        
        embedding = agent_a.embed("hello world")
        transfer = agent_a.transfer_embedding_to(agent_b.agent_id, embedding)
        
        assert transfer.embedding.shape == (768,)
    
    def test_transfer_without_calibration_raises(self, agent_a, agent_b):
        """Transfer without calibration should raise AgentNotCalibratedError."""
        with pytest.raises(AgentNotCalibratedError, match="No calibration found"):
            agent_a.transfer_to(agent_b.agent_id, "hello")
    
    def test_receive_transfer(self, calibrated_agents):
        """Test receiving a transfer."""
        agent_a, agent_b = calibrated_agents
        
        transfer = agent_a.transfer_to(agent_b.agent_id, "hello world")
        ack = agent_b.receive_transfer(transfer)
        
        assert ack["status"] == "success"
        assert "received_norm" in ack
    
    def test_receive_transfer_with_validation(self, calibrated_agents):
        """Test receiving transfer with quality validation."""
        agent_a, agent_b = calibrated_agents
        
        text = "hello world"
        transfer = agent_a.transfer_to(agent_b.agent_id, text)
        ack = agent_b.receive_transfer(transfer, original_text=text)
        
        assert ack["status"] == "success"
        assert ack["quality_metric"] is not None


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""
    
    def test_successful_result(self):
        """Test successful calibration result."""
        result = CalibrationResult(
            success=True,
            training_similarity=0.95,
            validation_similarity=0.90,
            worst_case_similarity=0.80,
        )
        
        assert result.meets_threshold(0.85)
        assert not result.meets_threshold(0.95)
    
    def test_failed_result(self):
        """Test failed calibration result."""
        result = CalibrationResult(
            success=False,
            error_message="Test error",
        )
        
        assert not result.meets_threshold(0.5)


class TestRecalibration:
    """Tests for recalibration functionality."""
    
    def test_requires_recalibration_no_calibration(self, agent_a, agent_b):
        """Should require recalibration if never calibrated."""
        assert agent_a.requires_recalibration(agent_b.agent_id)
    
    def test_requires_recalibration_after_calibration(self, calibrated_agents):
        """Should not require recalibration immediately after calibrating."""
        agent_a, agent_b = calibrated_agents
        
        # Check that calibration exists (matrix is stored)
        key = f"{agent_a.agent_id}_{agent_b.agent_id}"
        assert key in agent_a.transfer_matrices
        
        # Matrix should not be expired
        matrix = agent_a.transfer_matrices[key]
        assert not matrix.is_expired()
    
    def test_get_calibration_stats(self, calibrated_agents):
        """Test getting calibration statistics."""
        agent_a, agent_b = calibrated_agents
        
        stats = agent_a.get_calibration_stats(agent_b.agent_id)
        
        assert stats is not None
        assert "validation_similarity" in stats
        assert "valid_until" in stats
