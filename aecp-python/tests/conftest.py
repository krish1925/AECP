"""
Pytest configuration and fixtures for AECP tests.
"""

import pytest
import numpy as np
from typing import List

from aecp.adapters.mock import MockAdapter
from aecp import AECP


@pytest.fixture
def mock_adapter_384():
    """Create a mock adapter with 384 dimensions."""
    return MockAdapter(dimensions=384, model_name="mock-384")


@pytest.fixture
def mock_adapter_768():
    """Create a mock adapter with 768 dimensions."""
    return MockAdapter(dimensions=768, model_name="mock-768")


@pytest.fixture
def agent_a(mock_adapter_384):
    """Create agent A with 384-dimensional embeddings."""
    return AECP(mock_adapter_384, agent_id="agent_a")


@pytest.fixture
def agent_b(mock_adapter_768):
    """Create agent B with 768-dimensional embeddings."""
    return AECP(mock_adapter_768, agent_id="agent_b")


@pytest.fixture
def calibrated_agents(agent_a, agent_b):
    """Create a pair of calibrated agents."""
    # Use a small vocabulary for fast tests
    vocabulary = [
        "hello", "world", "test", "example", "data", "model",
        "system", "process", "function", "algorithm", "network",
        "machine", "learning", "artificial", "intelligence",
        "computer", "science", "technology", "information", "analysis",
    ]
    
    result = agent_a.calibrate_with(agent_b, vocabulary=vocabulary, verbose=False)
    assert result.success
    
    return agent_a, agent_b


@pytest.fixture
def sample_vocabulary() -> List[str]:
    """Sample vocabulary for testing."""
    return [
        "hello", "world", "test", "example", "data",
        "model", "system", "process", "function", "algorithm",
        "network", "machine", "learning", "artificial", "intelligence",
        "computer", "science", "technology", "information", "analysis",
        "research", "development", "optimization", "performance", "quality",
    ]


@pytest.fixture
def sample_embeddings_384() -> np.ndarray:
    """Sample embeddings with 384 dimensions."""
    np.random.seed(42)
    return np.random.randn(100, 384).astype(np.float32)


@pytest.fixture
def sample_embeddings_768() -> np.ndarray:
    """Sample embeddings with 768 dimensions."""
    np.random.seed(42)
    return np.random.randn(100, 768).astype(np.float32)
