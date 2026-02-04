"""
AECP Type Definitions

Core type definitions and protocols for the AECP library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All adapters (OpenAI, Voyage, Cohere, etc.) must implement this interface.
    """
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains invalid items
            RuntimeError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """
        Get the identifier for the embedding model.
        
        Returns:
            Model identifier string
        """
        pass


@dataclass
class AgentCapabilities:
    """
    Agent capabilities and configuration.
    
    Attributes:
        agent_id: Unique identifier for the agent
        embedding_model: Name of the embedding model used
        dimensions: Dimensionality of the embedding space
        max_batch_size: Maximum number of texts to embed in one batch
        min_quality_threshold: Minimum acceptable transfer quality
        protocol_version: AECP protocol version supported
    """
    agent_id: str
    embedding_model: str
    dimensions: int
    max_batch_size: int = 1000
    min_quality_threshold: float = 0.75
    protocol_version: str = "1.0"
    
    def __post_init__(self):
        """Validate capabilities after initialization."""
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not 0 < self.min_quality_threshold <= 1:
            raise ValueError("min_quality_threshold must be between 0 and 1")


@dataclass
class TransferMatrix:
    """
    Transfer matrix with quality metrics.
    
    Stores the learned transformation matrices and their quality metrics
    for transferring embeddings between two agents.
    
    Attributes:
        matrix_AB: Transfer matrix from agent A to agent B
        matrix_BA: Transfer matrix from agent B to agent A
        training_similarity: Cosine similarity on training data
        validation_similarity: Cosine similarity on held-out validation data
        worst_case_similarity: Minimum similarity observed
        valid_until: Expiration timestamp for the matrix
        created_at: When the matrix was created
    """
    matrix_AB: np.ndarray
    matrix_BA: np.ndarray
    training_similarity: float
    validation_similarity: float
    worst_case_similarity: float
    valid_until: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate matrix data after initialization."""
        if self.matrix_AB is None or self.matrix_BA is None:
            raise ValueError("Transfer matrices cannot be None")
        if not isinstance(self.matrix_AB, np.ndarray):
            self.matrix_AB = np.array(self.matrix_AB)
        if not isinstance(self.matrix_BA, np.ndarray):
            self.matrix_BA = np.array(self.matrix_BA)
        if self.matrix_AB.ndim != 2 or self.matrix_BA.ndim != 2:
            raise ValueError("Transfer matrices must be 2-dimensional")
    
    def is_expired(self) -> bool:
        """Check if the transfer matrix has expired."""
        try:
            valid_until = datetime.fromisoformat(self.valid_until)
            return datetime.now() > valid_until
        except (ValueError, TypeError):
            return True
    
    def meets_quality_threshold(self, threshold: float = 0.75) -> bool:
        """Check if the matrix meets the quality threshold."""
        return self.validation_similarity >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dictionary (excluding large matrices).
        
        Returns:
            Dictionary with matrix metadata and quality metrics
        """
        return {
            "matrix_AB_shape": list(self.matrix_AB.shape),
            "matrix_BA_shape": list(self.matrix_BA.shape),
            "training_similarity": float(self.training_similarity),
            "validation_similarity": float(self.validation_similarity),
            "worst_case_similarity": float(self.worst_case_similarity),
            "valid_until": self.valid_until,
            "created_at": self.created_at,
        }


@dataclass
class SemanticTransfer:
    """
    Semantic content transfer.
    
    Represents an embedding that has been transferred from one agent's
    embedding space to another's.
    
    Attributes:
        transfer_id: Unique identifier for this transfer
        embedding: The transferred embedding vector
        source_agent: ID of the source agent
        target_agent: ID of the target agent
        original_norm: L2 norm of the original embedding
        expected_similarity: Expected quality based on calibration
        timestamp: When the transfer occurred
    """
    transfer_id: str
    embedding: np.ndarray
    source_agent: str
    target_agent: str
    original_norm: float
    expected_similarity: float
    timestamp: str
    
    def __post_init__(self):
        """Validate transfer data after initialization."""
        if not self.transfer_id:
            raise ValueError("transfer_id cannot be empty")
        if self.embedding is None:
            raise ValueError("embedding cannot be None")
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
    
    def get_transferred_norm(self) -> float:
        """Get the L2 norm of the transferred embedding."""
        return float(np.linalg.norm(self.embedding))
    
    def get_norm_ratio(self) -> float:
        """
        Get the ratio of transferred norm to original norm.
        
        Values close to 1.0 indicate good norm preservation.
        """
        if self.original_norm == 0:
            return 0.0
        return self.get_transferred_norm() / self.original_norm


@dataclass
class CalibrationRequest:
    """
    Request for calibration between agents.
    
    Attributes:
        vocabulary_size: Number of vocabulary items to use
        validation_size: Number of items to hold out for validation
        quality_threshold: Minimum acceptable quality
        timestamp: When the request was created
    """
    vocabulary_size: int
    validation_size: int
    quality_threshold: float = 0.80
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if self.validation_size < 0:
            raise ValueError("validation_size cannot be negative")
        if self.validation_size >= self.vocabulary_size:
            raise ValueError("validation_size must be less than vocabulary_size")
        if not 0 < self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be between 0 and 1")
