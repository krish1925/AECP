"""
Mock Adapter for AECP

Provides a mock embedding provider for testing.
No API calls - generates deterministic pseudo-embeddings.
"""

from typing import List, Optional
import hashlib
import math

from .base import BaseAdapter


class MockAdapter(BaseAdapter):
    """
    Mock adapter for testing AECP without API calls.
    
    Generates deterministic pseudo-embeddings based on text hashing.
    Useful for:
    - Unit testing
    - Integration testing
    - Development without API costs
    
    Note: Embeddings are NOT semantically meaningful.
    
    Example:
        >>> adapter = MockAdapter()
        >>> embedding = adapter.embed("Hello world")
        >>> len(embedding)
        384
        
        >>> # Same text produces same embedding
        >>> adapter.embed("test") == adapter.embed("test")
        True
        
        >>> # Custom dimensions
        >>> adapter = MockAdapter(dimensions=768)
        >>> len(adapter.embed("test"))
        768
    """
    
    def __init__(
        self,
        dimensions: int = 384,
        model_name: str = "mock-model",
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize mock adapter.
        
        Args:
            dimensions: Embedding dimensionality
            model_name: Model identifier string
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to BaseAdapter
        """
        # Don't call super().__init__() to avoid retry logic overhead
        self._dimensions = dimensions
        self._model_name = model_name
        self._seed = seed
        self.max_retries = 1
        self.retry_delay = 0
        self.max_text_length = 10000
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate deterministic pseudo-embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Pseudo-embedding vector
        """
        return self._generate_embedding(text)
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate pseudo-embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of pseudo-embedding vectors
        """
        return [self._generate_embedding(text) for text in texts]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic pseudo-embedding from text.
        
        Uses SHA-256 hashing to create reproducible values.
        
        Args:
            text: Input text
            
        Returns:
            List of floats (pseudo-embedding)
        """
        # Create seed from text
        text_hash = hashlib.sha256(
            f"{text}{self._seed}".encode()
        ).hexdigest()
        
        # Generate embedding values
        embedding = []
        for i in range(self._dimensions):
            # Use different parts of hash for each dimension
            chunk_start = (i * 4) % 60
            chunk = text_hash[chunk_start:chunk_start + 4]
            
            # Convert to float in [-1, 1]
            value = int(chunk, 16) / 65535.0 * 2 - 1
            
            # Add some variation based on dimension
            value = math.sin(value * (i + 1) + self._seed)
            embedding.append(value)
        
        # Normalize to unit length
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 0:
            embedding = [v / norm for v in embedding]
        
        return embedding
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"mock:{self._model_name}"
