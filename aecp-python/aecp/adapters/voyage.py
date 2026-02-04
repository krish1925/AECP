"""
Voyage AI Adapter for AECP

Provides embedding generation using Voyage AI's embedding models.
"""

from typing import List, Optional
import os

from .base import BaseAdapter


# Model dimension mapping
VOYAGE_DIMENSIONS = {
    "voyage-2": 1024,
    "voyage-large-2": 1536,
    "voyage-code-2": 1536,
    "voyage-lite-02-instruct": 1024,
    "voyage-law-2": 1024,
    "voyage-finance-2": 1024,
}


class VoyageAdapter(BaseAdapter):
    """
    Adapter for Voyage AI embedding models.
    
    Supports:
    - voyage-2 (1024 dimensions, general purpose)
    - voyage-large-2 (1536 dimensions, highest quality)
    - voyage-code-2 (1536 dimensions, code-optimized)
    - voyage-lite-02-instruct (1024 dimensions, instruction-tuned)
    - voyage-law-2 (1024 dimensions, legal domain)
    - voyage-finance-2 (1024 dimensions, finance domain)
    
    Example:
        >>> adapter = VoyageAdapter(api_key="pa-...")
        >>> embedding = adapter.embed("Hello world")
        >>> len(embedding)
        1024
        
        >>> # Use environment variable
        >>> adapter = VoyageAdapter()  # Uses VOYAGE_API_KEY
        
        >>> # Use code-optimized model
        >>> adapter = VoyageAdapter(
        ...     api_key="pa-...",
        ...     model="voyage-code-2"
        ... )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-2",
        **kwargs
    ):
        """
        Initialize Voyage AI adapter.
        
        Args:
            api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            model: Model to use for embeddings
            **kwargs: Additional arguments passed to BaseAdapter
            
        Raises:
            ImportError: If voyageai package is not installed
            ValueError: If no API key is provided
        """
        super().__init__(**kwargs)
        
        # Lazy import
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Voyage AI package not installed. "
                "Install with: pip install aecp[voyage]"
            )
        
        # Get API key
        api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "Voyage API key required. Provide via api_key parameter "
                "or VOYAGE_API_KEY environment variable."
            )
        
        # Validate model
        if model not in VOYAGE_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: {list(VOYAGE_DIMENSIONS.keys())}"
            )
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self._dimensions = VOYAGE_DIMENSIONS[model]
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        result = self.client.embed(
            texts=[text],
            model=self.model,
        )
        return result.embeddings[0]
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        result = self.client.embed(
            texts=texts,
            model=self.model,
        )
        return result.embeddings
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"voyage:{self.model}"
