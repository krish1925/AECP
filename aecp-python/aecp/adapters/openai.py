"""
OpenAI Adapter for AECP

Provides embedding generation using OpenAI's embedding models.
"""

from typing import List, Optional
import os

from .base import BaseAdapter


# Model dimension mapping
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI embedding models.
    
    Supports:
    - text-embedding-3-small (1536 dimensions, recommended)
    - text-embedding-3-large (3072 dimensions, highest quality)
    - text-embedding-ada-002 (1536 dimensions, legacy)
    
    Example:
        >>> adapter = OpenAIAdapter(api_key="sk-...")
        >>> embedding = adapter.embed("Hello world")
        >>> len(embedding)
        1536
        
        >>> # Use environment variable
        >>> adapter = OpenAIAdapter()  # Uses OPENAI_API_KEY
        
        >>> # Use specific model
        >>> adapter = OpenAIAdapter(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-large"
        ... )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for embeddings
            organization: Optional organization ID
            base_url: Optional custom API base URL
            **kwargs: Additional arguments passed to BaseAdapter
            
        Raises:
            ImportError: If openai package is not installed
            ValueError: If no API key is provided
        """
        super().__init__(**kwargs)
        
        # Lazy import to avoid requiring openai for all users
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install aecp[openai]"
            )
        
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )
        
        # Validate model
        if model not in OPENAI_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: {list(OPENAI_DIMENSIONS.keys())}"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )
        self.model = model
        self._dimensions = OPENAI_DIMENSIONS[model]
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"openai:{self.model}"
