"""
Cohere Adapter for AECP

Provides embedding generation using Cohere's embedding models.
"""

from typing import List, Optional
import os

from .base import BaseAdapter


# Model dimension mapping
COHERE_DIMENSIONS = {
    "embed-english-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
}


class CohereAdapter(BaseAdapter):
    """
    Adapter for Cohere embedding models.
    
    Supports:
    - embed-english-v3.0 (1024 dimensions, English)
    - embed-english-light-v3.0 (384 dimensions, English, faster)
    - embed-multilingual-v3.0 (1024 dimensions, 100+ languages)
    - embed-multilingual-light-v3.0 (384 dimensions, multilingual, faster)
    
    Example:
        >>> adapter = CohereAdapter(api_key="...")
        >>> embedding = adapter.embed("Hello world")
        >>> len(embedding)
        1024
        
        >>> # Use environment variable
        >>> adapter = CohereAdapter()  # Uses COHERE_API_KEY
        
        >>> # Use multilingual model
        >>> adapter = CohereAdapter(
        ...     api_key="...",
        ...     model="embed-multilingual-v3.0"
        ... )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        **kwargs
    ):
        """
        Initialize Cohere adapter.
        
        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Model to use for embeddings
            input_type: Type of input ("search_document", "search_query", 
                       "classification", "clustering")
            **kwargs: Additional arguments passed to BaseAdapter
            
        Raises:
            ImportError: If cohere package is not installed
            ValueError: If no API key is provided
        """
        super().__init__(**kwargs)
        
        # Lazy import
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. "
                "Install with: pip install aecp[cohere]"
            )
        
        # Get API key
        api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key required. Provide via api_key parameter "
                "or COHERE_API_KEY environment variable."
            )
        
        # Validate model
        if model not in COHERE_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: {list(COHERE_DIMENSIONS.keys())}"
            )
        
        # Validate input type
        valid_input_types = [
            "search_document", "search_query", "classification", "clustering"
        ]
        if input_type not in valid_input_types:
            raise ValueError(
                f"Invalid input_type: {input_type}. "
                f"Valid types: {valid_input_types}"
            )
        
        self.client = cohere.Client(api_key=api_key)
        self.model = model
        self.input_type = input_type
        self._dimensions = COHERE_DIMENSIONS[model]
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type,
        )
        return response.embeddings[0]
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
        )
        return response.embeddings
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"cohere:{self.model}"
