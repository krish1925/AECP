"""
HuggingFace/SentenceTransformers Adapter for AECP

Provides embedding generation using local HuggingFace models.
No API key required - runs locally.
"""

from typing import List, Optional

from .base import BaseAdapter


# Common model dimension mapping
HUGGINGFACE_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "msmarco-MiniLM-L-6-v3": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class HuggingFaceAdapter(BaseAdapter):
    """
    Adapter for HuggingFace/SentenceTransformers embedding models.
    
    Runs locally - no API key required.
    
    Supports any SentenceTransformers-compatible model:
    - all-MiniLM-L6-v2 (384 dimensions, fast, recommended)
    - all-mpnet-base-v2 (768 dimensions, higher quality)
    - all-MiniLM-L12-v2 (384 dimensions, balanced)
    - Any model from HuggingFace Hub
    
    Example:
        >>> adapter = HuggingFaceAdapter()  # Default: all-MiniLM-L6-v2
        >>> embedding = adapter.embed("Hello world")
        >>> len(embedding)
        384
        
        >>> # Use specific model
        >>> adapter = HuggingFaceAdapter(model="all-mpnet-base-v2")
        
        >>> # Use GPU
        >>> adapter = HuggingFaceAdapter(device="cuda")
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize HuggingFace adapter.
        
        Args:
            model: Model name or path
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
            normalize_embeddings: Whether to L2-normalize embeddings
            **kwargs: Additional arguments passed to BaseAdapter
            
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__(**kwargs)
        
        # Lazy import
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformers package not installed. "
                "Install with: pip install aecp[huggingface]"
            )
        
        self.model_name = model
        self.normalize_embeddings = normalize_embeddings
        
        # Load model
        self.model = SentenceTransformer(model, device=device)
        
        # Get dimensions from model
        self._dimensions = self.model.get_sentence_embedding_dimension()
    
    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return f"huggingface:{self.model_name}"
