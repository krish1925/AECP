"""
Base Adapter Implementation

Provides common functionality for all adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time
import logging

from ..types import EmbeddingProvider

logger = logging.getLogger(__name__)


class BaseAdapter(EmbeddingProvider, ABC):
    """
    Base class for embedding adapters.
    
    Provides common functionality like:
    - Input validation
    - Retry logic
    - Rate limiting
    - Logging
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_text_length: int = 8192,
    ):
        """
        Initialize base adapter.
        
        Args:
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            max_text_length: Maximum text length (characters)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_text_length = max_text_length
        self._dimensions: Optional[int] = None
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with validation and retries.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If embedding fails after retries
        """
        # Validate input
        self._validate_text(text)
        
        # Truncate if needed
        text = self._truncate_text(text)
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._embed_impl(text)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embed attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(
            f"Failed to embed after {self.max_retries} attempts: {last_error}"
        )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with validation and retries.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is invalid
            RuntimeError: If embedding fails after retries
        """
        # Validate inputs
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        validated_texts = []
        for i, text in enumerate(texts):
            try:
                self._validate_text(text)
                validated_texts.append(self._truncate_text(text))
            except ValueError as e:
                raise ValueError(f"Invalid text at index {i}: {e}")
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._embed_batch_impl(validated_texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch embed attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(
            f"Failed to embed batch after {self.max_retries} attempts: {last_error}"
        )
    
    def _validate_text(self, text: str) -> None:
        """
        Validate input text.
        
        Args:
            text: Text to validate
            
        Raises:
            ValueError: If text is invalid
        """
        if text is None:
            raise ValueError("Text cannot be None")
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text)}")
        if len(text.strip()) == 0:
            raise ValueError("Text cannot be empty or whitespace only")
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text if it exceeds maximum length.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        if len(text) > self.max_text_length:
            logger.warning(
                f"Text truncated from {len(text)} to {self.max_text_length} chars"
            )
            return text[:self.max_text_length]
        return text
    
    @abstractmethod
    def _embed_impl(self, text: str) -> List[float]:
        """
        Implementation of single text embedding.
        
        Args:
            text: Pre-validated text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Implementation of batch text embedding.
        
        Args:
            texts: Pre-validated texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Get model identifier."""
        pass
