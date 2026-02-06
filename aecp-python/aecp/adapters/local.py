"""
Local Model Adapter for AECP

Wraps any pre-loaded model object (e.g. a SentenceTransformer instance)
as an AECP-compatible embedding provider. Unlike HuggingFaceAdapter which
loads a model by name, LocalModelAdapter accepts an already-instantiated
model — giving you full control over model loading, device placement,
quantization, and caching.

This is the easiest path to AECP when you have full control over your
model weights.

Example:
    >>> from sentence_transformers import SentenceTransformer
    >>> from aecp.adapters import LocalModelAdapter
    >>>
    >>> model = SentenceTransformer('all-MiniLM-L6-v2')
    >>> adapter = LocalModelAdapter(model)
    >>> embedding = adapter.embed("Hello world")
    >>> len(embedding)
    384

    >>> # Works with any object that has an .encode() method
    >>> adapter = LocalModelAdapter(model, model_id="my-custom-model")
"""

from typing import List, Optional, Callable, Any
import logging

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class LocalModelAdapter(BaseAdapter):
    """
    Adapter that wraps a pre-loaded local model for AECP.

    Accepts any model object that has an ``encode`` method returning
    numpy arrays (e.g. ``SentenceTransformer``, custom wrappers, etc.).

    For models without an ``encode`` method, you can supply a custom
    ``encode_fn`` callable that takes ``(model, texts, **kwargs)`` and
    returns a numpy-like array of embeddings.

    Attributes:
        model: The pre-loaded model object.
        model_id: Identifier string for this model.
        normalize_embeddings: Whether to L2-normalise embeddings.

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> from aecp.adapters import LocalModelAdapter
        >>>
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> adapter = LocalModelAdapter(model)
        >>>
        >>> # With a custom model that doesn't have .encode()
        >>> adapter = LocalModelAdapter(
        ...     my_model,
        ...     encode_fn=lambda m, texts, **kw: m.get_embeddings(texts),
        ...     dimensions=512,
        ...     model_id="my-custom-v1",
        ... )
    """

    def __init__(
        self,
        model: Any,
        model_id: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize_embeddings: bool = True,
        encode_fn: Optional[Callable] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize LocalModelAdapter.

        Args:
            model: Pre-loaded model object. Must have an ``encode`` method
                   unless ``encode_fn`` is provided.
            model_id: Human-readable model identifier. If not provided,
                      inferred from the model's class name or attributes.
            dimensions: Embedding dimensionality. If not provided,
                        auto-detected by encoding a probe text.
            normalize_embeddings: Whether to L2-normalise output embeddings.
            encode_fn: Optional custom encoding function with signature
                       ``(model, texts: List[str], **kwargs) -> array-like``.
                       If not provided, ``model.encode()`` is used.
            batch_size: Batch size for the underlying model's encode call.
            **kwargs: Additional arguments passed to BaseAdapter.

        Raises:
            TypeError: If model has no ``encode`` method and no ``encode_fn``
                       is provided.
            ValueError: If dimensions cannot be determined.
        """
        super().__init__(**kwargs)

        self.model = model
        self.normalize_embeddings = normalize_embeddings
        self._encode_fn = encode_fn
        self._batch_size = batch_size

        # Validate that we can encode
        if encode_fn is None and not hasattr(model, "encode"):
            raise TypeError(
                "Model must have an 'encode' method, or you must provide "
                "an 'encode_fn'. Got model of type: "
                f"{type(model).__name__}"
            )

        # Determine model ID
        if model_id is not None:
            self._model_id = model_id
        else:
            # Try to infer from common attributes
            self._model_id = self._infer_model_id(model)

        # Determine dimensions
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = self._detect_dimensions(model)

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _infer_model_id(model: Any) -> str:
        """Best-effort model ID from the model object."""
        # SentenceTransformer stores model name
        for attr in (
            "model_card_data",  # newer sentence-transformers
            "_model_card_data",
        ):
            card = getattr(model, attr, None)
            if card is not None:
                name = getattr(card, "model_name", None) or getattr(
                    card, "base_model", None
                )
                if name:
                    return f"local:{name}"

        # Fallback: class name
        return f"local:{type(model).__name__}"

    def _detect_dimensions(self, model: Any) -> int:
        """Auto-detect embedding dimensions by probing the model."""
        # SentenceTransformer exposes this directly
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
            if dim and dim > 0:
                return int(dim)

        # Fallback: encode a probe text and measure
        try:
            probe = self._encode_texts(["aecp dimension probe"])
            return len(probe[0])
        except Exception as e:
            raise ValueError(
                f"Cannot auto-detect embedding dimensions: {e}. "
                "Please provide the 'dimensions' argument explicitly."
            )

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts using the underlying model.

        Returns a list of lists of floats.
        """
        if self._encode_fn is not None:
            raw = self._encode_fn(self.model, texts)
        else:
            # Standard SentenceTransformer-compatible call
            encode_kwargs = {
                "show_progress_bar": False,
            }
            # Only pass normalize_embeddings if the model accepts it
            # (SentenceTransformer does, others might not)
            try:
                raw = self.model.encode(
                    texts,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=self._batch_size,
                    **encode_kwargs,
                )
            except TypeError:
                # Model doesn't accept normalize_embeddings or batch_size
                raw = self.model.encode(texts)

        # Convert to list-of-lists
        result = []
        for item in raw:
            if hasattr(item, "tolist"):
                result.append(item.tolist())
            elif isinstance(item, (list, tuple)):
                result.append(list(item))
            else:
                result.append(list(item))
        return result

    # ── BaseAdapter interface ────────────────────────────────────────

    def _embed_impl(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self._encode_texts([text])[0]

    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._encode_texts(texts)

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    def get_model_id(self) -> str:
        """Get model identifier."""
        return self._model_id
