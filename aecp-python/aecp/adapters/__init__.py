"""
AECP Provider Adapters

Adapters for various embedding providers.

Available Adapters:
- OpenAIAdapter: OpenAI embeddings (text-embedding-3-small, ada-002, etc.)
- VoyageAdapter: Voyage AI embeddings (voyage-2, voyage-code-2, etc.)
- CohereAdapter: Cohere embeddings (embed-english-v3.0, multilingual, etc.)
- HuggingFaceAdapter: Local HuggingFace/SentenceTransformers models
- MockAdapter: For testing (no API calls)

Example:
    >>> from aecp.adapters import OpenAIAdapter
    >>> adapter = OpenAIAdapter(api_key="sk-...")
    >>> embedding = adapter.embed("Hello world")
"""

from .openai import OpenAIAdapter
from .voyage import VoyageAdapter
from .cohere import CohereAdapter
from .huggingface import HuggingFaceAdapter
from .mock import MockAdapter

__all__ = [
    "OpenAIAdapter",
    "VoyageAdapter",
    "CohereAdapter",
    "HuggingFaceAdapter",
    "MockAdapter",
]
