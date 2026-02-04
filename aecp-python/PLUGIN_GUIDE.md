# AECP Plugin System Guide

## Overview

AECP includes a powerful plugin system that allows you to create custom embedding adapters without modifying the core library code. This enables:

- **Plug-and-play architecture** - Add new providers easily
- **Third-party integrations** - Create adapters for any embedding API
- **Clean separation** - Keep your custom code separate from AECP core
- **Easy distribution** - Package and share your adapters

## Quick Start

### Creating a Custom Adapter

```python
from aecp import register_adapter, adapter
from aecp.adapters.base import BaseAdapter
from typing import List

class MyCustomAdapter(BaseAdapter):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._dimensions = 384
    
    def _embed_impl(self, text: str) -> List[float]:
        # Your implementation here
        # Call your API, process response, etc.
        return [0.1] * self._dimensions
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        # Batch processing
        return [self._embed_impl(t) for t in texts]
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return "my-provider:v1"

# Register it
register_adapter("my-provider", MyCustomAdapter)

# Use it
from aecp import AECP, create_adapter

adapter = create_adapter("my-provider", api_key="...")
agent = AECP(adapter)
```

### Using the Decorator (Auto-Registration)

```python
from aecp import adapter
from aecp.adapters.base import BaseAdapter

@adapter("my-provider")
class MyCustomAdapter(BaseAdapter):
    # ... same implementation as above
    pass

# Already registered! Just use it:
from aecp import create_adapter
adapter = create_adapter("my-provider", api_key="...")
```

## Plugin API Reference

### Registration Functions

#### `register_adapter(name, adapter_class, factory=None, override=False)`

Register a custom adapter.

```python
from aecp import register_adapter

register_adapter("my-provider", MyAdapter)
register_adapter("my-provider", MyAdapter, override=True)  # Replace existing
```

#### `@adapter(name, override=False)`

Decorator for auto-registration.

```python
@adapter("my-provider")
class MyAdapter(BaseAdapter):
    pass
```

### Access Functions

#### `get_adapter(name) -> Type[EmbeddingProvider]`

Get a registered adapter class.

```python
from aecp import get_adapter

AdapterClass = get_adapter("my-provider")
adapter = AdapterClass(api_key="...")
```

#### `create_adapter(name, **kwargs) -> EmbeddingProvider`

Create an adapter instance.

```python
from aecp import create_adapter

adapter = create_adapter("my-provider", api_key="...")
```

#### `list_adapters() -> Dict[str, Type[EmbeddingProvider]]`

List all registered adapters.

```python
from aecp import list_adapters

adapters = list_adapters()
print(adapters.keys())  # ['openai', 'voyage', 'my-provider', ...]
```

## Complete Example: Anthropic Claude Adapter

```python
from aecp import adapter
from aecp.adapters.base import BaseAdapter
from typing import List
import anthropic

@adapter("anthropic")
class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic's Claude models.
    
    Usage:
        adapter = create_adapter("anthropic", api_key="sk-ant-...")
        agent = AECP(adapter)
    """
    
    DIMENSIONS = {
        "claude-3-opus": 1024,
        "claude-3-sonnet": 768,
        "claude-3-haiku": 512,
    }
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet", **kwargs):
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        if model not in self.DIMENSIONS:
            raise ValueError(f"Unknown model: {model}")
        
        self._dimensions = self.DIMENSIONS[model]
    
    def _embed_impl(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Note: Anthropic doesn't have embeddings API yet
        # This is pseudocode for when it becomes available
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.embedding
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def get_dimensions(self) -> int:
        return self._dimensions
    
    def get_model_id(self) -> str:
        return f"anthropic:{self.model}"

# Use it
from aecp import AECP, create_adapter

adapter = create_adapter("anthropic", api_key="sk-ant-...", model="claude-3-sonnet")
agent = AECP(adapter)

embedding = agent.embed("Hello world!")
print(f"Generated {len(embedding)}-dimensional embedding")
```

## Publishing Your Adapter as a Package

### Package Structure

```
my-aecp-adapter/
├── my_aecp_adapter/
│   ├── __init__.py
│   └── adapter.py
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
└── tests/
    └── test_adapter.py
```

### `my_aecp_adapter/adapter.py`

```python
from aecp import adapter
from aecp.adapters.base import BaseAdapter
from typing import List

@adapter("my-provider")
class MyProviderAdapter(BaseAdapter):
    """Adapter for MyProvider embedding API."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        # Your initialization
    
    def _embed_impl(self, text: str) -> List[float]:
        # Your implementation
        pass
    
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        pass
    
    def get_dimensions(self) -> int:
        return 384
    
    def get_model_id(self) -> str:
        return "my-provider:v1"
```

### `my_aecp_adapter/__init__.py`

```python
"""MyProvider adapter for AECP."""

from .adapter import MyProviderAdapter

__version__ = "1.0.0"
__all__ = ["MyProviderAdapter"]

# Adapter is already registered via decorator
```

### `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="my-aecp-adapter",
    version="1.0.0",
    description="MyProvider adapter for AECP",
    packages=find_packages(),
    install_requires=[
        "aecp>=1.0.0",
        "my-provider-sdk>=1.0.0",  # Your provider's SDK
    ],
    python_requires=">=3.8",
)
```

### Usage by End Users

```bash
pip install my-aecp-adapter
```

```python
from aecp import AECP, create_adapter

# Importing the package auto-registers the adapter via decorator
import my_aecp_adapter

# Use it
adapter = create_adapter("my-provider", api_key="...")
agent = AECP(adapter)
```

## Built-in Adapters

These adapters are automatically registered when you import AECP:

- `openai` - OpenAI embeddings
- `voyage` - Voyage AI embeddings
- `cohere` - Cohere embeddings
- `huggingface` - HuggingFace/SentenceTransformers
- `mock` - Mock adapter for testing

## Best Practices

### 1. Extend BaseAdapter

Always inherit from `BaseAdapter` to get:
- Input validation
- Retry logic with exponential backoff
- Text truncation
- Error handling

### 2. Handle API Errors Gracefully

```python
def _embed_impl(self, text: str) -> List[float]:
    try:
        response = self.client.embed(text)
        return response.embedding
    except APIConnectionError as e:
        raise RuntimeError(f"Failed to connect to API: {e}")
    except APIResponseError as e:
        raise RuntimeError(f"API returned error: {e}")
```

### 3. Support Batch Operations

```python
def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
    # Use provider's batch API if available
    return self.client.embed_batch(texts)
    
    # Otherwise, fall back to sequential (BaseAdapter handles retries)
    return [self._embed_impl(t) for t in texts]
```

### 4. Document Your Adapter

```python
@adapter("my-provider")
class MyAdapter(BaseAdapter):
    """
    Adapter for MyProvider embedding API.
    
    Supported models:
    - my-model-v1 (384 dimensions)
    - my-model-v2 (768 dimensions)
    
    Args:
        api_key: MyProvider API key
        model: Model to use (default: my-model-v1)
        
    Example:
        >>> adapter = create_adapter("my-provider", api_key="...")
        >>> agent = AECP(adapter)
    """
```

### 5. Write Tests

```python
# tests/test_adapter.py
import pytest
from my_aecp_adapter import MyProviderAdapter

def test_adapter_creation():
    adapter = MyProviderAdapter(api_key="test-key")
    assert adapter.get_dimensions() > 0

def test_embed_single():
    adapter = MyProviderAdapter(api_key="test-key")
    embedding = adapter.embed("test")
    assert len(embedding) == adapter.get_dimensions()

def test_embed_batch():
    adapter = MyProviderAdapter(api_key="test-key")
    embeddings = adapter.embed_batch(["test1", "test2"])
    assert len(embeddings) == 2
```

## Troubleshooting

### Adapter Already Registered

```python
# Use override=True to replace
register_adapter("name", MyAdapter, override=True)
```

### Import Errors

Make sure you have the provider's SDK installed:

```bash
pip install provider-sdk
```

### Adapter Not Found

```python
from aecp import list_adapters

# Check what's registered
print(list_adapters().keys())
```

## Examples

See `examples/custom_adapter.py` for complete working examples.
