"""
AECP Plugin System

Provides utilities for creating and registering custom embedding adapters
without modifying the core AECP codebase.
"""

from typing import Dict, Type, Optional, Callable
from .types import EmbeddingProvider


class PluginRegistry:
    """
    Registry for custom embedding provider plugins.
    
    Allows third-party developers to register their own adapters
    without modifying AECP's source code.
    
    Example:
        >>> from aecp.plugin import register_adapter
        >>> from aecp.adapters.base import BaseAdapter
        >>> 
        >>> class MyCustomAdapter(BaseAdapter):
        ...     def _embed_impl(self, text):
        ...         # Your implementation
        ...         pass
        >>> 
        >>> register_adapter("my-provider", MyCustomAdapter)
        >>> 
        >>> # Now use it
        >>> from aecp import AECP
        >>> from aecp.plugin import get_adapter
        >>> 
        >>> adapter = get_adapter("my-provider")(api_key="...")
        >>> agent = AECP(adapter)
    """
    
    _adapters: Dict[str, Type[EmbeddingProvider]] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        adapter_class: Optional[Type[EmbeddingProvider]] = None,
        factory: Optional[Callable] = None,
        override: bool = False
    ) -> None:
        """
        Register a custom adapter.
        
        Args:
            name: Unique identifier for the adapter (e.g., "my-provider")
            adapter_class: The adapter class (must implement EmbeddingProvider)
            factory: Optional factory function that returns an adapter instance
            override: Whether to override existing adapter with same name
            
        Raises:
            ValueError: If name already registered and override=False
            TypeError: If adapter_class doesn't implement EmbeddingProvider
        """
        if not override and name in cls._adapters:
            raise ValueError(
                f"Adapter '{name}' already registered. "
                f"Use override=True to replace it."
            )
        
        if adapter_class is not None:
            if not issubclass(adapter_class, EmbeddingProvider):
                raise TypeError(
                    f"Adapter class must implement EmbeddingProvider, "
                    f"got {adapter_class}"
                )
            cls._adapters[name] = adapter_class
        
        if factory is not None:
            if not callable(factory):
                raise TypeError(f"Factory must be callable, got {type(factory)}")
            cls._factories[name] = factory
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an adapter.
        
        Args:
            name: Adapter identifier to remove
        """
        cls._adapters.pop(name, None)
        cls._factories.pop(name, None)
    
    @classmethod
    def get(cls, name: str) -> Type[EmbeddingProvider]:
        """
        Get a registered adapter class.
        
        Args:
            name: Adapter identifier
            
        Returns:
            The adapter class
            
        Raises:
            KeyError: If adapter not found
        """
        if name not in cls._adapters:
            raise KeyError(
                f"Adapter '{name}' not found. "
                f"Available: {list(cls._adapters.keys())}"
            )
        return cls._adapters[name]
    
    @classmethod
    def get_factory(cls, name: str) -> Callable:
        """
        Get a registered adapter factory.
        
        Args:
            name: Adapter identifier
            
        Returns:
            The factory function
            
        Raises:
            KeyError: If factory not found
        """
        if name not in cls._factories:
            raise KeyError(
                f"Factory for '{name}' not found. "
                f"Available: {list(cls._factories.keys())}"
            )
        return cls._factories[name]
    
    @classmethod
    def list_adapters(cls) -> Dict[str, Type[EmbeddingProvider]]:
        """
        List all registered adapters.
        
        Returns:
            Dictionary mapping names to adapter classes
        """
        return cls._adapters.copy()
    
    @classmethod
    def has_adapter(cls, name: str) -> bool:
        """
        Check if an adapter is registered.
        
        Args:
            name: Adapter identifier
            
        Returns:
            True if adapter is registered
        """
        return name in cls._adapters
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (mainly for testing)."""
        cls._adapters.clear()
        cls._factories.clear()


# Convenience functions
def register_adapter(
    name: str,
    adapter_class: Optional[Type[EmbeddingProvider]] = None,
    factory: Optional[Callable] = None,
    override: bool = False
) -> None:
    """
    Register a custom adapter.
    
    Args:
        name: Unique identifier for the adapter
        adapter_class: The adapter class
        factory: Optional factory function
        override: Whether to override existing adapter
        
    Example:
        >>> from aecp.plugin import register_adapter
        >>> from aecp.adapters.base import BaseAdapter
        >>> 
        >>> class MyAdapter(BaseAdapter):
        ...     def _embed_impl(self, text):
        ...         return [0.1, 0.2, 0.3]
        ...     def _embed_batch_impl(self, texts):
        ...         return [[0.1, 0.2, 0.3]] * len(texts)
        ...     def get_dimensions(self):
        ...         return 3
        ...     def get_model_id(self):
        ...         return "my-model"
        >>> 
        >>> register_adapter("my-provider", MyAdapter)
    """
    PluginRegistry.register(name, adapter_class, factory, override)


def get_adapter(name: str) -> Type[EmbeddingProvider]:
    """
    Get a registered adapter class.
    
    Args:
        name: Adapter identifier
        
    Returns:
        The adapter class
        
    Example:
        >>> from aecp.plugin import get_adapter
        >>> MyAdapter = get_adapter("my-provider")
        >>> adapter = MyAdapter(api_key="...")
    """
    return PluginRegistry.get(name)


def get_adapter_factory(name: str) -> Callable:
    """
    Get a registered adapter factory.
    
    Args:
        name: Adapter identifier
        
    Returns:
        The factory function
    """
    return PluginRegistry.get_factory(name)


def create_adapter(name: str, **kwargs) -> EmbeddingProvider:
    """
    Create an adapter instance using registered factory or class.
    
    Args:
        name: Adapter identifier
        **kwargs: Arguments to pass to adapter constructor or factory
        
    Returns:
        Adapter instance
        
    Example:
        >>> from aecp.plugin import create_adapter
        >>> adapter = create_adapter("my-provider", api_key="...")
    """
    if PluginRegistry.has_adapter(name):
        if name in PluginRegistry._factories:
            return PluginRegistry.get_factory(name)(**kwargs)
        else:
            return PluginRegistry.get(name)(**kwargs)
    else:
        raise KeyError(f"Adapter '{name}' not found")


def list_adapters() -> Dict[str, Type[EmbeddingProvider]]:
    """
    List all registered adapters.
    
    Returns:
        Dictionary mapping names to adapter classes
        
    Example:
        >>> from aecp.plugin import list_adapters
        >>> adapters = list_adapters()
        >>> print(adapters.keys())
    """
    return PluginRegistry.list_adapters()


# Decorator for easy registration
def adapter(name: str, override: bool = False):
    """
    Decorator for registering an adapter class.
    
    Args:
        name: Adapter identifier
        override: Whether to override existing adapter
        
    Example:
        >>> from aecp.plugin import adapter
        >>> from aecp.adapters.base import BaseAdapter
        >>> 
        >>> @adapter("my-provider")
        ... class MyAdapter(BaseAdapter):
        ...     def _embed_impl(self, text):
        ...         return [0.1, 0.2, 0.3]
        ...     # ... implement other methods
    """
    def decorator(cls: Type[EmbeddingProvider]) -> Type[EmbeddingProvider]:
        register_adapter(name, cls, override=override)
        return cls
    return decorator


# Auto-register built-in adapters
def _register_builtin_adapters():
    """Register built-in adapters."""
    from .adapters import (
        OpenAIAdapter,
        VoyageAdapter,
        CohereAdapter,
        HuggingFaceAdapter,
        MockAdapter,
    )
    
    # Register with try-except in case imports fail
    builtin = {
        "openai": OpenAIAdapter,
        "voyage": VoyageAdapter,
        "cohere": CohereAdapter,
        "huggingface": HuggingFaceAdapter,
        "mock": MockAdapter,
    }
    
    for name, adapter_class in builtin.items():
        try:
            PluginRegistry.register(name, adapter_class)
        except Exception:
            # Silently skip if adapter can't be imported
            pass


# Register built-in adapters on import
_register_builtin_adapters()
