"""
AECP Integrations

Pre-built integration patterns for common agent frameworks and protocols:

- **AECPAgent**: Base class for agents with decoupled LLM + AECP embeddings.
  Works with Pydantic AI, LangChain, or any LLM framework.
- **AECPEnabledAgent**: Concrete agent wrapper with knowledge base and RAG.
- **AECPMCPServer**: MCP server with AECP-aware semantic search.
- **AECPMCPClient**: MCP client that auto-negotiates AECP with servers.

Example:
    >>> from aecp.integrations import AECPAgent, AECPEnabledAgent
    >>> from aecp.adapters import OpenAIAdapter
    >>>
    >>> # Simple decoupled agent
    >>> agent = AECPAgent(
    ...     llm_provider="openai:gpt-4",
    ...     embedder=OpenAIAdapter(model="text-embedding-3-small"),
    ... )
"""

from .base import AECPAgent
from .agent_framework import AECPEnabledAgent
from .mcp import AECPMCPServer, AECPMCPClient

__all__ = [
    "AECPAgent",
    "AECPEnabledAgent",
    "AECPMCPServer",
    "AECPMCPClient",
]
