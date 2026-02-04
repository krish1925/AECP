"""
AECP High-Level Patterns

Pre-built patterns for common AECP use cases:
- CostOptimizer: Minimize embedding costs while maintaining quality
- PrivacyBridge: Keep sensitive data local, share only semantics
- AgentHandoff: Seamless context transfer between specialist agents

Example:
    >>> from aecp.patterns import CostOptimizer
    >>> optimizer = CostOptimizer(
    ...     cheap_adapter=OpenAIAdapter(model="text-embedding-3-small"),
    ...     expensive_adapter=VoyageAdapter(model="voyage-large-2"),
    ... )
    >>> result = optimizer.embed("query", precision="high")
"""

from .cost_optimizer import CostOptimizer
from .privacy_bridge import PrivacyBridge
from .agent_handoff import AgentHandoff

__all__ = [
    "CostOptimizer",
    "PrivacyBridge",
    "AgentHandoff",
]
