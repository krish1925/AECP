"""
Agent Handoff Pattern

Enable seamless context transfer between specialist agents
that use different embedding models.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from ..types import EmbeddingProvider
from ..protocol import AECP, CalibrationResult


@dataclass
class AgentContext:
    """
    Context being passed between agents.
    
    Attributes:
        content: Original text content
        embedding: Current embedding (in current agent's space)
        history: List of agents that have handled this context
        metadata: Additional context metadata
        created_at: When context was created
    """
    content: str
    embedding: np.ndarray
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_to_history(self, agent_id: str) -> None:
        """Record that an agent handled this context."""
        self.history.append(agent_id)


class AgentHandoff:
    """
    Manage handoffs between specialist agents with different embedding models.
    
    Each agent can specialize in different domains (code, legal, medical)
    using the most appropriate embedding model, while maintaining
    seamless context transfer through AECP.
    
    Example:
        >>> from aecp.patterns import AgentHandoff
        >>> from aecp.adapters import VoyageAdapter, OpenAIAdapter, CohereAdapter
        >>> 
        >>> handoff = AgentHandoff({
        ...     'code': VoyageAdapter(model='voyage-code-2'),
        ...     'general': OpenAIAdapter(),
        ...     'multilingual': CohereAdapter(model='embed-multilingual-v3.0'),
        ... })
        >>> 
        >>> # Calibrate all agent pairs
        >>> handoff.calibrate_all()
        >>> 
        >>> # Start task with code agent
        >>> context = handoff.start("Debug this Python code", agent='code')
        >>> 
        >>> # Hand off to general agent
        >>> context = handoff.transfer(context, to_agent='general')
        >>> 
        >>> # Context preserved across different embedding models!
    """
    
    def __init__(
        self,
        agents: Dict[str, EmbeddingProvider],
        default_agent: Optional[str] = None,
    ):
        """
        Initialize agent handoff system.
        
        Args:
            agents: Dictionary mapping agent names to embedding providers
            default_agent: Default agent to use (first one if not specified)
            
        Raises:
            ValueError: If no agents provided
        """
        if not agents:
            raise ValueError("At least one agent must be provided")
        
        self.agent_names = list(agents.keys())
        self.default_agent = default_agent or self.agent_names[0]
        
        if self.default_agent not in self.agent_names:
            raise ValueError(
                f"Default agent '{self.default_agent}' not in agents"
            )
        
        # Create AECP instances for each agent
        self.agents: Dict[str, AECP] = {}
        for name, adapter in agents.items():
            self.agents[name] = AECP(adapter, agent_id=name)
        
        # Track calibration status
        self._calibration_status: Dict[str, Dict[str, bool]] = {
            name: {other: False for other in self.agent_names if other != name}
            for name in self.agent_names
        }
        
        self._stats = {
            "starts": 0,
            "transfers": 0,
            "contexts_created": 0,
        }
    
    def calibrate(
        self,
        agent1: str,
        agent2: str,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True
    ) -> CalibrationResult:
        """
        Calibrate transfer matrices between two agents.
        
        Args:
            agent1: First agent name
            agent2: Second agent name
            vocabulary: Custom calibration vocabulary
            verbose: Whether to print progress
            
        Returns:
            CalibrationResult with quality metrics
        """
        if agent1 not in self.agents:
            raise ValueError(f"Unknown agent: {agent1}")
        if agent2 not in self.agents:
            raise ValueError(f"Unknown agent: {agent2}")
        
        result = self.agents[agent1].calibrate_with(
            self.agents[agent2],
            vocabulary=vocabulary,
            verbose=verbose,
        )
        
        if result.success:
            self._calibration_status[agent1][agent2] = True
            self._calibration_status[agent2][agent1] = True
        
        return result
    
    def calibrate_all(
        self,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate all agent pairs.
        
        Args:
            vocabulary: Custom calibration vocabulary
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping agent pairs to calibration results
        """
        results = {}
        
        # Calibrate each unique pair
        for i, agent1 in enumerate(self.agent_names):
            for agent2 in self.agent_names[i + 1:]:
                pair_key = f"{agent1}_{agent2}"
                if verbose:
                    print(f"\nCalibrating {agent1} <-> {agent2}...")
                
                results[pair_key] = self.calibrate(
                    agent1, agent2,
                    vocabulary=vocabulary,
                    verbose=verbose,
                )
        
        return results
    
    def start(
        self,
        content: str,
        agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Start a new task with an agent.
        
        Args:
            content: Initial content/query
            agent: Agent to use (default agent if not specified)
            metadata: Optional metadata to attach
            
        Returns:
            AgentContext for tracking the task
        """
        agent = agent or self.default_agent
        
        if agent not in self.agents:
            raise ValueError(f"Unknown agent: {agent}")
        
        # Create embedding
        embedding = self.agents[agent].embed(content)
        
        # Create context
        context = AgentContext(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        context.add_to_history(agent)
        
        self._stats["starts"] += 1
        self._stats["contexts_created"] += 1
        
        return context
    
    def transfer(
        self,
        context: AgentContext,
        to_agent: str,
        update_content: Optional[str] = None
    ) -> AgentContext:
        """
        Transfer context to a different agent.
        
        Args:
            context: Current context
            to_agent: Target agent name
            update_content: Optional new content to embed
            
        Returns:
            Updated AgentContext with embedding in target agent's space
            
        Raises:
            ValueError: If agents not calibrated
        """
        if to_agent not in self.agents:
            raise ValueError(f"Unknown agent: {to_agent}")
        
        # Get source agent (last in history)
        from_agent = context.history[-1]
        
        if from_agent == to_agent:
            # No transfer needed
            return context
        
        # Check calibration
        if not self._calibration_status[from_agent][to_agent]:
            raise ValueError(
                f"Agents '{from_agent}' and '{to_agent}' not calibrated. "
                f"Call calibrate('{from_agent}', '{to_agent}') first."
            )
        
        # Transfer embedding
        if update_content:
            # Embed new content in target space
            new_embedding = self.agents[to_agent].embed(update_content)
            context.content = update_content
        else:
            # Transfer existing embedding
            transfer = self.agents[from_agent].transfer_embedding_to(
                to_agent,
                context.embedding,
            )
            new_embedding = transfer.embedding
        
        # Update context
        context.embedding = new_embedding
        context.add_to_history(to_agent)
        
        self._stats["transfers"] += 1
        
        return context
    
    def get_current_agent(self, context: AgentContext) -> str:
        """
        Get the current agent handling a context.
        
        Args:
            context: The context to check
            
        Returns:
            Name of the current agent
        """
        return context.history[-1]
    
    def is_calibrated(self, agent1: str, agent2: str) -> bool:
        """
        Check if two agents are calibrated.
        
        Args:
            agent1: First agent name
            agent2: Second agent name
            
        Returns:
            True if calibrated
        """
        if agent1 not in self._calibration_status:
            return False
        if agent2 not in self._calibration_status[agent1]:
            return agent1 == agent2
        return self._calibration_status[agent1][agent2]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage counts
        """
        return {
            **self._stats,
            "agents": self.agent_names,
            "calibration_status": {
                f"{a1}_{a2}": status
                for a1, statuses in self._calibration_status.items()
                for a2, status in statuses.items()
            },
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "starts": 0,
            "transfers": 0,
            "contexts_created": 0,
        }
