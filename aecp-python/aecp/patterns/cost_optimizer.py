"""
Cost Optimizer Pattern

Minimize embedding costs by using cheaper models for most work
and expensive models only when high precision is needed.
"""

from typing import List, Optional, Dict, Any, Literal
import numpy as np

from ..types import EmbeddingProvider
from ..protocol import AECP, CalibrationResult


PrecisionLevel = Literal["low", "medium", "high"]


class CostOptimizer:
    """
    Optimize embedding costs by intelligently routing between models.
    
    Uses a cheap model for most operations and an expensive model
    only when high precision is required. AECP enables seamless
    transfer between the two embedding spaces.
    
    Cost savings can be 70-90% compared to always using expensive models.
    
    Example:
        >>> from aecp.patterns import CostOptimizer
        >>> from aecp.adapters import OpenAIAdapter, VoyageAdapter
        >>> 
        >>> optimizer = CostOptimizer(
        ...     cheap_adapter=OpenAIAdapter(model="text-embedding-3-small"),
        ...     expensive_adapter=VoyageAdapter(model="voyage-large-2"),
        ... )
        >>> 
        >>> # Calibrate once
        >>> optimizer.calibrate()
        >>> 
        >>> # Use cheap model (default)
        >>> embedding = optimizer.embed("common query")
        >>> 
        >>> # Use expensive model when needed
        >>> precise_embedding = optimizer.embed("critical query", precision="high")
        >>> 
        >>> # Get cost statistics
        >>> print(optimizer.get_stats())
    """
    
    def __init__(
        self,
        cheap_adapter: EmbeddingProvider,
        expensive_adapter: EmbeddingProvider,
        cheap_cost_per_token: float = 0.00002,  # $0.02/1M tokens
        expensive_cost_per_token: float = 0.00012,  # $0.12/1M tokens
        avg_tokens_per_text: int = 10,
    ):
        """
        Initialize cost optimizer.
        
        Args:
            cheap_adapter: Low-cost embedding adapter
            expensive_adapter: High-quality embedding adapter
            cheap_cost_per_token: Cost per token for cheap model
            expensive_cost_per_token: Cost per token for expensive model
            avg_tokens_per_text: Average tokens per text (for cost estimation)
        """
        if cheap_adapter is None or expensive_adapter is None:
            raise ValueError("Both adapters must be provided")
        
        self.cheap_agent = AECP(cheap_adapter, agent_id="cheap_agent")
        self.expensive_agent = AECP(expensive_adapter, agent_id="expensive_agent")
        
        self.cheap_cost_per_token = cheap_cost_per_token
        self.expensive_cost_per_token = expensive_cost_per_token
        self.avg_tokens_per_text = avg_tokens_per_text
        
        self._calibrated = False
        self._stats = {
            "cheap_calls": 0,
            "expensive_calls": 0,
            "transfers": 0,
            "estimated_cost": 0.0,
            "estimated_savings": 0.0,
        }
    
    def calibrate(
        self,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True
    ) -> CalibrationResult:
        """
        Calibrate transfer matrices between cheap and expensive models.
        
        Args:
            vocabulary: Custom calibration vocabulary
            verbose: Whether to print progress
            
        Returns:
            CalibrationResult with quality metrics
        """
        result = self.cheap_agent.calibrate_with(
            self.expensive_agent,
            vocabulary=vocabulary,
            verbose=verbose,
        )
        
        if result.success:
            self._calibrated = True
        
        return result
    
    def embed(
        self,
        text: str,
        precision: PrecisionLevel = "low",
        target_space: Literal["cheap", "expensive"] = "cheap"
    ) -> np.ndarray:
        """
        Generate embedding with specified precision level.
        
        Args:
            text: Text to embed
            precision: Quality level ("low", "medium", "high")
            target_space: Which embedding space to return
            
        Returns:
            Embedding vector in the target space
            
        Raises:
            ValueError: If not calibrated and transfer is needed
        """
        if precision == "high":
            # Use expensive model directly
            embedding = self.expensive_agent.embed(text)
            self._stats["expensive_calls"] += 1
            self._update_cost(expensive=True)
            
            if target_space == "cheap" and self._calibrated:
                # Transfer to cheap space
                transfer = self.expensive_agent.transfer_embedding_to(
                    self.cheap_agent.agent_id, embedding
                )
                self._stats["transfers"] += 1
                return transfer.embedding
            
            return embedding
        
        elif precision == "medium":
            # Use cheap model, optionally transfer
            if not self._calibrated:
                raise ValueError(
                    "Calibration required for medium precision. "
                    "Call calibrate() first."
                )
            
            embedding = self.cheap_agent.embed(text)
            self._stats["cheap_calls"] += 1
            self._update_cost(expensive=False)
            
            if target_space == "expensive":
                transfer = self.cheap_agent.transfer_embedding_to(
                    self.expensive_agent.agent_id, embedding
                )
                self._stats["transfers"] += 1
                return transfer.embedding
            
            return embedding
        
        else:  # low precision
            # Always use cheap model
            embedding = self.cheap_agent.embed(text)
            self._stats["cheap_calls"] += 1
            self._update_cost(expensive=False)
            
            if target_space == "expensive" and self._calibrated:
                transfer = self.cheap_agent.transfer_embedding_to(
                    self.expensive_agent.agent_id, embedding
                )
                self._stats["transfers"] += 1
                return transfer.embedding
            
            return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        precision: PrecisionLevel = "low",
        target_space: Literal["cheap", "expensive"] = "cheap"
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            precision: Quality level
            target_space: Which embedding space to return
            
        Returns:
            List of embedding vectors
        """
        return [
            self.embed(text, precision=precision, target_space=target_space)
            for text in texts
        ]
    
    def _update_cost(self, expensive: bool) -> None:
        """Update cost statistics."""
        if expensive:
            cost = self.expensive_cost_per_token * self.avg_tokens_per_text
            savings = 0
        else:
            cost = self.cheap_cost_per_token * self.avg_tokens_per_text
            savings = (
                self.expensive_cost_per_token - self.cheap_cost_per_token
            ) * self.avg_tokens_per_text
        
        self._stats["estimated_cost"] += cost
        self._stats["estimated_savings"] += savings
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage and cost statistics.
        
        Returns:
            Dictionary with:
            - cheap_calls: Number of cheap model calls
            - expensive_calls: Number of expensive model calls
            - transfers: Number of AECP transfers
            - estimated_cost: Total estimated cost
            - estimated_savings: Estimated savings vs always using expensive
            - savings_percentage: Percentage saved
        """
        total_calls = self._stats["cheap_calls"] + self._stats["expensive_calls"]
        
        if total_calls > 0:
            all_expensive_cost = (
                total_calls * self.expensive_cost_per_token * self.avg_tokens_per_text
            )
            savings_pct = (
                self._stats["estimated_savings"] / all_expensive_cost * 100
                if all_expensive_cost > 0 else 0
            )
        else:
            savings_pct = 0
        
        return {
            **self._stats,
            "total_calls": total_calls,
            "savings_percentage": round(savings_pct, 1),
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "cheap_calls": 0,
            "expensive_calls": 0,
            "transfers": 0,
            "estimated_cost": 0.0,
            "estimated_savings": 0.0,
        }
