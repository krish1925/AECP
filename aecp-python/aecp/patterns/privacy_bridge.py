"""
Privacy Bridge Pattern

Keep sensitive data local while sharing semantic representations
with cloud services. Data never leaves your infrastructure.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from ..types import EmbeddingProvider
from ..protocol import AECP, CalibrationResult


class PrivacyBridge:
    """
    Bridge between local and cloud embedding models with privacy preservation.
    
    Sensitive data is embedded locally and only the semantic representation
    (transferred embedding) is sent to cloud services. The original text
    never leaves your infrastructure.
    
    Use cases:
    - Healthcare: Patient data stays local, semantics go to cloud AI
    - Finance: Transaction details local, patterns analyzed in cloud
    - Legal: Document content local, similarity search in cloud
    
    Example:
        >>> from aecp.patterns import PrivacyBridge
        >>> from aecp.adapters import HuggingFaceAdapter, OpenAIAdapter
        >>> 
        >>> bridge = PrivacyBridge(
        ...     local_adapter=HuggingFaceAdapter(),  # Runs on your server
        ...     cloud_adapter=OpenAIAdapter(api_key="sk-..."),
        ... )
        >>> 
        >>> # Calibrate (uses non-sensitive calibration data)
        >>> bridge.calibrate()
        >>> 
        >>> # Embed sensitive data locally
        >>> local_embedding = bridge.embed_local("Patient SSN: 123-45-6789")
        >>> 
        >>> # Transfer to cloud space (no text sent to cloud)
        >>> cloud_embedding = bridge.transfer_to_cloud(local_embedding)
        >>> 
        >>> # Use cloud embedding for similarity search, etc.
    """
    
    def __init__(
        self,
        local_adapter: EmbeddingProvider,
        cloud_adapter: EmbeddingProvider,
    ):
        """
        Initialize privacy bridge.
        
        Args:
            local_adapter: Adapter running locally (no external API calls)
            cloud_adapter: Adapter using cloud API
        """
        if local_adapter is None or cloud_adapter is None:
            raise ValueError("Both adapters must be provided")
        
        self.local_agent = AECP(local_adapter, agent_id="local_agent")
        self.cloud_agent = AECP(cloud_adapter, agent_id="cloud_agent")
        
        self._calibrated = False
        self._stats = {
            "local_embeddings": 0,
            "cloud_embeddings": 0,
            "transfers_to_cloud": 0,
            "transfers_to_local": 0,
        }
    
    def calibrate(
        self,
        vocabulary: Optional[List[str]] = None,
        verbose: bool = True
    ) -> CalibrationResult:
        """
        Calibrate transfer matrices.
        
        IMPORTANT: Use non-sensitive calibration vocabulary.
        The calibration process sends vocabulary to both models.
        
        Args:
            vocabulary: Non-sensitive calibration vocabulary
            verbose: Whether to print progress
            
        Returns:
            CalibrationResult with quality metrics
        """
        result = self.local_agent.calibrate_with(
            self.cloud_agent,
            vocabulary=vocabulary,
            verbose=verbose,
        )
        
        if result.success:
            self._calibrated = True
        
        return result
    
    def embed_local(self, text: str) -> np.ndarray:
        """
        Embed text using the local model.
        
        Text is processed entirely locally - no external API calls.
        
        Args:
            text: Sensitive text to embed
            
        Returns:
            Embedding in local model's space
        """
        embedding = self.local_agent.embed(text)
        self._stats["local_embeddings"] += 1
        return embedding
    
    def embed_local_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts locally.
        
        Args:
            texts: Sensitive texts to embed
            
        Returns:
            List of embeddings in local space
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_local(text))
        return embeddings
    
    def transfer_to_cloud(self, local_embedding: np.ndarray) -> np.ndarray:
        """
        Transfer a local embedding to cloud space.
        
        Only the embedding is transferred - original text is never sent.
        
        Args:
            local_embedding: Embedding from local model
            
        Returns:
            Embedding in cloud model's space
            
        Raises:
            ValueError: If not calibrated
        """
        if not self._calibrated:
            raise ValueError(
                "Calibration required. Call calibrate() first."
            )
        
        transfer = self.local_agent.transfer_embedding_to(
            self.cloud_agent.agent_id,
            local_embedding,
        )
        self._stats["transfers_to_cloud"] += 1
        return transfer.embedding
    
    def transfer_to_local(self, cloud_embedding: np.ndarray) -> np.ndarray:
        """
        Transfer a cloud embedding to local space.
        
        Args:
            cloud_embedding: Embedding from cloud model
            
        Returns:
            Embedding in local model's space
            
        Raises:
            ValueError: If not calibrated
        """
        if not self._calibrated:
            raise ValueError(
                "Calibration required. Call calibrate() first."
            )
        
        transfer = self.cloud_agent.transfer_embedding_to(
            self.local_agent.agent_id,
            cloud_embedding,
        )
        self._stats["transfers_to_local"] += 1
        return transfer.embedding
    
    def embed_and_transfer(self, text: str) -> np.ndarray:
        """
        Embed text locally and transfer to cloud space in one step.
        
        Convenience method that combines embed_local + transfer_to_cloud.
        
        Args:
            text: Sensitive text to embed
            
        Returns:
            Embedding in cloud model's space
        """
        local_embedding = self.embed_local(text)
        return self.transfer_to_cloud(local_embedding)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with embedding and transfer counts
        """
        return {
            **self._stats,
            "calibrated": self._calibrated,
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "local_embeddings": 0,
            "cloud_embeddings": 0,
            "transfers_to_cloud": 0,
            "transfers_to_local": 0,
        }
