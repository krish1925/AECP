"""
AECP Protocol Implementation

Agent Embedding Communication Protocol (AECP) v1.0 implementation.
Enables AI agents with different embedding models to communicate
with high semantic fidelity.
"""

import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .types import (
    EmbeddingProvider,
    AgentCapabilities,
    TransferMatrix,
    SemanticTransfer,
    CalibrationRequest,
)
from .matrix import (
    compute_transfer_matrices,
    transfer_embedding,
    cosine_similarity,
    evaluate_transfer_quality,
)
from .vocabulary import get_default_vocabulary


@dataclass
class CalibrationResult:
    """
    Result of a calibration between two agents.
    
    Attributes:
        success: Whether calibration succeeded
        transfer_matrix: The computed transfer matrix (if successful)
        training_similarity: Quality on training data
        validation_similarity: Quality on held-out data
        worst_case_similarity: Minimum observed similarity
        error_message: Error message (if failed)
    """
    success: bool
    transfer_matrix: Optional[TransferMatrix] = None
    training_similarity: float = 0.0
    validation_similarity: float = 0.0
    worst_case_similarity: float = 0.0
    error_message: Optional[str] = None
    
    def meets_threshold(self, threshold: float = 0.75) -> bool:
        """Check if calibration meets quality threshold."""
        return self.success and self.validation_similarity >= threshold


class ProtocolHandler:
    """
    Handles AECP protocol operations for a single agent.
    
    This class manages the complete lifecycle of agent communication:
    - Handshake and capability exchange
    - Calibration with other agents
    - Semantic transfer of embeddings
    - Quality monitoring and recalibration
    
    Example:
        >>> from aecp import ProtocolHandler
        >>> from aecp.adapters import OpenAIAdapter
        >>> 
        >>> adapter = OpenAIAdapter(api_key="sk-...")
        >>> agent = ProtocolHandler("agent_1", adapter)
        >>> 
        >>> # Calibrate with another agent
        >>> result = agent.calibrate(other_agent, vocabulary)
        >>> 
        >>> # Transfer embeddings
        >>> transfer = agent.transfer_to("other_agent", "Hello world")
    """
    
    def __init__(
        self,
        agent_id: str,
        embedder: EmbeddingProvider,
        max_batch_size: int = 1000,
        min_quality_threshold: float = 0.75,
        matrix_validity_days: int = 7
    ):
        """
        Initialize protocol handler.
        
        Args:
            agent_id: Unique identifier for this agent
            embedder: Embedding provider (adapter) to use
            max_batch_size: Maximum batch size for embedding operations
            min_quality_threshold: Minimum acceptable transfer quality
            matrix_validity_days: How long transfer matrices remain valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(embedder, EmbeddingProvider):
            raise TypeError("embedder must implement EmbeddingProvider")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not 0 < min_quality_threshold <= 1:
            raise ValueError("min_quality_threshold must be between 0 and 1")
        
        self.capabilities = AgentCapabilities(
            agent_id=agent_id,
            embedding_model=embedder.get_model_id(),
            dimensions=embedder.get_dimensions(),
            max_batch_size=max_batch_size,
            min_quality_threshold=min_quality_threshold,
        )
        self.embedder = embedder
        self.matrix_validity_days = matrix_validity_days
        self.transfer_matrices: Dict[str, TransferMatrix] = {}
        self.transfer_log: List[Dict] = []
        
    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self.capabilities.agent_id
    
    def send_handshake(self) -> Dict:
        """
        Create a handshake message to send to another agent.
        
        Returns:
            Dictionary containing agent capabilities and metadata
        """
        return {
            "message_type": "handshake",
            "protocol_version": self.capabilities.protocol_version,
            "agent_id": self.capabilities.agent_id,
            "embedding_model": {
                "name": self.capabilities.embedding_model,
                "dimensions": self.capabilities.dimensions
            },
            "capabilities": {
                "max_batch_size": self.capabilities.max_batch_size,
                "min_quality_threshold": self.capabilities.min_quality_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def receive_handshake(self, handshake: Dict) -> Tuple[bool, Optional[str]]:
        """
        Receive and validate a handshake from another agent.
        
        Args:
            handshake: Handshake message from another agent
            
        Returns:
            Tuple of (success, error_message)
        """
        # Validate required fields
        required_fields = ["protocol_version", "agent_id", "embedding_model"]
        for field in required_fields:
            if field not in handshake:
                return False, f"Missing required field: {field}"
        
        # Check protocol version compatibility
        if handshake.get("protocol_version") != self.capabilities.protocol_version:
            return False, (
                f"Protocol version mismatch: "
                f"{handshake.get('protocol_version')} vs "
                f"{self.capabilities.protocol_version}"
            )
        
        # Validate embedding model info
        embedding_model = handshake.get("embedding_model", {})
        if not isinstance(embedding_model, dict):
            return False, "Invalid embedding_model format"
        if "dimensions" not in embedding_model:
            return False, "Missing embedding dimensions"
        if embedding_model["dimensions"] <= 0:
            return False, "Invalid embedding dimensions"
        
        return True, None
    
    def create_calibration_request(
        self,
        vocabulary_size: int = 10000,
        validation_ratio: float = 0.1,
        quality_threshold: float = 0.80
    ) -> CalibrationRequest:
        """
        Create a calibration request.
        
        Args:
            vocabulary_size: Total vocabulary size to use
            validation_ratio: Fraction to hold out for validation
            quality_threshold: Minimum acceptable quality
            
        Returns:
            CalibrationRequest object
        """
        validation_size = int(vocabulary_size * validation_ratio)
        return CalibrationRequest(
            vocabulary_size=vocabulary_size,
            validation_size=validation_size,
            quality_threshold=quality_threshold,
        )
    
    def calibrate(
        self,
        partner_handler: 'ProtocolHandler',
        vocabulary: Optional[List[str]] = None,
        validation_vocabulary: Optional[List[str]] = None,
        quality_threshold: float = 0.80,
        verbose: bool = True
    ) -> CalibrationResult:
        """
        Perform calibration with a partner agent.
        
        This computes transfer matrices that enable embedding transfer
        between the two agents' embedding spaces.
        
        Args:
            partner_handler: The partner agent's protocol handler
            vocabulary: Training vocabulary (uses default if None)
            validation_vocabulary: Held-out validation vocabulary
            quality_threshold: Minimum acceptable quality
            verbose: Whether to print progress
            
        Returns:
            CalibrationResult with transfer matrices and quality metrics
        """
        try:
            # Use default vocabulary if not provided
            if vocabulary is None:
                train_vocab, val_vocab = get_default_vocabulary()
                vocabulary = train_vocab
                if validation_vocabulary is None:
                    validation_vocabulary = val_vocab
            
            if validation_vocabulary is None:
                # Split vocabulary into train/val
                split_idx = int(len(vocabulary) * 0.9)
                train_vocab = vocabulary[:split_idx]
                val_vocab = vocabulary[split_idx:]
            else:
                train_vocab = vocabulary
                val_vocab = validation_vocabulary
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"CALIBRATION: {self.agent_id} <-> {partner_handler.agent_id}")
                print(f"{'='*60}")
                print(f"Training vocabulary: {len(train_vocab):,} items")
                print(f"Validation vocabulary: {len(val_vocab):,} items")
            
            # Encode training vocabulary with both agents
            if verbose:
                print(f"\nEncoding training vocabulary...")
                print(f"  {self.agent_id}...")
            
            emb_A_train = self._encode_batch(train_vocab)
            
            if verbose:
                print(f"  {partner_handler.agent_id}...")
            
            emb_B_train = partner_handler._encode_batch(train_vocab)
            
            # Compute transfer matrices
            if verbose:
                print(f"\nComputing transfer matrices...")
            
            W_AB, W_BA = compute_transfer_matrices(
                emb_A_train, emb_B_train, method="lstsq"
            )
            
            # Evaluate on training data
            train_metrics = evaluate_transfer_quality(
                emb_A_train, emb_B_train, W_AB, W_BA, sample_size=1000
            )
            training_similarity = train_metrics["roundtrip_mean_similarity"]
            
            if verbose:
                print(f"  Training forward: {train_metrics['forward_mean_similarity']:.4f}")
                print(f"  Training round-trip: {training_similarity:.4f}")
            
            # Validate on held-out data
            if verbose:
                print(f"\nValidating on held-out vocabulary...")
            
            emb_A_val = self._encode_batch(val_vocab)
            emb_B_val = partner_handler._encode_batch(val_vocab)
            
            val_metrics = evaluate_transfer_quality(
                emb_A_val, emb_B_val, W_AB, W_BA
            )
            validation_similarity = val_metrics["roundtrip_mean_similarity"]
            worst_case = val_metrics["roundtrip_min_similarity"]
            
            if verbose:
                print(f"  Validation round-trip: {validation_similarity:.4f}")
                print(f"  Worst-case: {worst_case:.4f}")
            
            # Check quality threshold
            if validation_similarity < quality_threshold:
                if verbose:
                    print(f"\n⚠️  WARNING: Quality ({validation_similarity:.4f}) "
                          f"below threshold ({quality_threshold:.4f})")
            else:
                if verbose:
                    print(f"\n✓ Quality threshold met "
                          f"({validation_similarity:.4f} >= {quality_threshold:.4f})")
            
            # Create and store transfer matrix
            valid_until = (
                datetime.now() + timedelta(days=self.matrix_validity_days)
            ).isoformat()
            
            transfer_matrix = TransferMatrix(
                matrix_AB=W_AB,
                matrix_BA=W_BA,
                training_similarity=training_similarity,
                validation_similarity=validation_similarity,
                worst_case_similarity=worst_case,
                valid_until=valid_until,
            )
            
            # Store in both handlers
            key = f"{self.agent_id}_{partner_handler.agent_id}"
            self.transfer_matrices[key] = transfer_matrix
            
            partner_key = f"{partner_handler.agent_id}_{self.agent_id}"
            partner_handler.transfer_matrices[partner_key] = TransferMatrix(
                matrix_AB=W_BA,  # Reversed for partner
                matrix_BA=W_AB,
                training_similarity=training_similarity,
                validation_similarity=validation_similarity,
                worst_case_similarity=worst_case,
                valid_until=valid_until,
            )
            
            return CalibrationResult(
                success=True,
                transfer_matrix=transfer_matrix,
                training_similarity=training_similarity,
                validation_similarity=validation_similarity,
                worst_case_similarity=worst_case,
            )
            
        except Exception as e:
            return CalibrationResult(
                success=False,
                error_message=str(e),
            )
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts with the embedder.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of embeddings [n_texts, dimensions]
        """
        # Process in batches to respect max_batch_size
        all_embeddings = []
        batch_size = self.capabilities.max_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedder.embed_batch(batch)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def transfer_to(
        self,
        partner_agent_id: str,
        text: str
    ) -> SemanticTransfer:
        """
        Transfer semantic content to a partner agent.
        
        Args:
            partner_agent_id: ID of the target agent
            text: Text to transfer
            
        Returns:
            SemanticTransfer object with the transferred embedding
            
        Raises:
            ValueError: If not calibrated with the partner
        """
        # Get transfer matrix
        key = f"{self.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            raise ValueError(
                f"No calibration found for {key}. Run calibrate() first."
            )
        
        matrix = self.transfer_matrices[key]
        
        # Check if matrix is expired
        if matrix.is_expired():
            raise ValueError(
                f"Transfer matrix for {key} has expired. Recalibrate."
            )
        
        # Encode text
        embedding = np.array(self.embedder.embed(text))
        
        # Transform to partner's space
        transferred = transfer_embedding(embedding, matrix.matrix_AB)
        
        # Create transfer object
        transfer_id = self._generate_transfer_id(text)
        
        semantic_transfer = SemanticTransfer(
            transfer_id=transfer_id,
            embedding=transferred,
            source_agent=self.agent_id,
            target_agent=partner_agent_id,
            original_norm=float(np.linalg.norm(embedding)),
            expected_similarity=matrix.validation_similarity,
            timestamp=datetime.now().isoformat(),
        )
        
        # Log transfer
        self.transfer_log.append({
            "transfer_id": transfer_id,
            "source": self.agent_id,
            "target": partner_agent_id,
            "timestamp": semantic_transfer.timestamp,
            "expected_quality": matrix.validation_similarity,
        })
        
        return semantic_transfer
    
    def transfer_embedding_to(
        self,
        partner_agent_id: str,
        embedding: np.ndarray
    ) -> SemanticTransfer:
        """
        Transfer a pre-computed embedding to a partner agent.
        
        Args:
            partner_agent_id: ID of the target agent
            embedding: Pre-computed embedding vector
            
        Returns:
            SemanticTransfer object with the transferred embedding
        """
        key = f"{self.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            raise ValueError(
                f"No calibration found for {key}. Run calibrate() first."
            )
        
        matrix = self.transfer_matrices[key]
        
        if matrix.is_expired():
            raise ValueError(
                f"Transfer matrix for {key} has expired. Recalibrate."
            )
        
        embedding = np.asarray(embedding)
        transferred = transfer_embedding(embedding, matrix.matrix_AB)
        
        transfer_id = self._generate_transfer_id(str(embedding[:5]))
        
        return SemanticTransfer(
            transfer_id=transfer_id,
            embedding=transferred,
            source_agent=self.agent_id,
            target_agent=partner_agent_id,
            original_norm=float(np.linalg.norm(embedding)),
            expected_similarity=matrix.validation_similarity,
            timestamp=datetime.now().isoformat(),
        )
    
    def receive_transfer(
        self,
        transfer: SemanticTransfer,
        original_text: Optional[str] = None
    ) -> Dict:
        """
        Receive and validate a transferred embedding.
        
        Args:
            transfer: The received semantic transfer
            original_text: Optional original text for quality validation
            
        Returns:
            Acknowledgment dictionary with quality metrics
        """
        # Validate transfer
        if transfer.target_agent != self.agent_id:
            return {
                "status": "error",
                "message": f"Transfer intended for {transfer.target_agent}, "
                          f"not {self.agent_id}",
            }
        
        # Get reverse transfer matrix
        key = f"{self.agent_id}_{transfer.source_agent}"
        if key not in self.transfer_matrices:
            return {
                "status": "error",
                "message": "No calibration found for source agent",
            }
        
        matrix = self.transfer_matrices[key]
        
        # Compute received norm
        received_norm = float(np.linalg.norm(transfer.embedding))
        
        # Validate quality if original text provided
        quality_metric = None
        if original_text:
            our_embedding = np.array(self.embedder.embed(original_text))
            quality_metric = cosine_similarity(transfer.embedding, our_embedding)
        
        return {
            "message_type": "acknowledgment",
            "transfer_id": transfer.transfer_id,
            "status": "success",
            "received_norm": received_norm,
            "original_norm": transfer.original_norm,
            "norm_ratio": transfer.get_norm_ratio(),
            "quality_metric": float(quality_metric) if quality_metric else None,
            "timestamp": datetime.now().isoformat(),
        }
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using this agent's embedder.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return np.array(self.embedder.embed(text))
    
    def get_calibration_stats(self, partner_agent_id: str) -> Optional[Dict]:
        """
        Get calibration statistics for a partner agent.
        
        Args:
            partner_agent_id: ID of the partner agent
            
        Returns:
            Dictionary with calibration stats, or None if not calibrated
        """
        key = f"{self.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            return None
        
        return self.transfer_matrices[key].to_dict()
    
    def requires_recalibration(self, partner_agent_id: str) -> bool:
        """
        Check if recalibration is needed with a partner.
        
        Args:
            partner_agent_id: ID of the partner agent
            
        Returns:
            True if recalibration is needed
        """
        key = f"{self.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            return True
        
        matrix = self.transfer_matrices[key]
        
        # Check expiration
        if matrix.is_expired():
            return True
        
        # Check quality
        if not matrix.meets_quality_threshold(self.capabilities.min_quality_threshold):
            return True
        
        return False
    
    def _generate_transfer_id(self, content: str) -> str:
        """Generate a unique transfer ID."""
        data = f"{content}{datetime.now().isoformat()}{self.agent_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# Alias for simpler API
class AECP(ProtocolHandler):
    """
    AECP - Agent Embedding Communication Protocol
    
    Main interface for the AECP library. This is an alias for ProtocolHandler
    with a simpler API.
    
    Example:
        >>> from aecp import AECP
        >>> from aecp.adapters import OpenAIAdapter
        >>> 
        >>> agent = AECP(OpenAIAdapter(api_key="sk-..."), agent_id="my_agent")
        >>> 
        >>> # Calibrate with another agent
        >>> agent.calibrate_with(other_agent)
        >>> 
        >>> # Transfer text
        >>> transferred = agent.transfer_to(other_agent.agent_id, "Hello world")
    """
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        agent_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AECP agent.
        
        Args:
            embedder: Embedding provider (adapter)
            agent_id: Unique identifier (auto-generated if not provided)
            **kwargs: Additional arguments passed to ProtocolHandler
        """
        if agent_id is None:
            agent_id = f"agent_{hashlib.md5(str(id(embedder)).encode()).hexdigest()[:8]}"
        
        super().__init__(agent_id, embedder, **kwargs)
    
    def calibrate_with(
        self,
        other: 'AECP',
        vocabulary: Optional[List[str]] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Calibrate with another AECP agent.
        
        Args:
            other: The other AECP agent
            vocabulary: Optional custom vocabulary
            **kwargs: Additional arguments passed to calibrate()
            
        Returns:
            CalibrationResult
        """
        return self.calibrate(other, vocabulary=vocabulary, **kwargs)
