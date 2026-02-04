"""
Matrix Transfer Core Logic

Implements the transfer matrices computation and embedding transformation.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def compute_transfer_matrices(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    method: str = "lstsq"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer matrices between two embedding spaces.
    
    Args:
        embeddings_source: Source embeddings [n_samples, dim_source]
        embeddings_target: Target embeddings [n_samples, dim_target]
        method: Method for computing matrices ('lstsq' or 'pinv')
        
    Returns:
        Tuple of (W_source_to_target, W_target_to_source)
    """
    print(f"Computing transfer matrices using {method}...")
    print(f"  Source shape: {embeddings_source.shape}")
    print(f"  Target shape: {embeddings_target.shape}")
    
    if method == "lstsq":
        # Least squares solution: finds W that minimizes ||source @ W - target||^2
        W_source_to_target = np.linalg.lstsq(
            embeddings_source, 
            embeddings_target, 
            rcond=None
        )[0]
        
        W_target_to_source = np.linalg.lstsq(
            embeddings_target,
            embeddings_source,
            rcond=None
        )[0]
        
    elif method == "pinv":
        # Pseudo-inverse method
        W_source_to_target = np.linalg.pinv(embeddings_source) @ embeddings_target
        W_target_to_source = np.linalg.pinv(embeddings_target) @ embeddings_source
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  W_source_to_target shape: {W_source_to_target.shape}")
    print(f"  W_target_to_source shape: {W_target_to_source.shape}")
    
    return W_source_to_target, W_target_to_source


def transfer_embedding(
    embedding: np.ndarray,
    transfer_matrix: np.ndarray
) -> np.ndarray:
    """
    Transfer an embedding to a different space using a transfer matrix.
    
    Args:
        embedding: Input embedding [dim_source]
        transfer_matrix: Transfer matrix [dim_source, dim_target]
        
    Returns:
        Transformed embedding [dim_target]
    """
    return embedding @ transfer_matrix


def evaluate_transfer_quality(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    W_forward: np.ndarray,
    W_backward: Optional[np.ndarray] = None
) -> dict:
    """
    Evaluate the quality of transfer matrices.
    
    Args:
        embeddings_source: Source embeddings
        embeddings_target: Target embeddings
        W_forward: Forward transfer matrix
        W_backward: Optional backward transfer matrix for round-trip
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Forward transfer accuracy
    transferred = embeddings_source @ W_forward
    
    # Compute cosine similarities
    similarities = []
    for i in range(len(embeddings_source)):
        sim = cosine_similarity(transferred[i], embeddings_target[i])
        similarities.append(sim)
    
    results = {
        "forward_mean_similarity": float(np.mean(similarities)),
        "forward_median_similarity": float(np.median(similarities)),
        "forward_std_similarity": float(np.std(similarities)),
        "forward_min_similarity": float(np.min(similarities)),
        "forward_max_similarity": float(np.max(similarities)),
    }
    
    # Round-trip evaluation if backward matrix provided
    if W_backward is not None:
        roundtrip = transferred @ W_backward
        
        roundtrip_similarities = []
        for i in range(len(embeddings_source)):
            sim = cosine_similarity(embeddings_source[i], roundtrip[i])
            roundtrip_similarities.append(sim)
        
        results.update({
            "roundtrip_mean_similarity": float(np.mean(roundtrip_similarities)),
            "roundtrip_median_similarity": float(np.median(roundtrip_similarities)),
            "roundtrip_std_similarity": float(np.std(roundtrip_similarities)),
            "roundtrip_min_similarity": float(np.min(roundtrip_similarities)),
            "roundtrip_max_similarity": float(np.max(roundtrip_similarities)),
        })
    
    return results


class MatrixTransferSystem:
    """
    Complete system for embedding transfer using learned matrices.
    """
    
    def __init__(self, embedder1, embedder2):
        """
        Initialize with two embedding models.
        
        Args:
            embedder1: First embedding model
            embedder2: Second embedding model
        """
        self.embedder1 = embedder1
        self.embedder2 = embedder2
        self.W_12 = None  # embedder1 -> embedder2
        self.W_21 = None  # embedder2 -> embedder1
        
    def train(self, vocabulary: list):
        """
        Train the transfer matrices using vocabulary.
        
        Args:
            vocabulary: List of strings to use for learning alignment
        """
        print(f"\nTraining transfer matrices on {len(vocabulary)} vocabulary items...")
        
        # Encode vocabulary with both embedders
        print("Encoding with embedder1...")
        emb1 = self.embedder1.encode(vocabulary, show_progress_bar=True, batch_size=128)
        
        print("Encoding with embedder2...")
        emb2 = self.embedder2.encode(vocabulary, show_progress_bar=True, batch_size=128)
        
        # Compute transfer matrices
        self.W_12, self.W_21 = compute_transfer_matrices(emb1, emb2)
        
        # Evaluate training quality
        train_quality = evaluate_transfer_quality(emb1, emb2, self.W_12, self.W_21)
        print(f"\nTraining set transfer quality:")
        print(f"  Forward similarity: {train_quality['forward_mean_similarity']:.4f}")
        print(f"  Round-trip similarity: {train_quality['roundtrip_mean_similarity']:.4f}")
        
        return train_quality
    
    def transfer_1_to_2(self, text: str) -> np.ndarray:
        """Transfer text from embedder1 space to embedder2 space."""
        emb1 = self.embedder1.encode(text)
        return transfer_embedding(emb1, self.W_12)
    
    def transfer_2_to_1(self, text: str) -> np.ndarray:
        """Transfer text from embedder2 space to embedder1 space."""
        emb2 = self.embedder2.encode(text)
        return transfer_embedding(emb2, self.W_21)
    
    def roundtrip_1_2_1(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform round-trip: embedder1 -> embedder2 -> embedder1
        
        Returns:
            Tuple of (original_emb1, transferred_emb2, reconstructed_emb1)
        """
        emb1_original = self.embedder1.encode(text)
        emb2 = transfer_embedding(emb1_original, self.W_12)
        emb1_reconstructed = transfer_embedding(emb2, self.W_21)
        
        return emb1_original, emb2, emb1_reconstructed
    
    def roundtrip_2_1_2(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform round-trip: embedder2 -> embedder1 -> embedder2
        
        Returns:
            Tuple of (original_emb2, transferred_emb1, reconstructed_emb2)
        """
        emb2_original = self.embedder2.encode(text)
        emb1 = transfer_embedding(emb2_original, self.W_21)
        emb2_reconstructed = transfer_embedding(emb1, self.W_12)
        
        return emb2_original, emb1, emb2_reconstructed
