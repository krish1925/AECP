"""
AECP Matrix Operations

Core matrix operations for embedding transfer between agents.
Implements least-squares linear transformation for embedding space alignment.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector (any shape, will be flattened)
        vec2: Second vector (any shape, will be flattened)
        
    Returns:
        Cosine similarity score in range [-1, 1]
        
    Raises:
        ValueError: If vectors have different sizes after flattening
        
    Example:
        >>> vec1 = np.array([1, 0, 0])
        >>> vec2 = np.array([1, 0, 0])
        >>> cosine_similarity(vec1, vec2)
        1.0
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if vec1.size != vec2.size:
        raise ValueError(
            f"Vectors must have same size: {vec1.size} vs {vec2.size}"
        )
    
    if vec1.size == 0:
        raise ValueError("Vectors cannot be empty")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero vectors gracefully
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def compute_transfer_matrices(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    method: str = "lstsq",
    regularization: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer matrices between two embedding spaces.
    
    Uses least squares regression to find linear transformations that
    map embeddings from one space to another.
    
    Args:
        embeddings_source: Source embeddings [n_samples, dim_source]
        embeddings_target: Target embeddings [n_samples, dim_target]
        method: Method for computing matrices ('lstsq' or 'ridge')
        regularization: Regularization strength for ridge regression
        
    Returns:
        Tuple of (W_source_to_target, W_target_to_source)
        
    Raises:
        ValueError: If input shapes are incompatible
        RuntimeError: If matrix computation fails
        
    Example:
        >>> source = np.random.randn(1000, 384)
        >>> target = np.random.randn(1000, 768)
        >>> W_st, W_ts = compute_transfer_matrices(source, target)
        >>> W_st.shape
        (384, 768)
    """
    # Input validation
    embeddings_source = np.asarray(embeddings_source)
    embeddings_target = np.asarray(embeddings_target)
    
    if embeddings_source.ndim != 2 or embeddings_target.ndim != 2:
        raise ValueError("Embeddings must be 2-dimensional arrays")
    
    if embeddings_source.shape[0] != embeddings_target.shape[0]:
        raise ValueError(
            f"Number of samples must match: "
            f"{embeddings_source.shape[0]} vs {embeddings_target.shape[0]}"
        )
    
    n_samples = embeddings_source.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot compute transfer matrices with zero samples")
    
    dim_source = embeddings_source.shape[1]
    dim_target = embeddings_target.shape[1]
    
    # Check for sufficient samples
    min_samples = max(dim_source, dim_target)
    if n_samples < min_samples:
        warnings.warn(
            f"Number of samples ({n_samples}) is less than max dimension "
            f"({min_samples}). Results may be unstable."
        )
    
    # Check for degenerate inputs
    source_rank = np.linalg.matrix_rank(embeddings_source)
    if source_rank < min(n_samples, dim_source) * 0.9:
        warnings.warn(
            f"Source embeddings appear to be rank-deficient "
            f"(rank {source_rank} vs expected {min(n_samples, dim_source)}). "
            f"Consider using more diverse vocabulary."
        )
    
    if method == "lstsq":
        # Least squares solution: finds W that minimizes ||source @ W - target||^2
        try:
            W_source_to_target, residuals_st, rank_st, s_st = np.linalg.lstsq(
                embeddings_source, 
                embeddings_target, 
                rcond=None
            )
            
            W_target_to_source, residuals_ts, rank_ts, s_ts = np.linalg.lstsq(
                embeddings_target,
                embeddings_source,
                rcond=None
            )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Matrix computation failed: {e}")
            
    elif method == "ridge":
        # Ridge regression with regularization for stability
        try:
            # W = (X^T X + λI)^(-1) X^T Y
            XtX_source = embeddings_source.T @ embeddings_source
            XtX_target = embeddings_target.T @ embeddings_target
            
            # Add regularization
            reg_source = regularization * np.eye(dim_source)
            reg_target = regularization * np.eye(dim_target)
            
            W_source_to_target = np.linalg.solve(
                XtX_source + reg_source,
                embeddings_source.T @ embeddings_target
            )
            
            W_target_to_source = np.linalg.solve(
                XtX_target + reg_target,
                embeddings_target.T @ embeddings_source
            )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Ridge regression failed: {e}")
            
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lstsq' or 'ridge'")
    
    # Validate output shapes
    assert W_source_to_target.shape == (dim_source, dim_target), \
        f"Unexpected shape: {W_source_to_target.shape}"
    assert W_target_to_source.shape == (dim_target, dim_source), \
        f"Unexpected shape: {W_target_to_source.shape}"
    
    return W_source_to_target, W_target_to_source


def transfer_embedding(
    embedding: np.ndarray,
    transfer_matrix: np.ndarray
) -> np.ndarray:
    """
    Transfer an embedding to a different space using a transfer matrix.
    
    Args:
        embedding: Input embedding [dim_source] or [batch, dim_source]
        transfer_matrix: Transfer matrix [dim_source, dim_target]
        
    Returns:
        Transformed embedding [dim_target] or [batch, dim_target]
        
    Raises:
        ValueError: If shapes are incompatible
        
    Example:
        >>> embedding = np.random.randn(384)
        >>> W = np.random.randn(384, 768)
        >>> transferred = transfer_embedding(embedding, W)
        >>> transferred.shape
        (768,)
    """
    embedding = np.asarray(embedding)
    transfer_matrix = np.asarray(transfer_matrix)
    
    if transfer_matrix.ndim != 2:
        raise ValueError("Transfer matrix must be 2-dimensional")
    
    # Handle both single vectors and batches
    is_single = embedding.ndim == 1
    if is_single:
        embedding = embedding.reshape(1, -1)
    
    if embedding.shape[-1] != transfer_matrix.shape[0]:
        raise ValueError(
            f"Embedding dimension ({embedding.shape[-1]}) doesn't match "
            f"transfer matrix input dimension ({transfer_matrix.shape[0]})"
        )
    
    result = embedding @ transfer_matrix
    
    if is_single:
        result = result.flatten()
    
    return result


def evaluate_transfer_quality(
    embeddings_source: np.ndarray,
    embeddings_target: np.ndarray,
    W_forward: np.ndarray,
    W_backward: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate the quality of transfer matrices.
    
    Args:
        embeddings_source: Source embeddings [n_samples, dim_source]
        embeddings_target: Target embeddings [n_samples, dim_target]
        W_forward: Forward transfer matrix [dim_source, dim_target]
        W_backward: Optional backward transfer matrix for round-trip
        sample_size: Optional limit on number of samples to evaluate
        
    Returns:
        Dictionary of evaluation metrics including:
        - forward_mean_similarity: Average cosine similarity after forward transfer
        - forward_median_similarity: Median cosine similarity
        - forward_std_similarity: Standard deviation
        - forward_min_similarity: Worst-case similarity
        - forward_max_similarity: Best-case similarity
        - roundtrip_* (if W_backward provided): Same metrics for round-trip
        
    Example:
        >>> source = np.random.randn(100, 384)
        >>> target = np.random.randn(100, 768)
        >>> W_st, W_ts = compute_transfer_matrices(source, target)
        >>> metrics = evaluate_transfer_quality(source, target, W_st, W_ts)
        >>> print(f"Mean similarity: {metrics['forward_mean_similarity']:.4f}")
    """
    embeddings_source = np.asarray(embeddings_source)
    embeddings_target = np.asarray(embeddings_target)
    
    n_samples = len(embeddings_source)
    
    # Optionally subsample for efficiency
    if sample_size is not None and sample_size < n_samples:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        embeddings_source = embeddings_source[indices]
        embeddings_target = embeddings_target[indices]
        n_samples = sample_size
    
    # Forward transfer evaluation
    transferred = embeddings_source @ W_forward
    
    forward_similarities = []
    for i in range(n_samples):
        sim = cosine_similarity(transferred[i], embeddings_target[i])
        forward_similarities.append(sim)
    
    forward_similarities = np.array(forward_similarities)
    
    results = {
        "forward_mean_similarity": float(np.mean(forward_similarities)),
        "forward_median_similarity": float(np.median(forward_similarities)),
        "forward_std_similarity": float(np.std(forward_similarities)),
        "forward_min_similarity": float(np.min(forward_similarities)),
        "forward_max_similarity": float(np.max(forward_similarities)),
        "forward_percentile_5": float(np.percentile(forward_similarities, 5)),
        "forward_percentile_95": float(np.percentile(forward_similarities, 95)),
    }
    
    # Round-trip evaluation if backward matrix provided
    if W_backward is not None:
        roundtrip = transferred @ W_backward
        
        roundtrip_similarities = []
        for i in range(n_samples):
            sim = cosine_similarity(embeddings_source[i], roundtrip[i])
            roundtrip_similarities.append(sim)
        
        roundtrip_similarities = np.array(roundtrip_similarities)
        
        results.update({
            "roundtrip_mean_similarity": float(np.mean(roundtrip_similarities)),
            "roundtrip_median_similarity": float(np.median(roundtrip_similarities)),
            "roundtrip_std_similarity": float(np.std(roundtrip_similarities)),
            "roundtrip_min_similarity": float(np.min(roundtrip_similarities)),
            "roundtrip_max_similarity": float(np.max(roundtrip_similarities)),
            "roundtrip_percentile_5": float(np.percentile(roundtrip_similarities, 5)),
            "roundtrip_percentile_95": float(np.percentile(roundtrip_similarities, 95)),
        })
    
    return results


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics about a set of embeddings.
    
    Useful for diagnosing calibration issues.
    
    Args:
        embeddings: Embedding matrix [n_samples, dim]
        
    Returns:
        Dictionary with statistics about the embeddings
    """
    embeddings = np.asarray(embeddings)
    
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Compute pairwise similarities for a sample
    n_samples = min(100, len(embeddings))
    sample = embeddings[:n_samples]
    
    pairwise_sims = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairwise_sims.append(cosine_similarity(sample[i], sample[j]))
    
    pairwise_sims = np.array(pairwise_sims) if pairwise_sims else np.array([0.0])
    
    return {
        "n_samples": len(embeddings),
        "dimensions": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_pairwise_similarity": float(np.mean(pairwise_sims)),
        "std_pairwise_similarity": float(np.std(pairwise_sims)),
        "matrix_rank": int(np.linalg.matrix_rank(embeddings)),
    }
