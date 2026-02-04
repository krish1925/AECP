"""
Tests for AECP matrix operations.
"""

import pytest
import numpy as np

from aecp.matrix import (
    cosine_similarity,
    compute_transfer_matrices,
    transfer_embedding,
    evaluate_transfer_quality,
    compute_embedding_stats,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    
    def test_zero_vector(self):
        """Zero vector should return 0.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0
    
    def test_different_sizes_raises(self):
        """Different sized vectors should raise ValueError."""
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)
    
    def test_empty_vectors_raises(self):
        """Empty vectors should raise ValueError."""
        vec1 = np.array([])
        vec2 = np.array([])
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)
    
    def test_2d_vectors_flattened(self):
        """2D vectors should be flattened."""
        vec1 = np.array([[1.0, 2.0, 3.0]])
        vec2 = np.array([[1.0, 2.0, 3.0]])
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)


class TestComputeTransferMatrices:
    """Tests for compute_transfer_matrices function."""
    
    def test_same_dimensions(self):
        """Test transfer between same-dimensional spaces."""
        np.random.seed(42)
        source = np.random.randn(100, 64)
        target = np.random.randn(100, 64)
        
        W_st, W_ts = compute_transfer_matrices(source, target)
        
        assert W_st.shape == (64, 64)
        assert W_ts.shape == (64, 64)
    
    def test_different_dimensions(self):
        """Test transfer between different-dimensional spaces."""
        np.random.seed(42)
        source = np.random.randn(100, 32)
        target = np.random.randn(100, 64)
        
        W_st, W_ts = compute_transfer_matrices(source, target)
        
        assert W_st.shape == (32, 64)
        assert W_ts.shape == (64, 32)
    
    def test_ridge_method(self):
        """Test ridge regression method."""
        np.random.seed(42)
        source = np.random.randn(100, 32)
        target = np.random.randn(100, 64)
        
        W_st, W_ts = compute_transfer_matrices(
            source, target, method="ridge", regularization=0.01
        )
        
        assert W_st.shape == (32, 64)
        assert W_ts.shape == (64, 32)
    
    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        source = np.random.randn(10, 4)
        target = np.random.randn(10, 4)
        
        with pytest.raises(ValueError):
            compute_transfer_matrices(source, target, method="invalid")
    
    def test_mismatched_samples_raises(self):
        """Mismatched sample counts should raise ValueError."""
        source = np.random.randn(10, 4)
        target = np.random.randn(20, 4)
        
        with pytest.raises(ValueError):
            compute_transfer_matrices(source, target)
    
    def test_empty_input_raises(self):
        """Empty input should raise ValueError."""
        source = np.random.randn(0, 4)
        target = np.random.randn(0, 4)
        
        with pytest.raises(ValueError):
            compute_transfer_matrices(source, target)
    
    def test_1d_input_raises(self):
        """1D input should raise ValueError."""
        source = np.random.randn(10)
        target = np.random.randn(10)
        
        with pytest.raises(ValueError):
            compute_transfer_matrices(source, target)


class TestTransferEmbedding:
    """Tests for transfer_embedding function."""
    
    def test_single_embedding(self):
        """Test transferring a single embedding."""
        embedding = np.array([1.0, 2.0, 3.0])
        W = np.random.randn(3, 5)
        
        result = transfer_embedding(embedding, W)
        
        assert result.shape == (5,)
    
    def test_batch_embeddings(self):
        """Test transferring a batch of embeddings."""
        embeddings = np.random.randn(10, 3)
        W = np.random.randn(3, 5)
        
        result = transfer_embedding(embeddings, W)
        
        assert result.shape == (10, 5)
    
    def test_dimension_mismatch_raises(self):
        """Dimension mismatch should raise ValueError."""
        embedding = np.array([1.0, 2.0, 3.0])
        W = np.random.randn(5, 3)  # Wrong input dimension
        
        with pytest.raises(ValueError):
            transfer_embedding(embedding, W)
    
    def test_1d_matrix_raises(self):
        """1D matrix should raise ValueError."""
        embedding = np.array([1.0, 2.0, 3.0])
        W = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            transfer_embedding(embedding, W)


class TestEvaluateTransferQuality:
    """Tests for evaluate_transfer_quality function."""
    
    def test_basic_evaluation(self):
        """Test basic quality evaluation."""
        np.random.seed(42)
        source = np.random.randn(50, 32)
        target = np.random.randn(50, 64)
        
        W_st, W_ts = compute_transfer_matrices(source, target)
        
        metrics = evaluate_transfer_quality(source, target, W_st, W_ts)
        
        assert "forward_mean_similarity" in metrics
        assert "roundtrip_mean_similarity" in metrics
        assert -1 <= metrics["forward_mean_similarity"] <= 1
        assert -1 <= metrics["roundtrip_mean_similarity"] <= 1
    
    def test_forward_only(self):
        """Test evaluation without backward matrix."""
        np.random.seed(42)
        source = np.random.randn(50, 32)
        target = np.random.randn(50, 64)
        
        W_st, _ = compute_transfer_matrices(source, target)
        
        metrics = evaluate_transfer_quality(source, target, W_st)
        
        assert "forward_mean_similarity" in metrics
        assert "roundtrip_mean_similarity" not in metrics
    
    def test_with_sample_size(self):
        """Test evaluation with sample size limit."""
        np.random.seed(42)
        source = np.random.randn(100, 32)
        target = np.random.randn(100, 64)
        
        W_st, W_ts = compute_transfer_matrices(source, target)
        
        metrics = evaluate_transfer_quality(
            source, target, W_st, W_ts, sample_size=20
        )
        
        assert "forward_mean_similarity" in metrics


class TestComputeEmbeddingStats:
    """Tests for compute_embedding_stats function."""
    
    def test_basic_stats(self):
        """Test basic embedding statistics."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        
        stats = compute_embedding_stats(embeddings)
        
        assert stats["n_samples"] == 100
        assert stats["dimensions"] == 64
        assert "mean_norm" in stats
        assert "mean_pairwise_similarity" in stats
        assert "matrix_rank" in stats
    
    def test_normalized_embeddings(self):
        """Test stats for normalized embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        stats = compute_embedding_stats(embeddings)
        
        assert stats["mean_norm"] == pytest.approx(1.0, rel=0.01)
