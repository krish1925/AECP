"""
Generate Test Fixtures for TypeScript Tests
Creates known-good outputs from Python POC for validation
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from matrix_transfer import compute_transfer_matrices, cosine_similarity

def generate_fixtures():
    """Generate test fixtures from Python implementation."""
    print("Generating test fixtures from Python POC...")
    
    # Load models
    print("Loading models...")
    model_a = SentenceTransformer('all-MiniLM-L6-v2')  # 384D
    model_b = SentenceTransformer('all-mpnet-base-v2')  # 768D
    
    # Test vocabulary
    test_vocab = [
        "machine learning",
        "artificial intelligence",
        "neural network",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "embedding space",
        "semantic similarity",
        "vector representation"
    ]
    
    print(f"Encoding {len(test_vocab)} items with both models...")
    emb_a = model_a.encode(test_vocab, show_progress_bar=False)
    emb_b = model_b.encode(test_vocab, show_progress_bar=False)
    
    print("Computing transfer matrices...")
    W_AB, W_BA = compute_transfer_matrices(emb_a, emb_b)
    
    # Compute quality metrics
    transferred = emb_a @ W_AB
    roundtrip = transferred @ W_BA
    
    similarities = []
    for i in range(len(emb_a)):
        sim = cosine_similarity(emb_a[i], roundtrip[i])
        similarities.append(float(sim))
    
    quality = {
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities))
    }
    
    # Test single transfer
    test_text = "The model achieves high accuracy."
    test_emb_a = model_a.encode(test_text)
    test_emb_b = model_b.encode(test_text)
    test_transferred = test_emb_a @ W_AB
    test_similarity = float(cosine_similarity(test_transferred, test_emb_b))
    
    fixtures = {
        "metadata": {
            "model_a": {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384
            },
            "model_b": {
                "name": "all-mpnet-base-v2",
                "dimensions": 768
            },
            "vocabulary_size": len(test_vocab),
            "generated_at": "2026-02-04"
        },
        "vocabulary": test_vocab,
        "embeddings_a": emb_a.tolist(),
        "embeddings_b": emb_b.tolist(),
        "transfer_matrix_shape": {
            "W_AB": [W_AB.shape[0], W_AB.shape[1]],
            "W_BA": [W_BA.shape[0], W_BA.shape[1]]
        },
        "calibration_quality": quality,
        "test_transfer": {
            "text": test_text,
            "embedding_a": test_emb_a.tolist(),
            "embedding_b": test_emb_b.tolist(),
            "transferred": test_transferred.tolist(),
            "similarity": test_similarity
        },
        "validation": {
            "note": "TypeScript tests should achieve similar quality metrics",
            "expected_quality_range": [0.70, 0.95],
            "tolerance": 0.05
        }
    }
    
    output_file = "aecp-npm/packages/core/src/__tests__/fixtures.json"
    print(f"\nSaving fixtures to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(fixtures, f, indent=2)
    
    print("\n✅ Test fixtures generated successfully!")
    print(f"\nQuality Metrics:")
    print(f"  Mean similarity: {quality['mean']:.4f}")
    print(f"  Min similarity:  {quality['min']:.4f}")
    print(f"  Max similarity:  {quality['max']:.4f}")
    print(f"\nTest Transfer:")
    print(f"  Similarity: {test_similarity:.4f}")
    
    return fixtures

if __name__ == "__main__":
    generate_fixtures()
