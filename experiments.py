"""
Experiment Implementations

Implements Experiment 1 (text baseline) and Experiment 2 (matrix transfer).
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from matrix_transfer import cosine_similarity, MatrixTransferSystem


def experiment_1_text_baseline(
    embedder1,
    embedder2,
    test_corpus: List[str],
    W_21: np.ndarray
) -> Dict:
    """
    Experiment 1: Text as transfer medium (baseline)
    
    Measures how well two embedders AGREE on the same text.
    This is the baseline: if agents communicate via text strings,
    how much semantic information is preserved?
    
    Pipeline:
    - text → embedder1 → emb1
    - text → embedder2 → emb2
    - Project emb2 to emb1's space using W_21
    - Measure agreement: cosine_similarity(emb1, emb2_projected)
    
    Args:
        embedder1: First embedding model
        embedder2: Second embedding model
        test_corpus: List of test sentences
        W_21: Transfer matrix from embedder2 to embedder1 (for projection)
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Text Baseline (Cross-Embedder Agreement)")
    print("="*60)
    print(f"Testing on {len(test_corpus)} sentences...")
    
    similarities = []
    detailed_results = []
    
    for text in tqdm(test_corpus, desc="Text baseline"):
        # Both embedders encode the same text
        emb1 = embedder1.encode(text)
        emb2 = embedder2.encode(text)
        
        # Project emb2 into emb1's space for fair comparison
        emb2_projected = emb2 @ W_21
        
        # Measure agreement
        sim = cosine_similarity(emb1, emb2_projected)
        similarities.append(sim)
        
        detailed_results.append({
            "text": text[:100],  # Truncate for storage
            "similarity": float(sim),
            "emb1_norm": float(np.linalg.norm(emb1)),
            "emb2_norm": float(np.linalg.norm(emb2)),
        })
    
    # Compute statistics
    results = {
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "q25_similarity": float(np.percentile(similarities, 25)),
        "q75_similarity": float(np.percentile(similarities, 75)),
        "detailed_results": detailed_results,
        "all_similarities": similarities
    }
    
    print("\nResults:")
    print(f"  Mean similarity:   {results['mean_similarity']:.4f}")
    print(f"  Median similarity: {results['median_similarity']:.4f}")
    print(f"  Std deviation:     {results['std_similarity']:.4f}")
    print(f"  Min similarity:    {results['min_similarity']:.4f}")
    print(f"  Max similarity:    {results['max_similarity']:.4f}")
    print(f"  25th percentile:   {results['q25_similarity']:.4f}")
    print(f"  75th percentile:   {results['q75_similarity']:.4f}")
    
    return results


def experiment_2_matrix_transfer(
    embedder1,
    embedder2,
    test_corpus: List[str],
    W_12: np.ndarray,
    W_21: np.ndarray
) -> Dict:
    """
    Experiment 2: Matrix transfer (proposed method)
    
    Tests round-trip fidelity: can we preserve information through matrix transfer?
    
    Pipeline:
    - text → embedder1 → emb1_original
    - emb1_original @ W_12 → emb2
    - emb2 @ W_21 → emb1_reconstructed
    - Measure loss: cosine_similarity(emb1_original, emb1_reconstructed)
    
    Args:
        embedder1: First embedding model
        embedder2: Second embedding model  
        test_corpus: List of test sentences
        W_12: Transfer matrix from embedder1 to embedder2
        W_21: Transfer matrix from embedder2 to embedder1
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Matrix Transfer (Round-trip Fidelity)")
    print("="*60)
    print(f"Testing on {len(test_corpus)} sentences...")
    
    similarities = []
    detailed_results = []
    
    for text in tqdm(test_corpus, desc="Matrix transfer"):
        # Original embedding
        emb1_original = embedder1.encode(text)
        
        # Transfer to embedder2 space
        emb2 = emb1_original @ W_12
        
        # Transfer back to embedder1 space
        emb1_reconstructed = emb2 @ W_21
        
        # Measure round-trip fidelity
        sim = cosine_similarity(emb1_original, emb1_reconstructed)
        similarities.append(sim)
        
        # Also measure intermediate similarity
        emb2_reference = embedder2.encode(text)
        intermediate_sim = cosine_similarity(emb2, emb2_reference)
        
        detailed_results.append({
            "text": text[:100],
            "roundtrip_similarity": float(sim),
            "intermediate_similarity": float(intermediate_sim),
            "original_norm": float(np.linalg.norm(emb1_original)),
            "reconstructed_norm": float(np.linalg.norm(emb1_reconstructed)),
            "reconstruction_error": float(np.linalg.norm(emb1_original - emb1_reconstructed))
        })
    
    # Compute statistics
    results = {
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "q25_similarity": float(np.percentile(similarities, 25)),
        "q75_similarity": float(np.percentile(similarities, 75)),
        "detailed_results": detailed_results,
        "all_similarities": similarities
    }
    
    print("\nResults:")
    print(f"  Mean similarity:   {results['mean_similarity']:.4f}")
    print(f"  Median similarity: {results['median_similarity']:.4f}")
    print(f"  Std deviation:     {results['std_similarity']:.4f}")
    print(f"  Min similarity:    {results['min_similarity']:.4f}")
    print(f"  Max similarity:    {results['max_similarity']:.4f}")
    print(f"  25th percentile:   {results['q25_similarity']:.4f}")
    print(f"  75th percentile:   {results['q75_similarity']:.4f}")
    
    return results


def compare_experiments(
    results_exp1: Dict,
    results_exp2: Dict
) -> Dict:
    """
    Compare results from both experiments.
    
    Args:
        results_exp1: Results from Experiment 1 (text baseline)
        results_exp2: Results from Experiment 2 (matrix transfer)
        
    Returns:
        Comparison dictionary
    """
    print("\n" + "="*60)
    print("COMPARISON: Matrix Transfer vs Text Baseline")
    print("="*60)
    
    improvement = results_exp2["mean_similarity"] - results_exp1["mean_similarity"]
    improvement_pct = (improvement / results_exp1["mean_similarity"]) * 100
    
    comparison = {
        "text_baseline_mean": results_exp1["mean_similarity"],
        "matrix_transfer_mean": results_exp2["mean_similarity"],
        "absolute_improvement": improvement,
        "relative_improvement_pct": improvement_pct,
        "text_baseline_median": results_exp1["median_similarity"],
        "matrix_transfer_median": results_exp2["median_similarity"],
        "text_baseline_std": results_exp1["std_similarity"],
        "matrix_transfer_std": results_exp2["std_similarity"],
    }
    
    print(f"\nText Baseline:       {comparison['text_baseline_mean']:.4f}")
    print(f"Matrix Transfer:     {comparison['matrix_transfer_mean']:.4f}")
    print(f"Absolute Improvement: {comparison['absolute_improvement']:.4f}")
    print(f"Relative Improvement: {comparison['relative_improvement_pct']:.2f}%")
    
    # Determine verdict
    if improvement > 0.05:
        verdict = "STRONG WIN for Matrix Transfer"
        explanation = "Matrix transfer preserves significantly more information than text."
    elif improvement > 0.02:
        verdict = "MODERATE WIN for Matrix Transfer"
        explanation = "Matrix transfer shows meaningful improvement over text."
    elif improvement > 0:
        verdict = "SLIGHT WIN for Matrix Transfer"
        explanation = "Matrix transfer is marginally better than text."
    elif improvement > -0.02:
        verdict = "ROUGHLY EQUIVALENT"
        explanation = "Matrix transfer and text are comparable."
    else:
        verdict = "TEXT IS BETTER"
        explanation = "Text serialization preserves more information than matrix transfer."
    
    comparison["verdict"] = verdict
    comparison["explanation"] = explanation
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")
    print(f"{explanation}")
    print()
    
    return comparison


def run_both_experiments(
    embedder1,
    embedder2,
    vocabulary: List[str],
    test_corpus: List[str]
) -> Tuple[Dict, Dict, Dict]:
    """
    Run both experiments and compare results.
    
    Args:
        embedder1: First embedding model
        embedder2: Second embedding model
        vocabulary: Vocabulary for training transfer matrices
        test_corpus: Test sentences for evaluation
        
    Returns:
        Tuple of (exp1_results, exp2_results, comparison)
    """
    # Set up transfer system
    transfer_system = MatrixTransferSystem(embedder1, embedder2)
    
    # Train on vocabulary
    train_quality = transfer_system.train(vocabulary)
    
    # Run Experiment 1: Text baseline
    exp1_results = experiment_1_text_baseline(
        embedder1,
        embedder2,
        test_corpus,
        transfer_system.W_21
    )
    
    # Run Experiment 2: Matrix transfer
    exp2_results = experiment_2_matrix_transfer(
        embedder1,
        embedder2,
        test_corpus,
        transfer_system.W_12,
        transfer_system.W_21
    )
    
    # Compare results
    comparison = compare_experiments(exp1_results, exp2_results)
    
    # Add training quality to comparison
    comparison["training_forward_similarity"] = train_quality["forward_mean_similarity"]
    comparison["training_roundtrip_similarity"] = train_quality["roundtrip_mean_similarity"]
    
    return exp1_results, exp2_results, comparison
