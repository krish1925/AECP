"""
Main POC Runner

Execute the complete proof-of-concept including:
1. Load embedding models
2. Generate vocabulary and test corpus
3. Run both experiments
4. Generate comprehensive report
"""

import os
# Disable TensorFlow to avoid dependency conflicts
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
import numpy as np

from vocab_loader import load_common_words, load_test_corpus
from experiments import run_both_experiments
from report_generator import generate_full_report


def main():
    """
    Main execution function.
    """
    print("="*70)
    print(" EMBEDDING TRANSFER POC - Agent Communication via Matrix Transfer")
    print("="*70)
    print()
    print("This POC tests whether learned transfer matrices preserve more")
    print("information than text serialization for cross-embedder communication.")
    print()
    
    # Configuration
    VOCAB_SIZE = 30000
    TEST_SIZE = 1000
    
    print("\n" + "="*70)
    print("STEP 1: Loading Embedding Models")
    print("="*70)
    
    # Load two different sentence embedding models
    print("\nLoading embedder 1: all-MiniLM-L6-v2 (384 dimensions)...")
    embedder1 = SentenceTransformer('all-MiniLM-L6-v2')
    model1_name = "all-MiniLM-L6-v2 (384d)"
    print("✓ Loaded")
    
    print("\nLoading embedder 2: all-mpnet-base-v2 (768 dimensions)...")
    embedder2 = SentenceTransformer('all-mpnet-base-v2')
    model2_name = "all-mpnet-base-v2 (768d)"
    print("✓ Loaded")
    
    print("\n" + "="*70)
    print("STEP 2: Generating Vocabulary and Test Corpus")
    print("="*70)
    
    print(f"\nGenerating vocabulary ({VOCAB_SIZE:,} items)...")
    vocabulary = load_common_words(VOCAB_SIZE)
    print(f"✓ Generated {len(vocabulary):,} vocabulary items")
    print(f"  Sample: {vocabulary[:5]}")
    
    print(f"\nGenerating test corpus ({TEST_SIZE:,} sentences)...")
    test_corpus = load_test_corpus(TEST_SIZE)
    print(f"✓ Generated {len(test_corpus):,} test sentences")
    print(f"  Sample: {test_corpus[0][:100]}...")
    
    # Verify no overlap
    overlap = set(vocabulary) & set(test_corpus)
    print(f"\n  Vocabulary-Test overlap: {len(overlap)} items ({len(overlap)/len(test_corpus)*100:.2f}%)")
    
    print("\n" + "="*70)
    print("STEP 3: Running Experiments")
    print("="*70)
    
    print("\nThis will:")
    print("  1. Train transfer matrices on vocabulary")
    print("  2. Run Experiment 1: Text Baseline")
    print("  3. Run Experiment 2: Matrix Transfer")
    print("  4. Compare results")
    print()
    
    # Run both experiments
    exp1_results, exp2_results, comparison = run_both_experiments(
        embedder1,
        embedder2,
        vocabulary,
        test_corpus
    )
    
    print("\n" + "="*70)
    print("STEP 4: Generating Report")
    print("="*70)
    
    # Generate comprehensive report
    output_dir = generate_full_report(
        exp1_results,
        exp2_results,
        comparison,
        model1_name=model1_name,
        model2_name=model2_name,
        vocab_size=VOCAB_SIZE,
        test_size=TEST_SIZE
    )
    
    print("\n" + "="*70)
    print("POC COMPLETE!")
    print("="*70)
    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"✓ Read the full report: {output_dir}/REPORT.md")
    print()
    print("Summary:")
    print(f"  Text Baseline:       {comparison['text_baseline_mean']:.4f}")
    print(f"  Matrix Transfer:     {comparison['matrix_transfer_mean']:.4f}")
    print(f"  Improvement:         {comparison['absolute_improvement']:.4f} ({comparison['relative_improvement_pct']:.2f}%)")
    print(f"\n  Verdict: {comparison['verdict']}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
