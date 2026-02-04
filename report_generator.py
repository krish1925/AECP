"""
Report Generation

Creates comprehensive visualizations and analysis reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from datetime import datetime
import os


def create_output_dir():
    """Create output directory for reports."""
    os.makedirs("reports", exist_ok=True)
    return "reports"


def plot_similarity_distributions(
    exp1_results: Dict,
    exp2_results: Dict,
    output_dir: str
):
    """
    Plot similarity distributions for both experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp1_sims = exp1_results["all_similarities"]
    exp2_sims = exp2_results["all_similarities"]
    
    # Histogram comparison
    ax = axes[0, 0]
    ax.hist(exp1_sims, bins=50, alpha=0.6, label="Text Baseline", color="blue", edgecolor="black")
    ax.hist(exp2_sims, bins=50, alpha=0.6, label="Matrix Transfer", color="green", edgecolor="black")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Similarity Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax = axes[0, 1]
    data = pd.DataFrame({
        "Text Baseline": exp1_sims,
        "Matrix Transfer": exp2_sims
    })
    data.boxplot(ax=ax)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Distribution Comparison (Box Plot)")
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 0]
    sorted_exp1 = np.sort(exp1_sims)
    sorted_exp2 = np.sort(exp2_sims)
    cumulative_exp1 = np.arange(1, len(sorted_exp1) + 1) / len(sorted_exp1)
    cumulative_exp2 = np.arange(1, len(sorted_exp2) + 1) / len(sorted_exp2)
    ax.plot(sorted_exp1, cumulative_exp1, label="Text Baseline", linewidth=2)
    ax.plot(sorted_exp2, cumulative_exp2, label="Matrix Transfer", linewidth=2)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Violin plot
    ax = axes[1, 1]
    data_long = pd.DataFrame([
        {"Method": "Text Baseline", "Similarity": sim} for sim in exp1_sims
    ] + [
        {"Method": "Matrix Transfer", "Similarity": sim} for sim in exp2_sims
    ])
    sns.violinplot(data=data_long, x="Method", y="Similarity", ax=ax)
    ax.set_title("Similarity Distributions (Violin Plot)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_distributions.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/similarity_distributions.png")
    plt.close()


def plot_percentile_comparison(
    exp1_results: Dict,
    exp2_results: Dict,
    output_dir: str
):
    """
    Plot percentile comparison between experiments.
    """
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    exp1_percentiles = np.percentile(exp1_results["all_similarities"], percentiles)
    exp2_percentiles = np.percentile(exp2_results["all_similarities"], percentiles)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    ax.bar(x - width/2, exp1_percentiles, width, label="Text Baseline", alpha=0.8)
    ax.bar(x + width/2, exp2_percentiles, width, label="Matrix Transfer", alpha=0.8)
    
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Percentile Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in percentiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/percentile_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/percentile_comparison.png")
    plt.close()


def plot_sample_comparison(
    exp1_results: Dict,
    exp2_results: Dict,
    output_dir: str,
    n_samples: int = 50
):
    """
    Plot per-sample comparison of similarities.
    """
    exp1_sims = exp1_results["all_similarities"][:n_samples]
    exp2_sims = exp2_results["all_similarities"][:n_samples]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_samples)
    ax.plot(x, exp1_sims, 'o-', label="Text Baseline", alpha=0.7, markersize=4)
    ax.plot(x, exp2_sims, 's-', label="Matrix Transfer", alpha=0.7, markersize=4)
    
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Per-Sample Similarity (First {n_samples} samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/sample_comparison.png")
    plt.close()


def plot_scatter_comparison(
    exp1_results: Dict,
    exp2_results: Dict,
    output_dir: str
):
    """
    Scatter plot: text baseline vs matrix transfer similarities.
    """
    exp1_sims = np.array(exp1_results["all_similarities"])
    exp2_sims = np.array(exp2_results["all_similarities"])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(exp1_sims, exp2_sims, alpha=0.5, s=20)
    
    # Add diagonal line
    min_val = min(exp1_sims.min(), exp2_sims.min())
    max_val = max(exp1_sims.max(), exp2_sims.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y=x (equal performance)")
    
    ax.set_xlabel("Text Baseline Similarity")
    ax.set_ylabel("Matrix Transfer Similarity")
    ax.set_title("Text Baseline vs Matrix Transfer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add statistics
    correlation = np.corrcoef(exp1_sims, exp2_sims)[0, 1]
    above_diagonal = np.sum(exp2_sims > exp1_sims)
    total = len(exp1_sims)
    pct_better = (above_diagonal / total) * 100
    
    stats_text = f"Correlation: {correlation:.3f}\nMatrix better: {pct_better:.1f}% of samples"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/scatter_comparison.png")
    plt.close()


def plot_improvement_analysis(
    exp1_results: Dict,
    exp2_results: Dict,
    output_dir: str
):
    """
    Analyze where matrix transfer improves over text baseline.
    """
    exp1_sims = np.array(exp1_results["all_similarities"])
    exp2_sims = np.array(exp2_results["all_similarities"])
    
    improvements = exp2_sims - exp1_sims
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of improvements
    ax = axes[0]
    ax.hist(improvements, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label="No improvement")
    ax.axvline(np.mean(improvements), color='green', linestyle='-', linewidth=2, label=f"Mean: {np.mean(improvements):.4f}")
    ax.set_xlabel("Improvement (Matrix - Text)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Improvements")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improvement vs text baseline similarity
    ax = axes[1]
    ax.scatter(exp1_sims, improvements, alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Text Baseline Similarity")
    ax.set_ylabel("Improvement (Matrix - Text)")
    ax.set_title("Improvement vs Text Baseline")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/improvement_analysis.png")
    plt.close()


def generate_summary_statistics(
    exp1_results: Dict,
    exp2_results: Dict,
    comparison: Dict
) -> pd.DataFrame:
    """
    Generate summary statistics table.
    """
    stats = {
        "Metric": [
            "Mean Similarity",
            "Median Similarity", 
            "Std Deviation",
            "Min Similarity",
            "Max Similarity",
            "25th Percentile",
            "75th Percentile",
        ],
        "Text Baseline": [
            exp1_results["mean_similarity"],
            exp1_results["median_similarity"],
            exp1_results["std_similarity"],
            exp1_results["min_similarity"],
            exp1_results["max_similarity"],
            exp1_results["q25_similarity"],
            exp1_results["q75_similarity"],
        ],
        "Matrix Transfer": [
            exp2_results["mean_similarity"],
            exp2_results["median_similarity"],
            exp2_results["std_similarity"],
            exp2_results["min_similarity"],
            exp2_results["max_similarity"],
            exp2_results["q25_similarity"],
            exp2_results["q75_similarity"],
        ]
    }
    
    df = pd.DataFrame(stats)
    df["Difference"] = df["Matrix Transfer"] - df["Text Baseline"]
    
    return df


def generate_markdown_report(
    exp1_results: Dict,
    exp2_results: Dict,
    comparison: Dict,
    model1_name: str,
    model2_name: str,
    vocab_size: int,
    test_size: int,
    output_dir: str
):
    """
    Generate comprehensive markdown report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate statistics table
    stats_df = generate_summary_statistics(exp1_results, exp2_results, comparison)
    
    # Calculate additional metrics
    exp1_sims = np.array(exp1_results["all_similarities"])
    exp2_sims = np.array(exp2_results["all_similarities"])
    improvements = exp2_sims - exp1_sims
    pct_improved = (np.sum(improvements > 0) / len(improvements)) * 100
    pct_degraded = (np.sum(improvements < 0) / len(improvements)) * 100
    
    report = f"""# Embedding Transfer POC Report

**Generated:** {timestamp}

## Executive Summary

### Verdict: {comparison['verdict']}

{comparison['explanation']}

**Key Finding:** Matrix transfer achieved a mean similarity of {comparison['matrix_transfer_mean']:.4f} 
compared to text baseline of {comparison['text_baseline_mean']:.4f}, representing a 
{comparison['relative_improvement_pct']:.2f}% relative improvement.

---

## Experimental Setup

### Models
- **Embedder 1:** {model1_name}
- **Embedder 2:** {model2_name}

### Dataset
- **Vocabulary Size:** {vocab_size:,} items (used for training transfer matrices)
- **Test Corpus Size:** {test_size:,} sentences (held-out for evaluation)

### Methodology

#### Experiment 1: Text Baseline
Measures cross-embedder agreement when both encode the same text:
```
text → embedder1 → emb1
text → embedder2 → emb2 (projected to emb1 space)
similarity = cosine(emb1, emb2_projected)
```

This represents the information preservation when agents communicate via text strings.

#### Experiment 2: Matrix Transfer
Tests round-trip fidelity through learned transfer matrices:
```
text → embedder1 → emb1_original
emb1 @ W_12 → emb2
emb2 @ W_21 → emb1_reconstructed
similarity = cosine(emb1_original, emb1_reconstructed)
```

This represents the information preservation when agents communicate via embedding transfer.

---

## Results

### Summary Statistics

{stats_df.to_markdown(index=False, floatfmt=".4f")}

### Key Metrics

| Metric | Value |
|--------|-------|
| **Absolute Improvement** | {comparison['absolute_improvement']:.4f} |
| **Relative Improvement** | {comparison['relative_improvement_pct']:.2f}% |
| **Samples Improved** | {pct_improved:.1f}% |
| **Samples Degraded** | {pct_degraded:.1f}% |
| **Mean Improvement** | {np.mean(improvements):.4f} |
| **Median Improvement** | {np.median(improvements):.4f} |

### Training Quality

| Metric | Value |
|--------|-------|
| **Forward Transfer Similarity** | {comparison['training_forward_similarity']:.4f} |
| **Round-trip Similarity (train)** | {comparison['training_roundtrip_similarity']:.4f} |

---

## Visualizations

### Distribution Analysis
![Similarity Distributions](similarity_distributions.png)

The distribution plots show how similarities are spread across the test corpus for both methods.

### Percentile Comparison
![Percentile Comparison](percentile_comparison.png)

Percentile analysis reveals performance across the entire distribution, from worst to best cases.

### Per-Sample Analysis
![Sample Comparison](sample_comparison.png)

Individual sample trajectories show how each method performs on specific test cases.

### Correlation Analysis
![Scatter Comparison](scatter_comparison.png)

The scatter plot reveals the relationship between text baseline and matrix transfer performance.

### Improvement Analysis
![Improvement Analysis](improvement_analysis.png)

Analysis of where and how much matrix transfer improves over the text baseline.

---

## Interpretation

### What This Means

"""

    if comparison['absolute_improvement'] > 0.05:
        report += """
**Strong Validation:** Matrix transfer significantly outperforms text serialization. 
The learned transfer matrices preserve substantially more semantic information than 
converting embeddings to text and back. This suggests that agent communication via 
embedding transfer is a viable and superior alternative to text-based communication.

**Recommendation:** Implement embedding transfer protocol for agent communication.
"""
    elif comparison['absolute_improvement'] > 0.02:
        report += """
**Moderate Validation:** Matrix transfer shows meaningful improvement over text serialization.
The learned matrices capture important aspects of the cross-embedder relationship, though
there's still information loss. This suggests embedding transfer is beneficial but may need
refinement for critical applications.

**Recommendation:** Use embedding transfer for most use cases, fall back to text for critical scenarios.
"""
    elif comparison['absolute_improvement'] > 0:
        report += """
**Marginal Validation:** Matrix transfer slightly outperforms text baseline. The improvement
is real but modest. Consider whether the added complexity of maintaining transfer matrices
is worth the marginal gain.

**Recommendation:** Evaluate on a case-by-case basis based on performance requirements.
"""
    else:
        report += """
**Negative Result:** Text serialization preserves more information than matrix transfer.
This suggests the embedders are too incompatible for linear transfer, or the vocabulary
used for training wasn't representative enough.

**Recommendation:** Stick with text-based communication or explore non-linear transfer methods.
"""
    
    report += f"""

### Technical Insights

1. **Training vs Test Performance:** 
   - Training round-trip: {comparison['training_roundtrip_similarity']:.4f}
   - Test round-trip: {comparison['matrix_transfer_mean']:.4f}
   - Generalization gap: {comparison['training_roundtrip_similarity'] - comparison['matrix_transfer_mean']:.4f}

2. **Consistency:** 
   - Text baseline std: {exp1_results['std_similarity']:.4f}
   - Matrix transfer std: {exp2_results['std_similarity']:.4f}
   - Matrix transfer is {'more' if exp2_results['std_similarity'] < exp1_results['std_similarity'] else 'less'} consistent

3. **Robustness:**
   - Text baseline min: {exp1_results['min_similarity']:.4f}
   - Matrix transfer min: {exp2_results['min_similarity']:.4f}
   - Worst-case scenarios handled {'better' if exp2_results['min_similarity'] > exp1_results['min_similarity'] else 'worse'} by matrix transfer

---

## Conclusion

This POC {'successfully validates' if comparison['absolute_improvement'] > 0.02 else 'provides insights into'} 
the hypothesis that learned transfer matrices can {'preserve more semantic information than text serialization' if comparison['absolute_improvement'] > 0.02 else 'offer an alternative to text serialization'}.

The results demonstrate that {'embedding-based agent communication is a promising direction' if comparison['absolute_improvement'] > 0.02 else 'text-based communication remains competitive'} 
for multi-agent systems using different embedding models.

### Next Steps

"""

    if comparison['absolute_improvement'] > 0.02:
        report += """
1. **Scale Testing:** Evaluate on larger, more diverse corpora
2. **Model Diversity:** Test with more dissimilar embedding models
3. **Non-linear Transfer:** Explore neural network-based transfer functions
4. **Compression:** Investigate if matrices can be compressed further
5. **Real-world Application:** Deploy in actual multi-agent communication scenarios
"""
    else:
        report += """
1. **Root Cause Analysis:** Investigate why linear transfer underperforms
2. **Alternative Methods:** Explore non-linear transfer functions
3. **Vocabulary Optimization:** Refine vocabulary selection for better coverage
4. **Model Selection:** Test with more compatible embedding models
5. **Hybrid Approach:** Consider combining text and embedding transfer
"""
    
    report += f"""

---

## Detailed Results

### Best Performing Samples (Matrix Transfer)

Top 5 samples by round-trip similarity:

"""
    
    # Add top samples
    detailed = exp2_results["detailed_results"]
    sorted_detailed = sorted(detailed, key=lambda x: x["roundtrip_similarity"], reverse=True)
    for i, sample in enumerate(sorted_detailed[:5], 1):
        report += f"{i}. **Similarity: {sample['roundtrip_similarity']:.4f}**\n"
        report += f"   - Text: \"{sample['text']}\"\n"
        report += f"   - Reconstruction error: {sample['reconstruction_error']:.4f}\n\n"
    
    report += """
### Worst Performing Samples (Matrix Transfer)

Bottom 5 samples by round-trip similarity:

"""
    
    for i, sample in enumerate(sorted_detailed[-5:], 1):
        report += f"{i}. **Similarity: {sample['roundtrip_similarity']:.4f}**\n"
        report += f"   - Text: \"{sample['text']}\"\n"
        report += f"   - Reconstruction error: {sample['reconstruction_error']:.4f}\n\n"
    
    report += """
---

*Report generated by embedding_transfer_poc*
"""
    
    # Save report
    report_path = f"{output_dir}/REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nSaved comprehensive report: {report_path}")
    
    return report_path


def generate_full_report(
    exp1_results: Dict,
    exp2_results: Dict,
    comparison: Dict,
    model1_name: str = "embedder1",
    model2_name: str = "embedder2",
    vocab_size: int = 30000,
    test_size: int = 1000
):
    """
    Generate all visualizations and reports.
    """
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    output_dir = create_output_dir()
    
    # Generate all plots
    print("\nCreating visualizations...")
    plot_similarity_distributions(exp1_results, exp2_results, output_dir)
    plot_percentile_comparison(exp1_results, exp2_results, output_dir)
    plot_sample_comparison(exp1_results, exp2_results, output_dir)
    plot_scatter_comparison(exp1_results, exp2_results, output_dir)
    plot_improvement_analysis(exp1_results, exp2_results, output_dir)
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    report_path = generate_markdown_report(
        exp1_results, exp2_results, comparison,
        model1_name, model2_name,
        vocab_size, test_size,
        output_dir
    )
    
    # Save raw results as JSON
    results_data = {
        "experiment_1_text_baseline": exp1_results,
        "experiment_2_matrix_transfer": exp2_results,
        "comparison": comparison,
        "metadata": {
            "model1": model1_name,
            "model2": model2_name,
            "vocab_size": vocab_size,
            "test_size": test_size,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Remove large arrays for JSON storage
    results_data["experiment_1_text_baseline"]["all_similarities"] = len(exp1_results["all_similarities"])
    results_data["experiment_2_matrix_transfer"]["all_similarities"] = len(exp2_results["all_similarities"])
    
    json_path = f"{output_dir}/results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved raw results: {json_path}")
    
    print(f"\n{'='*60}")
    print(f"Report generation complete!")
    print(f"{'='*60}")
    print(f"\nView the full report at: {report_path}")
    print(f"All outputs saved to: {output_dir}/")
    
    return output_dir
