"""
Vector Store Dimension Comparison POC

This POC tests the impact of dimensional transformations on vector store accuracy:
- Scenario 1: VS1 (high-dim) -> English -> Embedder2 -> High-dim model
- Scenario 2: VS1 (high-dim) -> English -> Embedder2 -> Low-dim model

Compares:
1. Handshake Protocol (text-based transfer)
2. Transformation Matrix (matrix-based transfer)

Metrics:
- Accuracy: Cosine similarity to ground truth
- Precision retention: Top-K retrieval accuracy
- Latency: Time for transformation operations
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
import time
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime
from matrix_transfer import compute_transfer_matrices, cosine_similarity
from protocol import ProtocolHandler
import matplotlib.pyplot as plt
from tabulate import tabulate


class VectorStore:
    """Simple vector store for documents."""
    
    def __init__(self, name: str, embedder, documents: List[str]):
        """
        Initialize vector store.
        
        Args:
            name: Name of the vector store
            embedder: Embedding model
            documents: Documents to store
        """
        self.name = name
        self.embedder = embedder
        self.documents = documents
        self.dimensions = embedder.get_sentence_embedding_dimension()
        
        print(f"\nInitializing Vector Store: {name}")
        print(f"  Model: {embedder}")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Documents: {len(documents)}")
        
        # Create embeddings
        start = time.time()
        self.embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=64)
        elapsed = time.time() - start
        
        print(f"  Embedding time: {elapsed:.2f}s")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of (doc_idx, similarity) tuples
        """
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class DimensionComparisonPOC:
    """POC for comparing dimensional transformations."""
    
    def __init__(self, documents: List[str], test_queries: List[str]):
        """
        Initialize POC.
        
        Args:
            documents: Documents for vector store
            test_queries: Test queries for evaluation
        """
        self.documents = documents
        self.test_queries = test_queries
        self.results = {}
        
        print("="*80)
        print("VECTOR STORE DIMENSION COMPARISON POC")
        print("="*80)
        
        # Initialize models
        print("\n" + "="*80)
        print("INITIALIZING EMBEDDING MODELS")
        print("="*80)
        
        # VS1: High-dimensional base model
        print("\nLoading VS1 (Base High-Dim Model)...")
        self.model_vs1 = SentenceTransformer('all-mpnet-base-v2')  # 768 dim
        
        # Embedder2 - intermediate model
        print("\nLoading Embedder2 (Intermediate)...")
        self.model_embedder2 = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim
        
        # High-dim target
        print("\nLoading High-Dim Target...")
        self.model_highdim = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # 768 dim
        
        # Low-dim target
        print("\nLoading Low-Dim Target...")
        self.model_lowdim = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 384 dim
        
        print(f"\n✓ All models loaded")
        print(f"  VS1 dimensions: {self.model_vs1.get_sentence_embedding_dimension()}")
        print(f"  Embedder2 dimensions: {self.model_embedder2.get_sentence_embedding_dimension()}")
        print(f"  High-dim target dimensions: {self.model_highdim.get_sentence_embedding_dimension()}")
        print(f"  Low-dim target dimensions: {self.model_lowdim.get_sentence_embedding_dimension()}")
        
    def create_vector_store(self):
        """Create initial vector store with VS1."""
        print("\n" + "="*80)
        print("CREATING VECTOR STORE (VS1)")
        print("="*80)
        
        self.vs1 = VectorStore("VS1", self.model_vs1, self.documents)
        
    def scenario_handshake_highdim(self):
        """
        Scenario 1A: Handshake Protocol - High Dimensions
        VS1 -> English text -> Embedder2 -> High-dim model
        """
        print("\n" + "="*80)
        print("SCENARIO 1A: HANDSHAKE PROTOCOL (HIGH-DIM)")
        print("="*80)
        
        results = {
            "method": "handshake",
            "target_dims": self.model_highdim.get_sentence_embedding_dimension(),
            "similarities": [],
            "top_k_accuracies": [],
            "latencies": []
        }
        
        for query in self.test_queries:
            start = time.time()
            
            # Step 1: Get query embedding from VS1
            query_emb_vs1 = self.model_vs1.encode(query)
            
            # Step 2: Find most similar document in VS1 (this is our "text transfer")
            doc_idx = self.vs1.search(query_emb_vs1, top_k=1)[0][0]
            transferred_text = self.documents[doc_idx]
            
            # Step 3: Re-embed with Embedder2
            intermediate_emb = self.model_embedder2.encode(transferred_text)
            
            # Step 4: Re-embed with high-dim model
            final_emb = self.model_highdim.encode(transferred_text)
            
            latency = time.time() - start
            
            # Evaluation: Compare with ground truth (direct encoding)
            ground_truth = self.model_highdim.encode(query)
            similarity = cosine_similarity(final_emb, ground_truth)
            
            # Top-K accuracy (precision@5)
            vs1_results = self.vs1.search(query_emb_vs1, top_k=5)
            vs1_top_indices = [idx for idx, _ in vs1_results]
            
            # Create temporary vector store with high-dim for comparison
            highdim_embeddings = self.model_highdim.encode(self.documents, show_progress_bar=False)
            highdim_similarities = [(i, cosine_similarity(ground_truth, emb)) 
                                   for i, emb in enumerate(highdim_embeddings)]
            highdim_similarities.sort(key=lambda x: x[1], reverse=True)
            highdim_top_indices = [idx for idx, _ in highdim_similarities[:5]]
            
            # Calculate overlap
            overlap = len(set(vs1_top_indices) & set(highdim_top_indices))
            top_k_accuracy = overlap / 5.0
            
            results["similarities"].append(float(similarity))
            results["top_k_accuracies"].append(top_k_accuracy)
            results["latencies"].append(latency)
        
        results["mean_similarity"] = float(np.mean(results["similarities"]))
        results["mean_top_k_accuracy"] = float(np.mean(results["top_k_accuracies"]))
        results["mean_latency"] = float(np.mean(results["latencies"]))
        
        self.results["handshake_highdim"] = results
        
        print(f"\nResults:")
        print(f"  Mean Similarity: {results['mean_similarity']:.4f}")
        print(f"  Mean Top-K Accuracy: {results['mean_top_k_accuracy']:.4f}")
        print(f"  Mean Latency: {results['mean_latency']*1000:.2f}ms")
        
    def scenario_handshake_lowdim(self):
        """
        Scenario 1B: Handshake Protocol - Low Dimensions
        VS1 -> English text -> Embedder2 -> Low-dim model
        """
        print("\n" + "="*80)
        print("SCENARIO 1B: HANDSHAKE PROTOCOL (LOW-DIM)")
        print("="*80)
        
        results = {
            "method": "handshake",
            "target_dims": self.model_lowdim.get_sentence_embedding_dimension(),
            "similarities": [],
            "top_k_accuracies": [],
            "latencies": []
        }
        
        for query in self.test_queries:
            start = time.time()
            
            # Step 1: Get query embedding from VS1
            query_emb_vs1 = self.model_vs1.encode(query)
            
            # Step 2: Find most similar document in VS1
            doc_idx = self.vs1.search(query_emb_vs1, top_k=1)[0][0]
            transferred_text = self.documents[doc_idx]
            
            # Step 3: Re-embed with Embedder2
            intermediate_emb = self.model_embedder2.encode(transferred_text)
            
            # Step 4: Re-embed with low-dim model
            final_emb = self.model_lowdim.encode(transferred_text)
            
            latency = time.time() - start
            
            # Evaluation
            ground_truth = self.model_lowdim.encode(query)
            similarity = cosine_similarity(final_emb, ground_truth)
            
            # Top-K accuracy
            vs1_results = self.vs1.search(query_emb_vs1, top_k=5)
            vs1_top_indices = [idx for idx, _ in vs1_results]
            
            lowdim_embeddings = self.model_lowdim.encode(self.documents, show_progress_bar=False)
            lowdim_similarities = [(i, cosine_similarity(ground_truth, emb)) 
                                  for i, emb in enumerate(lowdim_embeddings)]
            lowdim_similarities.sort(key=lambda x: x[1], reverse=True)
            lowdim_top_indices = [idx for idx, _ in lowdim_similarities[:5]]
            
            overlap = len(set(vs1_top_indices) & set(lowdim_top_indices))
            top_k_accuracy = overlap / 5.0
            
            results["similarities"].append(float(similarity))
            results["top_k_accuracies"].append(top_k_accuracy)
            results["latencies"].append(latency)
        
        results["mean_similarity"] = float(np.mean(results["similarities"]))
        results["mean_top_k_accuracy"] = float(np.mean(results["top_k_accuracies"]))
        results["mean_latency"] = float(np.mean(results["latencies"]))
        
        self.results["handshake_lowdim"] = results
        
        print(f"\nResults:")
        print(f"  Mean Similarity: {results['mean_similarity']:.4f}")
        print(f"  Mean Top-K Accuracy: {results['mean_top_k_accuracy']:.4f}")
        print(f"  Mean Latency: {results['mean_latency']*1000:.2f}ms")
        
    def scenario_matrix_highdim(self):
        """
        Scenario 2A: Matrix Transfer - High Dimensions
        VS1 -> Matrix transform -> Embedder2 -> Matrix transform -> High-dim model
        """
        print("\n" + "="*80)
        print("SCENARIO 2A: MATRIX TRANSFER (HIGH-DIM)")
        print("="*80)
        
        # Calibration vocabulary (subset of documents)
        calibration_vocab = self.documents[:min(100, len(self.documents))]
        
        # Compute transfer matrices
        print("\nComputing transfer matrices...")
        print("  VS1 -> Embedder2")
        emb_vs1 = self.model_vs1.encode(calibration_vocab, show_progress_bar=True)
        emb_e2 = self.model_embedder2.encode(calibration_vocab, show_progress_bar=True)
        W_vs1_e2, W_e2_vs1 = compute_transfer_matrices(emb_vs1, emb_e2)
        
        print("  Embedder2 -> High-dim")
        emb_hd = self.model_highdim.encode(calibration_vocab, show_progress_bar=True)
        W_e2_hd, W_hd_e2 = compute_transfer_matrices(emb_e2, emb_hd)
        
        results = {
            "method": "matrix",
            "target_dims": self.model_highdim.get_sentence_embedding_dimension(),
            "similarities": [],
            "top_k_accuracies": [],
            "latencies": []
        }
        
        for query in self.test_queries:
            start = time.time()
            
            # Step 1: Get query embedding from VS1
            query_emb_vs1 = self.model_vs1.encode(query)
            
            # Step 2: Transform to Embedder2 space
            query_emb_e2 = query_emb_vs1 @ W_vs1_e2
            
            # Step 3: Transform to High-dim space
            query_emb_hd = query_emb_e2 @ W_e2_hd
            
            latency = time.time() - start
            
            # Evaluation
            ground_truth = self.model_highdim.encode(query)
            similarity = cosine_similarity(query_emb_hd, ground_truth)
            
            # Top-K accuracy
            vs1_results = self.vs1.search(query_emb_vs1, top_k=5)
            vs1_top_indices = [idx for idx, _ in vs1_results]
            
            highdim_embeddings = self.model_highdim.encode(self.documents, show_progress_bar=False)
            highdim_similarities = [(i, cosine_similarity(ground_truth, emb)) 
                                   for i, emb in enumerate(highdim_embeddings)]
            highdim_similarities.sort(key=lambda x: x[1], reverse=True)
            highdim_top_indices = [idx for idx, _ in highdim_similarities[:5]]
            
            overlap = len(set(vs1_top_indices) & set(highdim_top_indices))
            top_k_accuracy = overlap / 5.0
            
            results["similarities"].append(float(similarity))
            results["top_k_accuracies"].append(top_k_accuracy)
            results["latencies"].append(latency)
        
        results["mean_similarity"] = float(np.mean(results["similarities"]))
        results["mean_top_k_accuracy"] = float(np.mean(results["top_k_accuracies"]))
        results["mean_latency"] = float(np.mean(results["latencies"]))
        
        self.results["matrix_highdim"] = results
        
        print(f"\nResults:")
        print(f"  Mean Similarity: {results['mean_similarity']:.4f}")
        print(f"  Mean Top-K Accuracy: {results['mean_top_k_accuracy']:.4f}")
        print(f"  Mean Latency: {results['mean_latency']*1000:.2f}ms")
        
    def scenario_matrix_lowdim(self):
        """
        Scenario 2B: Matrix Transfer - Low Dimensions
        VS1 -> Matrix transform -> Embedder2 -> Matrix transform -> Low-dim model
        """
        print("\n" + "="*80)
        print("SCENARIO 2B: MATRIX TRANSFER (LOW-DIM)")
        print("="*80)
        
        # Calibration vocabulary
        calibration_vocab = self.documents[:min(100, len(self.documents))]
        
        # Compute transfer matrices
        print("\nComputing transfer matrices...")
        print("  VS1 -> Embedder2")
        emb_vs1 = self.model_vs1.encode(calibration_vocab, show_progress_bar=True)
        emb_e2 = self.model_embedder2.encode(calibration_vocab, show_progress_bar=True)
        W_vs1_e2, W_e2_vs1 = compute_transfer_matrices(emb_vs1, emb_e2)
        
        print("  Embedder2 -> Low-dim")
        emb_ld = self.model_lowdim.encode(calibration_vocab, show_progress_bar=True)
        W_e2_ld, W_ld_e2 = compute_transfer_matrices(emb_e2, emb_ld)
        
        results = {
            "method": "matrix",
            "target_dims": self.model_lowdim.get_sentence_embedding_dimension(),
            "similarities": [],
            "top_k_accuracies": [],
            "latencies": []
        }
        
        for query in self.test_queries:
            start = time.time()
            
            # Step 1: Get query embedding from VS1
            query_emb_vs1 = self.model_vs1.encode(query)
            
            # Step 2: Transform to Embedder2 space
            query_emb_e2 = query_emb_vs1 @ W_vs1_e2
            
            # Step 3: Transform to Low-dim space
            query_emb_ld = query_emb_e2 @ W_e2_ld
            
            latency = time.time() - start
            
            # Evaluation
            ground_truth = self.model_lowdim.encode(query)
            similarity = cosine_similarity(query_emb_ld, ground_truth)
            
            # Top-K accuracy
            vs1_results = self.vs1.search(query_emb_vs1, top_k=5)
            vs1_top_indices = [idx for idx, _ in vs1_results]
            
            lowdim_embeddings = self.model_lowdim.encode(self.documents, show_progress_bar=False)
            lowdim_similarities = [(i, cosine_similarity(ground_truth, emb)) 
                                  for i, emb in enumerate(lowdim_embeddings)]
            lowdim_similarities.sort(key=lambda x: x[1], reverse=True)
            lowdim_top_indices = [idx for idx, _ in lowdim_similarities[:5]]
            
            overlap = len(set(vs1_top_indices) & set(lowdim_top_indices))
            top_k_accuracy = overlap / 5.0
            
            results["similarities"].append(float(similarity))
            results["top_k_accuracies"].append(top_k_accuracy)
            results["latencies"].append(latency)
        
        results["mean_similarity"] = float(np.mean(results["similarities"]))
        results["mean_top_k_accuracy"] = float(np.mean(results["top_k_accuracies"]))
        results["mean_latency"] = float(np.mean(results["latencies"]))
        
        self.results["matrix_lowdim"] = results
        
        print(f"\nResults:")
        print(f"  Mean Similarity: {results['mean_similarity']:.4f}")
        print(f"  Mean Top-K Accuracy: {results['mean_top_k_accuracy']:.4f}")
        print(f"  Mean Latency: {results['mean_latency']*1000:.2f}ms")
        
    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        # Summary table
        table_data = []
        for scenario_name, scenario_results in self.results.items():
            method = scenario_results["method"].upper()
            dims = scenario_results["target_dims"]
            sim = scenario_results["mean_similarity"]
            acc = scenario_results["mean_top_k_accuracy"]
            lat = scenario_results["mean_latency"] * 1000
            
            table_data.append([
                scenario_name.replace("_", " ").title(),
                method,
                dims,
                f"{sim:.4f}",
                f"{acc:.4f}",
                f"{lat:.2f}ms"
            ])
        
        headers = ["Scenario", "Method", "Target Dims", "Similarity", "Top-K Acc", "Latency"]
        
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        # Handshake comparison
        hs_hd = self.results["handshake_highdim"]
        hs_ld = self.results["handshake_lowdim"]
        
        print("\n1. HANDSHAKE PROTOCOL (Text-based):")
        print(f"   High-dim vs Low-dim:")
        print(f"     Similarity: {hs_hd['mean_similarity']:.4f} vs {hs_ld['mean_similarity']:.4f}")
        print(f"     Difference: {(hs_hd['mean_similarity'] - hs_ld['mean_similarity']):.4f}")
        print(f"     Top-K Accuracy: {hs_hd['mean_top_k_accuracy']:.4f} vs {hs_ld['mean_top_k_accuracy']:.4f}")
        print(f"     Latency: {hs_hd['mean_latency']*1000:.2f}ms vs {hs_ld['mean_latency']*1000:.2f}ms")
        
        # Matrix comparison
        mx_hd = self.results["matrix_highdim"]
        mx_ld = self.results["matrix_lowdim"]
        
        print("\n2. MATRIX TRANSFER (Matrix-based):")
        print(f"   High-dim vs Low-dim:")
        print(f"     Similarity: {mx_hd['mean_similarity']:.4f} vs {mx_ld['mean_similarity']:.4f}")
        print(f"     Difference: {(mx_hd['mean_similarity'] - mx_ld['mean_similarity']):.4f}")
        print(f"     Top-K Accuracy: {mx_hd['mean_top_k_accuracy']:.4f} vs {mx_ld['mean_top_k_accuracy']:.4f}")
        print(f"     Latency: {mx_hd['mean_latency']*1000:.2f}ms vs {mx_ld['mean_latency']*1000:.2f}ms")
        
        # Method comparison
        print("\n3. HANDSHAKE vs MATRIX:")
        print(f"   High-dim:")
        print(f"     Similarity: Handshake {hs_hd['mean_similarity']:.4f} vs Matrix {mx_hd['mean_similarity']:.4f}")
        print(f"     Improvement: {(mx_hd['mean_similarity'] - hs_hd['mean_similarity']):.4f} ({(mx_hd['mean_similarity'] / hs_hd['mean_similarity'] - 1)*100:.1f}%)")
        print(f"     Latency: Handshake {hs_hd['mean_latency']*1000:.2f}ms vs Matrix {mx_hd['mean_latency']*1000:.2f}ms")
        print(f"     Speedup: {hs_hd['mean_latency'] / mx_hd['mean_latency']:.1f}x")
        
        print(f"   Low-dim:")
        print(f"     Similarity: Handshake {hs_ld['mean_similarity']:.4f} vs Matrix {mx_ld['mean_similarity']:.4f}")
        print(f"     Improvement: {(mx_ld['mean_similarity'] - hs_ld['mean_similarity']):.4f} ({(mx_ld['mean_similarity'] / hs_ld['mean_similarity'] - 1)*100:.1f}%)")
        print(f"     Latency: Handshake {hs_ld['mean_latency']*1000:.2f}ms vs Matrix {mx_ld['mean_latency']*1000:.2f}ms")
        print(f"     Speedup: {hs_ld['mean_latency'] / mx_ld['mean_latency']:.1f}x")
        
        # Dimensional impact
        print("\n4. DIMENSIONAL IMPACT:")
        print(f"   Matrix transfer maintains better quality:")
        print(f"     High-dim: {mx_hd['mean_similarity']:.4f}")
        print(f"     Low-dim: {mx_ld['mean_similarity']:.4f}")
        print(f"     Quality retention: {(mx_ld['mean_similarity'] / mx_hd['mean_similarity'])*100:.1f}%")
        
        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "num_queries": len(self.test_queries),
            "results": self.results,
            "summary": {
                "handshake_highdim": {
                    "similarity": hs_hd['mean_similarity'],
                    "top_k_accuracy": hs_hd['mean_top_k_accuracy'],
                    "latency_ms": hs_hd['mean_latency'] * 1000
                },
                "handshake_lowdim": {
                    "similarity": hs_ld['mean_similarity'],
                    "top_k_accuracy": hs_ld['mean_top_k_accuracy'],
                    "latency_ms": hs_ld['mean_latency'] * 1000
                },
                "matrix_highdim": {
                    "similarity": mx_hd['mean_similarity'],
                    "top_k_accuracy": mx_hd['mean_top_k_accuracy'],
                    "latency_ms": mx_hd['mean_latency'] * 1000
                },
                "matrix_lowdim": {
                    "similarity": mx_ld['mean_similarity'],
                    "top_k_accuracy": mx_ld['mean_top_k_accuracy'],
                    "latency_ms": mx_ld['mean_latency'] * 1000
                }
            }
        }
        
        with open('/Users/kpatel/Desktop/agent-communication/reports/dimension_comparison_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n✓ Results saved to reports/dimension_comparison_results.json")
        
        # Generate visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """Create comparison visualizations."""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scenarios = ["handshake_highdim", "handshake_lowdim", "matrix_highdim", "matrix_lowdim"]
        labels = ["HS High-Dim", "HS Low-Dim", "Matrix High-Dim", "Matrix Low-Dim"]
        colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']
        
        # 1. Similarity comparison
        ax = axes[0, 0]
        similarities = [self.results[s]["mean_similarity"] for s in scenarios]
        bars = ax.bar(labels, similarities, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title('Similarity Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.75, color='red', linestyle='--', label='Threshold (0.75)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # 2. Top-K Accuracy comparison
        ax = axes[0, 1]
        accuracies = [self.results[s]["mean_top_k_accuracy"] for s in scenarios]
        bars = ax.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Top-5 Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Retrieval Precision Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # 3. Latency comparison
        ax = axes[1, 0]
        latencies = [self.results[s]["mean_latency"] * 1000 for s in scenarios]
        bars = ax.bar(labels, latencies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom', fontweight='bold')
        
        # 4. Method vs Dimension matrix
        ax = axes[1, 1]
        methods = ['Handshake', 'Matrix']
        dimensions = ['High-Dim', 'Low-Dim']
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        handshake_vals = [
            self.results["handshake_highdim"]["mean_similarity"],
            self.results["handshake_lowdim"]["mean_similarity"]
        ]
        matrix_vals = [
            self.results["matrix_highdim"]["mean_similarity"],
            self.results["matrix_lowdim"]["mean_similarity"]
        ]
        
        bars1 = ax.bar(x - width/2, handshake_vals, width, label='Handshake', 
                      color='#FF6B6B', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, matrix_vals, width, label='Matrix',
                      color='#4ECDC4', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Mean Similarity', fontsize=12, fontweight='bold')
        ax.set_title('Method × Dimension Impact', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/Users/kpatel/Desktop/agent-communication/reports/dimension_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to reports/dimension_comparison.png")
        
    def run_all_scenarios(self):
        """Run all test scenarios."""
        self.create_vector_store()
        self.scenario_handshake_highdim()
        self.scenario_handshake_lowdim()
        self.scenario_matrix_highdim()
        self.scenario_matrix_lowdim()
        self.generate_report()


def generate_test_data():
    """Generate test documents and queries."""
    documents = [
        # Technology
        "Machine learning models require large amounts of training data",
        "Neural networks use backpropagation for learning",
        "Deep learning has revolutionized computer vision",
        "Natural language processing enables machines to understand text",
        "Transformers are the foundation of modern language models",
        
        # Science
        "Quantum mechanics describes the behavior of matter at atomic scales",
        "DNA contains the genetic instructions for all living organisms",
        "Climate change is caused by greenhouse gas emissions",
        "Photosynthesis converts sunlight into chemical energy",
        "The theory of relativity changed our understanding of space and time",
        
        # Business
        "Market research helps companies understand customer needs",
        "Digital transformation is reshaping traditional business models",
        "Data analytics provides insights for strategic decision making",
        "Customer experience is crucial for business success",
        "Sustainable practices are becoming essential for companies",
        
        # Health
        "Regular exercise improves cardiovascular health",
        "A balanced diet is essential for maintaining wellness",
        "Mental health is as important as physical health",
        "Preventive care reduces long-term healthcare costs",
        "Sleep quality affects cognitive function and mood",
        
        # Arts & Culture
        "Literature reflects the values and concerns of society",
        "Music has the power to evoke emotions and memories",
        "Visual arts communicate ideas beyond words",
        "Cultural diversity enriches human experience",
        "Creative expression is fundamental to human nature"
    ]
    
    test_queries = [
        "How do neural networks learn from data?",
        "What causes climate change?",
        "Why is customer experience important?",
        "How does exercise benefit health?",
        "What role does art play in society?",
        "Explain quantum physics basics",
        "How does DNA work?",
        "What is digital transformation?",
        "Why is sleep important?",
        "How does photosynthesis work?"
    ]
    
    return documents, test_queries


if __name__ == "__main__":
    print("Initializing Vector Store Dimension Comparison POC...")
    
    # Generate test data
    documents, test_queries = generate_test_data()
    
    print(f"\nTest Configuration:")
    print(f"  Documents: {len(documents)}")
    print(f"  Test Queries: {len(test_queries)}")
    
    # Run POC
    poc = DimensionComparisonPOC(documents, test_queries)
    poc.run_all_scenarios()
    
    print("\n" + "="*80)
    print("POC COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - reports/dimension_comparison_results.json")
    print("  - reports/dimension_comparison.png")
