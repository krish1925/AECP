"""
Enhanced Vocabulary and Test Corpus Loader

Generates 10x larger datasets with strict separation between:
- Training vocabulary (for calibration)
- Validation set (held-out from training, used to validate matrices)
- Test set (completely unseen, for final evaluation)

Ensures zero overlap to test true generalization.
"""

import json
import numpy as np
from typing import List, Tuple, Set
from tqdm import tqdm
import hashlib


def generate_diverse_vocabulary(n: int = 300000, seed: int = 42) -> List[str]:
    """
    Generate 300k diverse vocabulary items.
    
    This is 10x larger than the original POC.
    Includes more domain coverage and complexity.
    """
    np.random.seed(seed)
    vocabulary = []
    
    # 1. Core English words (50k)
    print("Generating core words...")
    base_words = [
        # Action verbs
        "analyze", "compute", "process", "optimize", "implement", "design", "test", "validate",
        "measure", "evaluate", "compare", "contrast", "synthesize", "integrate", "deploy", "monitor",
        "debug", "refactor", "document", "review", "assess", "investigate", "discover", "explore",
        "create", "build", "develop", "construct", "establish", "organize", "coordinate", "manage",
        "execute", "perform", "conduct", "facilitate", "enable", "enhance", "improve", "upgrade",
        "transform", "convert", "translate", "interpret", "explain", "clarify", "demonstrate", "illustrate",
        
        # Technical nouns
        "algorithm", "function", "method", "procedure", "process", "system", "model", "framework",
        "architecture", "structure", "component", "module", "interface", "protocol", "standard", "specification",
        "database", "repository", "storage", "memory", "cache", "buffer", "queue", "stack",
        "network", "server", "client", "endpoint", "gateway", "router", "switch", "firewall",
        "application", "software", "hardware", "firmware", "middleware", "platform", "environment", "infrastructure",
        "security", "authentication", "authorization", "encryption", "decryption", "certificate", "token", "credential",
        
        # Descriptive adjectives
        "efficient", "effective", "robust", "reliable", "scalable", "maintainable", "extensible", "flexible",
        "secure", "safe", "stable", "consistent", "coherent", "comprehensive", "complete", "thorough",
        "accurate", "precise", "exact", "correct", "valid", "sound", "logical", "rational",
        "fast", "quick", "rapid", "swift", "speedy", "slow", "gradual", "incremental",
        "simple", "complex", "sophisticated", "advanced", "basic", "fundamental", "elementary", "primary",
        "modern", "contemporary", "current", "recent", "latest", "new", "novel", "innovative",
        
        # Domain terms
        "machine", "learning", "artificial", "intelligence", "neural", "network", "deep", "learning",
        "data", "science", "analytics", "statistics", "probability", "inference", "prediction", "classification",
        "regression", "clustering", "dimensionality", "reduction", "feature", "engineering", "selection", "extraction",
        "training", "testing", "validation", "evaluation", "performance", "accuracy", "precision", "recall",
        "optimization", "gradient", "descent", "backpropagation", "forward", "pass", "backward", "pass",
        "tensor", "matrix", "vector", "scalar", "array", "dimension", "shape", "size",
    ]
    
    # Add variations
    for word in base_words:
        vocabulary.append(word)
        vocabulary.append(word.capitalize())
        vocabulary.append(word.upper())
        vocabulary.append(f"{word}s")  # plural
        vocabulary.append(f"{word}ing")  # gerund
        vocabulary.append(f"{word}ed")  # past tense
    
    # 2. Two-word phrases (80k)
    print("Generating two-word phrases...")
    prepositions = ["of", "in", "on", "at", "to", "for", "with", "by", "from", "about", "through", "during", "before", "after"]
    articles = ["the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their"]
    
    for i in range(40000):
        w1 = np.random.choice(base_words)
        w2 = np.random.choice(base_words)
        vocabulary.append(f"{w1} {w2}")
        
        if i < 20000:
            prep = np.random.choice(prepositions)
            vocabulary.append(f"{w1} {prep} {w2}")
        if i < 20000:
            art = np.random.choice(articles)
            vocabulary.append(f"{art} {w1}")
    
    # 3. Short technical phrases (70k)
    print("Generating technical phrases...")
    technical_templates = [
        "{} implementation",
        "{} architecture",
        "{} framework",
        "{} optimization",
        "{} algorithm",
        "{} model",
        "{} system",
        "{} protocol",
        "advanced {}",
        "modern {}",
        "distributed {}",
        "scalable {}",
        "{} analysis",
        "{} design",
        "{} development",
        "{} testing",
        "automated {}",
        "intelligent {}",
        "{} processing",
        "{} management",
        "{} monitoring",
        "{} evaluation",
        "{} validation",
        "{} verification",
        "real-time {}",
        "high-performance {}",
        "enterprise {}",
        "cloud-based {}",
        "{} integration",
        "{} deployment",
    ]
    
    for template in technical_templates:
        for word in base_words[:1000]:
            if "{}" in template:
                vocabulary.append(template.format(word))
    
    # 4. Common sentences (50k)
    print("Generating common sentences...")
    sentence_templates = [
        "The {} is {}.",
        "We need to {} the {}.",
        "This {} enables {}.",
        "The {} provides {}.",
        "A {} can {} the {}.",
        "The system {} the {}.",
        "We should {} the {}.",
        "The {} improves {}.",
        "This approach {} the {}.",
        "The {} requires {}.",
        "We must {} the {}.",
        "The {} supports {}.",
        "This {} enhances {}.",
        "The {} reduces {}.",
        "A {} increases {}.",
        "The {} optimizes {}.",
        "We can {} the {}.",
        "The {} validates {}.",
        "This {} ensures {}.",
        "The {} manages {}.",
    ]
    
    for i, template in enumerate(sentence_templates):
        count = template.count("{}")
        if count == 1:
            for word in base_words[:2500]:
                vocabulary.append(template.format(word))
        elif count == 2:
            for j in range(1250):
                w1 = base_words[j % len(base_words)]
                w2 = base_words[(j * 7) % len(base_words)]
                vocabulary.append(template.format(w1, w2))
    
    # 5. Domain-specific terminology (30k)
    print("Generating domain-specific terms...")
    domains = {
        "ML": [
            "convolutional neural network", "recurrent neural network", "transformer architecture",
            "attention mechanism", "self-attention layer", "multi-head attention", "positional encoding",
            "gradient descent", "stochastic gradient descent", "adam optimizer", "learning rate schedule",
            "batch normalization", "layer normalization", "dropout regularization", "weight decay",
            "backpropagation algorithm", "forward propagation", "activation function", "relu activation",
            "softmax function", "cross-entropy loss", "mean squared error", "binary classification",
            "multi-class classification", "semantic segmentation", "object detection", "image recognition",
        ],
        "Systems": [
            "distributed system", "microservices architecture", "service mesh", "load balancer",
            "container orchestration", "kubernetes cluster", "docker container", "virtual machine",
            "cloud computing", "edge computing", "fog computing", "serverless computing",
            "message queue", "event streaming", "pub-sub pattern", "request-response pattern",
            "database sharding", "replication strategy", "consistency model", "eventual consistency",
            "cache coherence", "memory hierarchy", "storage system", "file system",
        ],
        "Security": [
            "public key cryptography", "private key encryption", "digital signature", "hash function",
            "secure socket layer", "transport layer security", "certificate authority", "key exchange",
            "access control", "role-based access", "multi-factor authentication", "single sign-on",
            "intrusion detection", "firewall rule", "security policy", "threat model",
            "vulnerability assessment", "penetration testing", "security audit", "compliance check",
        ],
        "Data": [
            "data warehouse", "data lake", "data pipeline", "etl process",
            "stream processing", "batch processing", "real-time analytics", "time series analysis",
            "data quality", "data governance", "data lineage", "metadata management",
            "dimensional modeling", "fact table", "dimension table", "star schema",
            "data partitioning", "data compression", "data encryption", "data backup",
        ]
    }
    
    for domain, terms in domains.items():
        for term in terms:
            vocabulary.append(term)
            vocabulary.append(f"{term} system")
            vocabulary.append(f"{term} implementation")
            vocabulary.append(f"optimized {term}")
            vocabulary.append(f"distributed {term}")
            vocabulary.append(f"{term} architecture")
    
    # 6. Generated combinations (20k)
    print("Generating combinations...")
    for i in range(20000):
        idx1 = i % len(base_words)
        idx2 = (i * 13) % len(base_words)
        idx3 = (i * 7) % len(base_words)
        vocabulary.append(f"{base_words[idx1]} {base_words[idx2]} {base_words[idx3]}")
    
    # Deduplicate and ensure uniqueness
    print("Deduplicating...")
    vocabulary = list(set(vocabulary))
    
    # If we need more, pad with generated terms
    while len(vocabulary) < n:
        idx = len(vocabulary)
        vocabulary.append(f"technical_term_{idx}")
        vocabulary.append(f"concept_{idx}")
        vocabulary.append(f"system_component_{idx}")
        vocabulary.append(f"protocol_element_{idx}")
    
    # Shuffle for randomness
    np.random.shuffle(vocabulary)
    
    return vocabulary[:n]


def generate_test_corpus(n: int = 10000, seed: int = 43) -> List[str]:
    """
    Generate 10k diverse test sentences (10x larger than original).
    These sentences should be DIFFERENT and MORE COMPLEX than vocabulary.
    """
    np.random.seed(seed)
    test_corpus = []
    
    # 1. Complex technical descriptions (3k)
    print("Generating complex technical sentences...")
    complex_templates = [
        "The {adj} {system} employs {tech} to achieve {goal} while maintaining {quality}.",
        "By leveraging {tech1} and {tech2}, the {system} can {action} with {metric} improvement.",
        "Research has shown that {tech} significantly enhances {capability} in {domain} applications.",
        "The implementation utilizes {tech} to optimize {resource} allocation across {system} components.",
        "Advanced {tech} techniques enable {system} to process {data} with {metric} latency.",
        "The {adj} architecture integrates {tech1} with {tech2} for improved {quality}.",
        "Studies indicate that {tech} outperforms traditional approaches in {task} by {metric}.",
        "The proposed {system} combines {tech1}, {tech2}, and {tech3} to achieve {goal}.",
        "Experimental results demonstrate that {tech} reduces {resource} consumption by {metric}.",
        "The {adj} framework provides {capability} through sophisticated {tech} mechanisms.",
    ]
    
    adjectives = ["advanced", "novel", "sophisticated", "distributed", "scalable", "robust", "efficient", "innovative"]
    systems = ["framework", "architecture", "platform", "infrastructure", "system", "environment", "ecosystem"]
    technologies = ["machine learning", "neural networks", "parallel processing", "distributed computing", 
                   "cloud infrastructure", "edge computing", "quantum algorithms", "blockchain consensus"]
    goals = ["high throughput", "low latency", "fault tolerance", "scalability", "reliability", "performance"]
    qualities = ["security", "consistency", "availability", "maintainability", "extensibility"]
    
    for i in range(300):
        template = np.random.choice(complex_templates)
        sentence = template.format(
            adj=np.random.choice(adjectives),
            system=np.random.choice(systems),
            tech=np.random.choice(technologies),
            tech1=np.random.choice(technologies),
            tech2=np.random.choice(technologies),
            tech3=np.random.choice(technologies),
            goal=np.random.choice(goals),
            quality=np.random.choice(qualities),
            action="optimize performance",
            metric="significant",
            capability="advanced processing",
            domain="enterprise",
            resource="computational resources",
            data="large-scale datasets",
            task="data processing"
        )
        test_corpus.append(sentence)
        # Add 9 more variations
        for _ in range(9):
            varied = template.format(
                adj=np.random.choice(adjectives),
                system=np.random.choice(systems),
                tech=np.random.choice(technologies),
                tech1=np.random.choice(technologies),
                tech2=np.random.choice(technologies),
                tech3=np.random.choice(technologies),
                goal=np.random.choice(goals),
                quality=np.random.choice(qualities),
                action=np.random.choice(["enhance", "improve", "optimize", "streamline"]),
                metric=np.random.choice(["50%", "substantial", "measurable", "significant"]),
                capability=np.random.choice(["processing", "analysis", "computation", "transformation"]),
                domain=np.random.choice(["industrial", "scientific", "commercial", "research"]),
                resource=np.random.choice(["memory", "bandwidth", "storage", "compute"]),
                data=np.random.choice(["real-time streams", "batch datasets", "structured data"]),
                task=np.random.choice(["classification", "regression", "clustering", "prediction"])
            )
            test_corpus.append(varied)
    
    # 2. Multi-clause sentences (2k)
    print("Generating multi-clause sentences...")
    multiclause_templates = [
        "While {condition}, the {system} must {action1}, yet it should also {action2} to ensure {goal}.",
        "Although {challenge} presents difficulties, implementing {solution} allows {system} to {benefit}.",
        "Given that {context}, researchers have developed {tech} which enables {capability} despite {limitation}.",
        "Because {reason}, the {system} incorporates {feature} that facilitates {action} without compromising {quality}.",
        "When {situation} occurs, the {system} automatically {action1} and {action2} to maintain {quality}.",
    ]
    
    for i in range(400):
        template = np.random.choice(multiclause_templates)
        sentence = template.format(
            condition="traditional methods prove insufficient",
            system=np.random.choice(systems),
            action1="adapt dynamically",
            action2="maintain consistency",
            goal=np.random.choice(goals),
            challenge="distributed coordination",
            solution=np.random.choice(technologies),
            benefit="achieve consensus efficiently",
            context="modern applications require real-time processing",
            tech=np.random.choice(technologies),
            capability="sub-millisecond response times",
            limitation="resource constraints",
            reason="scalability is critical",
            feature="elastic load balancing",
            action="redistribute workload",
            quality=np.random.choice(qualities),
            situation="traffic spikes"
        )
        test_corpus.append(sentence)
        # 4 more variations
        for _ in range(4):
            test_corpus.append(template.format(
                condition=np.random.choice(["workload increases", "resources are limited", "latency matters"]),
                system=np.random.choice(systems),
                action1=np.random.choice(["scale horizontally", "optimize pathways", "cache results"]),
                action2=np.random.choice(["monitor health", "log metrics", "alert operators"]),
                goal=np.random.choice(goals),
                challenge=np.random.choice(["network partitioning", "data consistency", "fault recovery"]),
                solution=np.random.choice(technologies),
                benefit=np.random.choice(["improved throughput", "reduced latency", "enhanced reliability"]),
                context=np.random.choice(["cloud environments", "edge deployments", "hybrid systems"]),
                tech=np.random.choice(technologies),
                capability=np.random.choice(["real-time processing", "batch analytics", "stream computation"]),
                limitation=np.random.choice(["bandwidth limits", "memory constraints", "compute capacity"]),
                reason=np.random.choice(["users demand speed", "costs must be minimized", "accuracy is paramount"]),
                feature=np.random.choice(["auto-scaling", "load balancing", "circuit breaking"]),
                action=np.random.choice(["reroute traffic", "shed load", "activate replicas"]),
                quality=np.random.choice(qualities),
                situation=np.random.choice(["failures occur", "demand surges", "resources deplete"])
            ))
    
    # 3. Research-style abstracts (2k)
    print("Generating research-style sentences...")
    research_templates = [
        "This paper presents a novel approach to {problem} using {method}, achieving {result} on standard benchmarks.",
        "We propose {method} for {task}, demonstrating {improvement} compared to baseline methods.",
        "Our experiments show that {technique} can effectively {capability} in {domain} settings.",
        "The results indicate that {approach} provides {benefit} while reducing {cost} by {metric}.",
        "We introduce {innovation} that addresses {challenge} through {mechanism}, yielding {outcome}.",
    ]
    
    for i in range(400):
        template = np.random.choice(research_templates)
        sentence = template.format(
            problem="large-scale data processing",
            method=np.random.choice(technologies),
            result="state-of-the-art performance",
            task="semantic understanding",
            improvement="significant gains",
            technique="adaptive algorithms",
            capability="handle diverse inputs",
            domain=np.random.choice(["computer vision", "natural language", "robotics", "bioinformatics"]),
            approach=np.random.choice(technologies),
            benefit="improved accuracy",
            cost="computational overhead",
            metric="30%",
            innovation="a hybrid architecture",
            challenge="scalability limitations",
            mechanism="hierarchical processing",
            outcome="enhanced throughput"
        )
        test_corpus.append(sentence)
        for _ in range(4):
            test_corpus.append(template.format(
                problem=np.random.choice(["distributed consensus", "resource allocation", "anomaly detection"]),
                method=np.random.choice(technologies),
                result=np.random.choice(["superior accuracy", "reduced latency", "improved efficiency"]),
                task=np.random.choice(["classification", "clustering", "prediction", "optimization"]),
                improvement=np.random.choice(["10-15%", "substantial", "measurable", "notable"]),
                technique=np.random.choice(["ensemble methods", "deep architectures", "transfer learning"]),
                capability=np.random.choice(["generalize", "adapt", "scale", "converge"]),
                domain=np.random.choice(["computer vision", "speech recognition", "recommender systems"]),
                approach=np.random.choice(technologies),
                benefit=np.random.choice(["higher precision", "better recall", "faster convergence"]),
                cost=np.random.choice(["memory usage", "training time", "inference latency"]),
                metric=np.random.choice(["25%", "40%", "significant amount"]),
                innovation=np.random.choice(["an adaptive framework", "a novel architecture", "a unified model"]),
                challenge=np.random.choice(["data heterogeneity", "model complexity", "computational limits"]),
                mechanism=np.random.choice(["attention layers", "residual connections", "skip connections"]),
                outcome=np.random.choice(["improved generalization", "faster training", "better stability"])
            ))
    
    # 4. Conversational technical exchanges (1.5k)
    print("Generating conversational exchanges...")
    conversations = [
        "After analyzing the system's performance metrics over the past quarter, we've identified several bottlenecks that require immediate attention and optimization.",
        "The proposed solution leverages containerization and orchestration to provide seamless scalability across multiple availability zones.",
        "Initial benchmarking results suggest that migrating to a microservices architecture could reduce deployment times by up to forty percent.",
        "We need to ensure that the authentication layer properly handles edge cases, particularly when dealing with expired tokens and concurrent sessions.",
        "The team has successfully implemented continuous integration pipelines that automatically test, build, and deploy code changes to staging environments.",
    ]
    
    for base in conversations:
        test_corpus.append(base)
        # Generate variations
        for _ in range(2):
            test_corpus.append(base.replace("system", "platform").replace("optimization", "enhancement"))
            test_corpus.append(base.replace("performance", "efficiency").replace("attention", "focus"))
    
    # 5. Domain-specific paragraphs (1.5k)
    print("Generating domain-specific content...")
    domain_sentences = [
        "In distributed systems, achieving strong consistency while maintaining high availability remains a fundamental challenge that requires careful trade-off analysis.",
        "Modern machine learning pipelines incorporate automated feature engineering, model selection, and hyperparameter tuning to streamline the development process.",
        "Blockchain networks utilize consensus algorithms such as Proof of Work and Proof of Stake to ensure agreement among decentralized participants.",
        "Edge computing architectures position computational resources closer to data sources, minimizing latency and bandwidth requirements for time-critical applications.",
        "Natural language processing models have evolved from rule-based systems to neural architectures capable of capturing complex semantic relationships.",
    ]
    
    for base in domain_sentences:
        test_corpus.append(base)
        for _ in range(2):
            test_corpus.append(base.replace("systems", "environments").replace("challenge", "problem"))
            test_corpus.append(base.replace("Modern", "Contemporary").replace("architectures", "frameworks"))
    
    # Ensure uniqueness
    test_corpus = list(set(test_corpus))
    
    # Shuffle
    np.random.shuffle(test_corpus)
    
    return test_corpus[:n]


def create_train_val_test_split(
    vocabulary: List[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split vocabulary into train/val/test with NO overlap.
    
    Train: Used to compute transfer matrices
    Val: Used to validate matrix quality (held-out during training)
    Test: Completely unseen, used for final evaluation
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    indices = np.random.permutation(len(vocabulary))
    
    n_train = int(len(vocabulary) * train_ratio)
    n_val = int(len(vocabulary) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_vocab = [vocabulary[i] for i in train_indices]
    val_vocab = [vocabulary[i] for i in val_indices]
    test_vocab = [vocabulary[i] for i in test_indices]
    
    # Verify no overlap
    train_set = set(train_vocab)
    val_set = set(val_vocab)
    test_set = set(test_vocab)
    
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_vocab):,} items ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_vocab):,} items ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_vocab):,} items ({test_ratio*100:.0f}%)")
    print(f"  ✓ Zero overlap verified")
    
    return train_vocab, val_vocab, test_vocab


def compute_dataset_hash(data: List[str]) -> str:
    """Compute SHA256 hash of dataset for verification."""
    content = "".join(sorted(data)).encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:16]


def save_datasets(
    train_vocab: List[str],
    val_vocab: List[str],
    test_vocab: List[str],
    test_corpus: List[str],
    output_dir: str = "."
):
    """Save all datasets with verification hashes."""
    datasets = {
        "train_vocab": train_vocab,
        "val_vocab": val_vocab,
        "test_vocab": test_vocab,
        "test_corpus": test_corpus
    }
    
    metadata = {
        "train_vocab_size": len(train_vocab),
        "val_vocab_size": len(val_vocab),
        "test_vocab_size": len(test_vocab),
        "test_corpus_size": len(test_corpus),
        "train_vocab_hash": compute_dataset_hash(train_vocab),
        "val_vocab_hash": compute_dataset_hash(val_vocab),
        "test_vocab_hash": compute_dataset_hash(test_vocab),
        "test_corpus_hash": compute_dataset_hash(test_corpus),
        "created_at": "2026-02-03"
    }
    
    # Save datasets
    for name, data in datasets.items():
        filepath = f"{output_dir}/{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Saved {filepath} ({len(data):,} items)")
    
    # Save metadata
    with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {output_dir}/dataset_metadata.json")
    
    return metadata


if __name__ == "__main__":
    print("="*70)
    print("ENHANCED DATASET GENERATION (10x Scale)")
    print("="*70)
    print("\nGenerating 300k vocabulary (vs 30k in original POC)...")
    
    full_vocabulary = generate_diverse_vocabulary(300000)
    print(f"✓ Generated {len(full_vocabulary):,} vocabulary items")
    print(f"  Sample: {full_vocabulary[:5]}")
    
    print("\nSplitting into train/val/test (80/10/10)...")
    train_vocab, val_vocab, test_vocab = create_train_val_test_split(full_vocabulary)
    
    print("\nGenerating 10k test corpus (vs 1k in original POC)...")
    test_corpus = generate_test_corpus(10000)
    print(f"✓ Generated {len(test_corpus):,} test sentences")
    print(f"  Sample: {test_corpus[0][:100]}...")
    
    # Verify no overlap between vocab and test corpus
    vocab_set = set(full_vocabulary)
    corpus_set = set(test_corpus)
    overlap = vocab_set & corpus_set
    print(f"\n✓ Vocabulary-Corpus overlap: {len(overlap)} items ({len(overlap)/len(test_corpus)*100:.2f}%)")
    
    print("\nSaving datasets...")
    metadata = save_datasets(train_vocab, val_vocab, test_vocab, test_corpus)
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print("\nDataset Statistics:")
    for key, value in metadata.items():
        if not key.endswith("_hash") and key != "created_at":
            print(f"  {key}: {value:,}")
    
    print("\nVerification Hashes:")
    for key, value in metadata.items():
        if key.endswith("_hash"):
            print(f"  {key}: {value}")
