"""
AECP Default Vocabulary

Provides default vocabulary for calibration when custom vocabulary is not provided.
This vocabulary is designed to cover diverse semantic concepts.
"""

from typing import List, Tuple
import hashlib


def get_default_vocabulary(
    train_size: int = 5000,
    val_size: int = 500
) -> Tuple[List[str], List[str]]:
    """
    Get default training and validation vocabulary.
    
    The vocabulary is designed to cover:
    - Common words and phrases
    - Technical terminology
    - Domain-specific concepts
    - Varied sentence structures
    
    Args:
        train_size: Number of training vocabulary items
        val_size: Number of validation vocabulary items
        
    Returns:
        Tuple of (training_vocabulary, validation_vocabulary)
    """
    # Generate comprehensive vocabulary
    full_vocab = _generate_diverse_vocabulary()
    
    # Ensure we have enough items
    if len(full_vocab) < train_size + val_size:
        # Pad with generated items
        while len(full_vocab) < train_size + val_size:
            idx = len(full_vocab)
            full_vocab.append(f"concept_{idx}")
            full_vocab.append(f"term_{idx}")
    
    # Split into train and validation
    train_vocab = full_vocab[:train_size]
    val_vocab = full_vocab[train_size:train_size + val_size]
    
    return train_vocab, val_vocab


def _generate_diverse_vocabulary() -> List[str]:
    """Generate a diverse vocabulary covering multiple domains."""
    vocabulary = []
    
    # Core words
    core_words = [
        # Actions
        "analyze", "compute", "process", "optimize", "implement", "design",
        "test", "validate", "measure", "evaluate", "compare", "synthesize",
        "integrate", "deploy", "monitor", "debug", "refactor", "document",
        "create", "build", "develop", "organize", "manage", "execute",
        "transform", "convert", "translate", "interpret", "explain", "demonstrate",
        
        # Technical nouns
        "algorithm", "function", "method", "procedure", "system", "model",
        "framework", "architecture", "structure", "component", "module", "interface",
        "protocol", "database", "repository", "storage", "memory", "cache",
        "network", "server", "client", "endpoint", "gateway", "application",
        "software", "hardware", "platform", "environment", "infrastructure",
        "security", "authentication", "authorization", "encryption", "token",
        
        # Adjectives
        "efficient", "effective", "robust", "reliable", "scalable", "maintainable",
        "secure", "stable", "consistent", "comprehensive", "accurate", "precise",
        "fast", "quick", "rapid", "slow", "gradual", "incremental",
        "simple", "complex", "sophisticated", "advanced", "basic", "fundamental",
        "modern", "contemporary", "current", "recent", "new", "innovative",
        
        # Domain terms
        "machine", "learning", "artificial", "intelligence", "neural", "deep",
        "data", "science", "analytics", "statistics", "probability", "inference",
        "training", "testing", "validation", "performance", "accuracy", "precision",
        "optimization", "gradient", "descent", "backpropagation", "tensor", "matrix",
    ]
    
    # Add core words and variations
    for word in core_words:
        vocabulary.append(word)
        vocabulary.append(word.capitalize())
        vocabulary.append(f"the {word}")
        vocabulary.append(f"{word}s")
    
    # Two-word phrases
    phrase_templates = [
        "{} system", "{} model", "{} algorithm", "{} framework",
        "{} architecture", "{} implementation", "{} optimization",
        "advanced {}", "modern {}", "efficient {}", "robust {}",
        "{} analysis", "{} design", "{} development", "{} testing",
    ]
    
    for template in phrase_templates:
        for word in core_words[:50]:
            vocabulary.append(template.format(word))
    
    # Short sentences
    sentence_templates = [
        "The {} is important.",
        "We need to {} the system.",
        "This {} enables better performance.",
        "The {} provides reliable results.",
        "A {} can improve efficiency.",
        "The system uses {} effectively.",
        "We should optimize the {}.",
        "The {} enhances quality.",
    ]
    
    for template in sentence_templates:
        for word in core_words[:30]:
            vocabulary.append(template.format(word))
    
    # Domain-specific terms
    domain_terms = [
        # ML/AI
        "convolutional neural network", "recurrent neural network",
        "transformer architecture", "attention mechanism", "gradient descent",
        "batch normalization", "dropout regularization", "activation function",
        "cross-entropy loss", "mean squared error", "feature extraction",
        
        # Systems
        "distributed system", "microservices architecture", "load balancer",
        "container orchestration", "message queue", "event streaming",
        "database sharding", "cache coherence", "memory hierarchy",
        
        # Security
        "public key cryptography", "digital signature", "hash function",
        "access control", "authentication protocol", "encryption algorithm",
        
        # Data
        "data warehouse", "data pipeline", "stream processing",
        "batch processing", "data quality", "data governance",
    ]
    
    for term in domain_terms:
        vocabulary.append(term)
        vocabulary.append(f"{term} system")
        vocabulary.append(f"optimized {term}")
    
    # Conversational phrases
    conversations = [
        "How are you?", "I'm doing well", "Thanks for asking",
        "Can you help me?", "Sure, what do you need?",
        "That makes sense", "I understand now", "Could you explain more?",
        "Let me think about that", "Here's what I found", "Does this help?",
        "I'm not sure", "Let me check", "According to the data",
        "In my opinion", "Based on experience", "From what I know",
        "That's interesting", "Tell me more", "What do you mean?",
    ]
    
    vocabulary.extend(conversations)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vocab = []
    for item in vocabulary:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_vocab.append(item)
    
    return unique_vocab


def compute_vocabulary_hash(vocabulary: List[str]) -> str:
    """
    Compute a hash of the vocabulary for verification.
    
    Args:
        vocabulary: List of vocabulary items
        
    Returns:
        SHA256 hash of the vocabulary
    """
    content = "".join(sorted(vocabulary)).encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:16]


def validate_vocabulary(
    vocabulary: List[str],
    min_size: int = 100,
    max_item_length: int = 1000
) -> Tuple[bool, List[str]]:
    """
    Validate a vocabulary for use in calibration.
    
    Args:
        vocabulary: List of vocabulary items
        min_size: Minimum required vocabulary size
        max_item_length: Maximum length of individual items
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not vocabulary:
        errors.append("Vocabulary is empty")
        return False, errors
    
    if len(vocabulary) < min_size:
        errors.append(f"Vocabulary too small: {len(vocabulary)} < {min_size}")
    
    # Check for duplicates
    unique = set(vocabulary)
    if len(unique) < len(vocabulary):
        dup_count = len(vocabulary) - len(unique)
        errors.append(f"Vocabulary contains {dup_count} duplicates")
    
    # Check item lengths
    for i, item in enumerate(vocabulary):
        if not isinstance(item, str):
            errors.append(f"Item {i} is not a string: {type(item)}")
        elif len(item) == 0:
            errors.append(f"Item {i} is empty")
        elif len(item) > max_item_length:
            errors.append(f"Item {i} exceeds max length: {len(item)} > {max_item_length}")
    
    return len(errors) == 0, errors
