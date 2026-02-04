"""
Vocabulary and Test Corpus Loader

Provides common English vocabulary and diverse test sentences for experiments.
"""

import json
from typing import List
import numpy as np


def load_common_words(n: int = 30000) -> List[str]:
    """
    Load common English words/tokens as vocabulary.
    
    For POC, we'll generate a diverse vocabulary including:
    - Common words
    - Phrases
    - Technical terms
    - Conversational snippets
    
    Args:
        n: Number of vocabulary items (target ~30k)
        
    Returns:
        List of vocabulary strings
    """
    vocabulary = []
    
    # 1. Single common words (10k)
    common_words = [
        "hello", "world", "computer", "science", "technology", "information",
        "data", "algorithm", "network", "system", "process", "function",
        "variable", "memory", "storage", "database", "server", "client",
        "user", "interface", "design", "development", "software", "hardware",
        "programming", "language", "python", "java", "javascript", "code",
        "debug", "error", "exception", "class", "object", "method",
        "artificial", "intelligence", "machine", "learning", "neural", "network",
        "model", "training", "testing", "validation", "accuracy", "precision",
        "business", "management", "strategy", "planning", "execution", "analysis",
        "research", "experiment", "hypothesis", "theory", "evidence", "conclusion",
        "communication", "message", "information", "transfer", "protocol", "format",
        "security", "privacy", "encryption", "authentication", "authorization", "access",
        "performance", "optimization", "efficiency", "scalability", "reliability", "availability",
        "customer", "service", "product", "quality", "value", "satisfaction",
        "team", "collaboration", "project", "task", "goal", "objective",
        "time", "space", "resource", "constraint", "requirement", "specification",
        "architecture", "component", "module", "integration", "deployment", "maintenance",
        "innovation", "creativity", "problem", "solution", "approach", "methodology",
        "success", "failure", "risk", "opportunity", "challenge", "advantage",
        "knowledge", "skill", "experience", "expertise", "competence", "proficiency"
    ]
    
    # Expand with variations
    for word in common_words:
        vocabulary.append(word)
        vocabulary.append(word.capitalize())
        vocabulary.append(word.upper())
    
    # 2. Two-word phrases (5k)
    phrase_templates = [
        "{} {}", "the {}", "a {}", "an {}", "my {}", "your {}",
        "{} system", "{} data", "{} model", "{} algorithm",
        "good {}", "bad {}", "new {}", "old {}", "fast {}", "slow {}",
        "{} analysis", "{} process", "{} method", "{} approach"
    ]
    
    base_words = common_words[:100]
    for template in phrase_templates:
        for word in base_words[:50]:
            if "{}" in template and template.count("{}") == 1:
                vocabulary.append(template.format(word))
            elif template.count("{}") == 2:
                for word2 in base_words[:20]:
                    vocabulary.append(template.format(word, word2))
    
    # 3. Short sentences (5k)
    sentence_templates = [
        "The {} is {}.",
        "I like {}.",
        "This is a {}.",
        "How does {} work?",
        "Why is {} important?",
        "What is {}?",
        "{} helps with {}.",
        "{} requires {}.",
        "The best {} is {}.",
        "We need to {} the {}.",
    ]
    
    verbs = ["analyze", "process", "compute", "calculate", "optimize", "improve", 
             "design", "implement", "test", "validate", "verify", "measure"]
    adjectives = ["good", "better", "best", "fast", "efficient", "reliable", 
                  "secure", "scalable", "robust", "flexible", "simple", "complex"]
    
    for template in sentence_templates:
        count = template.count("{}")
        if count == 1:
            for word in base_words[:100]:
                vocabulary.append(template.format(word))
        elif count == 2:
            for w1 in base_words[:30]:
                for w2 in adjectives:
                    vocabulary.append(template.format(w1, w2))
                for w2 in verbs:
                    vocabulary.append(template.format(w1, w2))
    
    # 4. Domain-specific terms (5k)
    domains = {
        "AI/ML": ["neural network", "deep learning", "gradient descent", "backpropagation",
                  "convolutional layer", "recurrent network", "transformer model", "attention mechanism"],
        "Software": ["version control", "code review", "unit test", "integration test",
                     "continuous integration", "deployment pipeline", "microservice", "REST API"],
        "Data": ["data warehouse", "ETL pipeline", "data lake", "big data",
                 "real-time processing", "batch processing", "data quality", "data governance"],
        "Business": ["market research", "customer acquisition", "revenue growth", "cost reduction",
                     "competitive advantage", "value proposition", "business model", "strategic planning"]
    }
    
    for domain, terms in domains.items():
        for term in terms:
            vocabulary.append(term)
            vocabulary.append(f"{term} implementation")
            vocabulary.append(f"{term} strategy")
            vocabulary.append(f"optimizing {term}")
            vocabulary.append(f"improving {term}")
    
    # 5. Conversational snippets (5k)
    conversations = [
        "How are you?", "I'm doing well", "Thanks for asking",
        "Can you help me?", "Sure, what do you need?", "I need information about",
        "That makes sense", "I understand now", "Could you explain more?",
        "Let me think about that", "Here's what I found", "Does this help?",
        "I'm not sure", "Let me check", "According to the data",
        "In my opinion", "Based on experience", "From what I know",
        "That's interesting", "Tell me more", "What do you mean?",
        "I agree with you", "I see your point", "That's a good question"
    ]
    
    for conv in conversations:
        vocabulary.append(conv)
        vocabulary.append(conv.lower())
        vocabulary.append(f"{conv} about {base_words[0]}")
    
    # Deduplicate and limit to n items
    vocabulary = list(set(vocabulary))
    
    # If we don't have enough, pad with generated content
    while len(vocabulary) < n:
        idx = len(vocabulary)
        vocabulary.append(f"generated term {idx}")
        vocabulary.append(f"concept number {idx}")
        vocabulary.append(f"test phrase {idx}")
    
    return vocabulary[:n]


def load_test_corpus(n: int = 1000) -> List[str]:
    """
    Load diverse test sentences for evaluation.
    These should be DIFFERENT from the vocabulary.
    
    Args:
        n: Number of test sentences
        
    Returns:
        List of test sentences
    """
    test_corpus = []
    
    # 1. News-style sentences
    news_templates = [
        "Scientists have discovered a new method for {}.",
        "Recent studies show that {} is more effective than previously thought.",
        "Experts predict that {} will revolutionize the industry within five years.",
        "The latest research on {} reveals surprising insights about {}.",
        "A breakthrough in {} technology could lead to significant improvements in {}.",
    ]
    
    topics = [
        "machine learning", "quantum computing", "renewable energy", "biotechnology",
        "space exploration", "artificial intelligence", "climate modeling", "drug discovery",
        "materials science", "neural interfaces", "autonomous vehicles", "gene therapy"
    ]
    
    for template in news_templates:
        for topic in topics:
            if template.count("{}") == 1:
                test_corpus.append(template.format(topic))
            else:
                for topic2 in topics[:6]:
                    if topic != topic2:
                        test_corpus.append(template.format(topic, topic2))
    
    # 2. Technical descriptions
    tech_sentences = [
        "The algorithm iterates through the dataset and computes similarity scores for each pair of items.",
        "To optimize performance, we implemented a caching layer that reduces database queries by 80 percent.",
        "The neural network architecture consists of three convolutional layers followed by two fully connected layers.",
        "Data preprocessing includes normalization, tokenization, and removal of stop words.",
        "The system scales horizontally by distributing workload across multiple compute nodes.",
        "Error handling is implemented using a try-catch mechanism with custom exception classes.",
        "The API endpoint accepts JSON payloads and returns structured data in the same format.",
        "Authentication is performed using JWT tokens with a 24-hour expiration period.",
        "The database schema includes foreign key constraints to maintain referential integrity.",
        "Load balancing ensures even distribution of requests across available servers.",
    ]
    test_corpus.extend(tech_sentences)
    
    # 3. Conversational exchanges
    conversations = [
        "I've been working on this problem for hours and finally found a solution that works.",
        "Can you walk me through the steps needed to deploy this application to production?",
        "The main challenge we're facing is balancing performance with resource constraints.",
        "After analyzing the data, it became clear that we need to rethink our approach.",
        "What factors should we consider when choosing between these two alternatives?",
        "Based on user feedback, we've identified several areas for improvement in the interface.",
        "The team decided to postpone the release until we can address the critical bugs.",
        "Let's schedule a meeting to discuss the project timeline and resource allocation.",
        "I recommend we conduct a thorough code review before merging these changes.",
        "The stakeholders expressed concern about the security implications of this feature.",
    ]
    test_corpus.extend(conversations * 5)  # Repeat to get more samples
    
    # 4. Abstract concepts
    abstract = [
        "The relationship between cause and effect is often more complex than it initially appears.",
        "Understanding context is essential for making accurate interpretations of ambiguous information.",
        "Patterns emerge from repeated observation and analysis of similar phenomena across different domains.",
        "The balance between exploration and exploitation determines the efficiency of learning systems.",
        "Abstraction allows us to focus on essential features while ignoring irrelevant details.",
        "Feedback loops create dynamic systems where outputs influence subsequent inputs.",
        "Optimization involves finding the best solution within a space of possible alternatives.",
        "Uncertainty quantification helps assess the reliability of predictions and decisions.",
        "Hierarchical structures enable efficient organization and retrieval of information.",
        "The trade-off between complexity and interpretability affects model selection in practice.",
    ]
    test_corpus.extend(abstract * 3)
    
    # 5. Domain-specific content
    domains = [
        "The convolutional kernel slides across the input tensor, computing dot products at each position.",
        "Gradient descent updates parameters in the direction that minimizes the loss function.",
        "Attention mechanisms allow the model to focus on relevant parts of the input sequence.",
        "Regularization techniques like dropout prevent overfitting by randomly disabling neurons during training.",
        "The transformer architecture processes sequences in parallel using self-attention layers.",
        "Embeddings map discrete tokens to continuous vector representations in a learned semantic space.",
        "Cross-validation splits the dataset into folds to estimate generalization performance.",
        "Hyperparameter tuning searches the configuration space to find optimal model settings.",
        "The learning rate schedule adjusts the step size during optimization to improve convergence.",
        "Batch normalization stabilizes training by normalizing layer inputs across mini-batches.",
    ]
    test_corpus.extend(domains * 4)
    
    # 6. Varied sentence structures
    varied = [
        "While traditional methods work well in many cases, modern approaches offer significant advantages.",
        "Despite the challenges involved, the potential benefits make this worth pursuing.",
        "Not only does this improve efficiency, but it also reduces costs substantially.",
        "Given the constraints we're working with, this solution represents a reasonable compromise.",
        "Whether we choose option A or option B, we'll need to address the underlying issues.",
        "As technology continues to evolve, new opportunities and challenges will emerge.",
        "Unless we take action soon, the problem will become increasingly difficult to solve.",
        "In addition to the primary objective, we should also consider secondary goals.",
        "Rather than focusing solely on performance, we must also prioritize reliability.",
        "By combining multiple approaches, we can achieve better results than any single method.",
    ]
    test_corpus.extend(varied * 5)
    
    # Ensure uniqueness and limit
    test_corpus = list(set(test_corpus))
    
    # Shuffle to mix different types
    np.random.seed(42)
    indices = np.random.permutation(len(test_corpus))
    test_corpus = [test_corpus[i] for i in indices]
    
    return test_corpus[:n]


def save_vocabulary(vocabulary: List[str], filepath: str = "vocabulary.json"):
    """Save vocabulary to file."""
    with open(filepath, 'w') as f:
        json.dump(vocabulary, f, indent=2)


def load_vocabulary(filepath: str = "vocabulary.json") -> List[str]:
    """Load vocabulary from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test the loaders
    print("Loading vocabulary...")
    vocab = load_common_words(30000)
    print(f"Generated {len(vocab)} vocabulary items")
    print(f"Sample: {vocab[:10]}")
    
    print("\nLoading test corpus...")
    corpus = load_test_corpus(1000)
    print(f"Generated {len(corpus)} test sentences")
    print(f"Sample: {corpus[:5]}")
    
    # Check overlap
    overlap = set(vocab) & set(corpus)
    print(f"\nOverlap between vocab and test corpus: {len(overlap)} items")
