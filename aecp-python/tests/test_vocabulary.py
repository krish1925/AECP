"""
Tests for AECP vocabulary utilities.
"""

import pytest

from aecp.vocabulary import (
    get_default_vocabulary,
    compute_vocabulary_hash,
    validate_vocabulary,
)


class TestGetDefaultVocabulary:
    """Tests for get_default_vocabulary function."""
    
    def test_default_sizes(self):
        """Test default vocabulary sizes."""
        train, val = get_default_vocabulary()
        
        assert len(train) == 5000
        assert len(val) == 500
    
    def test_custom_sizes(self):
        """Test custom vocabulary sizes."""
        train, val = get_default_vocabulary(train_size=1000, val_size=100)
        
        assert len(train) == 1000
        assert len(val) == 100
    
    def test_no_overlap(self):
        """Training and validation should not overlap."""
        train, val = get_default_vocabulary()
        
        train_set = set(train)
        val_set = set(val)
        
        assert len(train_set & val_set) == 0
    
    def test_contains_diverse_content(self):
        """Vocabulary should contain diverse content."""
        train, val = get_default_vocabulary()
        
        # Should have single words
        assert any(len(item.split()) == 1 for item in train)
        
        # Should have phrases
        assert any(len(item.split()) > 1 for item in train)


class TestComputeVocabularyHash:
    """Tests for compute_vocabulary_hash function."""
    
    def test_deterministic(self):
        """Same vocabulary should produce same hash."""
        vocab = ["hello", "world", "test"]
        
        hash1 = compute_vocabulary_hash(vocab)
        hash2 = compute_vocabulary_hash(vocab)
        
        assert hash1 == hash2
    
    def test_different_vocab_different_hash(self):
        """Different vocabularies should produce different hashes."""
        vocab1 = ["hello", "world"]
        vocab2 = ["hello", "test"]
        
        hash1 = compute_vocabulary_hash(vocab1)
        hash2 = compute_vocabulary_hash(vocab2)
        
        assert hash1 != hash2
    
    def test_order_independent(self):
        """Hash should be order-independent."""
        vocab1 = ["hello", "world", "test"]
        vocab2 = ["test", "hello", "world"]
        
        hash1 = compute_vocabulary_hash(vocab1)
        hash2 = compute_vocabulary_hash(vocab2)
        
        assert hash1 == hash2


class TestValidateVocabulary:
    """Tests for validate_vocabulary function."""
    
    def test_valid_vocabulary(self):
        """Valid vocabulary should pass validation."""
        # Create unique vocabulary items
        vocab = [f"word_{i}" for i in range(150)]
        
        is_valid, errors = validate_vocabulary(vocab)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_empty_vocabulary(self):
        """Empty vocabulary should fail."""
        is_valid, errors = validate_vocabulary([])
        
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)
    
    def test_too_small(self):
        """Too small vocabulary should fail."""
        vocab = ["hello", "world"]
        
        is_valid, errors = validate_vocabulary(vocab, min_size=100)
        
        assert not is_valid
        assert any("small" in e.lower() for e in errors)
    
    def test_with_duplicates(self):
        """Vocabulary with duplicates should report them."""
        vocab = ["hello", "hello", "world"] * 50
        
        is_valid, errors = validate_vocabulary(vocab)
        
        assert any("duplicate" in e.lower() for e in errors)
    
    def test_with_empty_item(self):
        """Vocabulary with empty item should fail."""
        vocab = ["hello", "", "world"] * 50
        
        is_valid, errors = validate_vocabulary(vocab)
        
        assert not is_valid
        assert any("empty" in e.lower() for e in errors)
    
    def test_with_too_long_item(self):
        """Vocabulary with too long item should fail."""
        vocab = ["hello", "a" * 2000, "world"] * 50
        
        is_valid, errors = validate_vocabulary(vocab, max_item_length=1000)
        
        assert not is_valid
        assert any("length" in e.lower() for e in errors)
