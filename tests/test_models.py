"""Tests for protein structure prediction models."""

import pytest
import torch
import numpy as np

from src.models.protein_models import (
    BiLSTMProteinModel,
    TransformerProteinModel,
    CNNLSTMProteinModel,
    create_model,
)
from src.data.dataset import SyntheticProteinDataset
from src.utils.protein import encode_sequence, encode_structure
from src.losses.metrics import ProteinMetrics


class TestProteinModels:
    """Test protein structure prediction models."""
    
    def test_bilstm_model(self):
        """Test BiLSTM model forward pass."""
        model = BiLSTMProteinModel(
            vocab_size=20,
            embed_dim=64,
            hidden_dim=128,
            num_classes=3,
        )
        
        # Create dummy batch
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "attention_mask": torch.ones(2, 50, dtype=torch.bool),
        }
        
        output = model(batch)
        assert output.shape == (2, 50, 3)
    
    def test_transformer_model(self):
        """Test Transformer model forward pass."""
        model = TransformerProteinModel(
            vocab_size=20,
            embed_dim=128,
            num_classes=3,
            max_length=100,
        )
        
        # Create dummy batch
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "attention_mask": torch.ones(2, 50, dtype=torch.bool),
        }
        
        output = model(batch)
        assert output.shape == (2, 50, 3)
    
    def test_cnnlstm_model(self):
        """Test CNN-LSTM model forward pass."""
        model = CNNLSTMProteinModel(
            vocab_size=20,
            embed_dim=64,
            num_classes=3,
        )
        
        # Create dummy batch
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "attention_mask": torch.ones(2, 50, dtype=torch.bool),
        }
        
        output = model(batch)
        assert output.shape == (2, 50, 3)
    
    def test_model_with_features(self):
        """Test model with amino acid features."""
        model = BiLSTMProteinModel(
            vocab_size=20,
            embed_dim=64,
            hidden_dim=128,
            num_classes=3,
            include_features=True,
        )
        
        # Create dummy batch with features
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "attention_mask": torch.ones(2, 50, dtype=torch.bool),
            "features": torch.randn(2, 50, 4),
        }
        
        output = model(batch)
        assert output.shape == (2, 50, 3)
    
    def test_create_model_function(self):
        """Test model creation function."""
        # Test BiLSTM
        model = create_model("bilstm", vocab_size=20, num_classes=3)
        assert isinstance(model, BiLSTMProteinModel)
        
        # Test Transformer
        model = create_model("transformer", vocab_size=20, num_classes=3)
        assert isinstance(model, TransformerProteinModel)
        
        # Test CNN-LSTM
        model = create_model("cnnlstm", vocab_size=20, num_classes=3)
        assert isinstance(model, CNNLSTMProteinModel)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model("invalid", vocab_size=20, num_classes=3)


class TestDataset:
    """Test dataset classes."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticProteinDataset(num_samples=10, min_length=10, max_length=20)
        
        assert len(dataset) == 10
        
        # Test getting an item
        item = dataset[0]
        assert "sequence" in item
        assert "structure" in item
        assert "attention_mask" in item
        assert "length" in item
        
        # Check shapes
        assert item["sequence"].shape[0] == 100  # max_length
        assert item["structure"].shape[0] == 100
        assert item["attention_mask"].shape[0] == 100
    
    def test_synthetic_dataset_with_features(self):
        """Test synthetic dataset with features."""
        dataset = SyntheticProteinDataset(
            num_samples=10,
            min_length=10,
            max_length=20,
            include_features=True,
        )
        
        item = dataset[0]
        assert "features" in item
        assert item["features"].shape == (100, 4)  # max_length, feature_dim


class TestProteinUtils:
    """Test protein utility functions."""
    
    def test_encode_sequence(self):
        """Test sequence encoding."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        encoded = encode_sequence(sequence)
        
        assert encoded.shape == (20,)
        assert encoded.dtype == torch.long
    
    def test_encode_structure(self):
        """Test structure encoding."""
        structure = "HHHHEEEECCCCHHHH"
        encoded = encode_structure(structure)
        
        assert encoded.shape == (16,)
        assert encoded.dtype == torch.long
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        from src.utils.protein import validate_sequence
        
        # Valid sequence
        assert validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        
        # Invalid sequence
        assert not validate_sequence("ACDEFGHIKLMNPQRSTVWYX")  # X is not valid
    
    def test_structure_validation(self):
        """Test structure validation."""
        from src.utils.protein import validate_structure
        
        # Valid structure
        assert validate_structure("HHHHEEEECCCCHHHH")
        
        # Invalid structure
        assert not validate_structure("HHHHEEEECCCCHHHHX")  # X is not valid


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_protein_metrics(self):
        """Test protein metrics calculation."""
        metrics = ProteinMetrics()
        
        # Create dummy predictions and targets
        predictions = torch.tensor([[0, 1, 2, 0, 1]])  # H, E, C, H, E
        targets = torch.tensor([[0, 1, 2, 0, 1]])       # H, E, C, H, E
        attention_mask = torch.tensor([[True, True, True, True, True]])
        
        # Test Q3 accuracy
        accuracy = metrics.compute_q3_accuracy(predictions, targets, attention_mask)
        assert accuracy == 1.0  # Perfect accuracy
        
        # Test per-class accuracy
        per_class_acc = metrics.compute_per_class_accuracy(predictions, targets, attention_mask)
        assert len(per_class_acc) == 3
        assert all(acc == 1.0 for acc in per_class_acc.values())
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        metrics = ProteinMetrics()
        
        predictions = torch.tensor([[0, 1, 2, 0, 1]])
        targets = torch.tensor([[0, 1, 2, 0, 1]])
        attention_mask = torch.tensor([[True, True, True, True, True]])
        
        cm = metrics.compute_confusion_matrix(predictions, targets, attention_mask)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # Two H predictions
        assert cm[1, 1] == 2  # Two E predictions
        assert cm[2, 2] == 1  # One C prediction


if __name__ == "__main__":
    pytest.main([__file__])
