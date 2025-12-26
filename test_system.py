#!/usr/bin/env python3
"""Test script to verify the protein structure prediction system works."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.models.protein_models import create_model
        from src.data.dataset import SyntheticProteinDataset
        from src.utils.core import set_seed, get_device
        from src.utils.protein import encode_sequence, validate_sequence
        from src.losses.metrics import ProteinMetrics
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from src.models.protein_models import create_model
        
        # Test BiLSTM
        model = create_model("bilstm", vocab_size=20, num_classes=3)
        print(f"✅ BiLSTM model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test Transformer
        model = create_model("transformer", vocab_size=20, num_classes=3)
        print(f"✅ Transformer model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test CNN-LSTM
        model = create_model("cnnlstm", vocab_size=20, num_classes=3)
        print(f"✅ CNN-LSTM model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    
    try:
        from src.data.dataset import SyntheticProteinDataset
        
        dataset = SyntheticProteinDataset(num_samples=10, min_length=10, max_length=20)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test getting an item
        item = dataset[0]
        print(f"✅ Dataset item has keys: {list(item.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from src.utils.core import set_seed, get_device
        from src.utils.protein import encode_sequence, validate_sequence, get_sequence_statistics
        
        # Test seeding
        set_seed(42)
        print("✅ Random seeding works")
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection works: {device}")
        
        # Test sequence encoding
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        encoded = encode_sequence(sequence)
        print(f"✅ Sequence encoding works: {encoded.shape}")
        
        # Test sequence validation
        is_valid = validate_sequence(sequence)
        print(f"✅ Sequence validation works: {is_valid}")
        
        # Test sequence statistics
        stats = get_sequence_statistics(sequence)
        print(f"✅ Sequence statistics works: length={stats['length']}")
        
        return True
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test model forward pass."""
    print("\nTesting forward pass...")
    
    try:
        import torch
        from src.models.protein_models import create_model
        
        # Create model
        model = create_model("bilstm", vocab_size=20, num_classes=3)
        model.eval()
        
        # Create dummy input
        batch = {
            "sequence": torch.randint(0, 20, (2, 50)),
            "attention_mask": torch.ones(2, 50, dtype=torch.bool),
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(batch)
        
        print(f"✅ Forward pass works: output shape {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧬 Protein Structure Prediction System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_dataset,
        test_utilities,
        test_forward_pass,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Start demo: streamlit run demo/app.py")
        print("3. Check notebooks: jupyter notebook notebooks/")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
