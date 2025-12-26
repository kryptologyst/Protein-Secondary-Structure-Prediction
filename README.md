# Protein Secondary Structure Prediction

A research-grade deep learning system for predicting protein secondary structure from amino acid sequences. This project implements multiple neural network architectures (BiLSTM, Transformer, CNN-LSTM) for accurate secondary structure classification.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION TOOL ONLY**

- **NOT INTENDED FOR CLINICAL OR DIAGNOSTIC USE**
- **NOT FOR MEDICAL DECISION-MAKING**
- **RESEARCH AND EDUCATIONAL PURPOSES ONLY**
- **ALWAYS CONSULT QUALIFIED HEALTHCARE PROFESSIONALS FOR MEDICAL ADVICE**

## Features

- **Multiple Model Architectures**: BiLSTM, Transformer, and CNN-LSTM models
- **Comprehensive Evaluation**: Q3 accuracy, SOV score, per-class metrics, confusion matrices
- **Interactive Demo**: Streamlit web application for real-time predictions
- **Modern ML Stack**: PyTorch 2.x, Lightning, proper device handling (CUDA/MPS/CPU)
- **Reproducible Research**: Deterministic seeding, comprehensive logging, structured configs
- **Production Ready**: Type hints, documentation, testing, CI/CD setup

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Protein-Secondary-Structure-Prediction.git
cd Protein-Secondary-Structure-Prediction

# Install dependencies
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,serve]"
```

### Training a Model

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom.yaml --model-type transformer --num-epochs 100

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoints/best_model.pt
```

### Running the Demo

```bash
# Start the Streamlit demo
streamlit run demo/app.py

# Or with custom port
streamlit run demo/app.py --server.port 8502
```

## Project Structure

```
protein-structure-prediction/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   └── protein_models.py     # BiLSTM, Transformer, CNN-LSTM
│   ├── data/                     # Data handling
│   │   └── dataset.py           # Dataset classes and loaders
│   ├── losses/                   # Loss functions and metrics
│   │   └── metrics.py           # Q3, SOV, per-class metrics
│   ├── train/                    # Training utilities
│   │   └── trainer.py           # Training loop and checkpointing
│   ├── eval/                     # Evaluation utilities
│   │   └── evaluator.py          # Comprehensive evaluation
│   └── utils/                    # Utilities
│       ├── core.py              # Core utilities (device, seeding)
│       └── protein.py            # Protein sequence utilities
├── configs/                      # Configuration files
│   └── default.yaml             # Default training configuration
├── scripts/                      # Training and evaluation scripts
│   └── train.py                 # Main training script
├── demo/                        # Interactive demo
│   └── app.py                   # Streamlit application
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks for analysis
├── assets/                      # Generated plots and visualizations
├── data/                        # Dataset storage (not in repo)
├── outputs/                     # Training outputs and checkpoints
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Model Architectures

### BiLSTM Model
- Bidirectional LSTM for sequence modeling
- Embedding layer for amino acid representation
- Optional amino acid property features
- Configurable hidden dimensions and layers

### Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for sequence order
- Layer normalization and residual connections
- Configurable attention heads and layers

### CNN-LSTM Model
- CNN layers for local pattern extraction
- LSTM for long-range dependencies
- Multiple kernel sizes for different patterns
- Hybrid architecture benefits

## Evaluation Metrics

- **Q3 Accuracy**: Overall 3-state accuracy (Helix/Sheet/Coil)
- **SOV Score**: Segment Overlap score for structure segments
- **Per-Class Metrics**: Precision, recall, F1-score for each structure type
- **Confusion Matrix**: Detailed classification breakdown
- **Macro/Weighted Averages**: Balanced performance assessment

## Configuration

The system uses YAML configuration files for easy experimentation:

```yaml
# Model configuration
model:
  type: "bilstm"  # bilstm, transformer, cnnlstm
  vocab_size: 20
  num_classes: 3
  include_features: false

# Training configuration
training:
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-5
  use_early_stopping: true
  early_stopping_patience: 10

# Data configuration
data:
  dataset_type: "synthetic"  # synthetic, cb513
  batch_size: 32
  splits:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
```

## Usage Examples

### Basic Training

```python
from src.data.dataset import SyntheticProteinDataset, create_data_loaders
from src.models.protein_models import create_model
from src.train.trainer import create_trainer

# Create dataset
dataset = SyntheticProteinDataset(num_samples=1000)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(dataset)

# Create model
model = create_model("bilstm", vocab_size=20, num_classes=3)

# Create trainer
trainer = create_trainer(model, train_loader, val_loader, config)

# Train model
history = trainer.train(num_epochs=50)
```

### Model Evaluation

```python
from src.eval.evaluator import evaluate_model

# Evaluate model
metrics = evaluate_model(model, test_loader, save_dir="results/")

print(f"Q3 Accuracy: {metrics['q3_accuracy']:.4f}")
print(f"SOV Score: {metrics['sov_score']:.4f}")
```

### Sequence Analysis

```python
from src.utils.protein import encode_sequence, get_sequence_statistics

# Analyze protein sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
stats = get_sequence_statistics(sequence)

print(f"Length: {stats['length']}")
print(f"Hydrophobic fraction: {stats['hydrophobic_fraction']:.3f}")
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type hints** throughout the codebase
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit hooks** for automated quality checks

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Type check
mypy src/

# Run tests
pytest tests/
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::test_bilstm_model
```

## Performance

Typical performance on synthetic data:

| Model | Q3 Accuracy | SOV Score | Training Time |
|-------|-------------|-----------|--------------|
| BiLSTM | ~0.75 | ~0.65 | ~2 min |
| Transformer | ~0.78 | ~0.68 | ~5 min |
| CNN-LSTM | ~0.77 | ~0.67 | ~3 min |

*Results may vary based on data and hyperparameters*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{protein_structure_prediction,
  title={Protein Secondary Structure Prediction with Deep Learning},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Protein-Secondary-Structure-Prediction}
}
```

## Acknowledgments

- Built with PyTorch and PyTorch Lightning
- Inspired by AlphaFold and other protein structure prediction methods
- Uses standard bioinformatics datasets and evaluation metrics

## Support

For questions and support:

- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example notebooks in `notebooks/`

---

**Remember: This tool is for research and educational purposes only. Not for clinical use.**
# Protein-Secondary-Structure-Prediction
