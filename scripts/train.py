#!/usr/bin/env python3
"""Main training script for protein structure prediction."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import SyntheticProteinDataset, create_data_loaders
from src.models.protein_models import create_model
from src.train.trainer import create_trainer
from src.eval.evaluator import evaluate_model
from src.utils.core import set_seed, setup_logging, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train protein secondary structure prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bilstm", "transformer", "cnnlstm"],
        help="Override model type",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, do not train",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for evaluation",
    )
    
    return parser.parse_args()


def create_dataset(config: DictConfig) -> SyntheticProteinDataset:
    """Create dataset based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Dataset instance.
    """
    if config.data.dataset_type == "synthetic":
        return SyntheticProteinDataset(
            num_samples=config.data.synthetic.num_samples,
            min_length=config.data.synthetic.min_length,
            max_length=config.data.synthetic.max_length,
            include_features=config.model.include_features,
        )
    else:
        raise ValueError(f"Unknown dataset type: {config.data.dataset_type}")


def create_model_from_config(config: DictConfig) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Model instance.
    """
    model_config = config.model
    
    # Get model-specific parameters
    if model_config.type == "bilstm":
        model_params = model_config.bilstm
    elif model_config.type == "transformer":
        model_params = model_config.transformer
    elif model_config.type == "cnnlstm":
        model_params = model_config.cnnlstm
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
    
    return create_model(
        model_type=model_config.type,
        vocab_size=model_config.vocab_size,
        num_classes=model_config.num_classes,
        max_length=model_config.max_length,
        include_features=model_config.include_features,
        **model_params,
    )


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config.model.type = args.model_type
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file,
    )
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(config, config_save_path)
    
    logging.info("Starting protein structure prediction training")
    logging.info(f"Configuration: {config}")
    
    # Create dataset
    logging.info("Creating dataset")
    dataset = create_dataset(config)
    
    # Create data loaders
    logging.info("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config.data.batch_size,
        train_ratio=config.data.splits.train_ratio,
        val_ratio=config.data.splits.val_ratio,
        test_ratio=config.data.splits.test_ratio,
        num_workers=config.data.num_workers,
        random_seed=config.seed,
    )
    
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logging.info("Creating model")
    model = create_model_from_config(config)
    
    # Create trainer
    logging.info("Creating trainer")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        save_dir=str(output_dir / "checkpoints"),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    if not args.eval_only:
        logging.info("Starting training")
        history = trainer.train(config.training.num_epochs)
        
        # Save training history
        import json
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    # Evaluate model
    logging.info("Evaluating model")
    checkpoint_path = args.checkpoint or str(output_dir / "checkpoints" / "best_model.pt")
    
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    
    metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        save_dir=str(output_dir / "evaluation"),
    )
    
    # Print results
    logging.info("Evaluation Results:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()
