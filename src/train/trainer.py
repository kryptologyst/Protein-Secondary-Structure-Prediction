"""Training utilities for protein structure prediction models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.core import EarlyStopping, get_device, count_parameters
from ..losses.metrics import ProteinMetrics, compute_class_weights


class ProteinTrainer:
    """Trainer class for protein structure prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        early_stopping: Optional[EarlyStopping] = None,
        metrics: Optional[ProteinMetrics] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to train on.
            save_dir: Directory to save checkpoints.
            early_stopping: Early stopping utility.
            metrics: Metrics calculator.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.save_dir = Path(save_dir) if save_dir else None
        self.early_stopping = early_stopping
        self.metrics = metrics or ProteinMetrics()
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logging.info(f"Model has {count_parameters(self.model):,} trainable parameters")
        logging.info(f"Training on device: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch["structure"].view(-1),
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * batch["sequence"].size(0)
            total_samples += batch["sequence"].size(0)
            
            # Store predictions for metrics
            predictions = logits.argmax(dim=-1)
            all_predictions.append(predictions)
            all_targets.append(batch["structure"])
            all_masks.append(batch["attention_mask"])
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Compute accuracy
        accuracy = self.metrics.compute_q3_accuracy(all_predictions, all_targets, all_masks)
        
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    batch["structure"].view(-1),
                )
                
                # Update metrics
                total_loss += loss.item() * batch["sequence"].size(0)
                total_samples += batch["sequence"].size(0)
                
                # Store predictions for metrics
                predictions = logits.argmax(dim=-1)
                all_predictions.append(predictions)
                all_targets.append(batch["structure"])
                all_masks.append(batch["attention_mask"])
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets, all_masks)
        metrics["val_loss"] = avg_loss
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Dictionary of training history.
        """
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["train_loss"])
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics["val_loss"])
            self.val_accuracies.append(val_metrics["q3_accuracy"])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['q3_accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_accuracy = val_metrics["q3_accuracy"]
                self.save_checkpoint("best_model.pt", val_metrics)
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics["val_loss"], self.model):
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Save final model
        self.save_checkpoint("final_model.pt", val_metrics)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
            metrics: Current metrics.
        """
        if self.save_dir is None:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "metrics": metrics,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    save_dir: Optional[str] = None,
) -> ProteinTrainer:
    """Create a trainer with configuration.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        save_dir: Directory to save checkpoints.
        
    Returns:
        Configured trainer.
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    # Scheduler
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=config.get("scheduler_patience", 5),
            verbose=True,
        )
    
    # Early stopping
    early_stopping = None
    if config.get("use_early_stopping", False):
        early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            min_delta=config.get("early_stopping_min_delta", 0.0),
        )
    
    # Metrics
    metrics = ProteinMetrics()
    
    return ProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=get_device(),
        save_dir=save_dir,
        early_stopping=early_stopping,
        metrics=metrics,
    )
