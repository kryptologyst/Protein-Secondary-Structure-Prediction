"""Evaluation utilities for protein structure prediction models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.metrics import ProteinMetrics
from ..utils.core import get_device


class ProteinEvaluator:
    """Evaluator class for protein structure prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        metrics: Optional[ProteinMetrics] = None,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            test_loader: Test data loader.
            device: Device to run evaluation on.
            metrics: Metrics calculator.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device()
        self.metrics = metrics or ProteinMetrics()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        logging.info("Starting model evaluation")
        
        all_predictions = []
        all_targets = []
        all_masks = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(batch)
                predictions = logits.argmax(dim=-1)
                
                # Store results
                all_predictions.append(predictions)
                all_targets.append(batch["structure"])
                all_masks.append(batch["attention_mask"])
                all_logits.append(logits)
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets, all_masks)
        
        # Add additional metrics
        metrics.update(self._compute_additional_metrics(all_predictions, all_targets, all_masks))
        
        logging.info("Evaluation completed")
        return metrics
    
    def _compute_additional_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute additional evaluation metrics.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Dictionary of additional metrics.
        """
        metrics = {}
        
        # Per-class precision, recall, F1
        pred_flat = predictions[attention_mask].cpu().numpy()
        target_flat = targets[attention_mask].cpu().numpy()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            target_flat, pred_flat, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.metrics.class_names):
            metrics[f"precision_{class_name}"] = precision[i]
            metrics[f"recall_{class_name}"] = recall[i]
            metrics[f"f1_{class_name}"] = f1[i]
            metrics[f"support_{class_name}"] = support[i]
        
        # Macro averages
        metrics["precision_macro"] = np.mean(precision)
        metrics["recall_macro"] = np.mean(recall)
        metrics["f1_macro"] = np.mean(f1)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average="weighted", zero_division=0
        )
        
        metrics["precision_weighted"] = precision_weighted
        metrics["recall_weighted"] = recall_weighted
        metrics["f1_weighted"] = f1_weighted
        
        return metrics
    
    def generate_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Generate and optionally save confusion matrix.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            save_path: Optional path to save confusion matrix plot.
            
        Returns:
            Confusion matrix.
        """
        cm = self.metrics.compute_confusion_matrix(predictions, targets, attention_mask)
        
        if save_path:
            self._plot_confusion_matrix(cm, save_path)
        
        return cm
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix.
            save_path: Path to save plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.metrics.class_names,
            yticklabels=self.metrics.class_names,
            title="Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def generate_classification_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate detailed classification report.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            save_path: Optional path to save report.
            
        Returns:
            Classification report string.
        """
        report = self.metrics.compute_classification_report(predictions, targets, attention_mask)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
        
        return report
    
    def create_leaderboard(self, metrics: Dict[str, float], save_path: Optional[str] = None) -> pd.DataFrame:
        """Create a metrics leaderboard.
        
        Args:
            metrics: Dictionary of metrics.
            save_path: Optional path to save leaderboard.
            
        Returns:
            DataFrame with metrics.
        """
        # Organize metrics into categories
        leaderboard_data = []
        
        # Overall metrics
        leaderboard_data.append({
            "Metric": "Q3 Accuracy",
            "Value": metrics.get("q3_accuracy", 0.0),
            "Category": "Overall",
        })
        
        leaderboard_data.append({
            "Metric": "SOV Score",
            "Value": metrics.get("sov_score", 0.0),
            "Category": "Overall",
        })
        
        # Per-class accuracy
        for class_name in self.metrics.class_names:
            leaderboard_data.append({
                "Metric": f"Accuracy ({class_name})",
                "Value": metrics.get(f"accuracy_{class_name}", 0.0),
                "Category": "Per-Class",
            })
        
        # Per-class F1
        for class_name in self.metrics.class_names:
            leaderboard_data.append({
                "Metric": f"F1 ({class_name})",
                "Value": metrics.get(f"f1_{class_name}", 0.0),
                "Category": "Per-Class",
            })
        
        # Macro averages
        leaderboard_data.append({
            "Metric": "Precision (Macro)",
            "Value": metrics.get("precision_macro", 0.0),
            "Category": "Macro Average",
        })
        
        leaderboard_data.append({
            "Metric": "Recall (Macro)",
            "Value": metrics.get("recall_macro", 0.0),
            "Category": "Macro Average",
        })
        
        leaderboard_data.append({
            "Metric": "F1 (Macro)",
            "Value": metrics.get("f1_macro", 0.0),
            "Category": "Macro Average",
        })
        
        df = pd.DataFrame(leaderboard_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a protein structure prediction model.
    
    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        save_dir: Optional directory to save evaluation results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    evaluator = ProteinEvaluator(model, test_loader)
    metrics = evaluator.evaluate()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_df.to_csv(save_path / "evaluation_metrics.csv", index=False)
        
        # Create leaderboard
        leaderboard = evaluator.create_leaderboard(metrics, save_path / "leaderboard.csv")
        
        logging.info(f"Evaluation results saved to {save_path}")
    
    return metrics
