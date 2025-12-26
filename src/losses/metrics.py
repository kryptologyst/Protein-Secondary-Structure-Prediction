"""Loss functions and metrics for protein structure prediction."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            inputs: Predicted logits.
            targets: Ground truth labels.
            
        Returns:
            Dice loss value.
        """
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=-1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Compute Dice coefficient for each class
        intersection = (inputs * targets_one_hot).sum(dim=1)
        union = inputs.sum(dim=1) + targets_one_hot.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class ProteinMetrics:
    """Metrics for protein secondary structure prediction."""
    
    def __init__(self, class_names: List[str] = ["H", "E", "C"]):
        """Initialize metrics calculator.
        
        Args:
            class_names: Names of secondary structure classes.
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def compute_q3_accuracy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Compute Q3 accuracy (3-state accuracy).
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Q3 accuracy score.
        """
        # Flatten predictions and targets
        pred_flat = predictions[attention_mask]
        target_flat = targets[attention_mask]
        
        # Compute accuracy
        accuracy = (pred_flat == target_flat).float().mean().item()
        return accuracy
    
    def compute_per_class_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-class accuracy.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Dictionary of per-class accuracies.
        """
        pred_flat = predictions[attention_mask]
        target_flat = targets[attention_mask]
        
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = target_flat == i
            if class_mask.sum() > 0:
                class_acc = (pred_flat[class_mask] == target_flat[class_mask]).float().mean().item()
                per_class_acc[class_name] = class_acc
            else:
                per_class_acc[class_name] = 0.0
                
        return per_class_acc
    
    def compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """Compute confusion matrix.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Confusion matrix.
        """
        pred_flat = predictions[attention_mask].cpu().numpy()
        target_flat = targets[attention_mask].cpu().numpy()
        
        return confusion_matrix(target_flat, pred_flat, labels=range(self.num_classes))
    
    def compute_classification_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> str:
        """Compute detailed classification report.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Classification report string.
        """
        pred_flat = predictions[attention_mask].cpu().numpy()
        target_flat = targets[attention_mask].cpu().numpy()
        
        return classification_report(
            target_flat, 
            pred_flat, 
            target_names=self.class_names,
            digits=4,
        )
    
    def compute_sov_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Compute Segment Overlap (SOV) score.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            SOV score.
        """
        pred_flat = predictions[attention_mask].cpu().numpy()
        target_flat = targets[attention_mask].cpu().numpy()
        
        # Convert to strings for easier processing
        pred_str = "".join([self.class_names[i] for i in pred_flat])
        target_str = "".join([self.class_names[i] for i in target_flat])
        
        return self._compute_sov(pred_str, target_str)
    
    def _compute_sov(self, pred_str: str, target_str: str) -> float:
        """Compute SOV score for two structure strings."""
        if len(pred_str) != len(target_str):
            return 0.0
        
        # Find segments for each class
        pred_segments = self._find_segments(pred_str)
        target_segments = self._find_segments(target_str)
        
        total_overlap = 0
        total_length = len(pred_str)
        
        for pred_seg in pred_segments:
            for target_seg in target_segments:
                if pred_seg["class"] == target_seg["class"]:
                    overlap = self._compute_segment_overlap(pred_seg, target_seg)
                    total_overlap += overlap
        
        return total_overlap / total_length if total_length > 0 else 0.0
    
    def _find_segments(self, structure_str: str) -> List[Dict]:
        """Find continuous segments of the same class."""
        segments = []
        if not structure_str:
            return segments
        
        current_class = structure_str[0]
        start = 0
        
        for i in range(1, len(structure_str)):
            if structure_str[i] != current_class:
                segments.append({
                    "class": current_class,
                    "start": start,
                    "end": i,
                    "length": i - start,
                })
                current_class = structure_str[i]
                start = i
        
        # Add the last segment
        segments.append({
            "class": current_class,
            "start": start,
            "end": len(structure_str),
            "length": len(structure_str) - start,
        })
        
        return segments
    
    def _compute_segment_overlap(self, seg1: Dict, seg2: Dict) -> float:
        """Compute overlap between two segments."""
        overlap_start = max(seg1["start"], seg2["start"])
        overlap_end = min(seg1["end"], seg2["end"])
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        union_length = max(seg1["end"], seg2["end"]) - min(seg1["start"], seg2["start"])
        
        return overlap_length / union_length if union_length > 0 else 0.0
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all available metrics.
        
        Args:
            predictions: Predicted class indices.
            targets: Ground truth class indices.
            attention_mask: Mask for valid positions.
            
        Returns:
            Dictionary of all computed metrics.
        """
        metrics = {}
        
        # Q3 accuracy
        metrics["q3_accuracy"] = self.compute_q3_accuracy(predictions, targets, attention_mask)
        
        # Per-class accuracy
        per_class_acc = self.compute_per_class_accuracy(predictions, targets, attention_mask)
        for class_name, acc in per_class_acc.items():
            metrics[f"accuracy_{class_name}"] = acc
        
        # SOV score
        metrics["sov_score"] = self.compute_sov_score(predictions, targets, attention_mask)
        
        return metrics


def create_loss_function(
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """Create loss function for protein structure prediction.
    
    Args:
        loss_type: Type of loss function ('cross_entropy', 'focal', 'dice').
        class_weights: Optional class weights for imbalanced data.
        **kwargs: Additional loss-specific arguments.
        
    Returns:
        Loss function.
    """
    if loss_type == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Compute class weights for imbalanced data.
    
    Args:
        targets: Ground truth labels.
        attention_mask: Mask for valid positions.
        num_classes: Number of classes.
        
    Returns:
        Class weights tensor.
    """
    # Flatten targets and mask
    targets_flat = targets[attention_mask]
    
    # Count occurrences of each class
    class_counts = torch.bincount(targets_flat, minlength=num_classes).float()
    
    # Compute inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Handle zero counts
    class_weights[class_counts == 0] = 0.0
    
    return class_weights
