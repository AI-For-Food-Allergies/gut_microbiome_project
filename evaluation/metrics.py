"""
metrics.py - Comprehensive metrics for classification evaluation

Implements functions to compute various classification metrics including:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrices with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from typing import Dict, Tuple, Optional
import os


class ClassificationMetrics:
    """Class to compute and store classification metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None):
        """
        Initialize metrics calculator
        
        Args:
            y_true: Ground truth labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            y_pred_proba: Predicted probabilities for positive class (optional, for ROC-AUC)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
        
        # Validate inputs
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        self.metrics = {}
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all metrics"""
        # Accuracy
        self.metrics['accuracy'] = self.compute_accuracy()
        
        # Precision
        self.metrics['precision'] = self.compute_precision()
        
        # Recall
        self.metrics['recall'] = self.compute_recall()
        
        # F1-score
        self.metrics['f1_score'] = self.compute_f1_score()
        
        # AUC-ROC (if probabilities provided)
        if self.y_pred_proba is not None:
            self.metrics['auc_roc'] = self.compute_auc_roc()
        else:
            self.metrics['auc_roc'] = None
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = self.compute_confusion_matrix()
    
    def compute_accuracy(self) -> float:
        """
        Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
        
        Returns:
            float: Accuracy score between 0 and 1
        """
        return accuracy_score(self.y_true, self.y_pred)
    
    def compute_precision(self, zero_division: str = 'warn') -> float:
        """
        Compute precision: TP / (TP + FP)
        
        Args:
            zero_division: Handling of zero division ('warn', 0, or 1)
            
        Returns:
            float: Precision score between 0 and 1
        """
        return precision_score(self.y_true, self.y_pred, zero_division=zero_division, average='binary')
    
    def compute_recall(self, zero_division: str = 'warn') -> float:
        """
        Compute recall: TP / (TP + FN)
        
        Args:
            zero_division: Handling of zero division ('warn', 0, or 1)
            
        Returns:
            float: Recall score between 0 and 1
        """
        return recall_score(self.y_true, self.y_pred, zero_division=zero_division, average='binary')
    
    def compute_f1_score(self, zero_division: str = 'warn') -> float:
        """
        Compute F1-score: 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            zero_division: Handling of zero division ('warn', 0, or 1)
            
        Returns:
            float: F1-score between 0 and 1
        """
        return f1_score(self.y_true, self.y_pred, zero_division=zero_division, average='binary')
    
    def compute_auc_roc(self) -> float:
        """
        Compute Area Under the ROC Curve
        
        Returns:
            float: AUC-ROC score between 0 and 1
            
        Raises:
            ValueError: If y_pred_proba is not provided
        """
        if self.y_pred_proba is None:
            raise ValueError("Predicted probabilities (y_pred_proba) are required for AUC-ROC computation")
        
        return roc_auc_score(self.y_true, self.y_pred_proba)
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix
        
        Returns:
            np.ndarray: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Get all computed metrics as a dictionary
        
        Returns:
            Dict: Dictionary containing all metrics
        """
        return self.metrics.copy()
    
    def print_metrics_summary(self):
        """Print a summary of all computed metrics"""
        print("\n" + "="*50)
        print("CLASSIFICATION METRICS SUMMARY")
        print("="*50)
        print(f"Accuracy:    {self.metrics['accuracy']:.4f}")
        print(f"Precision:   {self.metrics['precision']:.4f}")
        print(f"Recall:      {self.metrics['recall']:.4f}")
        print(f"F1-Score:    {self.metrics['f1_score']:.4f}")
        if self.metrics['auc_roc'] is not None:
            print(f"AUC-ROC:     {self.metrics['auc_roc']:.4f}")
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        print("="*50 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels (default: [0, 1])
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    if labels is None:
        labels = ['Negative', 'Positive']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Tuple[plt.Figure, float]:
    """
    Plot ROC curve
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Tuple[plt.Figure, float]: Figure object and AUC-ROC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    
    return fig, roc_auc


def plot_metrics_comparison(
    metrics_dict: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot bar chart comparing different metrics
    
    Args:
        metrics_dict: Dictionary of metric names and values
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Filter out None values
    metrics_filtered = {k: v for k, v in metrics_dict.items() if v is not None and isinstance(v, (int, float))}
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(
        range(len(metrics_filtered)),
        list(metrics_filtered.values()),
        color=colors[:len(metrics_filtered)],
        alpha=0.7,
        edgecolor='black'
    )
    
    ax.set_xticks(range(len(metrics_filtered)))
    ax.set_xticklabels(
        [k.replace('_', ' ').title() for k in metrics_filtered.keys()],
        fontsize=11,
        fontweight='bold'
    )
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
    
    return fig


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    """
    Generate detailed classification report
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        str: Formatted classification report
    """
    if target_names is None:
        target_names = ['Negative', 'Positive']
    
    return classification_report(y_true, y_pred, target_names=target_names)