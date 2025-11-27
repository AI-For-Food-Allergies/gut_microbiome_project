"""
train.py - Training script with K-fold cross-validation

This script implements training functionality for the MicrobiomeClassifier with
comprehensive K-fold cross-validation evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional, Any
import os
import json
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import yaml

from modules.classifier import MicrobiomeClassifier
from evaluation.metrics import ClassificationMetrics, plot_confusion_matrix, plot_roc_curve


class TrainingResults:
    """Container for training results and metrics"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize training results container
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir or './training_results'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.trained_model = None
        self.fold_results = {}
        self.best_fold = None
        self.best_metrics = None
        self.training_history = []
        self.cv_summary = {}
    
    def save_results(self, filename: str = 'training_results.json'):
        """Save training results to JSON"""
        results_data = {
            'best_fold': self.best_fold,
            'best_metrics': self.best_metrics,
            'cv_summary': self.cv_summary,
            'num_folds': len(self.fold_results)
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Training results saved to {filepath}")
    
    def save_model(self, filename: str = 'trained_model.pt'):
        """Save trained model checkpoint"""
        if self.trained_model is None:
            print("No trained model to save")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(self.trained_model.state_dict(), filepath)
        print(f"Trained model saved to {filepath}")


def train_classifier(
    classifier: MicrobiomeClassifier,
    X: Dict[str, torch.Tensor],
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cpu',
    output_dir: Optional[str] = None,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Tuple[MicrobiomeClassifier, TrainingResults]:
    """
    Train classifier with K-fold cross-validation
    
    Args:
        classifier: MicrobiomeClassifier instance
        X: Dictionary containing input tensors
           - 'embeddings_type1': (N, seq_len1, input_dim_type1)
           - 'embeddings_type2': (N, seq_len2, input_dim_type2)
           - 'mask': (N, total_seq_len)
           - 'type_indicators': (N, total_seq_len)
        y: Binary labels (0 or 1)
        n_splits: Number of K-fold splits (default: 5)
        epochs: Maximum epochs per fold (default: 50)
        batch_size: Training batch size (default: 32)
        learning_rate: Learning rate (default: 1e-3)
        weight_decay: L2 regularization (default: 1e-5)
        device: 'cpu' or 'cuda'
        output_dir: Directory to save results
        early_stopping_patience: Patience for early stopping (default: 10)
        verbose: Print training progress (default: True)
        
    Returns:
        Tuple[MicrobiomeClassifier, TrainingResults]: Trained model and results
    """
    results = TrainingResults(output_dir)
    
    # Validate inputs
    n_samples = len(y)
    if n_samples < n_splits:
        raise ValueError(f"Number of samples ({n_samples}) must be >= n_splits ({n_splits})")
    
    # Initialize K-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    best_fold_model = None
    best_fold_f1 = 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING WITH {n_splits}-FOLD CROSS-VALIDATION")
        print(f"{'='*70}")
    
    # Perform K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(y)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold + 1}/{n_splits}")
            print(f"{'='*70}")
        
        # Split data
        X_train_fold = {k: v[train_idx] for k, v in X.items()}
        X_val_fold = {k: v[val_idx] for k, v in X.items()}
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Train on this fold
        fold_model, fold_history = train_single_fold(
            classifier=deepcopy(classifier),
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
            fold_num=fold + 1
        )
        
        # Evaluate on validation set
        y_pred, y_pred_proba = get_predictions(
            fold_model, X_val_fold, y_val_fold, device, batch_size
        )
        
        # Compute metrics
        metrics_obj = ClassificationMetrics(y_val_fold, y_pred, y_pred_proba)
        fold_metrics_dict = metrics_obj.get_metrics_dict()
        
        fold_metrics.append(fold_metrics_dict)
        results.fold_results[f'fold_{fold + 1}'] = {
            'metrics': fold_metrics_dict,
            'train_size': len(y_train_fold),
            'val_size': len(y_val_fold)
        }
        results.training_history.append(fold_history)
        
        if verbose:
            print(f"\nFold {fold + 1} Validation Results:")
            print(f"  Accuracy:  {fold_metrics_dict['accuracy']:.4f}")
            print(f"  Precision: {fold_metrics_dict['precision']:.4f}")
            print(f"  Recall:    {fold_metrics_dict['recall']:.4f}")
            print(f"  F1-Score:  {fold_metrics_dict['f1_score']:.4f}")
        
        # Track best fold
        if fold_metrics_dict['f1_score'] > best_fold_f1:
            best_fold_f1 = fold_metrics_dict['f1_score']
            best_fold_model = fold_model
            results.best_fold = fold + 1
            results.best_metrics = fold_metrics_dict
    
    # Compute cross-validation summary
    compute_cv_summary(fold_metrics, results)
    
    results.trained_model = best_fold_model
    
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        print_cv_summary(results.cv_summary)
        print(f"Best fold: Fold {results.best_fold}")
    
    # Save results
    results.save_results()
    results.save_model()
    
    return best_fold_model, results


def train_single_fold(
    classifier: MicrobiomeClassifier,
    X_train: Dict[str, torch.Tensor],
    y_train: np.ndarray,
    X_val: Dict[str, torch.Tensor],
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cpu',
    early_stopping_patience: int = 10,
    verbose: bool = True,
    fold_num: int = 1
) -> Tuple[MicrobiomeClassifier, Dict[str, List]]:
    """
    Train classifier on a single fold
    
    Args:
        classifier: MicrobiomeClassifier instance
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        early_stopping_patience: Early stopping patience
        verbose: Print progress
        fold_num: Fold number (for logging)
        
    Returns:
        Tuple: Trained model and training history
    """
    classifier = classifier.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Create data loaders
    train_dataset = create_dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = create_dataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        train_loss = train_epoch(classifier, train_loader, optimizer, loss_fn, device)
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(classifier, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(classifier.state_dict())
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    classifier.load_state_dict(best_model_state)
    
    return classifier, history


def train_epoch(
    classifier: MicrobiomeClassifier,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str
) -> float:
    """Train for one epoch"""
    classifier.train()
    total_loss = 0
    
    for batch in train_loader:
        # Move to device
        batch_data = {
            'embeddings_type1': batch[0].to(device),
            'embeddings_type2': batch[1].to(device),
            'mask': batch[2].to(device),
            'type_indicators': batch[3].to(device)
        }
        y_true = batch[4].to(device).float()
        
        # Forward pass
        output = classifier(batch_data)  # (batch_size, seq_len)
        
        # Aggregate predictions (mean over sequence)
        output_agg = output.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Compute loss
        loss = loss_fn(output_agg, y_true.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(
    classifier: MicrobiomeClassifier,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch"""
    classifier.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_data = {
                'embeddings_type1': batch[0].to(device),
                'embeddings_type2': batch[1].to(device),
                'mask': batch[2].to(device),
                'type_indicators': batch[3].to(device)
            }
            y_true = batch[4].to(device).float()
            
            # Forward pass
            output = classifier(batch_data)
            output_agg = output.mean(dim=1, keepdim=True)
            
            # Loss
            loss = loss_fn(output_agg, y_true.unsqueeze(1))
            total_loss += loss.item()
            
            # Predictions
            preds = (torch.sigmoid(output_agg) >= 0.5).long().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y_true.cpu().numpy().flatten())
    
    # Compute metrics
    metrics_obj = ClassificationMetrics(np.array(all_labels), np.array(all_preds))
    metrics = metrics_obj.get_metrics_dict()
    
    return total_loss / len(val_loader), metrics


def get_predictions(
    classifier: MicrobiomeClassifier,
    X: Dict[str, torch.Tensor],
    y: np.ndarray,
    device: str,
    batch_size: int,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from classifier"""
    classifier.eval()
    
    dataset = create_dataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds_proba = []
    
    with torch.no_grad():
        for batch in loader:
            batch_data = {
                'embeddings_type1': batch[0].to(device),
                'embeddings_type2': batch[1].to(device),
                'mask': batch[2].to(device),
                'type_indicators': batch[3].to(device)
            }
            
            output = classifier(batch_data)
            output_agg = output.mean(dim=1)
            proba = torch.sigmoid(output_agg).cpu().numpy()
            all_preds_proba.extend(proba)
    
    y_pred_proba = np.array(all_preds_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba


def create_dataset(X: Dict[str, torch.Tensor], y: np.ndarray) -> TensorDataset:
    """Create PyTorch dataset"""
    return TensorDataset(
        X['embeddings_type1'],
        X['embeddings_type2'],
        X['mask'],
        X['type_indicators'],
        torch.from_numpy(y).float()
    )


def compute_cv_summary(fold_metrics: List[Dict], results: TrainingResults):
    """Compute cross-validation summary statistics"""
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    for metric_key in metrics_keys:
        values = [fm.get(metric_key) for fm in fold_metrics if fm.get(metric_key) is not None]
        if values:
            results.cv_summary[metric_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }


def print_cv_summary(cv_summary: Dict):
    """Print cross-validation summary"""
    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    for metric_name, stats in cv_summary.items():
        print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} {stats['max']:<10.4f}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Example usage
if __name__ == "__main__":
    print("Training script - ready to use with MicrobiomeClassifier")
    print("Import this module and call train_classifier() with your model and data")