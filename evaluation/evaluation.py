"""
evaluation.py - Comprehensive evaluation script for microbiome classifier

Runs evaluation functions and stores relevant metrics and plots.
"""

import numpy as np
import torch
import os
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt

from modules.classifier import MicrobiomeClassifier
from metrics import (
    ClassificationMetrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_comparison,
    get_classification_report
)


class EvaluationResults:
    """Container for storing evaluation results and plots"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize evaluation results container
        
        Args:
            output_dir: Directory to save plots and results (default: './evaluation_results')
        """
        self.output_dir = output_dir or './evaluation_results'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.plots = {}
        self.classification_report = None
        self.predictions = {}
    
    def save_metrics_to_json(self, filename: str = 'metrics.json'):
        """Save metrics to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        print(f"Metrics saved to {filepath}")
    
    def save_classification_report(self, filename: str = 'classification_report.txt'):
        """Save classification report to text file"""
        if self.classification_report is None:
            print("No classification report to save")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(self.classification_report)
        print(f"Classification report saved to {filepath}")
    
    def save_all_plots(self):
        """Save all generated plots"""
        for plot_name, fig in self.plots.items():
            if fig is not None:
                filepath = os.path.join(self.output_dir, f'{plot_name}.png')
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Plot saved to {filepath}")


def evaluate_classifier(
    classifier: MicrobiomeClassifier,
    X: Dict[str, torch.Tensor],
    y: np.ndarray,
    output_dir: Optional[str] = None,
    device: str = 'cpu',
    batch_size: int = 32,
    threshold: float = 0.5
) -> EvaluationResults:
    """
    Evaluate classifier on test data and compute all metrics
    
    Args:
        classifier: MicrobiomeClassifier instance
        X: Dictionary containing input data
           - 'embeddings_type1': (N, seq_len1, input_dim_type1)
           - 'embeddings_type2': (N, seq_len2, input_dim_type2)
           - 'mask': (N, total_seq_len)
           - 'type_indicators': (N, total_seq_len)
        y: Ground truth labels (N,)
        output_dir: Directory to save results
        device: Device to run inference on ('cpu' or 'cuda')
        batch_size: Batch size for inference
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        EvaluationResults: Object containing metrics, plots, and reports
    """
    results = EvaluationResults(output_dir)
    
    # Move classifier to device and set to eval mode
    classifier = classifier.to(device)
    classifier.eval()
    
    # Get predictions
    y_pred, y_pred_proba = get_predictions(
        classifier, X, y, device, batch_size, threshold
    )
    
    # Store predictions
    results.predictions['y_true'] = y
    results.predictions['y_pred'] = y_pred
    results.predictions['y_pred_proba'] = y_pred_proba
    
    # Compute metrics
    metrics_obj = ClassificationMetrics(y, y_pred, y_pred_proba)
    results.metrics = metrics_obj.get_metrics_dict()
    
    # Print metrics summary
    metrics_obj.print_metrics_summary()
    
    # Generate classification report
    results.classification_report = get_classification_report(y, y_pred)
    print("\nClassification Report:")
    print(results.classification_report)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Confusion matrix plot
    fig_cm = plot_confusion_matrix(
        y, y_pred,
        labels=['No Allergy', 'Allergy'],
        save_path=os.path.join(results.output_dir, 'confusion_matrix.png')
    )
    results.plots['confusion_matrix'] = fig_cm
    plt.close(fig_cm)
    
    # ROC curve plot (if probabilities available)
    if y_pred_proba is not None:
        fig_roc, auc_score = plot_roc_curve(
            y, y_pred_proba,
            save_path=os.path.join(results.output_dir, 'roc_curve.png')
        )
        results.plots['roc_curve'] = fig_roc
        plt.close(fig_roc)
        print(f"AUC-ROC Score: {auc_score:.4f}")
    
    # Metrics comparison plot
    fig_metrics = plot_metrics_comparison(
        results.metrics,
        save_path=os.path.join(results.output_dir, 'metrics_comparison.png')
    )
    results.plots['metrics_comparison'] = fig_metrics
    plt.close(fig_metrics)
    
    # Save all results
    print("\nSaving results...")
    results.save_metrics_to_json()
    results.save_classification_report()
    
    return results


def get_predictions(
    classifier: MicrobiomeClassifier,
    X: Dict[str, torch.Tensor],
    y: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 32,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from classifier
    
    Args:
        classifier: MicrobiomeClassifier instance
        X: Input data dictionary
        y: Ground truth labels
        device: Device to run inference on
        batch_size: Batch size for inference
        threshold: Classification threshold
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (y_pred, y_pred_proba)
    """
    classifier.eval()
    
    all_pred_proba = []
    
    n_samples = len(y)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            # Extract batch
            batch = {
                'embeddings_type1': X['embeddings_type1'][start_idx:end_idx].to(device),
                'embeddings_type2': X['embeddings_type2'][start_idx:end_idx].to(device),
                'mask': X['mask'][start_idx:end_idx].to(device),
                'type_indicators': X['type_indicators'][start_idx:end_idx].to(device),
            }
            
            # Forward pass
            output = classifier(batch)  # (batch_size, seq_len)
            
            # Aggregate predictions (mean over sequence dimension)
            batch_proba = output.mean(dim=1).cpu().numpy()  # (batch_size,)
            all_pred_proba.extend(batch_proba)
    
    y_pred_proba = np.array(all_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba


def evaluate_with_train_val_test(
    classifier: MicrobiomeClassifier,
    X_train: Dict[str, torch.Tensor],
    y_train: np.ndarray,
    X_val: Dict[str, torch.Tensor],
    y_val: np.ndarray,
    X_test: Dict[str, torch.Tensor],
    y_test: np.ndarray,
    output_dir: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[str, EvaluationResults]:
    """
    Evaluate classifier on train, validation, and test sets
    
    Args:
        classifier: MicrobiomeClassifier instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        output_dir: Base output directory for results
        device: Device to run inference on
        
    Returns:
        Dict[str, EvaluationResults]: Results for each split
    """
    results = {}
    
    if output_dir is None:
        output_dir = './evaluation_results'
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("EVALUATING ON TRAINING SET")
    print("="*60)
    train_output_dir = os.path.join(output_dir, 'train')
    results['train'] = evaluate_classifier(
        classifier, X_train, y_train, train_output_dir, device
    )
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATING ON VALIDATION SET")
    print("="*60)
    val_output_dir = os.path.join(output_dir, 'validation')
    results['validation'] = evaluate_classifier(
        classifier, X_val, y_val, val_output_dir, device
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_output_dir = os.path.join(output_dir, 'test')
    results['test'] = evaluate_classifier(
        classifier, X_test, y_test, test_output_dir, device
    )
    
    # Create summary comparison
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print_evaluation_summary(results)
    
    return results


def print_evaluation_summary(results: Dict[str, EvaluationResults]):
    """Print summary comparison of evaluation results"""
    print(f"\n{'Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print("-" * 65)
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    for metric_name in metric_names:
        values = []
        for split in ['train', 'validation', 'test']:
            if split in results:
                value = results[split].metrics.get(metric_name)
                if value is not None:
                    values.append(f"{value:.4f}")
                else:
                    values.append("N/A")
            else:
                values.append("N/A")
        
        print(f"{metric_name:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")


def compare_models(
    models: Dict[str, MicrobiomeClassifier],
    X_test: Dict[str, torch.Tensor],
    y_test: np.ndarray,
    output_dir: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[str, EvaluationResults]:
    """
    Evaluate and compare multiple models on test data
    
    Args:
        models: Dictionary mapping model names to classifier instances
        X_test: Test data
        y_test: Test labels
        output_dir: Base output directory for results
        device: Device to run inference on
        
    Returns:
        Dict[str, EvaluationResults]: Results for each model
    """
    results = {}
    
    if output_dir is None:
        output_dir = './model_comparison'
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, classifier in models.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")
        
        model_output_dir = os.path.join(output_dir, model_name)
        results[model_name] = evaluate_classifier(
            classifier, X_test, y_test, model_output_dir, device
        )
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print_model_comparison(results)
    
    return results


def print_model_comparison(results: Dict[str, EvaluationResults]):
    """Print comparison of multiple models"""
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Find max length for model name
    max_model_len = max(len(name) for name in results.keys())
    
    print(f"\n{'Model':<{max_model_len}} ", end='')
    for metric in metric_names:
        print(f"{metric:<12} ", end='')
    print()
    
    print("-" * (max_model_len + len(metric_names) * 12 + 1))
    
    for model_name, result in results.items():
        print(f"{model_name:<{max_model_len}} ", end='')
        for metric in metric_names:
            value = result.metrics.get(metric)
            if value is not None:
                print(f"{value:<12.4f} ", end='')
            else:
                print(f"{'N/A':<12} ", end='')
        print()