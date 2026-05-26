"""
Evaluation utilities for storing and visualizing classification results.

Results are stored in the following structure:
    results_output_dir/
        dataset_name/
            specific_data/
                classification_report.csv
                {classifier}_roc_curve.png
                {classifier}_confusion_matrix.png
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib
from omegaconf import DictConfig, OmegaConf, ListConfig
from rich.console import Console

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics from a classifier run."""

    classifier_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    cv_folds: int
    roc_auc: Optional[float] = None
    classification_report_dict: Optional[Dict] = None
    best_params: Optional[Dict] = field(default=None)

    def __post_init__(self):
        """Compute derived metrics after initialization."""
        # Compute classification report
        self.classification_report_dict = classification_report(
            self.y_true, self.y_pred, output_dict=True
        )
        # Compute ROC-AUC if probabilities available
        if self.y_prob is not None:
            try:
                n_classes = len(np.unique(self.y_true))
                if n_classes == 2:
                    self.roc_auc = roc_auc_score(self.y_true, self.y_prob)
                else:
                    self.roc_auc = roc_auc_score(
                        self.y_true, self.y_prob, multi_class="ovr", average="macro"
                    )
            except Exception:
                self.roc_auc = None


@dataclass
class EvaluationResult:
    """Container for evaluation results from a single classifier run."""

    classifier_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    cv_folds: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_probabilities(self) -> bool:
        return self.y_prob is not None

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters if stored in additional_metrics."""
        return self.additional_metrics.get("best_params")


class ResultsManager:
    """
    Manages storage of evaluation results in an organized directory structure.

    Results are stored in: results_output_dir/dataset_name/data_source/
    """

    def __init__(
        self,
        config: DictConfig,
        console: Console,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the ResultsManager.

        Args:
            results_output_dir: Base directory for storing results
            dataset_path: Path to the dataset CSV file (used to extract dataset_name and data_source)
            class_names: Optional list of class names for labeling plots
        """
        self.results_output_dir = Path(config.evaluation.results_output_dir)
        self.dataset_name = config.data.hugging_face.dataset_name
        self.data_source = Path(config.data.hugging_face.csv_filename).stem
        self.class_names = class_names

        # Build output directory path
        self.output_dir = self.results_output_dir / self.dataset_name / self.data_source

        # Storage for results from multiple classifiers
        self.results: Dict[str, EvaluationResult] = {}

        # Log initialization
        console.print(
            f"ResultsManager initialized. Results will be saved to: {self.output_dir}",
            style="bold",
        )
        self.console = console

        # Create output directory
        self._ensure_output_dir()

    def to_native(self, obj):
        """Convert OmegaConf objects to native Python types."""
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: self.to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_native(v) for v in obj]
        else:
            return obj

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console.print(f"Results will be saved to: {self.output_dir}", style="bold")

    def add_result(self, result: EvaluationResult):
        """Add an evaluation result for a classifier."""
        self.results[result.classifier_name] = result

    def save_classification_report(
        self, result: EvaluationResult, filename: Optional[str] = None
    ) -> Path:
        """
        Save classification report as CSV file.

        Args:
            result: EvaluationResult object containing predictions
            filename: Optional custom filename (defaults to {classifier}_classification_report.csv)

        Returns:
            Path to saved CSV file
        """
        # Generate classification report as dict
        report_dict = classification_report(
            result.y_true,
            result.y_pred,
            target_names=self.class_names,
            output_dict=True,
        )

        # Convert to DataFrame for easy CSV export
        report_df = pd.DataFrame(report_dict).transpose()

        # Add additional info
        report_df["classifier"] = result.classifier_name
        report_df["cv_folds"] = result.cv_folds
        report_df["evaluation_mode"] = (
            "holdout" if result.cv_folds == 0 else "cross_validation"
        )

        # Add ROC-AUC if probabilities available
        if result.has_probabilities:
            try:
                n_classes = len(np.unique(result.y_true))
                if n_classes == 2:
                    roc_auc = roc_auc_score(result.y_true, result.y_prob)
                else:
                    roc_auc = roc_auc_score(
                        result.y_true, result.y_prob, multi_class="ovr", average="macro"
                    )
                report_df["roc_auc"] = roc_auc
            except Exception:
                pass

        # Add best params if available (from grid search)
        if result.best_params:
            native_best_params = self.to_native(result.best_params)
            report_df["best_params"] = json.dumps(native_best_params)

        # Determine filename
        if filename is None:
            filename = f"{result.classifier_name}_classification_report.csv"

        # Save to CSV
        output_path = self.output_dir / filename
        report_df.to_csv(output_path)
        self.console.print(
            f"  ✓ Classification report saved: {output_path}", style="success"
        )

        return output_path

    def save_roc_curve(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
        figsize: tuple = (8, 6),
    ) -> Optional[Path]:
        """
        Save ROC curve as PNG image.

        Args:
            result: EvaluationResult object containing predictions and probabilities
            filename: Optional custom filename (defaults to {classifier}_roc_curve.png)
            figsize: Figure size tuple

        Returns:
            Path to saved image, or None if probabilities not available
        """
        if not result.has_probabilities:
            self.console.print(
                f"  ⚠ Skipping ROC curve for {result.classifier_name}: no probability predictions",
                style="warning",
            )
            return None

        try:
            n_classes = len(np.unique(result.y_true))
            fig, ax = plt.subplots(figsize=figsize)

            if n_classes == 2:
                # Binary classification: single ROC curve
                fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
                roc_auc = roc_auc_score(result.y_true, result.y_prob)
                ax.plot(
                    fpr,
                    tpr,
                    color="#2563eb",
                    lw=2,
                    label=f"{result.classifier_name} (AUC = {roc_auc:.3f})",
                )
            else:
                # Multi-class: per-class One-vs-Rest ROC curves
                classes = np.unique(result.y_true)
                colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
                
                for idx, (cls, color) in enumerate(zip(classes, colors)):
                    # Binarize for this class
                    y_binary = (result.y_true == cls).astype(int)
                    y_score = result.y_prob[:, idx]
                    
                    fpr, tpr, _ = roc_curve(y_binary, y_score)
                    class_auc = roc_auc_score(y_binary, y_score)
                    
                    class_name = self.class_names[idx] if self.class_names else str(cls)
                    ax.plot(
                        fpr,
                        tpr,
                        color=color,
                        lw=2,
                        label=f"{class_name} (AUC = {class_auc:.3f})",
                    )
                
                # Compute macro-averaged AUC for title
                roc_auc = roc_auc_score(
                    result.y_true, result.y_prob, multi_class="ovr", average="macro"
                )

            # Plot diagonal reference line
            ax.plot(
                [0, 1],
                [0, 1],
                color="#94a3b8",
                lw=1.5,
                linestyle="--",
                label="Random Classifier",
            )

            # Styling
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            tuned_text = " (tuned)" if result.best_params else ""
            eval_text = (
                "Holdout Test Set" if result.cv_folds == 0
                else f"{result.cv_folds}-Fold CV"
            )
            title_suffix = "" if n_classes == 2 else f" (Macro AUC = {roc_auc:.3f})"
            ax.set_title(
                f"ROC Curve - {result.classifier_name}{tuned_text}{title_suffix} [{eval_text}]\n"
                f"Dataset: {self.dataset_name}/{self.data_source}",
                fontsize=13,
            )
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Tight layout
            plt.tight_layout()

            # Determine filename
            if filename is None:
                filename = f"{result.classifier_name}_roc_curve.png"

            # Save figure
            output_path = self.output_dir / filename
            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)

            self.console.print(f"  ✓ ROC curve saved: {output_path}", style="success")
            return output_path

        except Exception as e:
            self.console.print(f"  ✗ Error saving ROC curve: {e}", style="error")
            return None

    def save_confusion_matrix(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
        figsize: tuple = (8, 6),
        normalize: Optional[str] = None,
        cmap: str = "Blues",
    ) -> Path:
        """
        Save confusion matrix as PNG image.

        Args:
            result: EvaluationResult object containing predictions
            filename: Optional custom filename (defaults to {classifier}_confusion_matrix.png)
            figsize: Figure size tuple
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            cmap: Colormap name

        Returns:
            Path to saved image
        """
        # Compute confusion matrix
        cm = confusion_matrix(result.y_true, result.y_pred, normalize=normalize)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot(ax=ax, cmap=cmap, values_format=".2f" if normalize else "d")

        # Styling
        norm_text = f" (normalized: {normalize})" if normalize else ""
        tuned_text = " (tuned)" if result.best_params else ""
        eval_text = (
            "Holdout Test Set" if result.cv_folds == 0
            else f"{result.cv_folds}-Fold CV"
        )
        ax.set_title(
            f"Confusion Matrix - {result.classifier_name}{tuned_text}{norm_text} [{eval_text}]\n"
            f"Dataset: {self.dataset_name}/{self.data_source}",
            fontsize=13,
        )

        plt.tight_layout()

        # Determine filename
        if filename is None:
            suffix = f"_norm_{normalize}" if normalize else ""
            filename = f"{result.classifier_name}_confusion_matrix{suffix}.png"

        # Save figure
        output_path = self.output_dir / filename
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        self.console.print(
            f"  ✓ Confusion matrix saved: {output_path}", style="success"
        )
        return output_path

    def save_all_results(
        self, result: Optional[EvaluationResult] = None, save_normalized_cm: bool = True
    ) -> Dict[str, Path]:
        """
        Save all result files for a classifier (or all stored classifiers).

        Args:
            result: Optional specific result to save. If None, saves all stored results.
            save_normalized_cm: Whether to also save a normalized confusion matrix

        Returns:
            Dictionary mapping result type to file path
        """
        results_to_save = [result] if result else list(self.results.values())

        all_paths = {}

        for res in results_to_save:
            self.console.print(
                f"\nSaving results for {res.classifier_name}...", style="path"
            )

            # Save classification report
            report_path = self.save_classification_report(res)
            all_paths[f"{res.classifier_name}_report"] = report_path

            # Save ROC curve
            roc_path = self.save_roc_curve(res)
            if roc_path:
                all_paths[f"{res.classifier_name}_roc"] = roc_path

            # Save confusion matrix
            cm_path = self.save_confusion_matrix(res)
            all_paths[f"{res.classifier_name}_cm"] = cm_path

            # Save normalized confusion matrix
            if save_normalized_cm:
                cm_norm_path = self.save_confusion_matrix(res, normalize="true")
                all_paths[f"{res.classifier_name}_cm_normalized"] = cm_norm_path

        return all_paths

    def save_combined_report(
        self, filename: str = "combined_classification_report.csv"
    ) -> Path:
        """
        Save a combined classification report CSV with results from all classifiers.

        Returns:
            Path to saved CSV file
        """
        if not self.results:
            raise ValueError(
                "No results to save. Add results using add_result() first."
            )

        all_reports = []

        for classifier_name, result in self.results.items():
            # Generate report dict
            report_dict = classification_report(
                result.y_true,
                result.y_pred,
                target_names=self.class_names,
                output_dict=True,
            )

            # Flatten and add classifier info
            for metric_name, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    row = {"classifier": classifier_name, "class": metric_name}
                    row.update(metrics)
                else:
                    row = {
                        "classifier": classifier_name,
                        "class": metric_name,
                        "value": metrics,
                    }

                # Add ROC-AUC if available
                if result.has_probabilities:
                    try:
                        n_classes = len(np.unique(result.y_true))
                        if n_classes == 2:
                            row["roc_auc"] = roc_auc_score(result.y_true, result.y_prob)
                        else:
                            row["roc_auc"] = roc_auc_score(
                                result.y_true,
                                result.y_prob,
                                multi_class="ovr",
                                average="macro",
                            )
                    except Exception:
                        pass

                all_reports.append(row)

        # Create DataFrame and save
        combined_df = pd.DataFrame(all_reports)
        output_path = self.output_dir / filename
        combined_df.to_csv(output_path, index=False)

        self.console.print(
            f"\n✓ Combined classification report saved: {output_path}", style="success"
        )
        return output_path

    def save_comparison_roc_curves(
        self, filename: str = "comparison_roc_curves.png", figsize: tuple = (10, 8)
    ) -> Optional[Path]:
        """
        Save a single plot comparing ROC curves from all classifiers.

        Returns:
            Path to saved image, or None if no classifiers have probabilities
        """
        results_with_probs = [r for r in self.results.values() if r.has_probabilities]

        if not results_with_probs:
            self.console.print(
                "No classifiers with probability predictions to compare.",
                style="warning",
            )
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Determine if we're in multi-class mode (check first result)
        first_result = results_with_probs[0]
        n_classes = len(np.unique(first_result.y_true))

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_with_probs)))

        # Plot each classifier
        for result, color in zip(results_with_probs, colors):
            if n_classes == 2:
                # Binary: single curve per classifier
                fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
                roc_auc = roc_auc_score(result.y_true, result.y_prob)
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    lw=2,
                    label=f"{result.classifier_name} (AUC = {roc_auc:.3f})",
                )
            else:
                # Multi-class: macro-averaged AUC, plot mean ROC across classes
                roc_auc = roc_auc_score(
                    result.y_true, result.y_prob, multi_class="ovr", average="macro"
                )
                # For comparison plot, compute mean ROC across all classes
                classes = np.unique(result.y_true)
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                for idx, cls in enumerate(classes):
                    y_binary = (result.y_true == cls).astype(int)
                    y_score = result.y_prob[:, idx]
                    fpr, tpr, _ = roc_curve(y_binary, y_score)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                mean_tpr = np.mean(tprs, axis=0)
                ax.plot(
                    mean_fpr,
                    mean_tpr,
                    color=color,
                    lw=2,
                    label=f"{result.classifier_name} (Macro AUC = {roc_auc:.3f})",
                )

        # Plot diagonal
        ax.plot(
            [0, 1],
            [0, 1],
            color="#94a3b8",
            lw=1.5,
            linestyle="--",
            label="Random Classifier",
        )

        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        # Determine eval mode from first result
        first_result = results_with_probs[0]
        eval_text = (
            "Holdout Test Set" if first_result.cv_folds == 0
            else f"{first_result.cv_folds}-Fold CV"
        )
        ax.set_title(
            f"ROC Curve Comparison [{eval_text}]\nDataset: {self.dataset_name}/{self.data_source}",
            fontsize=14,
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        self.console.print(
            f"\n✓ Comparison ROC curves saved: {output_path}", style="success"
        )
        return output_path


def compute_and_store_roc(
    y: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    classifier_name: str = "Classifier",
) -> Dict[str, Any]:
    """
    Compute and store ROC curve data and plot.

    For binary classification, stores single ROC curve.
    For multi-class, stores per-class OvR ROC curves with macro-averaged AUC.

    Args:
        y: True labels
        y_prob: Predicted probabilities (1D for binary, 2D for multi-class)
        output_path: Base path for output (without extension)
        classifier_name: Name of the classifier for labeling

    Returns:
        Dictionary with ROC data (fpr, tpr, thresholds, roc_auc)
    """
    n_classes = len(np.unique(y))
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        # Binary classification
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = roc_auc_score(y, y_prob)

        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "roc_auc": roc_auc}

        ax.plot(
            fpr,
            tpr,
            color="#2563eb",
            lw=2,
            label=f"{classifier_name} (AUC = {roc_auc:.3f})",
        )
    else:
        # Multi-class: per-class OvR ROC curves
        classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        roc_data = {"roc_auc_macro": None, "per_class": {}}

        for idx, (cls, color) in enumerate(zip(classes, colors)):
            y_binary = (y == cls).astype(int)
            y_score = y_prob[:, idx]

            fpr, tpr, thresholds = roc_curve(y_binary, y_score)
            class_auc = roc_auc_score(y_binary, y_score)

            roc_data["per_class"][str(cls)] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "roc_auc": class_auc,
            }

            ax.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                label=f"Class {cls} (AUC = {class_auc:.3f})",
            )

        roc_auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
        roc_data["roc_auc_macro"] = roc_auc
        roc_data["roc_auc"] = roc_auc

    # Store ROC data as NPZ
    npz_path = str(output_path) + ".npz"
    np.savez(npz_path, **{k: v for k, v in roc_data.items() if not isinstance(v, dict)})

    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    title_suffix = "" if n_classes == 2 else f" (Macro AUC = {roc_auc:.3f})"
    ax.set_title(f"Receiver Operating Characteristic{title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    png_path = str(output_path) + ".png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return roc_data
