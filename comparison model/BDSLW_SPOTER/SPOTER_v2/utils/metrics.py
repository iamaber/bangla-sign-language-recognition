"""SPOTER v2 evaluation metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("metrics")


class SPOTERv2Metrics:
    """Comprehensive evaluation metrics for SPOTER v2."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_logits: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels (batch_size,)
            y_pred: Predicted labels (batch_size,)
            y_logits: Predicted logits (batch_size, num_classes)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(sk_metrics.accuracy_score(y_true, y_pred))

        # Top-k accuracies
        if y_logits is not None:
            metrics["top3_accuracy"] = self._top_k_accuracy(y_true, y_logits, k=3)
            metrics["top5_accuracy"] = self._top_k_accuracy(y_true, y_logits, k=5)

        # Precision, Recall, F1 (macro and per-class)
        precision_macro = sk_metrics.precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = sk_metrics.recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1_macro = sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics["precision_macro"] = precision_macro
        metrics["recall_macro"] = recall_macro
        metrics["f1_macro"] = f1_macro

        # Per-class metrics
        precision_per_class = sk_metrics.precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = sk_metrics.recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1_per_class = sk_metrics.f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        for i, (prec, rec, f1) in enumerate(
            zip(precision_per_class, recall_per_class, f1_per_class)
        ):
            if i < len(self.class_names):
                class_name = self.class_names[i]
                metrics[f"precision_{class_name}"] = prec
                metrics[f"recall_{class_name}"] = rec
                metrics[f"f1_{class_name}"] = f1

        # Confusion matrix
        cm = sk_metrics.confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm

        # Additional metrics
        metrics["per_class_accuracy"] = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y_true == i
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            else:
                class_acc = 0.0
            metrics[f"accuracy_{class_name}"] = float(class_acc)

        # Class distribution
        class_counts = Counter(y_true)
        metrics["class_support"] = {}
        for i, class_name in enumerate(self.class_names):
            metrics[f"support_{class_name}"] = class_counts.get(i, 0)

        return metrics

    def _top_k_accuracy(
        self, y_true: np.ndarray, y_logits: np.ndarray, k: int
    ) -> float:
        """Compute top-k accuracy."""
        top_k_preds = np.argsort(y_logits, axis=1)[:, -k:]
        top_k_correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                top_k_correct += 1
        return top_k_correct / len(y_true)

    def log_metrics(
        self, metrics: Dict[str, float], epoch: int = None, step: int = None
    ):
        """Log metrics with nice formatting."""
        LOGGER.info(f"\n{'=' * 60}")
        if epoch is not None:
            LOGGER.info(f"Metrics after epoch {epoch}")
        if step is not None:
            LOGGER.info(f"Metrics at step {step}")
        LOGGER.info(f"{'=' * 60}\n")

        # Overall metrics
        LOGGER.info(f"Overall Accuracy: {metrics['accuracy']:.2%}")

        if "top3_accuracy" in metrics:
            LOGGER.info(f"Top-3 Accuracy: {metrics['top3_accuracy']:.2%}")
        if "top5_accuracy" in metrics:
            LOGGER.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")

        LOGGER.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        LOGGER.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        LOGGER.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")

        # Per-class metrics
        LOGGER.info(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names[:10]):  # Show first 10 classes
            if f"precision_{class_name}" in metrics:
                LOGGER.info(
                    f"  {class_name:15}: "
                    f"Acc={metrics.get(f'accuracy_{class_name}', 0):.2%}, "
                    f"P={metrics.get(f'precision_{class_name}', 0):.4f}, "
                    f"R={metrics.get(f'recall_{class_name}', 0):.4f}, "
                    f"F1={metrics.get(f'f1_{class_name}', 0):.4f}"
                )

        if len(self.class_names) > 10:
            LOGGER.info(f"  ... and {len(self.class_names) - 10} more classes")

        LOGGER.info(f"{'=' * 60}\n")

    def save_confusion_matrix(
        self, cm: np.ndarray, output_dir: Path, epoch: int = None
    ):
        """Save confusion matrix as image and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        if epoch is not None:
            cm_path = output_dir / f"confusion_matrix_epoch_{epoch}.csv"
        else:
            cm_path = output_dir / "confusion_matrix.csv"
        np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
        LOGGER.info(f"Saved confusion matrix to {cm_path}")

        # Save as image
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Only show first 20 classes to avoid overcrowding
        num_classes_to_show = min(20, len(self.class_names))
        cm_subset = cm[:num_classes_to_show, :num_classes_to_show]
        class_names_subset = self.class_names[:num_classes_to_show]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_subset,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names_subset,
            yticklabels=class_names_subset,
            ax=ax,
        )
        ax.set_title("Confusion Matrix (showing first 20 classes)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()

        if epoch is not None:
            img_path = output_dir / f"confusion_matrix_epoch_{epoch}.png"
        else:
            img_path = output_dir / "confusion_matrix.png"
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info(f"Saved confusion matrix image to {img_path}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    metrics_calculator: SPOTERv2Metrics,
) -> Dict[str, float]:
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    all_true = []
    all_pred = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            # Assuming batch has 'sign_label' key
            if "sign_label" in batch:
                outputs = model(
                    batch["hand_left"],
                    batch["hand_right"],
                    batch["face"],
                    batch["pose"],
                )
                loss = criterion(outputs, batch["sign_label"])

                total_loss += loss.item() * batch["sign_label"].size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == batch["sign_label"]).sum().item()

                all_true.extend(batch["sign_label"].cpu().numpy().tolist())
                all_pred.extend(pred.cpu().numpy().tolist())
                all_logits.extend(outputs.cpu().numpy().tolist())
            elif "label" in batch:
                # Fallback for different key name
                outputs = model(
                    batch["hand_left"],
                    batch["hand_right"],
                    batch["face"],
                    batch["pose"],
                )
                loss = criterion(outputs, batch["label"])

                total_loss += loss.item() * batch["label"].size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == batch["label"]).sum().item()

                all_true.extend(batch["label"].cpu().numpy().tolist())
                all_pred.extend(pred.cpu().numpy().tolist())
                all_logits.extend(outputs.cpu().numpy().tolist())

    total_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    metrics = metrics_calculator.compute_metrics(
        np.array(all_true), np.array(all_pred), np.array(all_logits)
    )

    metrics["loss"] = total_loss
    metrics["accuracy"] = accuracy

    return metrics
