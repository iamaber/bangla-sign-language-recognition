"""Shared WandB utilities for both models."""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

GRAMMAR_IDX_TO_TAG = ["neutral", "question", "negation", "happy", "sad"]


def init_wandb(
    project: str, entity: str, experiment_name: str, config: dict = None
) -> None:
    wandb.init(
        project=project,
        entity=entity,
        name=experiment_name,
        config=config,
        settings=wandb.Settings(start_method="thread"),
    )


def log_confusion_matrix(
    y_true, y_pred, class_names: list, step: int = None, tag: str = ""
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_path = Path(f"confusion_matrix_{tag}.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
    wandb.save(
        cm_path,
        f"confusion_matrix_{tag}_step_{step}" if step else f"confusion_matrix_{tag}",
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {tag}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    img_path = Path(f"confusion_matrix_{tag}.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    wandb.log({f"confusion_matrix_{tag}": wandb.Image(str(img_path))}, step=step)
    wandb.save(
        img_path,
        f"confusion_matrix_{tag}_image_step_{step}"
        if step
        else f"confusion_matrix_{tag}_image",
    )


def log_metrics(metrics: dict, step: int = None) -> None:
    for key, value in metrics.items():
        wandb.log({key: value}, step=step)


def log_classification_report(
    y_true, y_pred, class_names: list, step: int = None, tag: str = ""
) -> None:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    for class_name in class_names:
        class_metrics = report.get(class_name, {})
        log_metrics(
            {
                f"{tag}precision_{class_name}": class_metrics.get("precision", 0),
                f"{tag}recall_{class_name}": class_metrics.get("recall", 0),
                f"{tag}f1_{class_name}": class_metrics.get("f1-score", 0),
            },
            step=step,
        )


def save_checkpoint(
    model: torch.nn.Module, checkpoint_name: str, is_best: bool = False
) -> Path:
    checkpoint_path = Path(f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path, name=checkpoint_name, type="model")
    if is_best:
        wandb.save(checkpoint_path, name=f"best_{checkpoint_name}", type="model")
    return checkpoint_path


def log_model_summary(model: torch.nn.Module, model_name: str, tag: str = "") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update(
        {
            f"{tag}{model_name}_total_params": total_params,
            f"{tag}{model_name}_architecture": str(model),
        }
    )
