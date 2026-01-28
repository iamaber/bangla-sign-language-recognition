"""Training script for multimodal fusion BdSL model with WandB integration."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from models.fusion import FusionModel
from train.dataset import BdSLDataset, SignerSplits
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from wandb_utils import (
    init_wandb,
    log_confusion_matrix,
    log_metrics,
    save_checkpoint,
    log_model_summary,
    GRAMAR_IDX_TO_TAG,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_fusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker count."
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable DataLoader pin_memory (enabled by default for GPU training).",
    )
    parser.add_argument(
        "--run-name", type=str, default="fusion_v2", help="WandB run name"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="BB3lAowfaCGkIlsby",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="wandb_v1", help="WandB entity name"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)
    signer_splits = SignerSplits(
        args.train_signers, args.val_signers, args.test_signers
    )
    train_dataset = BdSLDataset(
        args.manifest, args.landmarks, signer_splits, split="train"
    )
    val_dataset = BdSLDataset(args.manifest, args.landmarks, signer_splits, split="val")

    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )

    model = FusionModel().to(device)

    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            experiment_name=args.run_name,
            config={
                "model": "Multimodal Fusion",
                "version": "v2_emotion_integrated",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "signer_splits": f"train: {args.train_signers}, val: {args.val_signers}, test: {args.test_signers}",
                "hand_points": 21,
                "face_points": 468,
                "pose_points": 33,
                "device": args.device,
            },
        )
        log_model_summary(model, "fusion_v2", tag="")
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader_train:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            optimizer.zero_grad()
            sign_logits, grammar_logits = model(batch)
            loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                grammar_logits, batch["grammar_label"]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["sign_label"].size(0)
        scheduler.step()

        val_loss, val_acc, y_true, y_pred = evaluate_with_preds(
            model, loader_val, device, criterion
        )
        LOGGER.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f",
            epoch + 1,
            total_loss / len(loader_train.dataset),
            val_loss,
            val_acc,
        )

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / len(loader_train.dataset),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "sign_task_loss": total_loss
                    / len(loader_train.dataset),  # Approximate
                    "grammar_task_loss": 0.0,  # Not separately tracked
                }
            )

            log_confusion_matrix(
                y_true,
                y_pred[0],
                class_names=train_dataset.vocab.idx_to_label,
                step=epoch + 1,
                tag="sign",
            )
            log_confusion_matrix(
                y_true, y_pred[1], GRAMAR_IDX_TO_TAG, step=epoch + 1, tag="grammar"
            )

            save_checkpoint(model, f"fusion_v2_epoch_{epoch + 1}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, "fusion_v2_best", is_best=True)

    if wandb_enabled:
        save_checkpoint(model, "fusion_v2_final", is_best=True)
        wandb.finish()
    else:
        torch.save(model.state_dict(), Path("fusion_model.pt"))

    LOGGER.info("Training complete! Best val accuracy: %.3f", best_val_acc)


def evaluate_with_preds(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float, List[int], Tuple[List[int], List[int]]]:
    """Evaluate model and return loss, accuracy, and predictions for confusion matrices."""
    model.eval()
    total_loss = 0.0
    correct = 0
    all_true = []
    all_pred_sign = []
    all_pred_grammar = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            sign_logits, grammar_logits = model(batch)
            loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                grammar_logits, batch["grammar_label"]
            )
            total_loss += loss.item() * batch["sign_label"].size(0)
            correct += (sign_logits.argmax(dim=1) == batch["sign_label"]).sum().item()

            all_true.extend(batch["sign_label"].cpu().numpy().tolist())
            all_pred_sign.extend(sign_logits.argmax(dim=1).cpu().numpy().tolist())
            all_pred_grammar.extend(grammar_logits.argmax(dim=1).cpu().numpy().tolist())

    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset),
        all_true,
        (all_pred_sign, all_pred_grammar),
    )


if __name__ == "__main__":
    train()
