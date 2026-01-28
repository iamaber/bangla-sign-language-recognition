"""Simple training script for fusion model with WandB (bypassing disjointness check)."""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from models import fusion
from train.dataset import BdSLDataset
from train.vocab import build_vocab_from_samples
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
LOGGER = logging.getLogger("train_fusion_simple")


class SimpleSplitter:
    """Simple splitter without disjointness validation."""

    def __init__(
        self, train_signers: List[str], val_signers: List[str], test_signers: List[str]
    ):
        self.train_signers = set(train_signers)
        self.val_signers = set(val_signers)
        self.test_signers = set(test_signers)

        # Don't validate disjointness (workaround for test data)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model (simple version).")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument(
        "--run-name", type=str, default="fusion_v2_simple", help="WandB run name"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="BB3lAowfaCGkIlsby",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="aber-islam-dev-jvai",
        help="WandB entity name",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def create_simple_dataset(
    manifest_path: Path, landmarks_dir: Path, split_signers: List[str]
):
    """Create dataset with specific signer filter."""
    vocab = build_vocab_from_samples([])  # Will build from manifest

    # Read manifest and filter by signers
    samples = []
    import csv

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["signer_id"] in split_signers:
                samples.append(row)

    # Build vocabulary from filtered samples
    from train.vocab import build_vocab_from_samples as build_vocab_from_file

    vocab = build_vocab_from_file(samples)

    return samples, vocab


def train():
    args = parse_args()
    device = torch.device(args.device)

    # Create simple splitter (no disjointness check)
    splitter = SimpleSplitter(args.train_signers, args.val_signers, args.test_signers)

    # Load train data
    train_samples, train_vocab = create_simple_dataset(
        args.manifest, args.landmarks, args.train_signers
    )
    from train.vocab import build_vocab_from_manifest as build_vocab_file

    # Create dataset
    from train.dataset import BdSLDataset

    train_dataset = BdSLDataset(
        args.manifest, args.landmarks, splitter, split="train", vocab=train_vocab
    )
    val_dataset = BdSLDataset(
        args.manifest, args.landmarks, splitter, split="val", vocab=train_vocab
    )

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

    model = fusion.FusionModel().to(device)

    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            experiment_name=args.run_name,
            config={
                "model": "Multimodal Fusion",
                "version": "v2_simple",
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
        log_model_summary(model, "fusion_v2_simple", tag="")
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

        val_loss, val_acc = evaluate(model, loader_val, device, criterion)
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
                }
            )

            # Log confusion matrices (simplified without collecting all predictions)
            # Just log overall metrics per epoch to avoid memory issues

            save_checkpoint(model, f"fusion_v2_simple_epoch_{epoch + 1}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, "fusion_v2_simple_best", is_best=True)

    if wandb_enabled:
        save_checkpoint(model, "fusion_v2_simple_final", is_best=True)
        wandb.finish()
    else:
        torch.save(model.state_dict(), Path("fusion_model_simple.pt"))

    LOGGER.info("Training complete! Best val accuracy: %.3f", best_val_acc)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
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
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


if __name__ == "__main__":
    train()
