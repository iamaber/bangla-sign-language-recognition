"""Training script for Multi-Modal SPOTER v2 model with WandB integration.

Enhanced SPOTER that uses all available features:
- Hand landmarks: 126 dimensions
- Face landmarks: 1404 dimensions
- Pose landmarks: 99 dimensions
- Total: 1629 dimensions (100% feature utilization)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

# Add parent directories to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import multimodal_spoter
from dataset import BdSLDataset, SignerSplits
from vocab import build_vocab_from_samples

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_spoter_v2")


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-modal SPOTER v2")
    parser.add_argument("train_data", type=Path)
    parser.add_argument("val_data", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-classes", type=int, default=77)
    parser.add_argument(
        "--run-name", type=str, default="spoter_v2_multimodal", help="WandB run name"
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


class CustomScheduler:
    """Custom learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs=5, total_epochs=40, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.warmup_factor = (self.base_lr - self.min_lr) / self.warmup_epochs
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + self.warmup_factor * (self.current_epoch - 1)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def train():
    args = parse_args()
    device = torch.device(args.device)

    wandb_enabled = not args.no_wandb

    if wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={
                "model": "SPOTER v2 (Multi-Modal)",
                "version": "v2_multimodal",
                "architecture": "Transformer with cross-modal attention",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "device": args.device,
                "input_features": {
                    "hands": 126,
                    "face": 1404,
                    "pose": 99,
                    "total": 1629,
                },
                "feature_utilization": "100% (all modalities used)",
                "improvements": [
                    "Multi-modal input (hands + face + pose)",
                    "Cross-modal attention",
                    "Temporal convolution for sign dynamics",
                    "Sign boundary detection",
                ],
            },
        )

    # Load data
    LOGGER.info(f"Loading training data from {args.train_data}")
    LOGGER.info(f"Loading validation data from {args.val_data}")

    train_data = np.load(args.train_data)
    val_data = np.load(args.val_data)

    train_x = torch.from_numpy(train_data["x"]).float()
    train_y = torch.from_numpy(train_data["y"]).long()
    val_x = torch.from_numpy(val_data["x"]).float()
    val_y = torch.from_numpy(val_data["y"]).long()

    LOGGER.info(f"Train data shape: {train_x.shape}")
    LOGGER.info(f"Val data shape: {val_x.shape}")

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    # Initialize multi-modal SPOTER model
    model = multimodal_spoter.MultimodalSPOTERModel(
        hand_dim=126,
        face_dim=1404,
        pose_dim=99,
        d_model=512,
        nhead=12,
        num_classes=args.num_classes,
        num_layers=6,
    ).to(device)

    if wandb_enabled:
        wandb.watch(model, log_freq=100)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CustomScheduler(optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        for batch_idx, (batch_x, batch_y) in enumerate(loader_train):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, boundary_scores = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch_y.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_time = time.time() - start_time

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader_val:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits, boundary_scores = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_y.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Logging
        LOGGER.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} time={train_time:.1f}s"
        )

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_time": train_time,
                }
            )

        # Save best model
        if val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss

            # Save checkpoint
            checkpoint_path = Path(f"spoter_v2_epoch_{epoch + 1}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)

            if wandb_enabled:
                wandb.save(
                    checkpoint_path,
                    name=f"spoter_v2_epoch_{epoch + 1}_best",
                    type="model",
                )

    # Save final model
    final_checkpoint_path = Path("spoter_v2_final.pt")
    torch.save(model.state_dict(), final_checkpoint_path)

    if wandb_enabled:
        wandb.save(final_checkpoint_path, name="spoter_v2_final", type="model")
        wandb.finish()

    LOGGER.info(f"Training complete! Best val accuracy: {best_val_acc:.3f}")
    LOGGER.info(f"Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    train()
