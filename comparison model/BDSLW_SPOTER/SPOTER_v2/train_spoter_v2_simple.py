"""Training script for SPOTER v2 with WandB integration (99 dims - pose only for direct comparison).

Enhanced SPOTER that uses only pose landmarks (99 dimensions)
for fair comparison with SPOTER v1, but with improved architecture.
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
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Add parent directories to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from multimodal_spoter import MultimodalSPOTERModel, PositionalEncoding

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_spoter_v2_simple")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPOTER v2 (simple, 99 dims)")
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
        "--run-name", type=str, default="spoter_v2_simple", help="WandB run name"
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


class SPOTERv2Model(nn.Module):
    """Simplified SPOTER v2 with 99-dim pose input for direct comparison with SPOTER v1.

    Improvements over SPOTER v1:
    - Deeper architecture (512 dims vs 256)
    - More heads (12 vs 9)
    - More layers (6 vs 4)
    - Better positional encoding
    - Enhanced classifier with dropout
    """

    def __init__(
        self,
        pose_dim=99,
        d_model=512,
        nhead=12,
        num_classes=77,
        num_layers=6,
        dropout=0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head (deeper than SPOTER v1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Pose landmarks (batch, seq_len, 99)
        """
        # Input projection
        x = self.input_projection(x)

        # Apply positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


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
                "model": "SPOTER v2 (Simple, 99 dims)",
                "version": "v2_improved",
                "architecture": "Enhanced Transformer",
                "improvements": [
                    "Deeper architecture (512 vs 256 dims)",
                    "More heads (12 vs 9)",
                    "More layers (6 vs 4)",
                    "Better positional encoding",
                    "Enhanced classifier (4 layers vs 1)",
                ],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "device": args.device,
                "input_features": {
                    "pose_landmarks": 99,
                    "multi_modal": False,  # Simple comparison
                    "feature_utilization": "100% (pose only)",
                },
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
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

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

    # Initialize SPOTER v2 model
    model = SPOTERv2Model(
        pose_dim=99,
        d_model=512,
        nhead=12,
        num_classes=args.num_classes,
        num_layers=6,
        dropout=0.15,
    ).to(device)

    if wandb_enabled:
        wandb.watch(model, log_freq=100)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
            logits = model(batch_x)
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

                logits = model(batch_x)
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
            checkpoint_path = Path(f"spoter_v2_simple_epoch_{epoch + 1}_best.pt")
            torch.save(model.state_dict(), checkpoint_path)

            if wandb_enabled:
                wandb.save(
                    checkpoint_path,
                    name=f"spoter_v2_simple_epoch_{epoch + 1}_best",
                    type="model",
                )

    # Save final model
    final_checkpoint_path = Path("spoter_v2_simple_final.pt")
    torch.save(model.state_dict(), final_checkpoint_path)

    if wandb_enabled:
        wandb.save(final_checkpoint_path, name="spoter_v2_simple_final", type="model")
        wandb.finish()

    LOGGER.info(f"Training complete! Best val accuracy: {best_val_acc:.3f}")
    LOGGER.info(f"Final model saved to {final_checkpoint_path}")
    LOGGER.info(f"\n🎯 SPOTER v2 Simple Summary:")
    LOGGER.info(f"  Input: 99 dims (pose only)")
    LOGGER.info(f"  Architecture: Enhanced Transformer (512 dims, 6 layers, 12 heads)")
    LOGGER.info(
        f"  Improvements: Deeper, more heads, better positional encoding, deeper classifier"
    )
    LOGGER.info(f"  Best Val Accuracy: {best_val_acc:.3f}%")


if __name__ == "__main__":
    train()
