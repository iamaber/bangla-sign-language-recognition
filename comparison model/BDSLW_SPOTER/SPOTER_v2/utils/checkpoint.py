"""Checkpoint management utility for SPOTER v2."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import wandb

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("checkpoint")


def save_checkpoint(model: torch.nn.Module, checkpoint_name: str,
                   output_dir: Path, is_best: bool = False,
                   epoch: int = None, metrics: Optional[dict] = None) -> Path:
    """Save model checkpoint as both local file and WandB artifact.

    Args:
        model: PyTorch model
        checkpoint_name: Name for checkpoint file
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        epoch: Current epoch number
        metrics: Dictionary of metrics (accuracy, loss, etc.)

    Returns:
        Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{checkpoint_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGGER.info(f"✅ Saved checkpoint: {checkpoint_name}")
    if is_best:
        LOGGER.info(f"   ⭐ Best model so far (epoch {epoch})")
    if epoch is not None:
        LOGGER.info(f"   Epoch: {epoch}")
    if metrics is not None:
        if 'val_accuracy' in metrics:
            LOGGER.info(f"   Val Acc: {metrics['val_accuracy']:.4f}")
        if 'val_loss' in metrics:
            LOGGER.info(f"   Val Loss: {metrics['val_loss']:.4f}")

    # Save to WandB
    try:
        wandb.save(str(checkpoint_path), name=checkpoint_name, type="model")
        if is_best:
            wandb.save(str(checkpoint_path), name=f"best_{checkpoint_name}", type="model")
        LOGGER.info(f"   📤 Uploaded to WandB")
    except Exception as e:
        LOGGER.warning(f"   ⚠️  Failed to upload to WandB: {e}")

    # Keep only best and last 3 checkpoints
    if is_best or (epoch is not None and epoch % 10 == 0):
        try:
            _cleanup_old_checkpoints(output_dir, keep=3, best_name=checkpoint_name)
        except Exception as e:
            LOGGER.warning(f"   ⚠️  Failed to cleanup old checkpoints: {e}")

    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module,
                   device: torch.device) -> None:
    """Load model checkpoint."""
    if not checkpoint_path.exists():
        LOGGER.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    LOGGER.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    LOGGER.info("✅ Checkpoint loaded successfully")


def _cleanup_old_checkpoints(output_dir: Path, keep: int = 3, best_name: str = "best"):
    """Remove old checkpoints, keeping only best and last N."""
    try:
        # Get all checkpoint files
        checkpoint_files = sorted(output_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

        # Identify best checkpoint
        best_file = None
        if f"best_{best_name}.pt" in [f.name for f in checkpoint_files]:
            best_file = next(f for f in checkpoint_files if f.name == f"best_{best_name}.pt")

        # Files to keep
        files_to_keep = []
        if best_file:
            files_to_keep.append(best_file)
        files_to_keep.extend(checkpoint_files[:keep-1])
        else:
            files_to_keep.extend(checkpoint_files[:keep])

        # Remove old checkpoints
        for checkpoint_file in checkpoint_files:
            if checkpoint_file not in files_to_keep:
                checkpoint_file.unlink()
                LOGGER.info(f"   🗑️  Removed old checkpoint: {checkpoint_file.name}")

    except Exception as e:
        LOGGER.warning(f"Cleanup failed: {e}")


def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Get the most recent checkpoint."""
    checkpoint_files = sorted(output_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoint_files[0] if checkpoint_files else None
