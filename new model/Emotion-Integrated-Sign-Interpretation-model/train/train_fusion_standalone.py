import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion import FusionModel
from dataset import BdSLDataset
from vocab import build_vocab_from_samples

# Now we can safely import other modules that depend on these
import numpy as np
import sklearn.metrics as sk_metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_fusion_standalone")

GRAMMAR_IDX_TO_TAG = ["neutral", "question", "negation", "happy", "sad"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model (standalone).")
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
    parser.add_argument("--run-name", type=str, default="fusion_v2_standalone")
    parser.add_argument("--wandb-project", type=str, default="BB3lAowfaCGkIlsby")
    parser.add_argument("--wandb-entity", type=str, default="aber-islam-dev-jvai")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def save_checkpoint(
    model: nn.Module, checkpoint_name: str, is_best: bool = False
) -> None:
    """Save model checkpoint as WandB artifact."""
    checkpoint_path = Path(f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path, name=checkpoint_name, type="model")
    if is_best:
        wandb.save(checkpoint_path, name=f"best_{checkpoint_name}", type="model")


def log_confusion_matrix(
    y_true, y_pred, class_names: list, step: int = None, tag: str = ""
) -> None:
    """Log confusion matrix as WandB artifact."""
    cm = sk_metrics.confusion_matrix(y_true, y_pred)
    cm_path = Path(f"confusion_matrix_{tag}.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
    wandb.save(
        cm_path,
        f"confusion_matrix_{tag}_step_{step}" if step else f"confusion_matrix_{tag}",
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

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


def log_model_summary(model: nn.Module, model_name: str, tag: str = "") -> None:
    """Log model architecture parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update(
        {
            f"{tag}{model_name}_total_params": total_params,
            f"{tag}{model_name}_architecture": str(model),
        }
    )


def train():
    args = parse_args()
    device = torch.device(args.device)

    # Create custom splitter that doesn't enforce disjointness
    class SimpleSplitter:
        def __init__(self, train_signers, val_signers, test_signers):
            self.train_signers = set(train_signers)
            self.val_signers = set(val_signers)
            self.test_signers = set(test_signers)

    splitter = SimpleSplitter(args.train_signers, args.val_signers, args.test_signers)
    train_dataset = BdSLDataset(args.manifest, args.landmarks, splitter, split="train")
    val_dataset = BdSLDataset(args.manifest, args.landmarks, splitter, split="val")

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

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={
                "model": "Multimodal Fusion",
                "version": "v2_standalone",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "signer_splits": f"train: {args.train_signers}, val: {args.val_signers}, test: {args.test_signers}",
                "hand_points": 21,
                "face_points": 468,
                "pose_points": 33,
                "device": args.device,
            },
            settings=wandb.Settings(start_method="thread"),
        )
        log_model_summary(model, "fusion_v2_standalone", tag="")
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

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        all_true = []
        all_pred_sign = []
        all_pred_grammar = []

        with torch.no_grad():
            for batch in loader_val:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                sign_logits, grammar_logits = model(batch)
                loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                    grammar_logits, batch["grammar_label"]
                )
                val_loss += loss.item() * batch["sign_label"].size(0)

                all_true.extend(batch["sign_label"].cpu().numpy().tolist())
                all_pred_sign.extend(sign_logits.argmax(dim=1).cpu().numpy().tolist())
                all_pred_grammar.extend(
                    grammar_logits.argmax(dim=1).cpu().numpy().tolist()
                )
                val_correct += (
                    (sign_logits.argmax(dim=1) == batch["sign_label"]).sum().item()
                )

        val_loss = val_loss / len(loader_val.dataset)
        val_acc = val_correct / len(loader_val.dataset)

        LOGGER.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f",
            epoch + 1,
            total_loss / len(loader_train.dataset),
            val_loss,
            val_acc,
        )

        # Log to WandB
        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / len(loader_train.dataset),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Log confusion matrices
            log_confusion_matrix(
                all_true,
                all_pred_sign,
                class_names=train_dataset.vocab.idx_to_label,
                step=epoch + 1,
                tag="sign",
            )
            log_confusion_matrix(
                all_true,
                all_pred_grammar,
                GRAMMAR_IDX_TO_TAG,
                step=epoch + 1,
                tag="grammar",
            )

            # Save checkpoints
            save_checkpoint(model, f"fusion_v2_epoch_{epoch + 1}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, "fusion_v2_best", is_best=True)

    # Save final model and finish WandB
    if not args.no_wandb:
        save_checkpoint(model, "fusion_v2_final", is_best=True)
        wandb.finish()
    else:
        torch.save(model.state_dict(), Path("fusion_model_standalone.pt"))

    LOGGER.info("Training complete! Best val accuracy: %.3f", best_val_acc)


if __name__ == "__main__":
    train()
