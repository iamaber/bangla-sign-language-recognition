"""Custom optimizers for SPOTER v2 training."""

from __future__ import annotations

import torch
from torch.optim import AdamW


class AdamWWithWarmup:
    """AdamW optimizer with linear warmup and cosine annealing."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        total_epochs: int = 40,
        min_lr: float = 1e-6,
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = lr
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
                1 + torch.cos(torch.tensor(torch.pi * progress, device="cpu"))
            )
            lr = lr.item()

        for param_group in self.param_groups:
            param_group["lr"] = lr


class CosineAnnealingWithWarmRestarts:
    """Cosine annealing with warm restarts."""

    def __init__(
        self,
        optimizer,
        T_max: int,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        eta_max: float = 1e-3,
    ):
        """
        Args:
            optimizer: Wrapped optimizer (e.g., AdamW)
            T_max: Maximum number of iterations
            T_mult: Factor for increasing T_max after each restart
            eta_min: Minimum learning rate
            eta_max: Maximum learning rate
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = 0

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_max:
            self.T_cur = 0
            self.T_max *= self.T_mult

        eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1
            + torch.cos(torch.tensor(torch.pi * self.T_cur / self.T_max, device="cpu"))
        )
        eta_t = eta_t.item()

        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param.data = param.data * (eta_t / (self.base_lr + 1e-8))  # Update lr
