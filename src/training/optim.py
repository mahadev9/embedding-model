"""StableAdamW and learning-rate schedule.

StableAdamW (Wortsman et al., 2023) is AdamW with Adafactor-style update clipping: each tensor's
learning rate is divided by max(1, RMS), where RMS = sqrt(mean(grad^2 / v_hat)). It replaces global
gradient clipping and is what ModernBERT trained with.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class StableAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable | list[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        # Per-tensor update RMS from the latest step (a GPU tensor; read it only when logging).
        self.last_rms: torch.Tensor | None = None

    @torch.no_grad()
    def step(self, closure=None) -> None:
        rms_values: list[torch.Tensor] = []
        for group in self.param_groups:
            lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias1 = 1 - beta1**step
                bias2 = 1 - beta2**step
                v_hat = exp_avg_sq / bias2
                rms = (grad.pow(2) / v_hat.clamp(min=eps**2)).mean().sqrt()
                rms_values.append(rms)
                lr_t = lr / rms.clamp(min=1.0)  # a 0-dim tensor: no host synchronization

                if group["weight_decay"]:
                    p.mul_(1 - lr_t * group["weight_decay"])
                p.sub_(lr_t * (exp_avg / bias1) / (v_hat.sqrt() + eps))
        if rms_values:
            self.last_rms = torch.stack(rms_values)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay only 2-D weight matrices of linear layers; never norms, embeddings or biases."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_matrix = param.ndim == 2 and "embed" not in name
        (decay if is_matrix else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def wsd_lr(
    step: int, total_steps: int, peak_lr: float, warmup_steps: int, decay_steps: int
) -> float:
    """Warmup-stable-decay: linear warmup, constant, then 1 - sqrt decay to zero."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return peak_lr
    progress = (step - decay_start) / max(decay_steps, 1)
    return peak_lr * (1.0 - math.sqrt(min(progress, 1.0)))
