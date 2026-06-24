from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def attack_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long())


def l1_loss(delta: torch.Tensor) -> torch.Tensor:
    return delta.abs().mean()


def zero_mean_loss(delta: torch.Tensor) -> torch.Tensor:
    return delta.mean(dim=(1, 2, 3)).abs().mean()


def qinr_total_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    loss_config,
) -> Dict[str, torch.Tensor]:
    attack = attack_loss(logits, target)
    l1 = l1_loss(delta)
    zero_mean = zero_mean_loss(delta)
    total = (
        attack
        + float(loss_config.lambda_l1) * l1
        + float(loss_config.lambda_zero_mean) * zero_mean
    )
    return {
        "loss": total,
        "attack_loss": attack.detach(),
        "l1_loss": l1.detach(),
        "zero_mean_loss": zero_mean.detach(),
    }
