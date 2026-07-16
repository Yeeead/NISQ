from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from configs.default import ExperimentConfig
from methods.common import maybe_resize, method_namespace
from methods.poisoning import make_poison_mask
from utils.eval_helpers import normalize_for_victim


NAME = "fgsm"


def targeted_fgsm(
    victim: torch.nn.Module,
    x: torch.Tensor,
    target_label: int,
    epsilon: float,
    config: ExperimentConfig,
) -> torch.Tensor:
    if victim is None:
        raise ValueError("FGSM poisoning requires the current victim model.")
    was_training = victim.training
    victim.eval()

    x_adv = x.detach().clone().requires_grad_(True)
    target = torch.full(
        (x_adv.size(0),),
        int(target_label),
        dtype=torch.long,
        device=x_adv.device,
    )
    logits = victim(normalize_for_victim(x_adv, config))
    loss = F.cross_entropy(logits, target)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    low, high = config.data.input_range
    poisoned = torch.clamp(
        x_adv - float(epsilon) * grad.sign(),
        min=float(low),
        max=float(high),
    ).detach()
    victim.train(was_training)
    return poisoned


def poison_batch(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "eval",
    resolution: Optional[int] = None,
    config: ExperimentConfig = None,
    generator=None,
    victim: Optional[torch.nn.Module] = None,
):
    x = maybe_resize(x, resolution)
    target_label = int(config.train.target_label)
    poisoned = x.clone()

    if y is None:
        poison_y = None
        source = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
    elif str(mode).lower() == "train":
        source = make_poison_mask(
            y,
            poison_rate=float(config.backdoor.poison_rate),
            target_label=target_label,
            exclude_target=True,
        )
        poison_y = y.clone()
        poison_y[source] = target_label
    else:
        source = y != target_label
        poison_y = y.clone()
        poison_y[source] = target_label

    if source.any():
        poisoned[source] = targeted_fgsm(
            victim=victim,
            x=x[source],
            target_label=target_label,
            epsilon=float(config.train.epsilon),
            config=config,
        )
    return poisoned, poison_y, source


def train(config: ExperimentConfig, *args, **kwargs):
    from training.train_baselines import train_backdoor_baseline

    return train_backdoor_baseline(NAME, config, *args, **kwargs)


def build_generator(config: ExperimentConfig, device):
    return None


def build_method(config: ExperimentConfig, generator=None):
    return method_namespace(NAME, config, poison_batch, train, generator=generator)
