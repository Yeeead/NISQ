from __future__ import annotations

from typing import Optional

import torch

from configs.default import ExperimentConfig
from methods.common import maybe_resize, method_namespace, poison_labels
from methods.poisoning import apply_poison_to_batch, apply_trigger


NAME = "blended"


def poison_batch(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "eval",
    resolution: Optional[int] = None,
    config: ExperimentConfig = None,
    generator=None,
):
    x = maybe_resize(x, resolution)
    if str(mode).lower() == "train" and y is not None:
        return apply_poison_to_batch(x, y, NAME, config)

    poison_y, mask = poison_labels(y, config, poison_all=False)
    poisoned = x.clone()
    source = torch.ones(x.size(0), dtype=torch.bool, device=x.device) if y is None else mask
    if source.any():
        poisoned[source] = apply_trigger(x[source], NAME, config)
    return poisoned, poison_y, source


def train(config: ExperimentConfig, *args, **kwargs):
    from training.train_baselines import train_backdoor_baseline

    return train_backdoor_baseline(NAME, config, *args, **kwargs)


def build_generator(config: ExperimentConfig, device):
    return None




def build_method(config: ExperimentConfig, generator=None):
    return method_namespace(NAME, config, poison_batch, train, eval, generator=generator)
