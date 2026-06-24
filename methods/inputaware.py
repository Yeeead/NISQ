from __future__ import annotations

from typing import Optional

import torch

from configs.default import ExperimentConfig
from methods.common import maybe_resize, method_namespace, poison_labels
from training.poison import constrain_actual_delta, rebuild_poisoned_from_actual_delta


NAME = "inputaware"


def poison_batch(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "eval",
    resolution: Optional[int] = None,
    config: ExperimentConfig = None,
    generator=None,
):
    if generator is None:
        raise ValueError("Input-aware poison_batch requires a generator.")

    x = maybe_resize(x, resolution)
    poison_y, mask = poison_labels(y, config, poison_all=False)
    poisoned = x.clone()
    source = torch.ones(x.size(0), dtype=torch.bool, device=x.device) if y is None else mask
    if source.any():
        raw_delta, _, _ = generator.trigger_delta(
            x=x[source],
            trigger_source=x[source],
            input_range=tuple(config.data.input_range),
        )
        delta = constrain_actual_delta(raw_delta, config)
        poisoned[source] = rebuild_poisoned_from_actual_delta(x[source], delta, config)
    return poisoned, poison_y, source


def train(config: ExperimentConfig, *args, **kwargs):
    from training.train_input_aware import run_input_aware_training

    return run_input_aware_training(config, *args, **kwargs)


def build_generator(config: ExperimentConfig, device):
    from models.factory import build_input_aware_generator

    return build_input_aware_generator(config.model, config.inputaware).to(device)




def build_method(config: ExperimentConfig, generator=None):
    return method_namespace(NAME, config, poison_batch, train, generator=generator)
