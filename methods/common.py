from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import torch

from configs.default import ExperimentConfig
from training.poison import resize_image
from utils.eval_helpers import target_label


def canonical_method(method: str) -> str:
    method = str(method).lower()
    if method == "input_aware":
        return "inputaware"
    return "wanet" if method == "wanets" else method


def poison_labels(
    y: Optional[torch.Tensor],
    config: ExperimentConfig,
    poison_all: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if y is None:
        return None, torch.empty(0, dtype=torch.bool)

    labels = y.clone()
    target = target_label(config)
    mask = torch.ones_like(y, dtype=torch.bool) if bool(poison_all) else (y != int(target))
    if mask.any():
        labels[mask] = int(target)
    return labels, mask


def maybe_resize(x: torch.Tensor, resolution: Optional[int]) -> torch.Tensor:
    if resolution is None:
        return x
    return resize_image(x, int(resolution))


def attack_input_resolution(config: ExperimentConfig, method: str) -> int:
    method = canonical_method(method)
    if method == "wanet":
        return int(config.data.train_resolution)
    return int(config.data.victim_resolution)


def method_namespace(
    name: str,
    config: ExperimentConfig,
    poison_fn: Callable,
    train_fn: Callable,
    eval_fn: Optional[Callable] = None,
    generator=None,
):
    result = SimpleNamespace(
        name=canonical_method(name),
        config=config,
        generator=generator,
        poison_batch=lambda x, y=None, mode="eval", resolution=None: poison_fn(
            x=x,
            y=y,
            mode=mode,
            resolution=resolution,
            config=config,
            generator=generator,
        ),
        train=lambda *args, **kwargs: train_fn(config, *args, **kwargs),
    )
    if eval_fn is not None:
        result.eval = lambda *args, **kwargs: eval_fn(config, canonical_method(name), *args, **kwargs)
    return result,
        config=config,
        generator=generator,
        poison_batch=lambda x, y=None, mode="eval", resolution=None: poison_fn(
            x=x,
            y=y,
            mode=mode,
            resolution=resolution,
            config=config,
            generator=generator,
        ),
        train=lambda *args, **kwargs: train_fn(config, *args, **kwargs),
        eval=lambda *args, **kwargs: eval_fn(config, canonical_method(name), *args, **kwargs),
    )
