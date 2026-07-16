from __future__ import annotations

from typing import Optional

import torch

from configs.default import ExperimentConfig
from methods.common import maybe_resize, method_namespace, poison_labels
from methods.poisoning import apply_trigger, make_poison_mask
from methods.triggers import apply_wanet_cross_trigger
from utils.seed import seed_wanet_config


NAME = "wanet"


def _backdoor_params(config: ExperimentConfig):
    backdoor = getattr(config, "backdoor", None)
    data_range = getattr(getattr(config, "data", None), "input_range", (0.0, 1.0))
    clamp_min = float(getattr(backdoor, "clamp_min", data_range[0]))
    clamp_max = float(getattr(backdoor, "clamp_max", data_range[1]))
    poison_rate = float(getattr(backdoor, "poison_rate", getattr(config.train, "poison_rate", 0.05)))
    target_label = int(getattr(backdoor, "target_label", getattr(config.train, "target_label", 0)))
    return clamp_min, clamp_max, poison_rate, target_label


def _cross_mask(y: torch.Tensor, poison_mask: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    mask = torch.zeros_like(y, dtype=torch.bool)
    poison_count = int(poison_mask.sum().item())
    cross_ratio = float(getattr(config.wanet, "cross_ratio", 0.0))
    cross_count = int(round(poison_count * max(cross_ratio, 0.0)))
    if cross_count <= 0:
        return mask

    candidates = (~poison_mask).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return mask
    count = min(cross_count, int(candidates.numel()))
    perm = torch.randperm(candidates.numel(), device=y.device)[:count]
    mask[candidates.index_select(0, perm)] = True
    return mask


def _apply_cross_trigger(x: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    clamp_min, clamp_max, _, _ = _backdoor_params(config)
    wanet = getattr(config, "wanet")
    return apply_wanet_cross_trigger(
        x,
        s=float(wanet.s),
        grid_res=int(wanet.grid_res),
        noise_s=float(getattr(wanet, "noise_s", 1.0)),
        align_corners=bool(wanet.align_corners),
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        seed=seed_wanet_config(config),
        scale_mode=str(getattr(wanet, "scale_mode", "wanet")),
        normalize=str(getattr(wanet, "normalize", "mean_abs")),
        upsample_mode=str(getattr(wanet, "upsample_mode", "bicubic")),
        grid_rescale=float(getattr(wanet, "grid_rescale", 1.0)),
        sample_mode=str(getattr(wanet, "sample_mode", "bilinear")),
        padding_mode=str(getattr(wanet, "padding_mode", "zeros")),
    )


def _poison_train_batch(x: torch.Tensor, y: torch.Tensor, config: ExperimentConfig):
    _, _, poison_rate, target_label = _backdoor_params(config)
    poison_mask = make_poison_mask(
        y,
        poison_rate=poison_rate,
        target_label=target_label,
        exclude_target=True,
    )

    poisoned = x.clone()
    poison_y = y.clone()
    if poison_mask.any():
        poisoned[poison_mask] = apply_trigger(x[poison_mask], NAME, config)
        poison_y[poison_mask] = int(target_label)

    cross = _cross_mask(y, poison_mask, config)
    if cross.any():
        poisoned[cross] = _apply_cross_trigger(x[cross], config)
    return poisoned, poison_y, poison_mask


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
        return _poison_train_batch(x, y, config)

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
    return method_namespace(NAME, config, poison_batch, train, generator=generator)
