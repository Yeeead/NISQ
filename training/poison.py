from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from evaluation.perturbation_stats import perturbation_stats
from utils.coords import make_coord_grid


def input_range_tuple(input_range) -> Tuple[float, float]:
    return float(input_range[0]), float(input_range[1])


def resize_image(x: torch.Tensor, resolution: int) -> torch.Tensor:
    if x.shape[-2:] == (int(resolution), int(resolution)):
        return x
    return F.interpolate(x, size=(int(resolution), int(resolution)), mode="bilinear", align_corners=False)


def generate_delta_image(
    generator,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    coord_range: Tuple[float, float],
    device,
    dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    coords = make_coord_grid(
        height=int(height),
        width=int(width),
        batch_size=int(batch_size),
        coord_range=coord_range,
        device=device,
        dtype=dtype,
    )
    delta_flat, aux = generator(coords)
    if delta_flat.dim() != 3:
        raise ValueError("QINR generator must return [B, N, C], got {}".format(tuple(delta_flat.shape)))
    delta = delta_flat.permute(0, 2, 1).reshape(int(batch_size), -1, int(height), int(width))
    delta = delta.clamp(-1.0, 1.0)
    if delta.size(1) == 1 and int(channels) > 1:
        delta = delta.expand(-1, int(channels), -1, -1)
    if delta.size(1) != int(channels):
        raise ValueError("QINR out_channels={} does not match image channels={}".format(delta.size(1), channels))
    aux = dict(aux)
    aux["delta_raw"] = delta
    return delta, aux


def additive_poison(
    x: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float,
    input_range,
) -> torch.Tensor:
    valid_min, valid_max = input_range_tuple(input_range)
    return torch.clamp(x + float(epsilon) * delta, min=valid_min, max=valid_max)


def poison_batch(
    generator,
    x: torch.Tensor,
    epsilon: float,
    input_range,
    coord_range,
    shared_delta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    batch, channels, height, width = x.shape
    delta_batch = 1 if bool(shared_delta) and batch > 0 else batch
    delta, aux = generate_delta_image(
        generator=generator,
        batch_size=delta_batch,
        channels=channels,
        height=height,
        width=width,
        coord_range=tuple(coord_range),
        device=x.device,
        dtype=x.dtype,
    )
    if delta_batch == 1 and batch > 1:
        aux["delta_raw_base"] = delta
        delta = delta.expand(batch, -1, -1, -1)
        aux["delta_raw"] = delta
    poisoned = additive_poison(x, delta, epsilon=epsilon, input_range=input_range)
    aux["poisoned"] = poisoned
    return poisoned, delta, aux


def _config_float(obj, name: str, default: float) -> float:
    return float(getattr(obj, name, default))


def _input_bounds(config) -> Tuple[float, float]:
    low, high = config.data.input_range
    return float(low), float(high)


def constrain_delta(delta: torch.Tensor, config) -> torch.Tensor:
    """
    Apply the same hard QINR perturbation clipping used during training.
    """
    linf_clip = _config_float(config.loss, "delta_linf_clip", 1.0)
    if linf_clip > 0.0:
        return delta.clamp(-linf_clip, linf_clip)
    return delta


def rebuild_poisoned_from_delta(
    x: torch.Tensor,
    delta: torch.Tensor,
    config,
) -> torch.Tensor:
    low, high = _input_bounds(config)
    return torch.clamp(x + float(config.train.epsilon) * delta, min=low, max=high)


def constrain_actual_delta(delta: torch.Tensor, config) -> torch.Tensor:
    epsilon = float(config.train.epsilon)
    if epsilon <= 0.0:
        return torch.zeros_like(delta)
    return epsilon * constrain_delta(delta / epsilon, config)


def rebuild_poisoned_from_actual_delta(
    x: torch.Tensor,
    delta: torch.Tensor,
    config,
) -> torch.Tensor:
    low, high = _input_bounds(config)
    return torch.clamp(x + delta, min=low, max=high)


def poison_batch_constrained(
    generator,
    x: torch.Tensor,
    config,
    shared_delta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    _, raw_delta, aux = poison_batch(
        generator=generator,
        x=x,
        epsilon=config.train.epsilon,
        input_range=config.data.input_range,
        coord_range=config.data.coord_range,
        shared_delta=shared_delta,
    )
    delta = constrain_delta(raw_delta, config)
    poisoned = rebuild_poisoned_from_delta(x, delta, config)
    aux = dict(aux)
    aux["delta_constrained"] = delta
    aux["poisoned_constrained"] = poisoned
    return poisoned, delta, raw_delta, aux
