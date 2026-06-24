from __future__ import annotations

from typing import Iterable, Tuple

import torch

from configs.default import ExperimentConfig


def normalize_for_victim(x: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    if not config.data.normalize_mean or not config.data.normalize_std:
        return x
    mean = torch.tensor(config.data.normalize_mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(config.data.normalize_std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def target_label(config: ExperimentConfig) -> int:
    backdoor = getattr(config, "backdoor", None)
    return int(getattr(backdoor, "target_label", config.train.target_label))


def backdoor_bounds(config: ExperimentConfig) -> Tuple[float, float]:
    backdoor = getattr(config, "backdoor", None)
    data_range = getattr(config.data, "input_range", (0.0, 1.0))
    return (
        float(getattr(backdoor, "clamp_min", data_range[0])),
        float(getattr(backdoor, "clamp_max", data_range[1])),
    )


def fmt(value, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "{:.{}f}".format(value, int(precision))
    return str(value)


def kv_line(tag: str, items: Iterable[Tuple[str, object]], precision: int = 4) -> str:
    parts = [str(tag)]
    for key, value in items:
        parts.append("{}={}".format(key, fmt(value, precision=precision)))
    return " ".join(parts)
