from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from methods.triggers import (
    apply_badnets_trigger,
    apply_blended_trigger,
    apply_wanet_trigger,
)
from utils.seed import seed_config, seed_wanet_config


_BLENDED_TARGET_PATTERN_CACHE: Dict[Tuple, torch.Tensor] = {}


def make_poison_mask(
    y: torch.Tensor,
    poison_rate: float,
    target_label: int,
    exclude_target: bool = True,
) -> torch.Tensor:
    poison_rate = float(poison_rate)
    mask = torch.zeros_like(y, dtype=torch.bool)
    if poison_rate <= 0.0 or y.numel() == 0:
        return mask

    if bool(exclude_target):
        candidates = (y != int(target_label)).nonzero(as_tuple=False).flatten()
    else:
        candidates = torch.arange(y.numel(), device=y.device)
    if candidates.numel() == 0:
        return mask

    count = int(round(y.numel() * min(poison_rate, 1.0)))
    count = max(1, min(count, candidates.numel()))
    perm = torch.randperm(candidates.numel(), device=y.device)[:count]
    mask[candidates.index_select(0, perm)] = True
    return mask


def _backdoor_params(config):
    backdoor = getattr(config, "backdoor", None)
    data_range = getattr(getattr(config, "data", None), "input_range", (0.0, 1.0))
    clamp_min = float(getattr(backdoor, "clamp_min", data_range[0]))
    clamp_max = float(getattr(backdoor, "clamp_max", data_range[1]))
    poison_rate = float(getattr(backdoor, "poison_rate", getattr(config.train, "poison_rate", 0.05)))
    target_label = int(getattr(backdoor, "target_label", getattr(config.train, "target_label", 0)))
    return clamp_min, clamp_max, poison_rate, target_label


def _experiment_seed(config) -> int:
    return seed_config(config, seed_global=False, sync_wanet=False)


def _wanet_seed(config) -> int:
    return seed_wanet_config(config)


def _blended_target_pattern(config, target_label: int, seed: int) -> torch.Tensor:
    key = (
        str(config.data.dataset).lower(),
        str(config.data.data_root),
        int(config.data.train_resolution),
        int(target_label),
        int(seed),
    )
    pattern = _BLENDED_TARGET_PATTERN_CACHE.get(key)
    if pattern is None:
        from utils.mnist import load_mnist_class_image

        pattern = load_mnist_class_image(
            config.data,
            target_label=int(target_label),
            image_size=int(config.data.train_resolution),
            seed=int(seed),
            train=True,
        ).detach()
        _BLENDED_TARGET_PATTERN_CACHE[key] = pattern
    return pattern


def apply_trigger(x: torch.Tensor, method: str, config) -> torch.Tensor:
    method = str(method).lower()
    clamp_min, clamp_max, _, target_label = _backdoor_params(config)
    seed = _experiment_seed(config)

    if method == "badnets":
        badnets = getattr(config, "badnets")
        return apply_badnets_trigger(
            x,
            patch_size=int(badnets.patch_size),
            patch_value=float(badnets.patch_value),
            location=str(badnets.location),
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

    if method == "blended":
        blended = getattr(config, "blended")
        pattern_seed = int(getattr(blended, "pattern_seed", seed))
        pattern_type = str(blended.pattern_type)
        pattern = None
        if pattern_type.lower() in ("target_image", "target_class_image"):
            pattern = _blended_target_pattern(config, target_label=target_label, seed=pattern_seed)
        return apply_blended_trigger(
            x,
            alpha=float(config.train.epsilon),
            pattern=pattern,
            pattern_type=pattern_type,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            seed=pattern_seed,
        )

    if method in ("wanet", "wanets"):
        wanet = getattr(config, "wanet")
        return apply_wanet_trigger(
            x,
            s=float(wanet.s),
            grid_res=int(wanet.grid_res),
            align_corners=bool(wanet.align_corners),
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            seed=_wanet_seed(config),
            scale_mode=str(getattr(wanet, "scale_mode", "wanet")),
            normalize=str(getattr(wanet, "normalize", "mean_abs")),
            upsample_mode=str(getattr(wanet, "upsample_mode", "bicubic")),
            grid_rescale=float(getattr(wanet, "grid_rescale", 1.0)),
            sample_mode=str(getattr(wanet, "sample_mode", "bilinear")),
            padding_mode=str(getattr(wanet, "padding_mode", "zeros")),
        )

    raise ValueError("Unknown baseline backdoor method: {}".format(method))


def apply_poison_to_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    config,
    poison_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, poison_rate, target_label = _backdoor_params(config)
    if poison_mask is None:
        poison_mask = make_poison_mask(
            y,
            poison_rate=poison_rate,
            target_label=target_label,
            exclude_target=True,
        )
    poison_mask = poison_mask.to(device=y.device, dtype=torch.bool)

    poisoned_x = x.clone()
    poisoned_y = y.clone()
    if poison_mask.any():
        poisoned_x[poison_mask] = apply_trigger(x[poison_mask], method, config)
        poisoned_y[poison_mask] = int(target_label)
    return poisoned_x, poisoned_y, poison_mask
