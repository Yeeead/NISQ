from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ============================================================
# Common utilities
# ============================================================

def _check_image_batch(x: torch.Tensor) -> None:
    if x.dim() != 4:
        raise ValueError("trigger input must be [B, C, H, W], got {}".format(tuple(x.shape)))


def _clamp(x: torch.Tensor, clamp_min: float, clamp_max: float) -> torch.Tensor:
    return x.clamp(min=float(clamp_min), max=float(clamp_max))


# ============================================================
# BadNets trigger
# ============================================================

def _patch_slices(height: int, width: int, patch_size: int, location: str):
    patch_size = max(1, min(int(patch_size), int(height), int(width)))
    location = str(location).lower()

    if location == "bottom_right":
        top = height - patch_size
        left = width - patch_size
    elif location == "bottom_left":
        top = height - patch_size
        left = 0
    elif location == "top_right":
        top = 0
        left = width - patch_size
    elif location == "top_left":
        top = 0
        left = 0
    elif location == "center":
        top = (height - patch_size) // 2
        left = (width - patch_size) // 2
    else:
        raise ValueError("Unknown BadNets patch location: {}".format(location))

    return slice(top, top + patch_size), slice(left, left + patch_size)


def apply_badnets_trigger(
    x: torch.Tensor,
    patch_size: int = 4,
    patch_value: float = 1.0,
    location: str = "bottom_right",
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    _check_image_batch(x)
    poisoned = x.clone()
    h_slice, w_slice = _patch_slices(x.size(-2), x.size(-1), patch_size, location)
    poisoned[:, :, h_slice, w_slice] = torch.as_tensor(
        float(patch_value),
        device=x.device,
        dtype=x.dtype,
    )
    return _clamp(poisoned, clamp_min, clamp_max)


# ============================================================
# Blended trigger
# ============================================================

def build_blended_pattern(
    x: torch.Tensor,
    pattern_type: str = "checkerboard",
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    _check_image_batch(x)
    _, channels, height, width = x.shape
    pattern_type = str(pattern_type).lower()
    device = x.device
    dtype = x.dtype

    yy = torch.arange(height, device=device).view(height, 1)
    xx = torch.arange(width, device=device).view(1, width)

    if pattern_type == "checkerboard":
        base = ((yy + xx) % 2).to(dtype=dtype)
    elif pattern_type == "vertical_stripes":
        base = (xx % 2).expand(height, width).to(dtype=dtype)
    elif pattern_type == "horizontal_stripes":
        base = (yy % 2).expand(height, width).to(dtype=dtype)
    elif pattern_type in ("fixed_noise", "noise", "random"):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        base = torch.rand((height, width), generator=generator).to(device=device, dtype=dtype)
    elif pattern_type in ("target_image", "target_class_image"):
        raise ValueError("blended target_image pattern requires an explicit pattern tensor")
    else:
        raise ValueError("Unknown blended pattern_type: {}".format(pattern_type))

    pattern = float(clamp_min) + (float(clamp_max) - float(clamp_min)) * base
    return pattern.view(1, 1, height, width).expand(1, channels, height, width)


def apply_blended_trigger(
    x: torch.Tensor,
    alpha: float = 0.2,
    pattern: Optional[torch.Tensor] = None,
    pattern_type: str = "checkerboard",
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    _check_image_batch(x)
    alpha = float(alpha)

    if pattern is None:
        pattern = build_blended_pattern(
            x,
            pattern_type=pattern_type,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            seed=seed,
        )
    else:
        pattern = pattern.to(device=x.device, dtype=x.dtype)
        if pattern.dim() == 3:
            pattern = pattern.unsqueeze(0)
        if pattern.dim() != 4:
            raise ValueError("blended pattern must be [C, H, W] or [1, C, H, W]")
        if pattern.shape[-2:] != x.shape[-2:]:
            pattern = F.interpolate(pattern, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if pattern.size(1) == 1 and x.size(1) > 1:
            pattern = pattern.expand(-1, x.size(1), -1, -1)
        if pattern.size(0) == 1 and x.size(0) > 1:
            pattern = pattern.expand(x.size(0), -1, -1, -1)

    poisoned = (1.0 - alpha) * x + alpha * pattern
    return _clamp(poisoned, clamp_min, clamp_max)


# ============================================================
# WaNet trigger
# ============================================================
# Notes:
# 1. WaNet uses a smooth image-warping trigger, not an additive patch.
# 2. The attack grid is generated from a low-resolution random control grid,
#    normalized by mean absolute value and bicubic-upsampled.
# 3. During original WaNet training, besides backdoor samples, a "cross/noise"
#    branch is also used: random noisy warps keep their clean labels. This file
#    exposes build_wanet_cross_grid / apply_wanet_cross_trigger for that branch.


def _identity_grid(
    batch_size: int,
    height: int,
    width: int,
    device,
    dtype,
    align_corners: bool = True,
) -> torch.Tensor:
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("WaNet grid size must be positive, got {}x{}".format(height, width))

    if bool(align_corners):
        ys = torch.zeros(1, device=device, dtype=dtype) if height == 1 else torch.linspace(
            -1.0,
            1.0,
            steps=height,
            device=device,
            dtype=dtype,
        )
        xs = torch.zeros(1, device=device, dtype=dtype) if width == 1 else torch.linspace(
            -1.0,
            1.0,
            steps=width,
            device=device,
            dtype=dtype,
        )
    else:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / float(height)) - 1.0
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / float(width)) - 1.0

    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)

    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return grid.expand(int(batch_size), height, width, 2).clone()


def _pixel_to_normalized_scale(size: int, align_corners: bool) -> float:
    size = int(size)
    if size <= 0:
        raise ValueError("WaNet grid size must be positive, got {}".format(size))
    if bool(align_corners):
        return 2.0 / float(max(size - 1, 1))
    return 2.0 / float(size)


def _wanet_scale_tensor(
    height: int,
    width: int,
    s: float,
    device,
    dtype,
    align_corners: bool,
    scale_mode: str = "wanet",
) -> torch.Tensor:
    """
    Returns a [1, 1, 1, 2] scale for x/y displacement.

    scale_mode="wanet": close to the original WaNet convention, where the
    random field is divided by image size. For square inputs this matches the
    common official-style formula: grid + s * noise / input_height.

    scale_mode="pixel": interprets s as an approximate maximum pixel shift.
    This was closer to the previous implementation but is not the default here.
    """
    scale_mode = str(scale_mode).lower()
    if scale_mode == "wanet":
        sx = float(s) / float(max(int(width), 1))
        sy = float(s) / float(max(int(height), 1))
    elif scale_mode == "pixel":
        sx = float(s) * _pixel_to_normalized_scale(width, bool(align_corners))
        sy = float(s) * _pixel_to_normalized_scale(height, bool(align_corners))
    else:
        raise ValueError("Unknown WaNet scale_mode: {}".format(scale_mode))

    return torch.tensor([sx, sy], device=device, dtype=dtype).view(1, 1, 1, 2)


def _normalize_wanet_control_grid(control: torch.Tensor, normalize: str) -> torch.Tensor:
    normalize = str(normalize).lower()
    if normalize in ("mean_abs", "mean", "l1"):
        return control / control.abs().mean().clamp_min(1.0e-6)
    if normalize in ("max_abs", "max", "linf"):
        return control / control.abs().amax().clamp_min(1.0e-6)
    if normalize in ("none", "identity", "no"):
        return control
    raise ValueError("Unknown WaNet control-grid normalization: {}".format(normalize))


def build_wanet_displacement(
    height: int,
    width: int,
    s: float = 0.5,
    grid_res: int = 4,
    device=None,
    dtype=None,
    align_corners: bool = True,
    seed: int = 0,
    scale_mode: str = "wanet",
    normalize: str = "mean_abs",
    upsample_mode: str = "bicubic",
) -> torch.Tensor:
    """
    Build the fixed smooth WaNet displacement field.

    Returns:
        displacement: [1, H, W, 2], in normalized grid_sample coordinates.
    """
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    height = int(height)
    width = int(width)
    grid_res = max(2, int(grid_res))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    control = torch.rand((1, 2, grid_res, grid_res), generator=generator) * 2.0 - 1.0
    control = _normalize_wanet_control_grid(control, normalize=normalize)
    control = control.to(device=device, dtype=dtype)

    smooth_noise = F.interpolate(
        control,
        size=(height, width),
        mode=str(upsample_mode),
        align_corners=bool(align_corners),
    )
    smooth_noise = smooth_noise.permute(0, 2, 3, 1).contiguous()

    scale = _wanet_scale_tensor(
        height=height,
        width=width,
        s=s,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
        scale_mode=scale_mode,
    )
    return smooth_noise * scale


def build_wanet_grid(
    batch_size: int,
    height: int,
    width: int,
    s: float = 0.5,
    grid_res: int = 4,
    device=None,
    dtype=None,
    align_corners: bool = True,
    seed: int = 0,
    scale_mode: str = "wanet",
    normalize: str = "mean_abs",
    upsample_mode: str = "bicubic",
    grid_rescale: float = 1.0,
) -> torch.Tensor:
    """
    Build the fixed WaNet attack grid.

    Returns:
        grid: [B, H, W, 2], usable by torch.nn.functional.grid_sample.
    """
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    height = int(height)
    width = int(width)

    identity = _identity_grid(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
    )
    displacement = build_wanet_displacement(
        height=height,
        width=width,
        s=s,
        grid_res=grid_res,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
        seed=seed,
        scale_mode=scale_mode,
        normalize=normalize,
        upsample_mode=upsample_mode,
    )

    grid = (identity + displacement) * float(grid_rescale)
    return grid.clamp(-1.0, 1.0)


def _expand_grid_to_batch(grid: torch.Tensor, batch_size: int) -> torch.Tensor:
    if grid.size(0) == int(batch_size):
        return grid
    if grid.size(0) == 1:
        return grid.expand(int(batch_size), -1, -1, -1).clone()
    raise ValueError(
        "grid batch size must be 1 or match input batch size; got grid batch {} and input batch {}".format(
            grid.size(0), int(batch_size)
        )
    )


def build_wanet_cross_grid(
    batch_size: int,
    height: int,
    width: int,
    s: float = 0.5,
    grid_res: int = 4,
    noise_s: float = 1.0,
    device=None,
    dtype=None,
    align_corners: bool = True,
    seed: int = 0,
    noise_seed: Optional[int] = None,
    scale_mode: str = "wanet",
    normalize: str = "mean_abs",
    upsample_mode: str = "bicubic",
    grid_rescale: float = 1.0,
) -> torch.Tensor:
    """
    Build WaNet cross/noise-mode grids.

    In WaNet training, these noisy-warped samples should keep their original
    labels. They help prevent the model from treating arbitrary image warps as
    the target-class trigger.
    """
    device = torch.device("cpu") if device is None else device
    dtype = torch.float32 if dtype is None else dtype
    height = int(height)
    width = int(width)

    base_grid = build_wanet_grid(
        batch_size=1,
        height=height,
        width=width,
        s=s,
        grid_res=grid_res,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
        seed=seed,
        scale_mode=scale_mode,
        normalize=normalize,
        upsample_mode=upsample_mode,
        grid_rescale=grid_rescale,
    )
    base_grid = _expand_grid_to_batch(base_grid, batch_size)

    if noise_seed is None:
        random_noise = torch.rand((batch_size, height, width, 2), device=device, dtype=dtype) * 2.0 - 1.0
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(noise_seed))
        random_noise = torch.rand((batch_size, height, width, 2), generator=generator) * 2.0 - 1.0
        random_noise = random_noise.to(device=device, dtype=dtype)

    noise_scale = _wanet_scale_tensor(
        height=height,
        width=width,
        s=noise_s,
        device=device,
        dtype=dtype,
        align_corners=align_corners,
        scale_mode=scale_mode,
    )
    return (base_grid + random_noise * noise_scale).clamp(-1.0, 1.0)


def apply_wanet_trigger(
    x: torch.Tensor,
    s: float = 0.5,
    grid_res: int = 4,
    align_corners: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    seed: int = 0,
    scale_mode: str = "wanet",
    normalize: str = "mean_abs",
    upsample_mode: str = "bicubic",
    grid_rescale: float = 1.0,
    sample_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Apply the fixed WaNet backdoor trigger at the current input size."""
    _check_image_batch(x)

    grid = build_wanet_grid(
        batch_size=x.size(0),
        height=x.size(-2),
        width=x.size(-1),
        s=s,
        grid_res=grid_res,
        device=x.device,
        dtype=x.dtype,
        align_corners=align_corners,
        seed=seed,
        scale_mode=scale_mode,
        normalize=normalize,
        upsample_mode=upsample_mode,
        grid_rescale=grid_rescale,
    )

    poisoned = F.grid_sample(
        x,
        grid,
        mode=str(sample_mode),
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
    )
    return _clamp(poisoned, clamp_min, clamp_max)


def apply_wanet_cross_trigger(
    x: torch.Tensor,
    s: float = 0.5,
    grid_res: int = 4,
    noise_s: float = 1.0,
    align_corners: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    seed: int = 0,
    noise_seed: Optional[int] = None,
    scale_mode: str = "wanet",
    normalize: str = "mean_abs",
    upsample_mode: str = "bicubic",
    grid_rescale: float = 1.0,
    sample_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Apply the WaNet cross/noise-mode warp.

    Use this only for the auxiliary clean-label cross branch during training;
    do not count these samples as successful backdoor-triggered samples.
    """
    _check_image_batch(x)
    grid = build_wanet_cross_grid(
        batch_size=x.size(0),
        height=x.size(-2),
        width=x.size(-1),
        s=s,
        grid_res=grid_res,
        noise_s=noise_s,
        device=x.device,
        dtype=x.dtype,
        align_corners=align_corners,
        seed=seed,
        noise_seed=noise_seed,
        scale_mode=scale_mode,
        normalize=normalize,
        upsample_mode=upsample_mode,
        grid_rescale=grid_rescale,
    )
    warped = F.grid_sample(
        x,
        grid,
        mode=str(sample_mode),
        padding_mode=str(padding_mode),
        align_corners=bool(align_corners),
    )
    return _clamp(warped, clamp_min, clamp_max)
