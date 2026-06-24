from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def make_coord_grid(
    height: int,
    width: int,
    batch_size: Optional[int] = None,
    coord_range: Tuple[float, float] = (-1.0, 1.0),
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    low, high = float(coord_range[0]), float(coord_range[1])
    ys = torch.linspace(low, high, steps=int(height), device=device, dtype=dtype)
    xs = torch.linspace(low, high, steps=int(width), device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(ys, xs)
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(int(height) * int(width), 2)
    if batch_size is None:
        return coords
    return coords.unsqueeze(0).expand(int(batch_size), -1, -1)
