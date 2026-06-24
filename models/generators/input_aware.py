from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAEHead(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, activation: str = "sigmoid"):
        super().__init__()
        hidden_channels = max(4, int(hidden_channels))
        self.activation = str(activation).lower()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.decoder(self.encoder(x))
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        if self.activation == "tanh":
            return torch.tanh(out)
        raise ValueError("Unknown SimpleAEHead activation: {}".format(self.activation))


class InputAwareGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16):
        super().__init__()
        self._eps = 1.0e-7
        self.mask_g = SimpleAEHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation="sigmoid",
        )
        self.delta_g = SimpleAEHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation="sigmoid",
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mask_g(x), self.delta_g(x)

    def threshold_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mask * 20.0 - 10.0) / (2.0 + self._eps) + 0.5

    def normalize_pattern(self, pattern: torch.Tensor, input_range) -> torch.Tensor:
        low, high = float(input_range[0]), float(input_range[1])
        return low + (high - low) * pattern.clamp(0.0, 1.0)

    def trigger(self, trigger_source: torch.Tensor, input_range) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.threshold_mask(self.mask_g(trigger_source))
        pattern = self.normalize_pattern(self.delta_g(trigger_source), input_range)
        return pattern, mask

    def trigger_delta(
        self,
        x: torch.Tensor,
        trigger_source: torch.Tensor,
        input_range,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pattern, mask = self.trigger(trigger_source, input_range)
        if mask.size(1) == 1 and x.size(1) > 1:
            mask = mask.expand(-1, x.size(1), -1, -1)
        if pattern.size(1) == 1 and x.size(1) > 1:
            pattern = pattern.expand(-1, x.size(1), -1, -1)
        return (pattern - x) * mask, pattern, mask

    def poison(self, x: torch.Tensor, trigger_source: torch.Tensor, epsilon: float, input_range) -> torch.Tensor:
        delta, _, _ = self.trigger_delta(x, trigger_source, input_range)
        epsilon = float(epsilon)
        if epsilon > 0.0:
            delta = delta.clamp(-epsilon, epsilon)
        low, high = float(input_range[0]), float(input_range[1])
        return torch.clamp(x + delta, min=low, max=high)
