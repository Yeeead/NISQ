from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalINRGenerator(nn.Module):
    """Small coordinate MLP over fixed random Fourier features."""

    def __init__(
        self,
        out_channels: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        n_frequencies: int = 3,
        freq_scale: float = 1.0,
        freq_distribution: str = "normal",
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))
        self.n_frequencies = max(1, int(n_frequencies))

        freq_distribution = str(freq_distribution).strip().lower()
        freq_scale = float(freq_scale)
        if freq_distribution == "normal":
            rff_freqs = freq_scale * torch.randn(self.n_frequencies, 2)
        elif freq_distribution == "uniform":
            rff_freqs = torch.empty(self.n_frequencies, 2).uniform_(-freq_scale, freq_scale)
        else:
            raise ValueError("Unsupported classical_inr.freq_distribution: {}".format(freq_distribution))
        self.register_buffer("rff_freqs", rff_freqs)

        layers = []
        in_dim = 2 * self.n_frequencies
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=False))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.out_channels))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError("coords must have shape [N, 2] or [B, N, 2].")

        batch, n_points, _ = coords.shape
        flat_coords = coords.reshape(batch * n_points, 2)
        angles = flat_coords.matmul(self.rff_freqs.t())
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        delta = self.net(features).view(batch, n_points, self.out_channels)
        return delta, {"delta_flat": delta, "rff_freqs": self.rff_freqs.detach()}

    def quantum_parameters(self):
        return []

    def classical_parameters(self):
        return list(self.parameters())
