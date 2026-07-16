from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

from models.quantum.qinr_base import QINRBase, sample_frequencies


class SingleQubitLayerFreqQINRGenerator(QINRBase):
    def __init__(
        self,
        n_layers: int = 2,
        out_channels: int = 1,
        freq_scale: float = 1.0,
        freq_distribution: str = "normal",
        freq_trainable: bool = False,
        measurement: str = "pauli_z",
        shots: int | None = 32,
    ):
        super().__init__(
            n_qubits=1,
            n_layers=n_layers,
            out_channels=out_channels,
            measurement=measurement,
            shots=shots,
        )
        freqs = sample_frequencies(
            shape=(self.n_layers,),
            distribution=freq_distribution,
            scale=freq_scale,
        )
        if bool(freq_trainable):
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError("coords must have shape [N, 2] or [B, N, 2].")

        batch, n_points, _ = coords.shape
        coords = coords.reshape(batch * n_points, 2)
        freqs = self.freqs
        if freqs.dim() != 1 or freqs.numel() != self.n_layers:
            raise ValueError("Single-qubit layer-frequency QINR requires one frequency per data encoding layer.")

        qdev = tq.QuantumDevice(
            n_wires=1,
            bsz=coords.size(0),
            device=coords.device,
        )
        x_angle = coords[:, 0]
        y_angle = coords[:, 1]

        for layer in range(self.n_layers + 1):
            tqf.rz(qdev, 0, self.theta[layer, 0, 0].view(1))
            tqf.ry(qdev, 0, self.theta[layer, 0, 1].view(1))
            tqf.rz(qdev, 0, self.theta[layer, 0, 2].view(1))
            if layer < self.n_layers:
                freq = freqs[layer]
                tqf.rx(qdev, 0, freq * x_angle)
                tqf.ry(qdev, 0, freq * y_angle)

        measured = self._measure_first_qubit_low_shots(qdev)
        delta = measured.view(batch, n_points, 1)
        if self.out_channels > 1:
            delta = delta.expand(-1, -1, self.out_channels)
        aux = {
            "measurements": measured.view(batch, n_points, 1),
            "delta_flat": delta,
            "shots": self.shots,
            "frequency_granularity": "data_encoding_layer",
        }
        aux.update(self.frequency_aux(freqs))
        return delta, aux

    def quantum_parameters(self):
        return [self.theta]

    def classical_parameters(self):
        return list(self.frequency_parameters())
