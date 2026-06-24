from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

from models.quantum.qinr_base import QINRBase, sample_frequencies


class QINRNISQGenerator(QINRBase):
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 2,
        out_channels: int = 1,
        freq_scale: float = 1.0,
        freq_distribution: str = "normal",
        freq_trainable: bool = False,
        measurement: str = "pauli_z",
        shots: int | None = 32,
    ):
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            out_channels=out_channels,
            measurement=measurement,
            shots=shots,
        )
        freqs = sample_frequencies(
            shape=(self.n_qubits,),
            distribution=freq_distribution,
            scale=freq_scale,
        )
        if bool(freq_trainable):
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.entangle_phi = nn.Parameter(0.1 * torch.randn(n_layers))

    def forward(self, coords: torch.Tensor):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError("coords must have shape [N, 2] or [B, N, 2].")
        batch, n_points, _ = coords.shape
        coords = coords.reshape(batch * n_points, 2)
        freqs = self.freqs
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=coords.size(0),
            device=coords.device,
        )
        x_angle = coords[:, 0]
        y_angle = coords[:, 1]
        freq_dim = freqs.dim()
        for layer in range(self.n_layers + 1):
            for wire in range(self.n_qubits):
                tqf.rz(qdev, wire, self.theta[layer, wire, 0].view(1))
                tqf.ry(qdev, wire, self.theta[layer, wire, 1].view(1))
                tqf.rz(qdev, wire, self.theta[layer, wire, 2].view(1))
            for wire in range(self.n_qubits):
                next_wire = (wire + 1) % self.n_qubits
                phi = self.entangle_phi[min(layer, len(self.entangle_phi) - 1)]
                tqf.crz(qdev, wires=[wire, next_wire], params=phi.view(1))
            if layer < self.n_layers:
                for wire in range(self.n_qubits):
                    if freq_dim == 1:
                        if freqs.numel() == self.n_qubits:
                            freq_x = freq_y = freqs[wire]
                        else:
                            freq_x = freq_y = freqs[0] if freqs.numel() == 1 else freqs[layer]
                    else:
                        raise ValueError(f"Unsupported frequency tensor shape: {tuple(freqs.shape)}")
                    tqf.rx(qdev, wire, freq_x * x_angle)
                    tqf.ry(qdev, wire, freq_y * y_angle)
        measured = self._measure_first_qubit_low_shots(qdev)
        delta = measured.view(batch, n_points, 1)
        if self.out_channels > 1:
            delta = delta.expand(-1, -1, self.out_channels)
        aux = {
            "measurements": measured.view(batch, n_points, 1),
            "delta_flat": delta,
            "shots": self.shots,
        }
        aux.update(self.frequency_aux(freqs))
        return delta, aux

    def quantum_parameters(self):
        return [self.theta, self.entangle_phi]

    def classical_parameters(self):
        return list(self.frequency_parameters())
