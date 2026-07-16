from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def sample_frequencies(shape, distribution: str, scale: float) -> torch.Tensor:
    distribution = str(distribution).lower()
    if distribution == "normal":
        return float(scale) * torch.randn(*shape)
    if distribution == "uniform":
        return torch.empty(*shape).uniform_(-float(scale), float(scale))
    raise ValueError(f"Unsupported qinr_freq_distribution: {distribution}")


class QINRBase(nn.Module):
    """Shared QINR circuit for coordinate-to-perturbation generators."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        out_channels: int = 1,
        measurement: str = "pauli_z",
        shots: int | None = 64,
    ):
        super().__init__()
        if str(measurement) != "pauli_z":
            raise ValueError("Only pauli_z measurement is currently supported.")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.out_channels = int(out_channels)
        self.shots = None if shots is None else int(shots)
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be a positive integer or None.")

        self.theta = nn.Parameter(0.02 * torch.randn(self.n_layers + 1, self.n_qubits, 3))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def active_frequencies(self) -> torch.Tensor:
        if not hasattr(self, "freqs"):
            raise NotImplementedError
        return self.freqs

    def frequency_parameters(self):
        freqs = getattr(self, "freqs", None)
        return [freqs] if isinstance(freqs, nn.Parameter) else []

    def frequency_aux(self, freqs: torch.Tensor) -> dict:
        return {"freqs": freqs.detach()}

    def quantum_parameters(self):
        return [self.theta]

    def classical_parameters(self):
        return list(self.frequency_parameters())

    def _measure_first_qubit_low_shots(self, qdev) -> torch.Tensor:
        analytic_z = self.measure(qdev)
        first_qubit_z = analytic_z[:, :1]
        if self.shots is None:
            return first_qubit_z

        prob_one = ((1.0 - first_qubit_z) * 0.5).clamp(0.0, 1.0)
        count_one = torch.distributions.Binomial(total_count=self.shots, probs=prob_one).sample()
        sampled_z = 1.0 - 2.0 * count_one / float(self.shots)
        return first_qubit_z + (sampled_z - first_qubit_z).detach()

    def forward(self, coords: torch.Tensor):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError("coords must have shape [N, 2] or [B, N, 2].")

        batch, n_points, _ = coords.shape
        coords = coords.reshape(batch * n_points, 2)
        freqs = self.active_frequencies()

        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=coords.size(0),
            device=coords.device,
        )

        x_angle = coords[:, 0]
        y_angle = coords[:, 1]
        freq_dim = freqs.dim()
        cz_gate = getattr(tqf, "cz", None)
        if cz_gate is None:
            raise AttributeError("torchquantum.functional must provide cz for ring entanglement.")

        for layer in range(self.n_layers + 1):
            for wire in range(self.n_qubits):
                tqf.rz(qdev, wire, self.theta[layer, wire, 0].view(1))
                tqf.ry(qdev, wire, self.theta[layer, wire, 1].view(1))
                tqf.rz(qdev, wire, self.theta[layer, wire, 2].view(1))

            if self.n_qubits > 1:
                for wire in range(self.n_qubits):
                    cz_gate(qdev, wires=[wire, (wire + 1) % self.n_qubits])

            if layer < self.n_layers:
                for wire in range(self.n_qubits):
                    if freq_dim == 1:
                        if freqs.numel() == self.n_qubits:
                            freq_x = freq_y = freqs[wire]
                        elif freqs.numel() == self.n_layers:
                            freq_x = freq_y = freqs[layer]
                        else:
                            raise ValueError(f"Unsupported frequency tensor shape: {tuple(freqs.shape)}")
                    elif freq_dim == 3:
                        freq_x = freqs[layer, wire, 0]
                        freq_y = freqs[layer, wire, 1]
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
