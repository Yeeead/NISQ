from configs.default import ClassicalINRConfig, ModelConfig
from models.generators.classical_inr import ClassicalINRGenerator
from models.generators.input_aware import InputAwareGenerator
from models.quantum.qinr_base import QINRBase
from models.quantum.qinr_nisq import QINRNISQGenerator
from models.quantum.qinr_single_qubit_layer_freq import SingleQubitLayerFreqQINRGenerator


def build_classical_inr_generator(config: ClassicalINRConfig):
    return ClassicalINRGenerator(
        out_channels=int(config.out_channels),
        hidden_dim=int(config.hidden_dim),
        hidden_layers=int(config.hidden_layers),
        n_frequencies=int(config.n_frequencies),
        freq_scale=float(config.freq_scale),
        freq_distribution=str(config.freq_distribution),
    )


def build_single_qubit_layer_freq_qinr_generator(config: ModelConfig):
    return SingleQubitLayerFreqQINRGenerator(
        n_layers=int(config.qinr_n_layers),
        out_channels=int(config.qinr_out_channels),
        freq_scale=float(config.qinr_freq_scale),
        freq_distribution=str(config.qinr_freq_distribution),
        freq_trainable=bool(config.qinr_freq_trainable),
        measurement=getattr(config, "qinr_measurement", "pauli_z"),
        shots=getattr(config, "qinr_shots", 32),
    )


def build_generator(config: ModelConfig):
    if bool(getattr(config, "qinr_base", False)):
        generator = QINRBase(
            n_qubits=int(config.qinr_n_qubits),
            n_layers=int(config.qinr_n_layers),
            out_channels=int(config.qinr_out_channels),
            measurement=getattr(config, "qinr_measurement", "pauli_z"),
            shots=getattr(config, "qinr_shots", 32),
        )
        freqs = generator.theta.new_full(
            (generator.n_qubits,), float(config.qinr_baseline_freq)
        )
        generator.register_buffer("freqs", freqs)
        return generator

    mode = str(getattr(config, "qinr_freq_mode", "nisq")).strip().lower()
    if mode == "nisq":
        return QINRNISQGenerator(
            n_qubits=int(config.qinr_n_qubits),
            n_layers=int(config.qinr_n_layers),
            out_channels=int(config.qinr_out_channels),
            freq_scale=float(config.qinr_freq_scale),
            freq_distribution=str(config.qinr_freq_distribution),
            freq_trainable=bool(config.qinr_freq_trainable),
            measurement=getattr(config, "qinr_measurement", "pauli_z"),
            shots=getattr(config, "qinr_shots", 32),
        )
    raise ValueError("Unsupported qinr_freq_mode: {}".format(mode))


def build_input_aware_generator(model_config: ModelConfig, inputaware_config) -> InputAwareGenerator:
    return InputAwareGenerator(
        in_channels=int(getattr(model_config, "qinr_out_channels", 1)),
        hidden_channels=int(getattr(inputaware_config, "hidden_channels", 16)),
    )
