from configs.default import ModelConfig
from models.generators.input_aware import InputAwareGenerator
from models.quantum.qinr_nisq import QINRNISQGenerator


def build_generator(config: ModelConfig):
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
