from __future__ import annotations

from models.classifiers import build_victim
from models.generators import build_classical_inr_generator as _build_classical_inr_generator
from models.generators import build_generator
from models.generators import build_single_qubit_layer_freq_qinr_generator as _build_single_qubit_layer_freq_qinr_generator


def build_classifier(config):
    model_config = config.model if hasattr(config, "model") else config
    return build_victim(model_config)


def build_qinr_generator(config):
    model_config = config.model if hasattr(config, "model") else config
    return build_generator(model_config)


def build_classical_inr_generator(config):
    classical_inr_config = config.classical_inr if hasattr(config, "classical_inr") else config
    return _build_classical_inr_generator(classical_inr_config)


def build_single_qubit_layer_freq_qinr_generator(config):
    model_config = config.model if hasattr(config, "model") else config
    return _build_single_qubit_layer_freq_qinr_generator(model_config)


__all__ = [
    "build_classical_inr_generator",
    "build_classifier",
    "build_qinr_generator",
    "build_single_qubit_layer_freq_qinr_generator",
]
