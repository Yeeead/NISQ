from __future__ import annotations

from models.classifiers import build_victim
from models.generators import build_generator


def build_classifier(config):
    model_config = config.model if hasattr(config, "model") else config
    return build_victim(model_config)


def build_qinr_generator(config):
    model_config = config.model if hasattr(config, "model") else config
    return build_generator(model_config)


__all__ = [
    "build_classifier",
    "build_qinr_generator",
]
