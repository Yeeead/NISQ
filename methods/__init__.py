from __future__ import annotations

from importlib import import_module

from methods.common import canonical_method


METHOD_MODULES = {
    "badnets": "methods.badnets",
    "blended": "methods.blended",
    "wanet": "methods.wanet",
    "inputaware": "methods.inputaware",
}


def get_method_module(method: str):
    method = canonical_method(method)
    if method not in METHOD_MODULES:
        raise ValueError("Unknown backdoor method '{}'. Expected one of {}.".format(method, sorted(METHOD_MODULES)))
    return import_module(METHOD_MODULES[method])


def build_method(method: str, config, generator=None):
    return get_method_module(method).build_method(config, generator=generator)


def build_method_generator(method: str, config, device):
    module = get_method_module(method)
    return module.build_generator(config, device)


def poison_batch(method: str, x, y=None, config=None, generator=None, mode: str = "eval", resolution=None):
    return build_method(method, config, generator=generator).poison_batch(
        x,
        y,
        mode=mode,
        resolution=resolution,
    )


__all__ = [
    "METHOD_MODULES",
    "build_method",
    "build_method_generator",
    "canonical_method",
    "get_method_module",
    "poison_batch",
]
