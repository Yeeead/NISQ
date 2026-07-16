from __future__ import annotations

import torch

from utils.training import count_parameters


def build_adamw(parameters, lr: float, weight_decay: float = 0.0):
    return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))


def build_victim_optimizer(victim, config):
    return build_adamw(
        victim.parameters(),
        lr=float(config.train.victim_lr),
        weight_decay=float(config.train.weight_decay),
    )


def build_qinr_optimizer(generator, config, log: bool = True):
    quantum_params = list(generator.quantum_parameters())
    classical_params = list(generator.classical_parameters())
    param_groups = [
        {
            "params": quantum_params,
            "lr": float(config.train.quantum_lr),
            "weight_decay": 0.0,
            "name": "quantum",
        },
    ]
    if classical_params:
        param_groups.append(
            {
                "params": classical_params,
                "lr": float(config.train.classical_lr),
                "weight_decay": float(config.train.weight_decay),
                "name": "classical",
            }
        )

    if log:
        print(
            "qinr optimizer q_params={} q_lr={} c_params={} c_lr={}".format(
                count_parameters(quantum_params),
                config.train.quantum_lr,
                count_parameters(classical_params),
                config.train.classical_lr,
            )
        )
    return torch.optim.AdamW(param_groups)


def build_classical_inr_optimizer(generator, config, log: bool = True):
    params = list(generator.parameters())
    if log:
        print(
            "classical inr optimizer params={} lr={}".format(
                count_parameters(params),
                config.train.classical_lr,
            )
        )
    return build_adamw(
        params,
        lr=float(config.train.classical_lr),
        weight_decay=float(config.train.weight_decay),
    )


def build_input_aware_optimizer(victim, generator, config):
    return torch.optim.AdamW(
        [
            {
                "params": victim.parameters(),
                "lr": float(config.train.victim_lr),
                "weight_decay": float(config.train.weight_decay),
            },
            {
                "params": generator.delta_g.parameters(),
                "lr": float(getattr(config.inputaware, "delta_lr", config.train.classical_lr)),
                "weight_decay": float(config.train.weight_decay),
            },
        ]
    )


def build_input_aware_mask_optimizer(generator, config):
    return build_adamw(
        generator.mask_g.parameters(),
        lr=float(getattr(config.inputaware, "mask_lr", config.train.classical_lr)),
        weight_decay=float(config.train.weight_decay),
    )
