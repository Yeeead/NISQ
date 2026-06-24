import random
import secrets

import numpy as np
import torch


def _requests_random_seed(seed) -> bool:
    if seed is None:
        return True
    if seed is False:
        return True
    if isinstance(seed, str):
        return seed.strip().lower() in {"false", "random", "none", "null"}
    return False


def resolve_seed(seed) -> int:
    if _requests_random_seed(seed):
        return int(secrets.randbits(32))
    return int(seed)


def set_seed(seed) -> int:
    resolved = resolve_seed(seed)
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    torch.cuda.manual_seed_all(resolved)
    return resolved


def seed_wanet_config(config) -> int:
    resolved = resolve_seed(getattr(config.wanet, "seed", "random"))
    config.wanet.seed = int(resolved)
    return int(resolved)


def seed_config(config, seed_global: bool = True, sync_wanet: bool = True) -> int:
    resolved = resolve_seed(getattr(config.train, "seed", 0))
    if seed_global:
        set_seed(resolved)
    config.train.seed = int(resolved)
    if sync_wanet and hasattr(config, "wanet"):
        seed_wanet_config(config)
    return int(resolved)
