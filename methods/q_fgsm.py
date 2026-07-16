from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.default import ExperimentConfig
from datasets.builder import build_train_and_test_loaders
from methods.common import maybe_resize, method_namespace
from models.factory import build_classifier
from training.optim import build_victim_optimizer
from training.poison import resize_image
from training.steps import optimizer_step
from utils.eval_helpers import normalize_for_victim


NAME = "q_fgsm"


class QFGSMTrigger(nn.Module):
    def __init__(self, delta: torch.Tensor):
        super().__init__()
        self.register_buffer("delta", delta.detach().clone())

    def forward(self, x: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
        return apply_delta(x, self.delta, config)


class ProxyFNN(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super().__init__()
        flat_dim = int(image_size) * int(image_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _qcfg(config: ExperimentConfig):
    return getattr(config, "q_fgsm")


def _epsilon(config: ExperimentConfig) -> float:
    value = getattr(_qcfg(config), "epsilon", None)
    return float(config.train.epsilon if value is None else value)


def _target_label(config: ExperimentConfig) -> int:
    return int(getattr(config.backdoor, "target_label", config.train.target_label))


def _poison_rate(config: ExperimentConfig) -> float:
    return float(getattr(config.backdoor, "poison_rate", config.train.poison_rate))


def _input_bounds(config: ExperimentConfig) -> Tuple[float, float]:
    low, high = config.data.input_range
    return float(low), float(high)


def apply_delta(x: torch.Tensor, delta: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    low, high = _input_bounds(config)
    return torch.clamp(x + delta.to(device=x.device, dtype=x.dtype), min=low, max=high)


def _target_poison_mask(y: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    target = _target_label(config)
    candidates = (y == target).nonzero(as_tuple=False).flatten()
    mask = torch.zeros_like(y, dtype=torch.bool)
    if candidates.numel() == 0:
        return mask
    count = int(round(y.numel() * min(_poison_rate(config), 1.0)))
    count = max(1, min(count, int(candidates.numel())))
    perm = torch.randperm(candidates.numel(), device=y.device)[:count]
    mask[candidates.index_select(0, perm)] = True
    return mask


def _build_proxy(config: ExperimentConfig, device) -> nn.Module:
    proxy_model = str(getattr(_qcfg(config), "proxy_model", "victim")).lower()
    if proxy_model == "victim":
        return build_classifier(config).to(device)
    if proxy_model == "fnn":
        return ProxyFNN(
            image_size=int(config.data.victim_resolution),
            num_classes=int(config.model.num_classes),
        ).to(device)
    raise ValueError("Unknown Q-FGSM proxy_model '{}'. Expected 'victim' or 'fnn'.".format(proxy_model))


def _train_proxy(proxy: nn.Module, train_loader, device, config: ExperimentConfig) -> None:
    optimizer = build_victim_optimizer(proxy, config)
    epochs = max(1, int(getattr(config.train, "clean_epochs", 1)))
    for _ in range(epochs):
        proxy.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = resize_image(x, int(config.data.victim_resolution))
            logits = proxy(normalize_for_victim(x, config))
            loss = F.cross_entropy(logits, y)
            optimizer_step(loss, optimizer)


def _collect_target_pool(train_loader, device, config: ExperimentConfig) -> torch.Tensor:
    target = _target_label(config)
    chunks = []
    for x, y in train_loader:
        mask = y == target
        if mask.any():
            x_t = x[mask].to(device, non_blocking=True)
            chunks.append(resize_image(x_t, int(config.data.victim_resolution)))
    if not chunks:
        raise ValueError("Q-FGSM requires at least one target-label sample to build fuzzy admix.")
    return torch.cat(chunks, dim=0)


def _sample_targets(target_pool: torch.Tensor, batch_size: int, admix_n: int) -> torch.Tensor:
    idx = torch.randint(
        target_pool.size(0),
        (int(batch_size), int(admix_n)),
        device=target_pool.device,
    )
    flat = target_pool.index_select(0, idx.flatten())
    return flat.view(int(batch_size), int(admix_n), *target_pool.shape[1:])


def _fuzzy_admix(x: torch.Tensor, target_pool: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    qcfg = _qcfg(config)
    if not bool(getattr(qcfg, "fuzzy_admix", True)):
        return x
    admix_n = max(1, int(getattr(qcfg, "admix_n", 3)))
    c = float(getattr(qcfg, "admix_c", 1.0))
    sigma = max(float(getattr(qcfg, "admix_sigma", 2.0)), 1.0e-6)

    targets = _sample_targets(target_pool, x.size(0), admix_n)
    mu = torch.exp(-((x - c) ** 2) / (2.0 * sigma * sigma))
    nu = 1.0 - mu
    mixed = mu * x + (nu.unsqueeze(1) * targets).mean(dim=1)
    low, high = _input_bounds(config)
    return torch.clamp(mixed, min=low, max=high)


@torch.no_grad()
def _fooling_rate(
    proxy: nn.Module,
    train_loader,
    delta: torch.Tensor,
    device,
    config: ExperimentConfig,
) -> float:
    proxy.eval()
    target = _target_label(config)
    total = 0
    fooled = 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = y != target
        if not mask.any():
            continue
        x_src = resize_image(x[mask], int(config.data.victim_resolution))
        logits = proxy(normalize_for_victim(apply_delta(x_src, delta, config), config))
        total += int(mask.sum().item())
        fooled += int((logits.argmax(dim=1) == target).sum().item())
    return fooled / max(total, 1)


def _cache_metadata(config: ExperimentConfig, shape) -> Dict:
    qcfg = _qcfg(config)
    return {
        "method": NAME,
        "shape": list(shape),
        "target_label": _target_label(config),
        "epsilon": _epsilon(config),
        "norm": str(getattr(qcfg, "norm", "linf")).lower(),
        "fuzzy_admix": bool(getattr(qcfg, "fuzzy_admix", True)),
        "admix_n": int(getattr(qcfg, "admix_n", 3)),
        "admix_c": float(getattr(qcfg, "admix_c", 1.0)),
        "admix_sigma": float(getattr(qcfg, "admix_sigma", 2.0)),
        "proxy_model": str(getattr(qcfg, "proxy_model", "victim")).lower(),
        "victim_resolution": int(config.data.victim_resolution),
    }


def _metadata_matches(actual: Dict, expected: Dict) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value is None:
            return False
        if isinstance(expected_value, float):
            if abs(float(actual_value) - expected_value) > 1.0e-12:
                return False
        elif actual_value != expected_value:
            return False
    return True


def _load_cached_delta(path: Path, metadata: Dict, device) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device)
    if isinstance(payload, torch.Tensor):
        delta = payload
        return delta if list(delta.shape) == metadata["shape"] else None
    if not isinstance(payload, dict) or "delta" not in payload:
        return None
    if not _metadata_matches(dict(payload.get("metadata", {})), metadata):
        return None
    return payload["delta"].to(device)


def _save_delta(path: Path, delta: torch.Tensor, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": delta.detach().cpu(), "metadata": metadata}, path)


def _generate_delta(proxy: nn.Module, train_loader, target_pool, device, config: ExperimentConfig) -> torch.Tensor:
    qcfg = _qcfg(config)
    norm = str(getattr(qcfg, "norm", "linf")).lower()
    if norm != "linf":
        raise ValueError("Q-FGSM currently supports only norm='linf'.")

    epsilon = _epsilon(config)
    target = _target_label(config)
    delta = torch.zeros(
        1,
        1,
        int(config.data.victim_resolution),
        int(config.data.victim_resolution),
        device=device,
    )
    max_iter = max(1, int(getattr(qcfg, "max_iter", 10)))
    threshold = float(getattr(qcfg, "fooling_threshold", 0.6))

    proxy.eval()
    for _ in range(max_iter):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = y != target
            if not mask.any():
                continue

            x_src = resize_image(x[mask], int(config.data.victim_resolution))
            x_mix = _fuzzy_admix(x_src, target_pool, config)
            labels = torch.full(
                (x_mix.size(0),),
                target,
                dtype=torch.long,
                device=device,
            )

            delta = delta.detach().requires_grad_(True)
            poisoned = apply_delta(x_mix, delta, config)
            logits = proxy(normalize_for_victim(poisoned, config))
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = (delta - epsilon * grad.sign()).clamp(-epsilon, epsilon).detach()

        if _fooling_rate(proxy, train_loader, delta, device, config) >= threshold:
            break
    return delta.detach()


def build_generator(config: ExperimentConfig, device):
    device = torch.device(device)
    shape = [1, 1, int(config.data.victim_resolution), int(config.data.victim_resolution)]
    metadata = _cache_metadata(config, shape)
    cache_path = Path(getattr(_qcfg(config), "trigger_cache", "outputs/triggers/q_fgsm_delta.pt"))
    delta = _load_cached_delta(cache_path, metadata, device)
    if delta is not None:
        return QFGSMTrigger(delta)

    train_loader, _ = build_train_and_test_loaders(
        config,
        train_image_size=int(config.data.victim_resolution),
        test_image_size=int(config.data.victim_resolution),
        normalize=False,
    )
    proxy = _build_proxy(config, device)
    _train_proxy(proxy, train_loader, device, config)
    target_pool = _collect_target_pool(train_loader, device, config)
    delta = _generate_delta(proxy, train_loader, target_pool, device, config)
    _save_delta(cache_path, delta, metadata)
    return QFGSMTrigger(delta)


def poison_batch(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "eval",
    resolution: Optional[int] = None,
    config: ExperimentConfig = None,
    generator: Optional[QFGSMTrigger] = None,
):
    if generator is None:
        raise ValueError("Q-FGSM poisoning requires a generated universal trigger.")

    x = maybe_resize(x, resolution)
    target = _target_label(config)
    poisoned = x.clone()

    if y is None:
        poison_y = None
        source = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
    elif str(mode).lower() == "train":
        source = _target_poison_mask(y, config)
        poison_y = y.clone()
    else:
        source = y != target
        poison_y = y.clone()
        poison_y[source] = target

    if source.any():
        poisoned[source] = generator(x[source], config)
    return poisoned, poison_y, source


def train(config: ExperimentConfig, *args, **kwargs):
    from training.train_baselines import train_backdoor_baseline

    return train_backdoor_baseline(NAME, config, *args, **kwargs)


def build_method(config: ExperimentConfig, generator=None):
    return method_namespace(NAME, config, poison_batch, train, generator=generator)
