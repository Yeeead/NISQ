from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from configs.default import ExperimentConfig
from methods.common import canonical_method
from methods.triggers import apply_badnets_trigger, apply_blended_trigger, apply_wanet_trigger
from training.poison import (
    additive_poison,
    constrain_actual_delta,
    constrain_delta,
    generate_delta_image,
    rebuild_poisoned_from_actual_delta,
)
from utils.eval_helpers import backdoor_bounds
from utils.mnist import load_mnist_class_image
from utils.seed import seed_wanet_config


@dataclass
class TriggerOutput:
    poisoned_images: torch.Tensor
    effective_delta: torch.Tensor
    metadata: Dict[str, Any]


_BLENDED_PATTERN_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def _train_resolution(config: ExperimentConfig) -> int:
    xres = getattr(config, "cross_resolution_eval", None)
    return int(getattr(xres, "train_resolution", config.data.train_resolution))


def _qinr_shots(config: ExperimentConfig):
    shots = getattr(config.model, "qinr_shots", None)
    if shots is None:
        return None
    if isinstance(shots, str) and shots.strip().lower() in {"", "none", "null"}:
        return None
    return int(shots)


def _resolved_wanet_seed(config: ExperimentConfig) -> int:
    seed = getattr(config.wanet, "seed", 0)
    if seed is False:
        return seed_wanet_config(config)
    if isinstance(seed, str) and seed.strip().lower() in {"false", "random"}:
        return seed_wanet_config(config)
    return int(seed)


def _target_image_pattern(config: ExperimentConfig, target_label: int) -> torch.Tensor:
    blended = config.blended
    seed = int(getattr(blended, "pattern_seed", 0))
    key = (
        str(config.data.dataset).lower(),
        str(config.data.data_root),
        _train_resolution(config),
        int(target_label),
        seed,
    )
    pattern = _BLENDED_PATTERN_CACHE.get(key)
    if pattern is None:
        pattern = load_mnist_class_image(
            config.data,
            target_label=int(target_label),
            image_size=_train_resolution(config),
            seed=seed,
            train=True,
        ).detach()
        _BLENDED_PATTERN_CACHE[key] = pattern
    return pattern


@contextmanager
def _manual_seed(seed: Optional[int], device: torch.device):
    if seed is None:
        yield
        return

    cuda_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


class TriggerAdapter:
    def __init__(self, method_name: str, config: ExperimentConfig, generator=None):
        self.method_name = canonical_method(method_name)
        self.config = config
        self.generator = generator

    @property
    def shots(self):
        if self.method_name != "qinr":
            return None
        if self.generator is not None and hasattr(self.generator, "shots"):
            return getattr(self.generator, "shots")
        return _qinr_shots(self.config)

    @property
    def is_stochastic(self) -> bool:
        return self.method_name == "qinr" and self.shots is not None

    def apply(
        self,
        images: torch.Tensor,
        *,
        target_label: int,
        repeat_seed: int | None = None,
    ) -> TriggerOutput:
        if images.dim() != 4:
            raise ValueError("trigger adapter input must be [B, C, H, W], got {}".format(tuple(images.shape)))

        if self.method_name == "badnets":
            poisoned = self._apply_badnets(images)
        elif self.method_name == "blended":
            poisoned = self._apply_blended(images, target_label=int(target_label))
        elif self.method_name == "wanet":
            poisoned = self._apply_wanet(images)
        elif self.method_name == "qinr":
            poisoned = self._apply_qinr(images, repeat_seed=repeat_seed)
        elif self.method_name == "inputaware":
            poisoned = self._apply_inputaware(images)
        else:
            raise ValueError("Unknown backdoor method: {}".format(self.method_name))

        if poisoned.shape != images.shape:
            raise ValueError(
                "{} trigger returned shape {}, expected {}".format(
                    self.method_name,
                    tuple(poisoned.shape),
                    tuple(images.shape),
                )
            )

        return TriggerOutput(
            poisoned_images=poisoned,
            effective_delta=poisoned - images,
            metadata={
                "method": self.method_name,
                "target_label": int(target_label),
                "repeat_seed": repeat_seed,
                "shots": self.shots,
            },
        )

    def _apply_badnets(self, images: torch.Tensor) -> torch.Tensor:
        base_resolution = max(_train_resolution(self.config), 1)
        scale = min(float(images.size(-2)), float(images.size(-1))) / float(base_resolution)
        patch_size = max(1, int(round(float(self.config.badnets.patch_size) * scale)))
        low, high = backdoor_bounds(self.config)
        return apply_badnets_trigger(
            images,
            patch_size=patch_size,
            patch_value=float(self.config.badnets.patch_value),
            location=str(self.config.badnets.location),
            clamp_min=low,
            clamp_max=high,
        )

    def _apply_blended(self, images: torch.Tensor, target_label: int) -> torch.Tensor:
        blended = self.config.blended
        pattern_type = str(blended.pattern_type)
        pattern = None
        if pattern_type.lower() in {"target_image", "target_class_image"}:
            pattern = _target_image_pattern(self.config, target_label=int(target_label))
        low, high = backdoor_bounds(self.config)
        return apply_blended_trigger(
            images,
            alpha=float(self.config.train.epsilon),
            pattern=pattern,
            pattern_type=pattern_type,
            clamp_min=low,
            clamp_max=high,
            seed=int(getattr(blended, "pattern_seed", 0)),
        )

    def _apply_wanet(self, images: torch.Tensor) -> torch.Tensor:
        wanet = self.config.wanet
        low, high = backdoor_bounds(self.config)
        return apply_wanet_trigger(
            images,
            s=float(wanet.s),
            grid_res=int(wanet.grid_res),
            align_corners=bool(wanet.align_corners),
            clamp_min=low,
            clamp_max=high,
            seed=_resolved_wanet_seed(self.config),
            scale_mode=str(getattr(wanet, "scale_mode", "wanet")),
            normalize=str(getattr(wanet, "normalize", "mean_abs")),
            upsample_mode=str(getattr(wanet, "upsample_mode", "bicubic")),
            grid_rescale=float(getattr(wanet, "grid_rescale", 1.0)),
            sample_mode=str(getattr(wanet, "sample_mode", "bilinear")),
            padding_mode=str(getattr(wanet, "padding_mode", "zeros")),
        )

    def _apply_qinr(self, images: torch.Tensor, repeat_seed: int | None) -> torch.Tensor:
        if self.generator is None:
            raise ValueError("QINR trigger adapter requires a generator.")
        with _manual_seed(repeat_seed, images.device):
            raw_delta, _ = generate_delta_image(
                generator=self.generator,
                batch_size=1,
                channels=images.size(1),
                height=images.size(-2),
                width=images.size(-1),
                coord_range=tuple(self.config.data.coord_range),
                device=images.device,
                dtype=images.dtype,
            )
        if images.size(0) > 1:
            raw_delta = raw_delta.expand(images.size(0), -1, -1, -1)
        delta = constrain_delta(raw_delta, self.config)
        return additive_poison(
            images,
            delta,
            epsilon=float(self.config.train.epsilon),
            input_range=backdoor_bounds(self.config),
        )

    def _apply_inputaware(self, images: torch.Tensor) -> torch.Tensor:
        if self.generator is None:
            raise ValueError("Input-aware trigger adapter requires a generator.")
        raw_delta, _, _ = self.generator.trigger_delta(
            x=images,
            trigger_source=images,
            input_range=tuple(self.config.data.input_range),
        )
        delta = constrain_actual_delta(raw_delta, self.config)
        return rebuild_poisoned_from_actual_delta(images, delta, self.config)

    def apply_inputaware_train_resolution_trigger(
        self,
        images: torch.Tensor,
        train_images: torch.Tensor,
        *,
        interpolation: str,
        antialias: bool,
    ) -> TriggerOutput:
        if self.method_name != "inputaware":
            raise ValueError("train-resolution input-aware trigger is only valid for inputaware.")
        if self.generator is None:
            raise ValueError("Input-aware trigger adapter requires a generator.")
        if images.dim() != 4 or train_images.dim() != 4:
            raise ValueError(
                "input-aware trigger inputs must be [B, C, H, W], got {} and {}".format(
                    tuple(images.shape),
                    tuple(train_images.shape),
                )
            )
        if images.size(0) != train_images.size(0) or images.size(1) != train_images.size(1):
            raise ValueError(
                "source images and train-resolution images must share batch/channels, got {} and {}".format(
                    tuple(images.shape),
                    tuple(train_images.shape),
                )
            )

        raw_delta, pattern, mask = self.generator.trigger_delta(
            x=train_images,
            trigger_source=train_images,
            input_range=tuple(self.config.data.input_range),
        )
        actual_delta = constrain_actual_delta(raw_delta, self.config)
        source_size = tuple(images.shape[-2:])
        if tuple(actual_delta.shape[-2:]) != source_size:
            actual_delta = F.interpolate(
                actual_delta,
                size=source_size,
                mode=str(interpolation),
                align_corners=False,
                antialias=bool(antialias),
            )
        poisoned = rebuild_poisoned_from_actual_delta(images, actual_delta, self.config)
        return TriggerOutput(
            poisoned_images=poisoned,
            effective_delta=poisoned - images,
            metadata={
                "method": self.method_name,
                "trigger_generation_resolution": int(train_images.size(-1)),
                "source_resolution": int(images.size(-1)),
                "pattern_shape": tuple(pattern.shape),
                "mask_shape": tuple(mask.shape),
            },
        )


def build_trigger_adapter(method_name: str, config: ExperimentConfig, generator=None) -> TriggerAdapter:
    return TriggerAdapter(method_name=method_name, config=config, generator=generator)
