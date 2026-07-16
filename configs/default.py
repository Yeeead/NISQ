from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar, Union


@dataclass
class ModelConfig:
    victim_arch: str = "resnet18_mnist"
    num_classes: int = 10
    victim_channels: int = 32
    qinr_n_qubits: int = 3
    qinr_n_layers: int = 8
    qinr_base: bool = False
    qinr_freq_mode: str = "nisq"
    qinr_baseline_freq: float = 1.0
    qinr_freq_scale: float = 10
    qinr_freq_distribution: str = "normal"
    qinr_freq_trainable: bool = True
    qinr_freq_std_init: float = 1
    qinr_measurement: str = "pauli_z"
    qinr_shots: Union[int, None] = 8
    qinr_out_channels: int = 1


@dataclass
class TrainConfig:
    clean_epochs: int = 1
    qinr_epochs: int = 2
    backdoor_epochs: int = 2
    batch_size: int = 256
    poison_rate: float = 0.05
    target_label: int = 0
    epsilon: float = 0.03
    generator_k: int = 1
    quantum_lr: float = 1.0e-1
    classical_lr: float = 1.0e-3
    
    victim_lr: float = 1.0e-3
    weight_decay: float = 0.0
    seed: Optional[Union[int, bool, str]] = "random"
    device: str = "auto"
    log_every: int = 100
    save_dir: str = "runs/nisq_backdoor"


@dataclass
class LossConfig:
    lambda_l1: float = 10
    lambda_zero_mean: float = 0
    delta_linf_clip: float = 1.0


@dataclass
class DataConfig:
    dataset: str = "mnist"
    data_root: str = "data"
    train_resolution: int = 32
    victim_resolution: int = 32
    input_range: Tuple[float, float] = (0.0, 1.0)
    coord_range: Tuple[float, float] = (-1.0, 1.0)
    normalize_mean: Tuple[float, ...] = ()
    normalize_std: Tuple[float, ...] = ()
    num_workers: int = 2
    pin_memory: bool = True
    download: bool = True
    train_subset: int = 0
    test_subset: int = 0


@dataclass
class EvalConfig:
    save_dir: str = "runs/nisq_backdoor"


@dataclass
class BackdoorConfig:
    method: str = "q_fgsm"
    methods: List[str] = field(default_factory=lambda: ["badnets", "blended", "wanet", "qinr", "inputaware"])
    target_label: int = 0
    poison_rate: float = 0.05
    clamp_min: float = 0.0
    clamp_max: float = 1.0


@dataclass
class BadNetsConfig:
    patch_size: int = 3
    patch_value: float = 1.0
    location: str = "bottom_right"


@dataclass
class BlendedConfig:
    pattern_type: str = "target_image"
    pattern_seed: int = 0


@dataclass
class WaNetConfig:
    s: float = 0.5
    grid_res: int = 3
    align_corners: bool = True
    seed: Optional[Union[int, bool, str]] = "random"
    cross_ratio: float = 2.0
    noise_s: float = 1.0
    scale_mode: str = "wanet"
    normalize: str = "mean_abs"
    upsample_mode: str = "bicubic"
    grid_rescale: float = 1.0
    sample_mode: str = "bilinear"
    padding_mode: str = "zeros"


@dataclass
class ClassicalINRConfig:
    out_channels: int = 1
    hidden_dim: int = 128
    hidden_layers: int = 3
    n_frequencies: int = 5
    freq_scale: float = 10
    freq_distribution: str = "normal"


@dataclass
class QFGSMConfig:
    proxy_model: str = "victim"
    epsilon: float = 0.1
    max_iter: int = 10
    fooling_threshold: float = 0.6
    norm: str = "linf"
    fuzzy_admix: bool = True
    admix_n: int = 3
    admix_c: float = 1.0
    admix_sigma: float = 2.0
    trigger_cache: str = "outputs/triggers/q_fgsm_delta.pt"


@dataclass
class InputAwareConfig:
    hidden_channels: int = 16
    mask_pretrain_epochs: int = 1
    mask_l1_budget: float = 0.1
    mask_l1_weight: float = 0.01
    lambda_div: float = 0.0001
    mask_lr: float = 1.0e-4
    delta_lr: float = 1.0e-4
    rho_attack: float = 0.05
    rho_cross: float = 0.05
    train_mask_after_pretrain: bool = False
    diversity_eps: float = 0.1


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)
    badnets: BadNetsConfig = field(default_factory=BadNetsConfig)
    blended: BlendedConfig = field(default_factory=BlendedConfig)
    wanet: WaNetConfig = field(default_factory=WaNetConfig)
    classical_inr: ClassicalINRConfig = field(default_factory=ClassicalINRConfig)
    q_fgsm: QFGSMConfig = field(default_factory=QFGSMConfig)
    inputaware: InputAwareConfig = field(default_factory=InputAwareConfig)
    checkpoint_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))


T = TypeVar("T")


def _serialize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize(v) for v in value]
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def _coerce_sequence(value: Any, target: Sequence[Any]) -> Any:
    if isinstance(target, tuple):
        return tuple(value)
    if isinstance(target, list):
        return list(value)
    return value


def _update_dataclass(instance: T, values: Mapping[str, Any]) -> T:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError("Unknown config key '{}' for {}".format(key, type(instance).__name__))

        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise TypeError("Config section '{}' must be a mapping".format(key))
            _update_dataclass(current, value)
        elif isinstance(current, (tuple, list)) and isinstance(value, (tuple, list)):
            setattr(instance, key, _coerce_sequence(value, current))
        else:
            setattr(instance, key, value)
    return instance


def _sync_backdoor_defaults(config: ExperimentConfig, values: Mapping[str, Any]) -> None:
    backdoor_values = values.get("backdoor", {})
    if not isinstance(backdoor_values, Mapping):
        backdoor_values = {}

    if "target_label" not in backdoor_values:
        config.backdoor.target_label = int(config.train.target_label)
    if "poison_rate" not in backdoor_values:
        config.backdoor.poison_rate = float(config.train.poison_rate)
    if "clamp_min" not in backdoor_values:
        config.backdoor.clamp_min = float(config.data.input_range[0])
    if "clamp_max" not in backdoor_values:
        config.backdoor.clamp_max = float(config.data.input_range[1])


def config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig()
    _update_dataclass(config, values)
    _sync_backdoor_defaults(config, values)
    return config


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            raw = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML config files require PyYAML. Install requirements.txt or use JSON."
                ) from exc
            raw = yaml.safe_load(f) or {}

    if not isinstance(raw, MutableMapping):
        raise TypeError("Top-level config must be a mapping.")
    return config_from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
