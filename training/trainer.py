from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from configs.default import save_config
from utils.checkpoint import save_checkpoint
from utils.io import write_json


class MetricTracker:
    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Dict[str, float], weight: int = 1) -> None:
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * int(weight)
        self.count += int(weight)

    def compute(self) -> Dict[str, float]:
        denom = max(int(self.count), 1)
        return {key: value / denom for key, value in self.totals.items()}


def average_totals(totals: Dict[str, float], count: int) -> Dict[str, float]:
    denom = max(int(count), 1)
    return {key: float(value) / denom for key, value in totals.items()}


def checkpoint_paths(output_dir, prefix: str = ""):
    checkpoint_dir = Path(output_dir) / "checkpoints"
    return {
        "best": checkpoint_dir / "best.pt",
        "latest": checkpoint_dir / "latest.pt",
        "final": checkpoint_dir / "final.pt",
    }


def checkpoint_state(**items):
    return dict(items)


def _data_split_summary(config) -> Dict:
    data = config.data
    return {
        "dataset": str(data.dataset),
        "data_root": str(data.data_root),
        "train_subset": int(getattr(data, "train_subset", 0)),
        "test_subset": int(getattr(data, "test_subset", 0)),
        "train_resolution": int(data.train_resolution),
        "victim_resolution": int(data.victim_resolution),
        "standard_split": "train/test",
    }


class Trainer:
    """Small run helper for metadata and checkpoint bookkeeping."""

    def __init__(self, config, output_dir, seed) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.paths = checkpoint_paths(self.output_dir)

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, self.output_dir / "config.json")
        write_json(
            self.output_dir / "run_metadata.json",
            {
                "seed": self.seed,
                "started_at": self.started_at,
                "output_dir": str(self.output_dir),
            },
        )
        write_json(self.output_dir / "data_split.json", _data_split_summary(self.config))

    def save_checkpoint(self, name: str, state: Dict) -> Path:
        path = self.paths[name]
        save_checkpoint(state, path)
        return path


def _metric_value(metrics: Dict, *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return 0.0


def checkpoint_selection_score(metrics: Dict) -> float:
    clean_acc = max(0.0, _metric_value(metrics, "clean_acc", "clean_accuracy"))
    asr = max(0.0, _metric_value(metrics, "asr", "ASR"))
    if clean_acc <= 0.0 or asr <= 0.0:
        return 0.0
    return 2.0 * clean_acc * asr / (clean_acc + asr)


def checkpoint_selection_key(metrics: Dict):
    clean_acc = _metric_value(metrics, "clean_acc", "clean_accuracy")
    asr = _metric_value(metrics, "asr", "ASR")
    return checkpoint_selection_score(metrics), clean_acc, asr


def is_better_checkpoint(metrics: Dict, best_metrics: Optional[Dict]) -> bool:
    if best_metrics is None:
        return True
    return checkpoint_selection_key(metrics) > checkpoint_selection_key(best_metrics)


def clone_state_dict(state_dict):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def checkpoint_selection_summary(epoch: int, metrics: Dict) -> Dict:
    return {
        "epoch": int(epoch),
        "clean_acc": _metric_value(metrics, "clean_acc", "clean_accuracy"),
        "asr": _metric_value(metrics, "asr", "ASR"),
        "checkpoint_score": checkpoint_selection_score(metrics),
        "selection_rule": "harmonic_mean(clean_acc,asr)",
    }
