from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default import ExperimentConfig, load_config
from datasets.builder import build_test_loader
from methods import build_method_generator, get_method_module
from models.factory import build_classifier, build_qinr_generator
from training.train_baselines import (
    evaluate_asr as evaluate_baseline_asr,
    evaluate_clean as evaluate_baseline_clean,
    train_backdoor_baseline,
)
from training.train_nisq import (
    evaluate_asr as evaluate_nisq_asr,
    evaluate_clean as evaluate_nisq_clean,
    run_nisq_training,
)
from training.trainer import checkpoint_selection_score
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.io import write_json


METHODS = ["nisq", "badnets", "wanet", "blended"]
POISON_RATES = [0.01, 0.02, 0.03, 0.04, 0.05]
Y_AXIS_STRETCH_THRESHOLD = 0.15
Y_AXIS_MIN_SPAN = 0.05


def default_output_dir(config: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.train.save_dir) / "poison_rate_sweep_{}".format(stamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep poisoning rate and compare NISQ with BadNets, WaNet, and Blended."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _set_poison_rate(config: ExperimentConfig, poison_rate: float) -> None:
    config.train.poison_rate = float(poison_rate)
    config.backdoor.poison_rate = float(poison_rate)
    config.backdoor.target_label = int(config.train.target_label)
    config.backdoor.clamp_min = float(config.data.input_range[0])
    config.backdoor.clamp_max = float(config.data.input_range[1])


def _method_label(method: str) -> str:
    labels = {
        "nisq": "NISQ",
        "badnets": "BadNets",
        "wanet": "WaNet",
        "blended": "Blended",
    }
    return labels.get(method, method)


def _evaluate_nisq_checkpoint(checkpoint_path: str, config: ExperimentConfig) -> Dict:
    device = resolve_device(config.train.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    test_loader = build_test_loader(
        config,
        image_size=config.data.victim_resolution,
        normalize=False,
    )

    victim = build_classifier(config).to(device)
    victim.load_state_dict(checkpoint["victim"])
    victim.eval()

    generator = build_qinr_generator(config).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    clean_metrics = evaluate_nisq_clean(victim, test_loader, device, config)
    asr_metrics = evaluate_nisq_asr(victim, generator, test_loader, device, config)
    metrics = dict(clean_metrics)
    metrics.update(asr_metrics)
    return {
        "metric_split": "test",
        "clean_acc": float(metrics["clean_acc"]),
        "asr": float(metrics["asr"]),
        "checkpoint_score": float(checkpoint_selection_score(metrics)),
    }


def _evaluate_baseline_checkpoint(
    method: str,
    checkpoint_path: str,
    config: ExperimentConfig,
    constrain_to_epsilon: bool,
) -> Dict:
    device = resolve_device(config.train.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    test_loader = build_test_loader(
        config,
        image_size=config.data.victim_resolution,
        normalize=False,
    )

    victim = build_classifier(config).to(device)
    victim.load_state_dict(checkpoint["victim"])
    victim.eval()

    method_module = get_method_module(method)
    generator = build_method_generator(method, config, device)
    if generator is not None:
        generator = generator.to(device)
        generator.eval()

    clean_metrics = evaluate_baseline_clean(victim, test_loader, device, config)
    asr_metrics = evaluate_baseline_asr(
        victim,
        method_module,
        test_loader,
        device,
        config,
        generator=generator,
        constrain_to_epsilon=constrain_to_epsilon,
    )
    metrics = dict(clean_metrics)
    metrics.update(asr_metrics)
    return {
        "metric_split": "test",
        "clean_acc": float(metrics["clean_acc"]),
        "asr": float(metrics["asr"]),
        "checkpoint_score": float(checkpoint_selection_score(metrics)),
    }


def _train_one(
    method: str,
    config: ExperimentConfig,
    output_dir: Path,
) -> Dict:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    if method == "nisq":
        result = run_nisq_training(config=config, output_dir=method_dir)
        metrics = _evaluate_nisq_checkpoint(result["final_checkpoint"], config)
        metrics["method"] = method
        return metrics

    if method == "blended":
        config.blended.alpha = float(config.train.epsilon)

    constrain_to_epsilon = method not in ("blended", "wanet")
    result = train_backdoor_baseline(
        method,
        config,
        output_dir=method_dir,
        constrain_to_epsilon=constrain_to_epsilon,
    )
    metrics = _evaluate_baseline_checkpoint(
        method,
        result["final_checkpoint"],
        config,
        constrain_to_epsilon=constrain_to_epsilon,
    )
    metrics["method"] = method
    return metrics


def _write_csv(rows: List[Dict], output_dir: Path) -> Path:
    path = output_dir / "poison_rate_sweep_results.csv"
    fieldnames = [
        "poison_rate",
        "method",
        "metric_split",
        "clean_acc",
        "asr",
        "checkpoint_score",
        "epsilon",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _set_metric_ylim(ax, values: List[float]) -> None:
    if not values:
        ax.set_ylim(0.0, 1.0)
        return

    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span > Y_AXIS_STRETCH_THRESHOLD:
        ax.set_ylim(0.0, 1.0)
        return

    padded_span = max(span, Y_AXIS_MIN_SPAN)
    padding = padded_span * 0.2
    center = (min_value + max_value) / 2.0
    half_span = padded_span / 2.0 + padding
    lower = max(0.0, center - half_span)
    upper = min(1.0, center + half_span)

    if upper - lower < Y_AXIS_MIN_SPAN:
        if lower <= 0.0:
            upper = min(1.0, lower + Y_AXIS_MIN_SPAN)
        else:
            lower = max(0.0, upper - Y_AXIS_MIN_SPAN)

    ax.set_ylim(lower, upper)


def _plot_results(rows: List[Dict], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    all_asr: List[float] = []
    all_clean_acc: List[float] = []
    for method in METHODS:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["poison_rate"],
        )
        if not method_rows:
            continue
        poison_rates = [row["poison_rate"] for row in method_rows]
        asr = [row["asr"] for row in method_rows]
        clean_acc = [row["clean_acc"] for row in method_rows]
        all_asr.extend(asr)
        all_clean_acc.extend(clean_acc)
        label = _method_label(method)
        axes[0].plot(poison_rates, asr, marker="o", label=label)
        axes[1].plot(poison_rates, clean_acc, marker="o", label=label)

    axes[0].set_title("ASR vs Poisoning Rate")
    axes[0].set_ylabel("ASR")
    axes[1].set_title("Clean Accuracy vs Poisoning Rate")
    axes[1].set_ylabel("Clean accuracy")
    for ax in axes:
        ax.set_xlabel("Poisoning rate")
        ax.set_xticks(POISON_RATES)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    _set_metric_ylim(axes[0], all_asr)
    _set_metric_ylim(axes[1], all_clean_acc)
    axes[1].legend(loc="best")
    fig.tight_layout()
    path = output_dir / "poison_rate_sweep_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_sweep(config: ExperimentConfig, output_dir: Path | None = None) -> Dict:
    output_dir = Path(output_dir or default_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for poison_rate in POISON_RATES:
        for method in METHODS:
            run_config = copy.deepcopy(config)
            _set_poison_rate(run_config, poison_rate)
            run_dir = output_dir / "poison_rate_{:.2f}".format(poison_rate)

            print(
                "running method={} poison_rate={:.2f} epsilon={}".format(
                    method,
                    poison_rate,
                    run_config.train.epsilon,
                )
            )
            result = _train_one(method, run_config, output_dir=run_dir)
            row = {
                "poison_rate": float(poison_rate),
                "epsilon": float(run_config.train.epsilon),
            }
            row.update(result)
            rows.append(row)
            print(
                "  {:<8s} clean_acc={:.4f} asr={:.4f}".format(
                    method,
                    row["clean_acc"],
                    row["asr"],
                )
            )

    _write_csv(rows, output_dir)
    json_path = output_dir / "poison_rate_sweep_results.json"
    _plot_results(rows, output_dir)
    summary = {
        "methods": METHODS,
        "poison_rates": POISON_RATES,
        "metric_split": "test",
        "epsilon": float(config.train.epsilon),
        "hard_epsilon_constraint": False,
        "post_epsilon_clipped_methods": ["badnets"],
        "unclipped_warp_methods": ["wanet"],
        "rows": rows,
    }
    write_json(json_path, summary)
    print("=" * 60)
    print("Poison-rate sweep complete.")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_sweep(config, output_dir=Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
