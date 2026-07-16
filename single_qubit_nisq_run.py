from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from configs.default import load_config
from evaluation.visualize import visualize_method_per_class
from models.factory import (
    build_classifier,
    build_single_qubit_layer_freq_qinr_generator,
)
from nisq_run import (
    DEFAULT_STRIP_ALPHA,
    DEFAULT_STRIP_CLEAN_FPR,
    DEFAULT_STRIP_REPEATS,
    DEFAULT_STRIP_SAMPLES,
    evaluate_strip_entropy,
)
from training.train_nisq import run_nisq_training
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.io import write_json


def default_output_dir(config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.train.save_dir) / "single_qubit_nisq_backdoor_{}".format(stamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-qubit QINR backdoor attack with layer-wise sampled frequency coefficients."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--strip-samples",
        type=int,
        default=DEFAULT_STRIP_SAMPLES,
        help="Number of test samples for STRIP; 0 means all test samples.",
    )
    parser.add_argument("--strip-repeats", type=int, default=DEFAULT_STRIP_REPEATS)
    parser.add_argument("--strip-alpha", type=float, default=DEFAULT_STRIP_ALPHA)
    parser.add_argument("--strip-clean-fpr", type=float, default=DEFAULT_STRIP_CLEAN_FPR)
    return parser.parse_args()


def _prepare_single_qubit_config(config):
    config.model.qinr_n_qubits = 1
    config.model.qinr_base = False
    return config


def _load_trained_models(config, checkpoint_path: str | Path, device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    victim = build_classifier(config).to(device)
    generator = build_single_qubit_layer_freq_qinr_generator(config).to(device)
    victim.load_state_dict(checkpoint["victim"])
    generator.load_state_dict(checkpoint["generator"])
    victim.eval()
    generator.eval()
    return victim, generator


def main() -> None:
    args = parse_args()
    config = _prepare_single_qubit_config(load_config(args.config))
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_nisq_training(
        config=config,
        output_dir=output_dir,
        generator_builder=build_single_qubit_layer_freq_qinr_generator,
        experiment_name="single_qubit_layer_freq_qinr_backdoor",
        log_label="single-qubit layer-freq qinr",
    )
    device = resolve_device(config.train.device)
    victim, generator = _load_trained_models(config, result["final_checkpoint"], device)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualization_path = visualize_method_per_class(
        method_name="single_qubit_qinr",
        config=config,
        victim=victim,
        output_dir=vis_dir,
        device=device,
        generator=generator,
        num_classes=int(config.model.num_classes),
    )

    strip_dir = output_dir / "strip"
    strip_result = evaluate_strip_entropy(
        victim=victim,
        generator=generator,
        config=config,
        device=device,
        output_dir=strip_dir,
        num_samples=int(args.strip_samples),
        repeats=int(args.strip_repeats),
        alpha=float(args.strip_alpha),
        clean_fpr=float(args.strip_clean_fpr),
    )
    post_training_artifacts = {
        "visualization": str(visualization_path),
        "visualization_metrics_json": str(
            vis_dir / "single_qubit_qinr_poisoned_samples_metrics.json"
        ),
        "strip_entropy_json": str(strip_dir / "strip_entropy.json"),
        "strip_entropy_histogram": str(strip_dir / "strip_entropy_hist.png"),
        "strip_entropy": {
            "sample_mode": strip_result["strip_sample_mode"],
            "sample_count": strip_result["strip_samples"],
            "clean_mean": strip_result["clean_entropy"]["mean"],
            "poisoned_mean": strip_result["poisoned_entropy"]["mean"],
            "clean_minus_poisoned": strip_result["entropy_gap_clean_minus_poisoned"],
            "threshold": strip_result["rejection"]["threshold"],
            "clean_reject_rate": strip_result["rejection"]["clean_reject_rate"],
            "poisoned_reject_rate": strip_result["rejection"]["poisoned_reject_rate"],
            "clean_fpr": strip_result["rejection"]["clean_fpr"],
        },
        "model": {
            "qinr_n_qubits": int(config.model.qinr_n_qubits),
            "frequency_count": int(config.model.qinr_n_layers),
            "frequency_granularity": "data_encoding_layer",
        },
    }
    write_json(output_dir / "post_training_artifacts.json", post_training_artifacts)

    print("=" * 60)
    print("Single-qubit layer-frequency QINR backdoor training complete.")
    print("  selected_epoch:   {}".format(result["checkpoint_selection"]["epoch"]))
    print("  clean_acc:        {:.4f}".format(result["checkpoint_selection"]["clean_acc"]))
    print("  asr:              {:.4f}".format(result["checkpoint_selection"]["asr"]))
    print("  score:            {:.4f}".format(result["checkpoint_selection"]["checkpoint_score"]))
    print("  qinr_n_qubits:    {}".format(config.model.qinr_n_qubits))
    print("  freq_count:       {} (one per data encoding layer)".format(config.model.qinr_n_layers))
    print(
        "  strip_samples:    {} ({})".format(
            strip_result["strip_samples"],
            strip_result["strip_sample_mode"],
        )
    )
    print(
        "  strip_gap:        clean_mean={:.4f} poisoned_mean={:.4f} gap={:.4f}".format(
            strip_result["clean_entropy"]["mean"],
            strip_result["poisoned_entropy"]["mean"],
            strip_result["entropy_gap_clean_minus_poisoned"],
        )
    )
    print(
        "  strip_reject:     fpr={:.4f} threshold={:.4f} clean={:.4f} poisoned={:.4f}".format(
            strip_result["rejection"]["clean_fpr"],
            strip_result["rejection"]["threshold"],
            strip_result["rejection"]["clean_reject_rate"],
            strip_result["rejection"]["poisoned_reject_rate"],
        )
    )


if __name__ == "__main__":
    main()
