from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default import load_config
from datasets.builder import build_test_loader
from evaluation.visualize import visualize_method_per_class
from models.factory import build_classifier, build_qinr_generator
from training.poison import poison_batch_constrained, resize_image
from training.train_nisq import run_nisq_training
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.eval_helpers import normalize_for_victim
from utils.io import write_json


DEFAULT_STRIP_SAMPLES = 300
DEFAULT_STRIP_REPEATS = 10
DEFAULT_STRIP_ALPHA = 0.5
DEFAULT_STRIP_CLEAN_FPR = 0.05


def _parse_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def default_output_dir(config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.train.save_dir) / "nisq_backdoor_{}".format(stamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NISQ low-shot implicit neural backdoor attack on ResNet18."
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
    parser.add_argument("--qinr-base", type=_parse_bool, default=None)
    return parser.parse_args()


def _load_trained_models(config, checkpoint_path: str | Path, device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    victim = build_classifier(config).to(device)
    generator = build_qinr_generator(config).to(device)
    victim.load_state_dict(checkpoint["victim"])
    generator.load_state_dict(checkpoint["generator"])
    victim.eval()
    generator.eval()
    return victim, generator


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1.0e-12)
    return -(probs * probs.log()).sum(dim=1)


def _summary_stats(values) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _calibrate_strip_rejection(clean_entropies, poisoned_entropies, clean_fpr: float) -> dict:
    if not clean_entropies:
        return {
            "clean_fpr": float(clean_fpr),
            "threshold": 0.0,
            "clean_reject_rate": 0.0,
            "poisoned_reject_rate": 0.0,
        }
    clean_fpr = min(max(float(clean_fpr), 0.0), 1.0)
    clean_tensor = torch.tensor(clean_entropies, dtype=torch.float32)
    poisoned_tensor = torch.tensor(poisoned_entropies, dtype=torch.float32)
    threshold = float(torch.quantile(clean_tensor, clean_fpr).item())
    clean_reject = float((clean_tensor < threshold).float().mean().item())
    if poisoned_tensor.numel() == 0:
        poisoned_reject = 0.0
    else:
        poisoned_reject = float((poisoned_tensor < threshold).float().mean().item())
    return {
        "clean_fpr": clean_fpr,
        "threshold": threshold,
        "clean_reject_rate": clean_reject,
        "poisoned_reject_rate": poisoned_reject,
    }


def _take_reference_images(loader, device, needed: int) -> torch.Tensor:
    refs = []
    for x, _ in loader:
        refs.append(x.to(device, non_blocking=True))
        if int(needed) > 0 and sum(batch.size(0) for batch in refs) >= int(needed):
            break
    if not refs:
        raise ValueError("STRIP evaluation requires at least one test image.")
    return torch.cat(refs, dim=0)


@torch.no_grad()
def evaluate_strip_entropy(
    victim,
    generator,
    config,
    device,
    output_dir: Path,
    num_samples: int,
    repeats: int,
    alpha: float,
    clean_fpr: float,
) -> dict:
    if int(repeats) <= 0:
        raise ValueError("STRIP evaluation requires strip-repeats > 0.")
    if int(num_samples) < 0:
        raise ValueError("STRIP evaluation requires strip-samples >= 0.")
    loader = build_test_loader(
        config,
        image_size=config.data.train_resolution,
        normalize=False,
    )
    sample_limit = int(num_samples)
    reference_images = _take_reference_images(
        loader=loader,
        device=device,
        needed=max(sample_limit, int(repeats)) if sample_limit > 0 else 0,
    )
    sample_images = reference_images if sample_limit == 0 else reference_images[:sample_limit]
    if sample_images.numel() == 0:
        raise ValueError("STRIP evaluation requires at least one test sample.")

    clean_entropies = []
    poisoned_entropies = []
    alpha = float(alpha)
    valid_min = float(config.data.input_range[0])
    valid_max = float(config.data.input_range[1])

    for idx in range(sample_images.size(0)):
        clean = sample_images[idx : idx + 1]
        poisoned, _, _, _ = poison_batch_constrained(
            generator=generator,
            x=clean,
            config=config,
        )

        clean_probs = []
        poisoned_probs = []
        for repeat_idx in range(int(repeats)):
            ref_idx = (idx + repeat_idx) % reference_images.size(0)
            ref = reference_images[ref_idx : ref_idx + 1]
            mixed_clean = torch.clamp(
                alpha * clean + (1.0 - alpha) * ref,
                min=valid_min,
                max=valid_max,
            )
            mixed_poisoned = torch.clamp(
                alpha * poisoned + (1.0 - alpha) * ref,
                min=valid_min,
                max=valid_max,
            )
            mixed_clean = resize_image(mixed_clean, config.data.victim_resolution)
            mixed_poisoned = resize_image(mixed_poisoned, config.data.victim_resolution)
            mixed_clean = normalize_for_victim(mixed_clean, config)
            mixed_poisoned = normalize_for_victim(mixed_poisoned, config)
            clean_probs.append(F.softmax(victim(mixed_clean), dim=1))
            poisoned_probs.append(F.softmax(victim(mixed_poisoned), dim=1))

        clean_mean = torch.stack(clean_probs, dim=0).mean(dim=0)
        poisoned_mean = torch.stack(poisoned_probs, dim=0).mean(dim=0)
        clean_entropies.append(float(_entropy_from_probs(clean_mean).item()))
        poisoned_entropies.append(float(_entropy_from_probs(poisoned_mean).item()))

    clean_stats = _summary_stats(clean_entropies)
    poisoned_stats = _summary_stats(poisoned_entropies)
    rejection = _calibrate_strip_rejection(
        clean_entropies=clean_entropies,
        poisoned_entropies=poisoned_entropies,
        clean_fpr=clean_fpr,
    )
    result = {
        "strip_samples": int(sample_images.size(0)),
        "strip_sample_mode": "all" if sample_limit == 0 else "limited",
        "strip_repeats": int(repeats),
        "strip_alpha": alpha,
        "clean_entropy": clean_stats,
        "poisoned_entropy": poisoned_stats,
        "entropy_gap_clean_minus_poisoned": clean_stats["mean"] - poisoned_stats["mean"],
        "rejection": rejection,
        "clean_entropies": clean_entropies,
        "poisoned_entropies": poisoned_entropies,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "strip_entropy.json", result)
    _save_strip_histogram(
        clean_entropies=clean_entropies,
        poisoned_entropies=poisoned_entropies,
        output_path=output_dir / "strip_entropy_hist.png",
    )
    return result


def _save_strip_histogram(clean_entropies, poisoned_entropies, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clean_entropies, bins=20, alpha=0.6, label="clean", color="#2f6fbb")
    ax.hist(poisoned_entropies, bins=20, alpha=0.6, label="poisoned", color="#c73e3a")
    ax.set_xlabel("Prediction entropy")
    ax.set_ylabel("Count")
    ax.set_title("STRIP entropy distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.qinr_base is not None:
        config.model.qinr_base = bool(args.qinr_base)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_nisq_training(config=config, output_dir=output_dir)
    device = resolve_device(config.train.device)
    victim, generator = _load_trained_models(config, result["final_checkpoint"], device)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualization_path = visualize_method_per_class(
        method_name="nisq",
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
            vis_dir / "nisq_poisoned_samples_metrics.json"
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
    }
    write_json(output_dir / "post_training_artifacts.json", post_training_artifacts)

    print("=" * 60)
    print("NISQ backdoor training complete.")
    print("  selected_epoch:   {}".format(result["checkpoint_selection"]["epoch"]))
    print("  clean_acc:        {:.4f}".format(result["checkpoint_selection"]["clean_acc"]))
    print("  asr:              {:.4f}".format(result["checkpoint_selection"]["asr"]))
    print("  score:            {:.4f}".format(result["checkpoint_selection"]["checkpoint_score"]))
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
