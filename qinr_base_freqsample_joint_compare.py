from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from configs.default import load_config
from datasets.builder import build_test_loader, build_train_and_test_loaders
from models.factory import build_classifier, build_qinr_generator
from training.poison import poison_batch_constrained, resize_image
from utils.device import resolve_device
from utils.eval_helpers import normalize_for_victim
from utils.io import write_json
from utils.seed import resolve_seed, set_seed


def default_output_dir(config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.train.save_dir) / "qinr_base_freqsample_joint_{}".format(stamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare QINR_base and QINR+Freqsample triggers under joint training."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    return parser.parse_args()


def _target_labels(y: torch.Tensor, config) -> torch.Tensor:
    return torch.full_like(y, int(config.train.target_label))


def _build_joint_optimizer(victim, generator, config):
    param_groups = [
        {
            "params": victim.parameters(),
            "lr": float(config.train.victim_lr),
            "weight_decay": float(config.train.weight_decay),
        },
        {
            "params": generator.quantum_parameters(),
            "lr": float(config.train.quantum_lr),
            "weight_decay": 0.0,
        },
    ]
    classical_params = list(generator.classical_parameters())
    if classical_params:
        param_groups.append(
            {
                "params": classical_params,
                "lr": float(config.train.classical_lr),
                "weight_decay": float(config.train.weight_decay),
            }
        )
    return torch.optim.AdamW(param_groups)


def _joint_train_epoch(victim, generator, train_loader, optimizer, device, config, alpha: float) -> dict:
    victim.train()
    generator.train()
    totals = {
        "loss": 0.0,
        "poison_loss": 0.0,
        "clean_loss": 0.0,
        "clean_acc_batch": 0.0,
        "poison_target_acc_batch": 0.0,
        "actual_delta_l1": 0.0,
    }
    count = 0

    alpha = float(alpha)
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target = _target_labels(y, config)

        poisoned, _, _, _ = poison_batch_constrained(
            generator=generator,
            x=x,
            config=config,
        )
        clean_input = resize_image(x, config.data.victim_resolution)
        poisoned_input = resize_image(poisoned, config.data.victim_resolution)
        clean_input = normalize_for_victim(clean_input, config)
        poisoned_input = normalize_for_victim(poisoned_input, config)

        clean_logits = victim(clean_input)
        poisoned_logits = victim(poisoned_input)
        clean_loss = F.cross_entropy(clean_logits, y)
        poison_loss = F.cross_entropy(poisoned_logits, target)
        loss = alpha * poison_loss + (1.0 - alpha) * clean_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_acc = (clean_logits.argmax(dim=1) == y).float().mean()
            poison_target_acc = (poisoned_logits.argmax(dim=1) == target).float().mean()
            actual_delta_l1 = (poisoned - x).abs().mean()

        totals["loss"] += float(loss.item())
        totals["poison_loss"] += float(poison_loss.item())
        totals["clean_loss"] += float(clean_loss.item())
        totals["clean_acc_batch"] += float(clean_acc.item())
        totals["poison_target_acc_batch"] += float(poison_target_acc.item())
        totals["actual_delta_l1"] += float(actual_delta_l1.item())
        count += 1

    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def _evaluate(victim, generator, test_loader, device, config) -> dict:
    victim.eval()
    generator.eval()
    target_label = int(config.train.target_label)
    clean_total = 0
    clean_correct = 0
    asr_total = 0
    asr_success = 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        clean_input = resize_image(x, config.data.victim_resolution)
        clean_input = normalize_for_victim(clean_input, config)
        clean_preds = victim(clean_input).argmax(dim=1)
        clean_total += int(y.numel())
        clean_correct += int((clean_preds == y).sum().item())

        candidate_idx = (y != target_label).nonzero(as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            continue
        x_src = x.index_select(0, candidate_idx)
        poisoned, _, _, _ = poison_batch_constrained(
            generator=generator,
            x=x_src,
            config=config,
        )
        poisoned_input = resize_image(poisoned, config.data.victim_resolution)
        poisoned_input = normalize_for_victim(poisoned_input, config)
        poison_preds = victim(poisoned_input).argmax(dim=1)
        asr_total += int(candidate_idx.numel())
        asr_success += int((poison_preds == target_label).sum().item())

    return {
        "clean_acc": clean_correct / max(clean_total, 1),
        "asr": asr_success / max(asr_total, 1),
    }


def _first_samples_per_class(loader, device, num_classes: int) -> dict:
    samples = {}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        for cls in range(int(num_classes)):
            if cls in samples:
                continue
            idx = (y == cls).nonzero(as_tuple=False)
            if idx.numel() > 0:
                samples[cls] = x[idx[0].item()].unsqueeze(0)
        if len(samples) >= int(num_classes):
            break
    return samples


def _image_for_plot(x: torch.Tensor) -> torch.Tensor:
    image = x[0].detach().cpu()
    if image.dim() == 3 and image.size(0) == 1:
        image = image.squeeze(0)
    return image


@torch.no_grad()
def save_trigger_comparison(variants: dict, config, test_loader, device, output_dir: Path, num_classes: int) -> Path:
    samples = _first_samples_per_class(test_loader, device, num_classes=num_classes)
    rows = []

    for cls in range(int(num_classes)):
        if cls not in samples:
            continue
        clean = samples[cls]
        row = {"class": cls, "clean": clean}
        for name, state in variants.items():
            poisoned, _, _, _ = poison_batch_constrained(
                generator=state["generator"],
                x=clean,
                config=state["config"],
            )
            actual_delta = (poisoned - clean).abs()
            row[name] = {
                "poisoned": poisoned,
                "actual_delta": actual_delta,
            }
        rows.append(row)

    if not rows:
        raise ValueError("No class samples were available for visualization.")

    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 2.5 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, row in enumerate(rows):
        clean_img = _image_for_plot(row["clean"])
        axes[row_idx, 0].imshow(clean_img, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title("Class {} clean".format(row["class"]))

        base_delta = _image_for_plot(row["qinr_base"]["actual_delta"])
        freq_delta = _image_for_plot(row["freqsample"]["actual_delta"])
        axes[row_idx, 1].imshow(base_delta, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("QINR_base |actual delta|")
        axes[row_idx, 2].imshow(freq_delta, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("Freqsample |actual delta|")

        base_poison = _image_for_plot(row["qinr_base"]["poisoned"])
        freq_poison = _image_for_plot(row["freqsample"]["poisoned"])
        axes[row_idx, 3].imshow(base_poison, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 3].set_title("QINR_base poisoned")
        axes[row_idx, 4].imshow(freq_poison, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 4].set_title("Freqsample poisoned")

        for col in range(5):
            axes[row_idx, col].axis("off")

    fig.suptitle("Joint-trained QINR trigger comparison (black=0, white=1)", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qinr_base_vs_freqsample_triggers.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    variant_metrics = {}
    for name in variants:
        per_class = [
            {
                "class": int(row["class"]),
                "mean_abs_delta": float(row[name]["actual_delta"].mean().item()),
                "max_abs_delta": float(row[name]["actual_delta"].max().item()),
            }
            for row in rows
        ]
        variant_metrics[name] = {
            "mean_abs_delta": sum(row["mean_abs_delta"] for row in per_class) / len(per_class),
            "max_abs_delta": max(row["max_abs_delta"] for row in per_class),
            "classes": per_class,
        }
    metrics_path = output_dir / "qinr_base_vs_freqsample_trigger_metrics.json"
    write_json(metrics_path, variant_metrics)
    for name, metrics in variant_metrics.items():
        print(
            "  {} perturbation mean_abs_delta={:.6f} max_abs_delta={:.6f}".format(
                name,
                metrics["mean_abs_delta"],
                metrics["max_abs_delta"],
            )
        )
    return output_path


def run_variant(name: str, qinr_base: bool, base_config, seed: int, alpha: float, epochs: int, device) -> dict:
    config = copy.deepcopy(base_config)
    config.train.seed = int(seed)
    config.model.qinr_base = bool(qinr_base)
    set_seed(seed)

    train_loader, test_loader = build_train_and_test_loaders(
        config,
        train_image_size=config.data.train_resolution,
        test_image_size=config.data.victim_resolution,
        normalize=False,
    )
    victim = build_classifier(config).to(device)
    generator = build_qinr_generator(config).to(device)
    optimizer = _build_joint_optimizer(victim, generator, config)

    history = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _joint_train_epoch(
            victim=victim,
            generator=generator,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            alpha=alpha,
        )
        eval_metrics = _evaluate(victim, generator, test_loader, device, config)
        row = {"epoch": epoch}
        row.update(train_metrics)
        row.update(eval_metrics)
        history.append(row)
        print(
            "{} epoch={} loss={:.4f} clean_acc={:.4f} asr={:.4f} delta_l1={:.6f}".format(
                name,
                epoch,
                row["loss"],
                row["clean_acc"],
                row["asr"],
                row["actual_delta_l1"],
            )
        )

    return {
        "name": name,
        "config": config,
        "victim": victim,
        "generator": generator,
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else resolve_seed(config.train.seed))
    epochs = int(args.epochs if args.epochs is not None else config.train.backdoor_epochs)
    alpha = float(args.alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--alpha must be in [0, 1].")

    device = resolve_device(config.train.device)
    print(
        "joint qinr comparison seed={} epochs={} alpha={} output_dir={}".format(
            seed, epochs, alpha, output_dir
        )
    )

    qinr_base_result = run_variant(
        name="qinr_base",
        qinr_base=True,
        base_config=config,
        seed=seed,
        alpha=alpha,
        epochs=epochs,
        device=device,
    )
    freqsample_result = run_variant(
        name="freqsample",
        qinr_base=False,
        base_config=config,
        seed=seed,
        alpha=alpha,
        epochs=epochs,
        device=device,
    )

    vis_loader = build_test_loader(
        config,
        image_size=config.data.train_resolution,
        normalize=False,
    )
    num_classes = int(args.num_classes if args.num_classes is not None else config.model.num_classes)
    variants = {
        "qinr_base": qinr_base_result,
        "freqsample": freqsample_result,
    }
    visualization_path = save_trigger_comparison(
        variants=variants,
        config=config,
        test_loader=vis_loader,
        device=device,
        output_dir=output_dir,
        num_classes=num_classes,
    )

    summary = {
        "seed": seed,
        "epochs": epochs,
        "alpha": alpha,
        "target_label": int(config.train.target_label),
        "visualization": str(visualization_path),
        "visualization_metrics": str(
            output_dir / "qinr_base_vs_freqsample_trigger_metrics.json"
        ),
        "variants": {
            name: {
                "qinr_base": bool(state["config"].model.qinr_base),
                "history": state["history"],
                "final_metrics": state["final_metrics"],
            }
            for name, state in variants.items()
        },
    }
    summary_path = output_dir / "qinr_base_freqsample_joint_summary.json"
    write_json(summary_path, summary)

    print("=" * 60)
    print("QINR_base vs QINR+Freqsample joint comparison complete.")


if __name__ == "__main__":
    main()
