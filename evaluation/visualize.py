"""
Per-class poisoned image visualization for backdoor methods.

For each method and each class, produces a row showing:
  [clean image] [poisoned image] [overlay (delta)]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default import ExperimentConfig
from datasets.builder import build_test_loader
from methods import get_method_module
from methods.common import attack_input_resolution
from models.factory import build_qinr_generator
from training.poison import poison_batch_constrained
from utils.io import write_json


@torch.no_grad()
def visualize_method_per_class(
    method_name: str,
    config: ExperimentConfig,
    victim: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    generator: Optional[torch.nn.Module] = None,
    num_classes: int = 10,
) -> Path:
    victim.eval()

    loader = build_test_loader(config, image_size=config.data.train_resolution, normalize=False)
    class_samples: Dict[int, torch.Tensor] = {}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for cls in range(num_classes):
            if cls in class_samples:
                continue
            idx = (y == cls).nonzero(as_tuple=False)
            if idx.numel() > 0:
                class_samples[cls] = x[idx[0].item()].unsqueeze(0)
        if len(class_samples) >= num_classes:
            break

    if method_name in {"nisq", "classical_inr", "single_qubit_qinr"}:
        if generator is None:
            if method_name in {"classical_inr", "single_qubit_qinr"}:
                raise ValueError("{} visualization requires a generator.".format(method_name))
            gen = build_qinr_generator(config).to(device)
        else:
            gen = generator
        gen.eval()

        def poison_fn(clean):
            poisoned, delta, _, _ = poison_batch_constrained(gen, clean, config)
            return poisoned, delta
    else:
        method_module = get_method_module(method_name)
        resolution = attack_input_resolution(config, method_name)

        def poison_fn(clean):
            p, _, _ = method_module.poison_batch(clean, mode="eval", resolution=resolution, config=config, generator=generator)
            return p, (p - clean)

    ncols = 3
    fig, axes = plt.subplots(num_classes, ncols, figsize=(ncols * 3, num_classes * 2.5))
    fig.suptitle("Method: {} - Poisoned Samples Per Class".format(method_name), fontsize=14)
    class_metrics = []

    for cls in range(num_classes):
        if cls not in class_samples:
            continue
        clean = class_samples[cls]
        row = axes[cls]

        clean_img = clean[0].cpu()
        if clean_img.dim() == 3 and clean_img.size(0) == 1:
            clean_img = clean_img.squeeze(0)
        row[0].imshow(clean_img, cmap="gray", vmin=0, vmax=1)
        row[0].set_title("Class {} clean".format(cls))
        row[0].axis("off")

        poisoned, delta = poison_fn(clean)
        poison_img = poisoned[0].cpu()
        if poison_img.dim() == 3 and poison_img.size(0) == 1:
            poison_img = poison_img.squeeze(0)
        row[1].imshow(poison_img, cmap="gray", vmin=0, vmax=1)
        row[1].set_title("Class {} poisoned".format(cls))
        row[1].axis("off")

        overlay_img = delta[0].abs().cpu()
        if overlay_img.dim() == 3 and overlay_img.size(0) == 1:
            overlay_img = overlay_img.squeeze(0)
        row[2].imshow(overlay_img, cmap="gray", vmin=0, vmax=1)
        row[2].set_title("Overlay (|delta|)")
        row[2].axis("off")
        class_metrics.append(
            {
                "class": int(cls),
                "mean_abs_delta": float(overlay_img.mean().item()),
                "max_abs_delta": float(overlay_img.max().item()),
            }
        )

    plt.tight_layout()
    save_path = output_dir / "{}_poisoned_samples.png".format(method_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if class_metrics:
        mean_abs_delta = sum(row["mean_abs_delta"] for row in class_metrics) / len(class_metrics)
        max_abs_delta = max(row["max_abs_delta"] for row in class_metrics)
    else:
        mean_abs_delta = 0.0
        max_abs_delta = 0.0
    metrics_path = output_dir / "{}_poisoned_samples_metrics.json".format(method_name)
    write_json(
        metrics_path,
        {
            "method": method_name,
            "mean_abs_delta": mean_abs_delta,
            "max_abs_delta": max_abs_delta,
            "classes": class_metrics,
        },
    )
    print(
        "  {} perturbation mean_abs_delta={:.6f} max_abs_delta={:.6f}".format(
            method_name,
            mean_abs_delta,
            max_abs_delta,
        )
    )
    return save_path


@torch.no_grad()
def visualize_all_methods(
    methods: List[str],
    config: ExperimentConfig,
    victims: Dict[str, torch.nn.Module],
    output_dir: Path,
    device: torch.device,
    generators: Optional[Dict[str, Optional[torch.nn.Module]]] = None,
    num_classes: int = 10,
) -> Dict[str, Path]:
    if generators is None:
        generators = {}
    results = {}
    for method in methods:
        gen = generators.get(method)
        path = visualize_method_per_class(
            method_name=method, config=config, victim=victims[method],
            output_dir=output_dir, device=device, generator=gen,
            num_classes=num_classes,
        )
        results[method] = path
    return results
