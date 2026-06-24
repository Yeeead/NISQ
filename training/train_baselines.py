from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from configs.default import ExperimentConfig
from datasets.builder import build_test_loader, build_train_and_test_loaders
from methods import build_method_generator, get_method_module
from methods.common import attack_input_resolution
from models.factory import build_classifier
from training.optim import build_victim_optimizer
from training.poison import resize_image
from training.steps import optimizer_step
from training.trainer import (
    Trainer,
    checkpoint_selection_score,
    checkpoint_selection_summary,
    checkpoint_state,
    clone_state_dict,
    is_better_checkpoint,
)
from utils.device import resolve_device
from utils.eval_helpers import normalize_for_victim
from utils.io import write_jsonl
from utils.seed import seed_config, seed_wanet_config


@torch.no_grad()
def evaluate_clean(victim, loader, device, config: ExperimentConfig) -> Dict[str, float]:
    victim.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = normalize_for_victim(x, config)
        logits = victim(x)
        total += y.numel()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
    total = max(total, 1)
    return {"clean_acc": correct / total, "clean_loss": total_loss / total}


@torch.no_grad()
def evaluate_asr(victim, method_module, loader, device, config: ExperimentConfig, generator=None) -> Dict[str, float]:
    victim.eval()
    target_label = int(config.train.target_label)
    total = 0
    success = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        candidate_idx = (y != target_label).nonzero(as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            continue
        x_src = x.index_select(0, candidate_idx)
        resolution = attack_input_resolution(config, method_module.NAME)
        poisoned_x, _, _ = method_module.poison_batch(
            x_src, mode="eval", resolution=resolution, config=config, generator=generator,
        )
        poisoned_x = resize_image(poisoned_x, config.data.victim_resolution)
        poisoned_x = normalize_for_victim(poisoned_x, config)
        logits = victim(poisoned_x)
        preds = logits.argmax(dim=1)
        total += int(candidate_idx.numel())
        success += int((preds == target_label).sum().item())
    denom = max(total, 1)
    return {"asr": success / denom}


def train_backdoor_baseline(
    method_name: str,
    config: ExperimentConfig,
    output_dir: Optional[Path] = None,
) -> Dict:
    seed = seed_config(config)
    device = resolve_device(config.train.device)
    output_dir = Path(output_dir or config.train.save_dir)
    trainer = Trainer(config, output_dir, seed)
    trainer.prepare()
    output_dir = trainer.output_dir

    train_loader, _ = build_train_and_test_loaders(
        config, train_image_size=config.data.train_resolution,
        test_image_size=config.data.victim_resolution, normalize=False,
    )
    test_loader = build_test_loader(
        config, image_size=config.data.victim_resolution, normalize=False,
    )

    victim = build_classifier(config).to(device)
    method_module = get_method_module(method_name)
    generator = build_method_generator(method_name, config, device)
    if generator is not None:
        generator = generator.to(device)

    victim_optimizer = build_victim_optimizer(victim, config)
    epochs = int(getattr(config.train, "backdoor_epochs", config.train.qinr_epochs))

    print(
        "baseline {} seed={} epochs={} poison_rate={} target_label={}".format(
            method_name, seed, epochs, config.train.poison_rate, config.train.target_label,
        )
    )

    log_path = output_dir / "train_log.jsonl"
    selected_metrics: Optional[Dict[str, float]] = None
    selected_victim = clone_state_dict(victim.state_dict())

    resolution = attack_input_resolution(config, method_name)

    for epoch in range(1, epochs + 1):
        victim.train()
        totals = {}
        count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            poisoned_x, poison_y, _ = method_module.poison_batch(
                x, y, mode="train", resolution=resolution, config=config, generator=generator,
            )
            poisoned_x = resize_image(poisoned_x, config.data.victim_resolution)
            poisoned_x = normalize_for_victim(poisoned_x, config)

            logits = victim(poisoned_x)
            loss = F.cross_entropy(logits, poison_y)
            optimizer_step(loss, victim_optimizer)

            preds = logits.detach().argmax(dim=1)
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            totals["acc"] = totals.get("acc", 0.0) + (preds == poison_y).float().mean().item()
            count += 1

        clean_metrics = evaluate_clean(victim, test_loader, device, config)
        asr_metrics = evaluate_asr(victim, method_module, test_loader, device, config, generator=generator)

        row = {"stage": "baseline", "method": method_name, "epoch": epoch}
        row.update(clean_metrics)
        row.update(asr_metrics)

        candidate_metrics = dict(clean_metrics)
        candidate_metrics.update(asr_metrics)
        checkpoint_selected = is_better_checkpoint(candidate_metrics, selected_metrics)
        if checkpoint_selected:
            selected_metrics = candidate_metrics
            selected_victim = clone_state_dict(victim.state_dict())

        row["checkpoint_score"] = checkpoint_selection_score(candidate_metrics)
        row["checkpoint_selected"] = bool(checkpoint_selected)
        write_jsonl(log_path, [row], append=True)
        trainer.save_checkpoint("latest", checkpoint_state(
            kind="baseline_latest", checkpoint_type="latest", epoch=epoch,
            method=method_name, victim=clone_state_dict(victim.state_dict()),
            metrics=candidate_metrics, config=config.to_dict(),
        ))
        print(
            "{} epoch={} clean={:.4f} asr={:.4f}".format(
                method_name, epoch, clean_metrics.get("clean_acc", 0.0), asr_metrics.get("asr", 0.0),
            )
        )

    if selected_metrics is None:
        selected_metrics = {}
    victim.load_state_dict(selected_victim)
    trainer.save_checkpoint("best", checkpoint_state(
        kind="baseline_selected", checkpoint_type="best", epoch=epochs,
        method=method_name, victim=selected_victim, metrics=selected_metrics,
        config=config.to_dict(),
    ))
    final_state = {"kind": "baseline_final", "checkpoint_type": "final",
                   "method": method_name, "victim": selected_victim,
                   "metrics": selected_metrics, "config": config.to_dict()}
    trainer.save_checkpoint("final", final_state)

    return {
        "final_checkpoint": str(trainer.paths["final"]),
        "best_checkpoint": str(trainer.paths["best"]),
        "method": method_name,
        "clean_acc": selected_metrics.get("clean_acc", 0.0),
        "asr": selected_metrics.get("asr", 0.0),
    }
