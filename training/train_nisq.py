from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from configs.default import ExperimentConfig
from datasets.builder import build_test_loader, build_train_and_test_loaders
from models.factory import build_classifier, build_qinr_generator
from training.losses import qinr_total_loss
from training.poison import poison_batch_constrained, resize_image
from training.optim import build_qinr_optimizer, build_victim_optimizer
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
from utils.seed import seed_config
from utils.training import set_requires_grad


def _select_poison_indices(
    y: torch.Tensor,
    poison_rate: float,
    target_label: int,
) -> torch.Tensor:
    poison_rate = float(poison_rate)
    candidate_idx = (y != int(target_label)).nonzero(as_tuple=False).flatten()
    if candidate_idx.numel() == 0:
        return candidate_idx
    count = int(round(y.size(0) * min(poison_rate, 1.0)))
    count = max(1, min(count, candidate_idx.numel()))
    perm = torch.randperm(candidate_idx.numel(), device=y.device)[:count]
    return candidate_idx.index_select(0, perm)


@torch.no_grad()
def _build_nisq_batch(
    generator,
    x: torch.Tensor,
    y: torch.Tensor,
    poison_idx: torch.Tensor,
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int]:
    target_label = int(config.train.target_label)
    if poison_idx.numel() == 0:
        return x, y, None, x.size(0), 0
    keep_mask = torch.ones(y.size(0), dtype=torch.bool, device=y.device)
    keep_mask[poison_idx] = False
    clean_idx = keep_mask.nonzero(as_tuple=False).flatten()
    x_clean = x.index_select(0, clean_idx)
    y_clean = y.index_select(0, clean_idx)
    x_poison_src = x.index_select(0, poison_idx)
    poisoned_x, delta, _, _ = poison_batch_constrained(
        generator=generator, x=x_poison_src, config=config,
    )
    y_poison = torch.full(
        (poisoned_x.size(0),), target_label, dtype=torch.long, device=y.device,
    )
    mixed_x = torch.cat([x_clean, poisoned_x], dim=0)
    mixed_y = torch.cat([y_clean, y_poison], dim=0)
    return mixed_x, mixed_y, delta, x_clean.size(0), poisoned_x.size(0)


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
def evaluate_asr(victim, generator, loader, device, config: ExperimentConfig) -> Dict[str, float]:
    victim.eval()
    generator.eval()
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
        poisoned_x, delta, _, _ = poison_batch_constrained(
            generator=generator, x=x_src, config=config,
        )
        poisoned_x = resize_image(poisoned_x, config.data.victim_resolution)
        poisoned_x = normalize_for_victim(poisoned_x, config)
        logits = victim(poisoned_x)
        preds = logits.argmax(dim=1)
        total += int(candidate_idx.numel())
        success += int((preds == target_label).sum().item())
    denom = max(total, 1)
    return {"asr": success / denom}


def _train_epoch(
    victim, generator, train_loader,
    victim_optimizer, qinr_optimizer,
    device, config: ExperimentConfig,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target_label = int(config.train.target_label)
        poison_idx = _select_poison_indices(
            y=y, poison_rate=config.train.poison_rate, target_label=target_label,
        )

        # Fallback: no non-target samples
        if poison_idx.numel() == 0:
            victim.train()
            generator.eval()
            set_requires_grad(victim, True)
            set_requires_grad(generator, False)
            x_v = resize_image(x, config.data.victim_resolution)
            x_v = normalize_for_victim(x_v, config)
            loss = F.cross_entropy(victim(x_v), y)
            optimizer_step(loss, victim_optimizer)
            for k in ["victim_loss", "mixed_acc", "clean_acc_batch", "poison_asr_batch"]:
                totals[k] = totals.get(k, 0.0) + 0.0
            totals["victim_loss"] = totals.get("victim_loss", 0.0) + loss.item()
            mixed_acc = (victim(x_v).detach().argmax(dim=1) == y).float().mean().item()
            totals["mixed_acc"] = totals.get("mixed_acc", 0.0) + mixed_acc
            totals["clean_acc_batch"] = totals.get("clean_acc_batch", 0.0) + mixed_acc
            count += 1
            continue

        x_poison_src = x.index_select(0, poison_idx)
        y_poison = torch.full(
            (x_poison_src.size(0),), target_label, dtype=torch.long, device=device,
        )

        # Step A: freeze victim, train generator for k steps
        victim.eval()
        generator.train()
        set_requires_grad(victim, False)
        set_requires_grad(generator, True)

        for _ in range(max(1, int(config.train.generator_k))):
            poisoned_x, delta, _, _ = poison_batch_constrained(
                generator=generator, x=x_poison_src, config=config,
            )
            pv = resize_image(poisoned_x, config.data.victim_resolution)
            pv = normalize_for_victim(pv, config)
            losses = dict(qinr_total_loss(victim(pv), y_poison, delta, config.loss))
            optimizer_step(losses["loss"], qinr_optimizer)

        # Step B: freeze generator, train victim on mixed batch
        victim.train()
        generator.eval()
        set_requires_grad(victim, True)
        set_requires_grad(generator, False)

        mixed_x, mixed_y, delta_eval, clean_count, poison_count = _build_nisq_batch(
            generator=generator, x=x, y=y, poison_idx=poison_idx, config=config,
        )
        mixed_x = resize_image(mixed_x, config.data.victim_resolution)
        mixed_x = normalize_for_victim(mixed_x, config)
        logits = victim(mixed_x)
        v_loss = F.cross_entropy(logits, mixed_y)
        optimizer_step(v_loss, victim_optimizer)

        preds = logits.detach().argmax(dim=1)
        totals["victim_loss"] = totals.get("victim_loss", 0.0) + v_loss.item()
        totals["mixed_acc"] = totals.get("mixed_acc", 0.0) + (preds == mixed_y).float().mean().item()
        totals["clean_acc_batch"] = totals.get("clean_acc_batch", 0.0) + (
            (preds[:clean_count] == mixed_y[:clean_count]).float().mean().item() if clean_count > 0 else 0.0
        )
        totals["poison_asr_batch"] = totals.get("poison_asr_batch", 0.0) + (
            (preds[clean_count:] == mixed_y[clean_count:]).float().mean().item() if poison_count > 0 else 0.0
        )
        count += 1

    denom = max(count, 1)
    return {k: v / denom for k, v in totals.items()}


def run_nisq_training(
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
    generator = build_qinr_generator(config).to(device)
    victim_optimizer = build_victim_optimizer(victim, config)
    qinr_optimizer = build_qinr_optimizer(generator, config)

    print(
        "nisq backdoor seed={} n_qubits={} n_layers={} shots={} epochs={} "
        "generator_k={} poison_rate={} epsilon={} l1_lambda={}".format(
            seed, config.model.qinr_n_qubits, config.model.qinr_n_layers,
            getattr(generator, "shots", None),
            config.train.backdoor_epochs, config.train.generator_k,
            config.train.poison_rate, config.train.epsilon,
            config.loss.lambda_l1,
        )
    )

    log_path = output_dir / "train_log.jsonl"
    backdoor_epochs = int(getattr(config.train, "backdoor_epochs", config.train.qinr_epochs))

    selected_epoch = 0
    selected_metrics: Optional[Dict[str, float]] = None
    selected_victim = clone_state_dict(victim.state_dict())
    selected_generator = clone_state_dict(generator.state_dict())

    for epoch in range(1, backdoor_epochs + 1):
        train_metrics = _train_epoch(
            victim=victim, generator=generator, train_loader=train_loader,
            victim_optimizer=victim_optimizer, qinr_optimizer=qinr_optimizer,
            device=device, config=config,
        )

        clean_metrics = evaluate_clean(victim, test_loader, device, config)
        asr_metrics = evaluate_asr(victim, generator, test_loader, device, config)

        row = {"stage": "nisq_backdoor", "epoch": epoch}
        row.update(train_metrics)
        row.update(clean_metrics)
        row.update(asr_metrics)

        candidate_metrics = dict(clean_metrics)
        candidate_metrics.update(asr_metrics)
        checkpoint_selected = is_better_checkpoint(candidate_metrics, selected_metrics)
        if checkpoint_selected:
            selected_epoch = int(epoch)
            selected_metrics = candidate_metrics
            selected_victim = clone_state_dict(victim.state_dict())
            selected_generator = clone_state_dict(generator.state_dict())

        row["checkpoint_score"] = checkpoint_selection_score(candidate_metrics)
        row["checkpoint_selected"] = bool(checkpoint_selected)
        write_jsonl(log_path, [row], append=True)
        trainer.save_checkpoint("latest", checkpoint_state(
            kind="nisq_latest", checkpoint_type="latest", epoch=epoch,
            victim=clone_state_dict(victim.state_dict()),
            generator=clone_state_dict(generator.state_dict()),
            metrics=candidate_metrics,
            checkpoint_selection=checkpoint_selection_summary(epoch, candidate_metrics),
            config=config.to_dict(),
        ))
        print(
            "nisq epoch={} clean={:.4f} asr={:.4f} score={:.4f} selected={}".format(
                epoch, clean_metrics.get("clean_acc", 0.0), asr_metrics.get("asr", 0.0),
                checkpoint_selection_score(candidate_metrics),
                str(bool(checkpoint_selected)).lower(),
            )
        )

    if selected_metrics is None:
        selected_metrics = {}
    victim.load_state_dict(selected_victim)
    generator.load_state_dict(selected_generator)
    best_state = checkpoint_state(
        kind="nisq_selected", checkpoint_type="best", epoch=selected_epoch,
        selected_epoch=selected_epoch,
        victim=selected_victim, generator=selected_generator,
        metrics=selected_metrics,
        checkpoint_selection=checkpoint_selection_summary(selected_epoch, selected_metrics),
        config=config.to_dict(),
    )
    trainer.save_checkpoint("best", best_state)
    final_state = dict(best_state)
    final_state["checkpoint_type"] = "final"
    trainer.save_checkpoint("final", final_state)

    selection = checkpoint_selection_summary(selected_epoch, selected_metrics)
    print(
        "selected epoch={} clean_acc={:.4f} asr={:.4f} score={:.4f}".format(
            selected_epoch, selection["clean_acc"], selection["asr"],
            selection["checkpoint_score"],
        )
    )

    return {
        "final_checkpoint": str(trainer.paths["final"]),
        "best_checkpoint": str(trainer.paths["best"]),
        "latest_checkpoint": str(trainer.paths["latest"]),
        "checkpoint_selection": selection,
    }
