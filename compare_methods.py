\"\"\"
Complete experiment comparison: NISQ vs BadNets vs Blended vs WaNet.

Trains each method, evaluates clean_acc and ASR, generates per-class
poisoned-image visualizations, and writes all results to a single CSV table.
\"\"\"

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from configs.default import ExperimentConfig, load_config
from datasets.builder import build_test_loader
from evaluation.visualize import visualize_all_methods
from methods import build_method_generator, get_method_module
from methods.common import attack_input_resolution
from models.factory import build_classifier, build_qinr_generator
from training.poison import resize_image
from training.train_baselines import evaluate_clean, evaluate_asr
from training.train_nisq import evaluate_clean as nisq_eval_clean, evaluate_asr as nisq_eval_asr
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.io import write_json
from utils.seed import seed_config


METHODS = [\"nisq\", \"badnets\", \"blended\", \"wanet\"]


@torch.no_grad()
def evaluate_nisq(
    victim: torch.nn.Module,
    generator: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    test_loader = build_test_loader(config, image_size=config.data.victim_resolution, normalize=False)
    clean_metrics = nisq_eval_clean(victim, test_loader, device, config)
    asr_metrics = nisq_eval_asr(victim, generator, test_loader, device, config)
    result = dict(clean_metrics)
    result.update(asr_metrics)
    result[\"method\"] = \"nisq\"
    return result


@torch.no_grad()
def evaluate_baseline(
    method: str,
    victim: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    test_loader = build_test_loader(config, image_size=config.data.victim_resolution, normalize=False)
    clean_metrics = evaluate_clean(victim, test_loader, device, config)
    method_module = get_method_module(method)
    generator = build_method_generator(method, config, device)
    if generator is not None:
        generator = generator.to(device)
        generator.eval()
    asr_metrics = evaluate_asr(victim, method_module, test_loader, device, config, generator=generator)
    result = dict(clean_metrics)
    result.update(asr_metrics)
    result[\"method\"] = method
    return result


def default_output_dir() -> Path:
    stamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")
    return Path.cwd() / \"runs\" / \"compare_{}\".format(stamp)


def write_results_csv(results: List[Dict[str, float]], output_dir: Path):
    fieldnames = [\"method\", \"clean_acc\", \"asr\", \"clean_loss\"]
    path = output_dir / \"comparison_results.csv\"
    with open(path, \"w\", newline=\"\") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, \"\") for k in fieldnames})
    print(\"Results table saved: {}\".format(path))
    write_json(output_dir / \"comparison_results.json\", results)
    return path


def train_method(
    method: str,
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
):
    \"\"\"Train a method. Returns (victim, maybe_generator).\"\"\"
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    print(\"=\" * 60)
    print(\"Training method: {}\".format(method))
    print(\"=\" * 60)

    if method == \"nisq\":
        result = run_nisq_training(config=config, output_dir=method_dir)
        ckpt = load_checkpoint(result[\"final_checkpoint\"], device=device)
        victim = build_classifier(config).to(device)
        victim.load_state_dict(ckpt[\"victim\"])
        gen = build_qinr_generator(config).to(device)
        gen.load_state_dict(ckpt[\"generator\"])
        gen.eval()
        print(\"{} training complete. clean_acc={:.4f} asr={:.4f}\".format(
            method, result.get(\"checkpoint_selection\", {}).get(\"clean_acc\", 0.0),
            result.get(\"checkpoint_selection\", {}).get(\"asr\", 0.0),
        ))
        return victim, gen
    else:
        from training.train_baselines import train_backdoor_baseline
        result = train_backdoor_baseline(method, config, output_dir=method_dir)
        ckpt = load_checkpoint(result[\"final_checkpoint\"], device=device)
        victim = build_classifier(config).to(device)
        victim.load_state_dict(ckpt[\"victim\"])
        print(\"{} training complete. clean_acc={:.4f} asr={:.4f}\".format(
            method, result.get(\"clean_acc\", 0.0), result.get(\"asr\", 0.0),
        ))
        return victim, None


def run_comparison(
    config: ExperimentConfig,
    output_dir: Optional[Path] = None,
    skip_train: bool = False,
) -> Dict:
    seed_config(config)
    device = resolve_device(config.train.device)
    output_dir = Path(output_dir or default_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    config.train.qinr_epochs = config.train.backdoor_epochs

    print(\"Backdoor method comparison experiment\")
    print(\"Methods: {}\".format(METHODS))
    print(\"Output: {}\".format(output_dir))
    print(\"Dataset: {}  poison_rate={}  target_label={}  epsilon={}\".format(
        config.data.dataset, config.train.poison_rate,
        config.train.target_label, config.train.epsilon,
    ))
    print()

    victims: Dict[str, torch.nn.Module] = {}
    generators: Dict[str, Optional[torch.nn.Module]] = {}
    results: List[Dict[str, float]] = []

    # --- Train each method ---
    for method in METHODS:
        victim, gen = train_method(method, config, output_dir, device)
        victims[method] = victim
        generators[method] = gen

    # --- Evaluate all methods ---
    print()
    print(\"=\" * 60)
    print(\"Final Evaluation\")
    print(\"=\" * 60)
    for method in METHODS:
        if method == \"nisq\":
            result = evaluate_nisq(victims[method], generators[method], config, device)
        else:
            result = evaluate_baseline(method, victims[method], config, device)
        results.append(result)
        print(\"  {:<12s}  clean_acc={:.4f}  asr={:.4f}\".format(
            method, result.get(\"clean_acc\", 0.0), result.get(\"asr\", 0.0),
        ))

    # --- Write results table ---
    write_results_csv(results, output_dir)

    # --- Generate per-class visualizations ---
    print()
    print(\"Generating per-class poisoned-sample visualizations...\")
    vis_dir = output_dir / \"visualizations\"
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualize_all_methods(METHODS, config, victims, vis_dir, device, generators=generators)

    print()
    print(\"=\" * 60)
    print(\"Comparison complete.\")
    print(\"Results:     {}/comparison_results.csv\".format(output_dir))
    print(\"Visualizations: {}/visualizations/\".format(output_dir))
    print(\"=\" * 60)

    return {
        \"output_dir\": str(output_dir),
        \"results\": results,
        \"results_csv\": str(output_dir / \"comparison_results.csv\"),
        \"visualization_dir\": str(vis_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=\"Backdoor method comparison: NISQ vs BadNets vs Blended vs WaNet.\"
    )
    parser.add_argument(\"--config\", default=None, help=\"Path to YAML/JSON config.\")
    parser.add_argument(\"--output-dir\", default=None, help=\"Output directory.\")
    args = parser.parse_args()
    config = load_config(args.config)
    run_comparison(config, output_dir=args.output_dir)


if __name__ == \"__main__\":
    main()
