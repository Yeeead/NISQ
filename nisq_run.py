from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from configs.default import load_config
from training.train_nisq import run_nisq_training


def default_output_dir(config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.train.save_dir) / "nisq_backdoor_{}".format(stamp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NISQ low-shot implicit neural backdoor attack on ResNet18."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_nisq_training(config=config, output_dir=output_dir)

    print("=" * 60)
    print("NISQ backdoor training complete.")
    print("  final_checkpoint: {}".format(result["final_checkpoint"]))
    print("  best_checkpoint:  {}".format(result["best_checkpoint"]))
    print("  selected_epoch:   {}".format(result["checkpoint_selection"]["epoch"]))
    print("  clean_acc:        {:.4f}".format(result["checkpoint_selection"]["clean_acc"]))
    print("  asr:              {:.4f}".format(result["checkpoint_selection"]["asr"]))
    print("  score:            {:.4f}".format(result["checkpoint_selection"]["checkpoint_score"]))


if __name__ == "__main__":
    main()