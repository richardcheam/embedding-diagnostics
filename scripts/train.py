"""Train one condition and log diagnostics.

Usage:
    python scripts/train.py --condition sigreg_nostopgrad --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jepa_lens.config import load_config
from jepa_lens.data import build_dataloaders
from jepa_lens.hardware import require_device
from jepa_lens.logging_utils import RunLogger
from jepa_lens.training.trainer import Trainer, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one jepa-lens condition")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-steps", type=int, default=None, help="override config")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="override config")
    parser.add_argument("--tag", default="default", help="experiment group directory name")
    parser.add_argument(
        "--base-config",
        default="base.yaml",
        help="base config file under configs/ (base.yaml = CIFAR-10, bdd.yaml = BDD100K)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the run seed; pair runs across conditions by using the same value",
    )
    parser.add_argument(
        "--eval-split-seed",
        type=int,
        default=None,
        help="override data.eval_split_seed; keep fixed across conditions and run seeds",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=None,
        help="override strategy.sigreg_weight; ignored by conditions without SIGReg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Before the CIFAR-10 download and model construction: a mismatched CUDA
    # build otherwise surfaces minutes in, as a message about driver versions.
    require_device(args.device)
    config = load_config(args.condition, ROOT / "configs", base=args.base_config)
    if args.total_steps is not None:
        config["optim"]["total_steps"] = args.total_steps
    if args.checkpoint_every is not None:
        config["logging"]["checkpoint_every"] = args.checkpoint_every
    if args.sigreg_weight is not None:
        config["strategy"]["sigreg_weight"] = args.sigreg_weight
    if args.seed is not None:
        config["seed"] = args.seed
    if args.eval_split_seed is not None:
        config["data"]["eval_split_seed"] = args.eval_split_seed

    run_dir = ROOT / config["logging"]["output_root"] / args.tag / args.condition
    ssl_loader, probe_train, probe_test = build_dataloaders(config)
    trainer = Trainer(config, device=args.device)

    print(
        f"condition={args.condition} device={args.device} "
        f"seed={config['seed']} eval_split_seed={config['data'].get('eval_split_seed', 0)}"
    )
    print(f"sigreg_weight={config['strategy']['sigreg_weight']}")
    print(f"encoder params={count_parameters(trainer.context_encoder):,}")
    print(f"predictor params={count_parameters(trainer.predictor):,}")
    print(f"steps={config['optim']['total_steps']} -> {run_dir}")

    with RunLogger(run_dir) as logger:
        logger.write_config(config)
        trainer.fit(ssl_loader, probe_train, probe_test, logger)

    print(f"done: {run_dir / 'metrics.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
