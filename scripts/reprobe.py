"""Recompute probe endpoints from a finished run's saved encoder.

    uv run python scripts/reprobe.py --tag phaseB --seeds 0,1,2,3,4 --data-root ../100k
    uv run python scripts/reprobe.py --tag phaseB_s0 --condition ema_stopgrad --compare

WHY THIS EXISTS. The probe protocol used during training was defective: sklearn's
lbfgs stops when the gradient norm falls below `tol`, the gradient scales with
feature magnitude, and on severely scale-contracted embeddings the default
`tol=1e-4` is met almost immediately. The solver takes two or three steps, never
fits, predicts the majority class, reports chance -- and raises no
ConvergenceWarning. Every unstandardized probe endpoint measured on a contracted
representation is therefore suspect. See `jepa_lens.diagnostics.probe`.

Retraining to fix a measurement would be absurd, and the encoder is saved, so
this reloads it, re-embeds the same evaluation split, and refits the probes under
the corrected protocol. Inference only.

Output goes to `metrics_reprobed.jsonl` beside the original. The original is
never modified: the comparison between the two protocols is itself a result, and
overwriting the old numbers would destroy the evidence that the correction
mattered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_lens.data import build_dataloaders
from jepa_lens.diagnostics.probe import ProbeConfig, linear_probe_scores, majority_rate
from jepa_lens.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def load_encoder(run_dir: Path, config: dict, device: str) -> Trainer:
    """Rebuild the trainer and load the saved encoder weights into it.

    Goes through Trainer rather than constructing a bare ViTEncoder so that
    `encode_all` -- pooling, batching, eval mode, no_grad -- is the exact same
    tested code path that produced the original numbers. The predictor and
    optimizer it also builds are unused here and cost nothing.
    """
    trainer = Trainer(config, device=device)
    state = torch.load(run_dir / "encoder.pt", map_location=device, weights_only=True)
    trainer.context_encoder.load_state_dict(state)
    trainer.context_encoder.eval()
    return trainer


def reprobe_run(run_dir: Path, device: str, probe_config: ProbeConfig) -> dict:
    """Recompute every probe endpoint for one run directory."""
    config = json.loads((run_dir / "config.json").read_text())
    # The evaluation split is a deterministic function of `eval_split_seed`,
    # which lives in the config, so this is the same images the run scored.
    _, probe_train, probe_test = build_dataloaders(config)
    trainer = load_encoder(run_dir, config, device)

    train_features = trainer.encode_all(probe_train[0])
    test_features = trainer.encode_all(probe_test[0])
    train_labels, test_labels = probe_train[1], probe_test[1]

    label_sets = (
        {name: (train_labels[name], test_labels[name]) for name in train_labels}
        if isinstance(train_labels, dict)
        else {None: (train_labels, test_labels)}
    )

    record: dict[str, float | str] = {
        "condition": config["strategy"]["name"],
        "seed": config["seed"],
        "eval_split_seed": config["data"].get("eval_split_seed", 0),
        "feature_scale": float(np.mean(np.std(train_features, axis=0))),
        "total_variance": float(np.var(test_features, axis=0).sum()),
    }

    for name, (fit_labels, eval_labels) in label_sets.items():
        suffix = f"_{name}" if name is not None else ""
        for variant, standardize in (("", True), ("_unscaled", False)):
            scores = linear_probe_scores(
                train_features, fit_labels, test_features, eval_labels,
                seed=config["seed"], standardize=standardize, config=probe_config,
            )
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "selected_C",
                        "n_iter", "converged", "underfit_train", "train_accuracy"):
                record[f"probe_{key}{variant}{suffix}"] = scores[key]
            record[f"probe_warnings{variant}{suffix}"] = scores["warnings"]
        record[f"probe_majority{suffix}"] = majority_rate(eval_labels)

    return record


def original_endpoint(run_dir: Path, key: str) -> float | None:
    """The value the run itself logged at its final checkpoint."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if not records:
        return None
    return max(records, key=lambda r: r["step"]).get(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute probe endpoints from saved encoders")
    parser.add_argument("--tag", required=True, help="tag prefix, e.g. phaseB (or a full tag)")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds; omit for a full tag")
    parser.add_argument("--condition", default=None, help="limit to one condition")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=None, help="override data.root for re-embedding")
    parser.add_argument("--compare", action="store_true", help="print old vs new side by side")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    tags = (
        [f"{args.tag}_s{s.strip()}" for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.tag]
    )

    probe_config = ProbeConfig()
    rows = []
    for tag in tags:
        tag_dir = experiments_dir / tag
        if not tag_dir.is_dir():
            print(f"  skip {tag}: no such directory")
            continue
        for run_dir in sorted(tag_dir.iterdir()):
            if args.condition and run_dir.name != args.condition:
                continue
            if not (run_dir / "encoder.pt").exists():
                print(f"  SKIP {tag}/{run_dir.name}: no encoder.pt -- this run predates "
                      "artifact saving and cannot be reprobed without retraining")
                continue

            config = json.loads((run_dir / "config.json").read_text())
            if args.data_root:
                config["data"]["root"] = args.data_root
                (run_dir / "config.json").write_text(json.dumps(config, indent=2))

            record = reprobe_run(run_dir, args.device, probe_config)
            record["tag"] = tag
            (run_dir / "metrics_reprobed.jsonl").write_text(json.dumps(record) + "\n")
            rows.append(record)
            flag = " UNDERFIT" if record.get("probe_underfit_train_unscaled") else ""
            print(f"  {tag}/{run_dir.name}: reprobed{flag}")

    if args.compare and rows:
        print()
        print("old protocol vs corrected, unstandardized probe at the final checkpoint")
        print(f"{'run':<34}{'old':>9}{'new':>9}{'delta':>9}{'C':>9}{'underfit':>10}")
        for record in rows:
            run_dir = experiments_dir / str(record["tag"]) / str(record["condition"])
            old = original_endpoint(run_dir, "probe_accuracy_unscaled")
            new = record.get("probe_accuracy_unscaled")
            if old is None or new is None:
                continue
            label = f"{record['tag']}/{record['condition']}"
            print(
                f"{label:<34}{old:>9.4f}{new:>9.4f}{new - old:>+9.4f}"
                f"{record['probe_selected_C_unscaled']:>9g}"
                f"{'yes' if record['probe_underfit_train_unscaled'] else 'no':>10}"
            )
        print()
        print("A large positive delta means the old number was an optimiser artifact,")
        print("not a property of the representation.")

    print(f"\nreprobed {len(rows)} run(s); originals untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
