"""Check a BDD100K tree before spending GPU time on it.

    python scripts/inspect_dataset.py --root ../100k

Reports the detected layout, per-split counts, how many images carry usable
scenario attributes, and the attribute distributions. Exits non-zero if a split
is missing or nothing is labelled, so it can gate a run.

Run this first on a new machine. A misdetected layout otherwise shows up as an
empty index and a training crash minutes later, or — worse — as probe splits
quietly built from a handful of images.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB, discover_split_dir, load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a BDD100K tree")
    parser.add_argument("--root", required=True, help="directory holding the splits")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="rebuild the index instead of reading .embedding_diagnostics_index_<split>.json",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    print(f"root: {root.resolve()}")
    problems = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"\n=== {split} ===")
        try:
            image_dir, layout = discover_split_dir(root, split)
        except FileNotFoundError as error:
            print(f"  MISSING: {error}")
            problems += 1
            continue
        print(f"  layout: {layout}")
        print(f"  images dir: {image_dir}")

        index = load_index(root, split, use_cache=not args.no_cache)
        fully = [codes for _, codes in index if all(value >= 0 for value in codes.values())]
        print(f"  images indexed: {len(index)}")
        print(f"  fully labelled: {len(fully)}  (probe splits draw only from these)")
        if not index:
            print("  PROBLEM: empty index")
            problems += 1
            continue
        if not fully:
            print("  PROBLEM: no image carries all three scenario attributes")
            problems += 1

        for attribute, vocabulary in ATTRIBUTE_VOCAB.items():
            counts = Counter(codes[attribute] for _, codes in index)
            shown = ", ".join(
                f"{vocabulary[code]}={counts[code]}"
                for code in sorted(c for c in counts if c >= 0)
            )
            print(f"  {attribute:<10} unusable={counts.get(-1, 0):<6} {shown}")

        sample_path, sample_codes = index[0]
        print(f"  example: {sample_path.name} -> {sample_codes}")
        if layout == "per_image":
            sidecar = sample_path.with_suffix(".json")
            if sidecar.is_file():
                raw = json.loads(sidecar.read_text())
                keys = sorted(raw) if isinstance(raw, dict) else f"list[{len(raw)}]"
                print(f"  sidecar top-level keys: {keys}")

    if problems:
        print(f"\n{problems} problem(s). Fix these before training.")
        return 1
    print("\nOK — point data.root at this directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
