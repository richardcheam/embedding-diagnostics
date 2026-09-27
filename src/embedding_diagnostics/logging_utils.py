"""Append-only JSONL logging for experiment runs.

One JSON object per line keeps runs streamable, diffable, and small enough to
track in git, which is how results move from the GPU machine back to the
machine that builds the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLogger:
    """Write one JSON object per line to `<run_dir>/metrics.jsonl`."""

    def __init__(self, run_dir: Path, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        if not resume and self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
            raise FileExistsError(
                f"{self.metrics_path} already contains records. Appending a second run "
                "here would interleave two runs into one file, and the figure builders "
                "sort by step without deduplicating — the plot would silently mix them. "
                "Delete the run directory, choose a different --tag, or pass resume=True "
                "if you genuinely intend to continue this run."
            )
        self._handle = self.metrics_path.open("a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        """Append one record and flush, so a killed run keeps its history."""
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def write_config(self, config: dict[str, Any]) -> None:
        """Snapshot the resolved config next to the metrics."""
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
