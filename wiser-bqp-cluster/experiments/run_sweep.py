"""Run one sweep task by index (one SLURM array element = one config).

    python -m experiments.run_sweep --index 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sweeps import run_one, tasks

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", default=str(RESULTS / "sweeps" / "heat_ksweep"))
    args = parser.parse_args()

    n = len(tasks())
    if not 0 <= args.index < n:
        raise SystemExit(f"index {args.index} out of range [0, {n})")

    summary = run_one(args.index, Path(args.out))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
