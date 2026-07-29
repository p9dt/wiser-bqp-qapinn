"""Aggregate the Heat K-sweep: 18 summaries -> CSV + the headline elbow figure.

    python -m experiments.aggregate_sweep

Reads results/sweeps/heat_ksweep/K*/summary.json, writes:
  - ksweep.csv            one row per run (K, seed, n_params, rel_l2, ...)
  - ksweep_elbow.png      rel-L2 vs K, mean +/- std over seeds, with the two
                          reference lines the story rests on:
                            * Parseval floor 0.156  (error if mode-4 unreachable)
                            * measured K=4 baseline  1.2e-2
The prediction (PROJECT_LOG §6): error sits near the floor below K=4, drops
sharply at K=4, and may tick up beyond (capacity). A cliff landing on K=4 is the
causal signature of the bandwidth mechanism.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sweeps" / "heat_ksweep"
PARSEVAL_FLOOR = 0.156   # rel-L2 lower bound if wavenumber 4 is unreachable (PROJECT_LOG §6)
K4_BASELINE = 0.012      # measured rel-L2 at K=4 (heat_qapinn_q4)

FIELDS = ["run_name", "K", "seed", "n_qubits", "encoding", "n_layers",
          "n_params", "rel_l2", "max_abs", "process_seconds", "train_seconds"]


def load_rows(root: Path) -> list[dict]:
    rows = []
    for summ in sorted(root.glob("K*/summary.json")):
        d = json.loads(summ.read_text())
        rows.append({k: d.get(k) for k in FIELDS})
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def elbow_figure(rows: list[dict], out: Path) -> dict:
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["rel_l2"] is not None:
            by_k[r["K"]].append(r["rel_l2"])
    ks = sorted(by_k)
    means = [mean(by_k[k]) for k in ks]
    stds = [pstdev(by_k[k]) if len(by_k[k]) > 1 else 0.0 for k in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, lw=2, label="QAPINN rel-L2")
    ax.axhline(PARSEVAL_FLOOR, ls="--", color="crimson",
               label=f"Parseval floor ({PARSEVAL_FLOOR:.3f})")
    ax.axhline(K4_BASELINE, ls=":", color="gray",
               label=f"K=4 baseline ({K4_BASELINE:.3f})")
    ax.axvline(4, color="k", alpha=0.25)
    ax.set_yscale("log")
    ax.set_xlabel("quantum-layer bandwidth K")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Heat K-sweep: error vs quantum bandwidth (mean ± std, 3 seeds)")
    ax.set_xticks(ks)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return {k: (m, s) for k, m, s in zip(ks, means, stds)}


def main() -> None:
    rows = load_rows(RESULTS)
    if not rows:
        raise SystemExit(f"no summaries under {RESULTS} yet")
    write_csv(rows, RESULTS / "ksweep.csv")
    stats = elbow_figure(rows, RESULTS / "ksweep_elbow.png")

    print(f"{len(rows)} runs aggregated -> {RESULTS/'ksweep.csv'}")
    print(f"{'K':>3} {'mean rel-L2':>12} {'std':>10}  n")
    by_k: dict[int, int] = defaultdict(int)
    for r in rows:
        by_k[r["K"]] += 1
    for k in sorted(stats):
        m, s = stats[k]
        print(f"{k:>3} {m:>12.4e} {s:>10.2e}  {by_k[k]}")


if __name__ == "__main__":
    main()
