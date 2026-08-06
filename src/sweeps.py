"""Sweep definitions: enumerate (config, seed) tasks; run one by array index.

**Heat K-sweep** — the designed Phase-3 experiment (see PROJECT_LOG §6). We dial the
quantum layer's Fourier bandwidth ``K`` via (n_qubits, encoding, n_layers) and measure
how heat's rel-L2 responds. Heat's exact solution here has spectral lines *only* at
wavenumbers 1 and 4, so the prediction is an **elbow**: approximation-error floor below
K=4 (Parseval), a sharp drop at K=4, and a possible slight rise beyond (capacity, ref [1]).

Bandwidth counting (ref [4], validated in Phase 4a):
    K = (n_qubits / in_dim) * n_uploads,   n_uploads = n_layers if reupload else 1
with in_dim = 2 (x, t) for heat. The ladder below is the §9 table, verified in ``__main__``.

Everything except the bandwidth dials is held fixed (measurement=expectation for clean
Fourier channels; ring entanglement; the sweep training recipe).

PDE and training recipe load from ``configs/heat_ksweep.yaml`` so the sweep's recipe
can be diffed against the other configs instead of hiding in source.
"""
from __future__ import annotations

import json
from pathlib import Path

from .utils.config import load_config

IN_DIM = 2  # heat inputs are (x, t)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "heat_ksweep.yaml"
_CFG = load_config(CONFIG_PATH)

HEAT_PDE = _CFG["pde"]
HEAT_TRAIN = _CFG["training"]

# (K, n_qubits, encoding, n_layers) — bandwidth ladder, matches PROJECT_LOG §9.
K_LADDER = [
    (1, 2, "angle",    1),
    (2, 4, "angle",    1),
    (3, 2, "reupload", 3),
    (4, 4, "reupload", 2),
    (5, 2, "reupload", 5),
    (8, 4, "reupload", 4),
]
SEEDS = [1234, 2025, 7]


def tasks() -> list[dict]:
    """Flat list of 18 tasks (seed-major), one per SLURM array index."""
    return [
        {"K": K, "n_qubits": nq, "encoding": enc, "n_layers": L, "seed": seed}
        for seed in SEEDS
        for (K, nq, enc, L) in K_LADDER
    ]


def _model_cfg(task: dict) -> dict:
    return {
        "type": "qapinn", "hidden": 20, "depth": 4, "activation": "tanh",
        "quantum": {
            "n_qubits": task["n_qubits"], "n_layers": task["n_layers"],
            "encoding": task["encoding"], "entanglement": "ring",
            "measurement": "expectation", "trainable_scaling": True,
        },
    }


def run_one(index: int, out_root: Path) -> dict:
    """Train the task at ``index`` and write its summary + checkpoint."""
    import torch
    from .build import build_model, build_pde, build_train_config
    from .models.mlp import count_parameters
    from .training.trainer import train
    from .utils.seeds import set_seed

    task = tasks()[index]
    set_seed(task["seed"])
    pde = build_pde(HEAT_PDE)
    model = build_model(_model_cfg(task), pde)
    tcfg = build_train_config(HEAT_TRAIN)
    gen = torch.Generator().manual_seed(task["seed"])
    result = train(model, pde, tcfg, generator=gen, verbose=False)

    run_name = f"K{task['K']}_seed{task['seed']}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run_name": run_name, **task,
               "n_params": count_parameters(model), **result.final}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")
    return summary


if __name__ == "__main__":  # self-check: enumeration + K formula + recipe pinning
    ts = tasks()
    assert len(ts) == 18, len(ts)
    for t in ts:
        up = t["n_layers"] if t["encoding"] == "reupload" else 1
        assert t["K"] == (t["n_qubits"] // IN_DIM) * up, t
    assert len({t["K"] for t in ts}) == 6

    # The two recipe values that made the sweep incomparable to the single-run heat
    # baselines. Editing them in YAML invalidates results/sweeps/heat_ksweep/.
    assert HEAT_TRAIN["n_collocation"] == 2000 and HEAT_TRAIN["lbfgs_epochs"] == 0, HEAT_TRAIN

    print(f"{len(ts)} tasks OK; K ladder consistent: {sorted({t['K'] for t in ts})}")
