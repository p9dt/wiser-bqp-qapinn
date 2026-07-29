"""Train a single PINN/QAPINN run from a YAML config.

    python -m experiments.run_baseline --config configs/burgers_pinn.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.build import build_model, build_pde, build_train_config
from src.models.mlp import count_parameters
from src.training.trainer import train
from src.utils.config import load_config
from src.utils.plotting import plot_history, plot_solution
from src.utils.seeds import set_seed

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1234))

    pde = build_pde(cfg["pde"])
    model = build_model(cfg["model"], pde)
    train_cfg = build_train_config(cfg["training"])

    n_params = count_parameters(model)
    print(f"[{cfg['run_name']}] model={cfg['model'].get('type', 'mlp')} "
          f"params={n_params} pde={pde.name}")

    generator = torch.Generator().manual_seed(cfg.get("seed", 1234))
    result = train(model, pde, train_cfg, generator=generator)

    out_dir = RESULTS / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": cfg["run_name"],
        "pde": pde.name,
        "model_type": cfg["model"].get("type", "mlp"),
        "n_params": n_params,
        **result.final,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
    torch.save(model.state_dict(), out_dir / "model.pt")

    plot_solution(model, pde, out_dir / "solution.png")
    plot_history(result.history, out_dir / "history.png")

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
