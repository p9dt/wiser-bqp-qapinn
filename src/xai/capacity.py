"""Hu et al. spectral-complexity metric over a model's Linear layers.

Computes the product-of-spectral-norms capacity bound C = prod(‖Wˡ‖₂) ·
(Σ (‖Wˡ‖_{2,1} / ‖Wˡ‖₂)^(2/3))^(3/2) over all nn.Linear layers.

The VQC layer has no weight matrix and is naturally skipped, so this measures
the classical tail only — directly comparable between PINN and QAPINN.

Usage:
    python -m src.xai.capacity configs/heat_pinn.yaml   results/heat_pinn/model.pt
    python -m src.xai.capacity configs/heat_qapinn.yaml results/heat_qapinn_q4/model.pt
"""
from __future__ import annotations

import torch
from torch import nn


def spectral_complexity(model: nn.Module) -> dict:
    Ms, Ns = [], []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach()
            s = torch.linalg.matrix_norm(W, ord=2)   # ‖W‖₂
            n21 = W.norm(p=2, dim=0).sum()            # ‖W‖_{2,1}
            Ms.append(s)
            Ns.append(n21 / s)
    if not Ms:
        return {"prod_spectral": float("nan"), "ratio_term": float("nan"), "complexity": float("nan")}
    prod = torch.stack(Ms).prod()
    ratio = (torch.stack(Ns) ** (2 / 3)).sum() ** (3 / 2)
    return {
        "n_linear_layers": len(Ms),
        "prod_spectral": prod.item(),
        "ratio_term": ratio.item(),
        "complexity": (prod * ratio).item(),
    }


if __name__ == "__main__":
    import json
    import sys

    from src.build import build_model, build_pde
    from src.utils.config import load_config

    if len(sys.argv) != 3:
        print("Usage: python -m src.xai.capacity <config.yaml> <model.pt>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    pde = build_pde(cfg["pde"])
    model = build_model(cfg["model"], pde)
    model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))
    model.eval()

    result = spectral_complexity(model)
    print(json.dumps(result, indent=2))
