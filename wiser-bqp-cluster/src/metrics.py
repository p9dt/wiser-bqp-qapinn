"""Evaluation metrics shared across experiments."""
from __future__ import annotations

import torch

from .pdes.base import PDE, Model


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative L2 error ||pred - target|| / ||target||."""
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target)
    return (num / den).item()


def max_abs_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().max().item()


@torch.no_grad()
def evaluate_on_grid(
    model: Model, pde: PDE, nx: int = 256, nt: int = 100
) -> dict[str, float]:
    """Relative-L2 / max-error of ``model`` vs the PDE reference on a grid."""
    coords, _, _ = pde.grid(nx, nt)
    pred = model(coords)
    ref = pde.reference(coords)
    return {
        "rel_l2": relative_l2(pred, ref),
        "max_abs": max_abs_error(pred, ref),
    }


def pde_residual_error(model: Model, pde: PDE, n: int = 5000,
                       generator: torch.Generator | None = None) -> float:
    """Root-mean-square PDE residual at random interior collocation points."""
    coords = pde.sample_collocation(n, generator=generator)
    res = pde.residual(model, coords)
    return res.detach().pow(2).mean().sqrt().item()
