"""PINN loss = PDE-residual loss + supervised (IC/BC) loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..pdes.base import PDE, Model


@dataclass
class LossWeights:
    pde: float = 1.0
    data: float = 1.0


def pinn_loss(
    model: Model,
    pde: PDE,
    collocation: torch.Tensor,
    sup_coords: torch.Tensor,
    sup_target: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the total loss and a dict of its scalar components.

    Args:
        collocation: interior points for the PDE residual.
        sup_coords / sup_target: Dirichlet IC/BC points and their target u.
    """
    residual = pde.residual(model, collocation)
    loss_pde = residual.pow(2).mean()

    u_pred = model(sup_coords)
    loss_data = (u_pred - sup_target).pow(2).mean()

    total = weights.pde * loss_pde + weights.data * loss_data
    logs = {
        "loss": total.item(),
        "loss_pde": loss_pde.item(),
        "loss_data": loss_data.item(),
    }
    return total, logs
