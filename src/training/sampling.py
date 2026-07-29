"""Point-set sampling for PINN training.

The trainer works with a fixed pool of collocation + supervised points (needed
for the LBFGS closure and for reproducibility), optionally resampled every few
iterations during the Adam phase. These helpers centralize that draw.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..pdes.base import PDE


@dataclass
class PointSet:
    collocation: torch.Tensor
    sup_coords: torch.Tensor
    sup_target: torch.Tensor


def draw_points(
    pde: PDE,
    n_collocation: int,
    n_supervised: int,
    generator: torch.Generator | None = None,
) -> PointSet:
    """Draw a fresh pool of interior + supervised points."""
    collocation = pde.sample_collocation(n_collocation, generator=generator)
    sup_coords, sup_target = pde.supervised(n_supervised, generator=generator)
    return PointSet(collocation, sup_coords, sup_target)
