"""1D heat equation with an exact analytical reference solution.

    u_t = alpha * u_xx,     x in [0, 1],  t in [0, T]
    u(0, t) = u(1, t) = 0                 (Dirichlet)
    u(x, 0) = sum_k b_k sin(k*pi*x)        (initial condition)

Because the sine modes are eigenfunctions of the Laplacian with these BCs, the
exact solution is the term-by-term decayed series

    u(x, t) = sum_k b_k * exp(-alpha * (k*pi)^2 * t) * sin(k*pi*x).

The multi-mode initial condition gives the model non-trivial frequency content,
which ties directly into the Fourier-expressivity analysis (Phase 4).
"""
from __future__ import annotations

import math

import torch

from .base import PDE, Model, gradient


class HeatEquation(PDE):
    name = "heat"

    def __init__(
        self,
        alpha: float = 0.05,
        modes: list[tuple[int, float]] | None = None,
        t_max: float = 1.0,
    ) -> None:
        # (wavenumber k, coefficient b_k)
        self.modes = modes if modes is not None else [(1, 1.0), (4, 0.5)]
        self.alpha = float(alpha)
        self.domain = {"x": (0.0, 1.0), "t": (0.0, float(t_max))}

    # -- exact solution & IC ---------------------------------------------
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for k, b in self.modes:
            out = out + b * torch.sin(k * math.pi * x)
        return out

    def reference(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        t = coords[:, 1:2]
        out = torch.zeros_like(x)
        for k, b in self.modes:
            decay = torch.exp(-self.alpha * (k * math.pi) ** 2 * t)
            out = out + b * decay * torch.sin(k * math.pi * x)
        return out

    # -- PDE residual -----------------------------------------------------
    def residual(self, model: Model, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().requires_grad_(True)
        u = model(coords)
        grads = gradient(u, coords)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = gradient(u_x, coords)[:, 0:1]
        return u_t - self.alpha * u_xx

    # -- supervised (IC + Dirichlet BC) ----------------------------------
    def supervised(self, n: int, generator: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self.domain["x"]
        t0, t1 = self.domain["t"]

        n_ic = n // 2
        n_bc = n - n_ic  # split across the two boundaries

        # initial condition: t = t0, x uniform
        x_ic = x0 + (x1 - x0) * torch.rand(n_ic, 1, generator=generator)
        t_ic = torch.full((n_ic, 1), t0)
        coords_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic = self.initial_condition(x_ic)

        # boundaries: x in {x0, x1}, t uniform ; Dirichlet value 0
        t_bc = t0 + (t1 - t0) * torch.rand(n_bc, 1, generator=generator)
        left = torch.rand(n_bc, 1, generator=generator) < 0.5
        x_bc = torch.where(left, torch.full_like(t_bc, x0), torch.full_like(t_bc, x1))
        coords_bc = torch.cat([x_bc, t_bc], dim=1)
        u_bc = torch.zeros(n_bc, 1)

        coords = torch.cat([coords_ic, coords_bc], dim=0)
        target = torch.cat([u_ic, u_bc], dim=0)
        return coords, target
