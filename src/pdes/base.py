"""Base interface shared by every PDE benchmark.

A PDE knows four things:

1. its space-time ``domain`` (bounds used for collocation sampling);
2. how to compute its ``residual`` given a model that maps coords -> u, using
   automatic differentiation;
3. how to sample ``supervised`` points (Dirichlet initial + boundary conditions)
   together with their target values;
4. a ``reference`` solution used to score accuracy (analytical or precomputed).

All coordinate tensors have shape ``(N, input_dim)``. For the 1D time-dependent
problems here, columns are ``[x, t]``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch

Model = Callable[[torch.Tensor], torch.Tensor]


def gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs)/d(inputs) for a scalar-per-row output via autograd.

    ``outputs`` has shape ``(N, 1)`` and ``inputs`` shape ``(N, D)``; the result
    has shape ``(N, D)``. ``inputs`` must have ``requires_grad=True``.
    """
    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


class PDE(ABC):
    """Abstract base class for a PDE benchmark on a 1D (x) + time domain."""

    name: str = "pde"
    input_dim: int = 2  # [x, t]
    output_dim: int = 1

    #: bounds as {"x": (lo, hi), "t": (lo, hi)}; column order matches input_dim
    domain: dict[str, tuple[float, float]]

    # -- geometry helpers -------------------------------------------------
    @property
    def lower(self) -> np.ndarray:
        return np.array([self.domain["x"][0], self.domain["t"][0]], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.domain["x"][1], self.domain["t"][1]], dtype=np.float64)

    # -- core PDE contract ------------------------------------------------
    @abstractmethod
    def residual(self, model: Model, coords: torch.Tensor) -> torch.Tensor:
        """Return the PDE residual f(model) at ``coords`` (shape (N, 1))."""

    @abstractmethod
    def supervised(self, n: int, generator: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Dirichlet initial+boundary points and their target u values.

        Returns ``(coords, u_target)`` with shapes ``(n, input_dim)`` and
        ``(n, 1)``.
        """

    @abstractmethod
    def reference(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate the reference solution u at ``coords`` (shape (N, 1))."""

    # -- collocation sampling (shared) -----------------------------------
    def sample_collocation(self, n: int, generator: torch.Generator | None = None
                           ) -> torch.Tensor:
        """Uniformly sample ``n`` interior collocation points in the domain."""
        lo = torch.tensor(self.lower, dtype=torch.float32)
        hi = torch.tensor(self.upper, dtype=torch.float32)
        u = torch.rand(n, self.input_dim, generator=generator)
        return lo + (hi - lo) * u

    def grid(self, nx: int, nt: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a regular evaluation grid.

        Returns ``(coords, X, T)`` where ``coords`` is ``(nx*nt, 2)`` in row-major
        order over ``(t, x)`` and ``X``, ``T`` are ``(nt, nx)`` meshgrids.
        """
        x = torch.linspace(self.domain["x"][0], self.domain["x"][1], nx)
        t = torch.linspace(self.domain["t"][0], self.domain["t"][1], nt)
        T, X = torch.meshgrid(t, x, indexing="ij")
        coords = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        return coords, X, T
