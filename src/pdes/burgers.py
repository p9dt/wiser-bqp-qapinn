"""1D viscous Burgers equation (the ref [3] benchmark).

    u_t + u*u_x = nu * u_xx,   x in [-1, 1],  t in [0, 1]
    u(-1, t) = u(1, t) = 0
    u(x, 0) = -sin(pi*x),      nu = 0.01/pi

The initial condition is odd and periodic with period 2, so the exact solution
stays odd and respects the zero Dirichlet BCs automatically. That lets us build
a high-accuracy **reference** with a Fourier pseudo-spectral solver using an
integrating-factor RK4 time step (stable for the stiff diffusion term). The
reference is computed once and cached to ``results/reference/``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from .base import PDE, Model, gradient

_CACHE = Path(__file__).resolve().parents[2] / "results" / "reference"


class BurgersEquation(PDE):
    name = "burgers"

    def __init__(self, nu: float = 0.01 / math.pi, t_max: float = 1.0) -> None:
        self.nu = float(nu)
        self.domain = {"x": (-1.0, 1.0), "t": (0.0, float(t_max))}
        self._interp: RegularGridInterpolator | None = None

    # -- initial condition -----------------------------------------------
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(math.pi * x)

    # -- PDE residual -----------------------------------------------------
    def residual(self, model: Model, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clone().requires_grad_(True)
        u = model(coords)
        grads = gradient(u, coords)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = gradient(u_x, coords)[:, 0:1]
        return u_t + u * u_x - self.nu * u_xx

    # -- supervised (IC + Dirichlet BC) ----------------------------------
    def supervised(self, n: int, generator: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = self.domain["x"]
        t0, t1 = self.domain["t"]

        n_ic = n // 2
        n_bc = n - n_ic

        x_ic = x0 + (x1 - x0) * torch.rand(n_ic, 1, generator=generator)
        t_ic = torch.full((n_ic, 1), t0)
        coords_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic = self.initial_condition(x_ic)

        t_bc = t0 + (t1 - t0) * torch.rand(n_bc, 1, generator=generator)
        left = torch.rand(n_bc, 1, generator=generator) < 0.5
        x_bc = torch.where(left, torch.full_like(t_bc, x0), torch.full_like(t_bc, x1))
        coords_bc = torch.cat([x_bc, t_bc], dim=1)
        u_bc = torch.zeros(n_bc, 1)

        coords = torch.cat([coords_ic, coords_bc], dim=0)
        target = torch.cat([u_ic, u_bc], dim=0)
        return coords, target

    # -- reference solution ----------------------------------------------
    def reference(self, coords: torch.Tensor) -> torch.Tensor:
        interp = self._get_interpolator()
        pts = coords.detach().cpu().numpy()
        # interpolator expects (t, x) order
        query = np.stack([pts[:, 1], pts[:, 0]], axis=1)
        vals = interp(query)
        return torch.tensor(vals, dtype=torch.float32).reshape(-1, 1)

    def _get_interpolator(self) -> RegularGridInterpolator:
        if self._interp is not None:
            return self._interp
        t_grid, x_grid, U = self._load_or_solve()
        # U has shape (n_t, n_x); RegularGridInterpolator over (t, x)
        self._interp = RegularGridInterpolator(
            (t_grid, x_grid), U, method="linear", bounds_error=False, fill_value=None
        )
        return self._interp

    def _cache_path(self) -> Path:
        tag = f"burgers_nu{self.nu:.6f}_tmax{self.domain['t'][1]:.2f}.npz"
        return _CACHE / tag

    def _load_or_solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self._cache_path()
        if path.exists():
            data = np.load(path)
            return data["t"], data["x"], data["U"]
        t_grid, x_grid, U = self._spectral_solve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=t_grid, x=x_grid, U=U)
        return t_grid, x_grid, U

    # -- Fourier pseudo-spectral reference solver ------------------------
    def _spectral_solve(
        self, N: int = 256, dt: float = 1e-4, n_save: int = 101
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrating-factor RK4 on the periodic domain [-1, 1)."""
        nu = self.nu
        t_max = self.domain["t"][1]
        L = 2.0

        x = -1.0 + L * np.arange(N) / N            # periodic grid, excludes x=+1
        u = -np.sin(math.pi * x)

        # integer wavenumbers scaled by fundamental 2*pi/L = pi
        k = (2.0 * math.pi / L) * np.fft.fftfreq(N, d=1.0 / N)
        ik = 1j * k
        k2 = k * k

        # 2/3 dealiasing mask
        cutoff = (N // 2) * 2 // 3
        dealias = np.abs(np.fft.fftfreq(N, d=1.0 / N)) <= cutoff

        def nonlinear(u_hat: np.ndarray) -> np.ndarray:
            u_real = np.real(np.fft.ifft(u_hat))
            prod = np.fft.fft(u_real * u_real)
            prod *= dealias
            return -0.5 * ik * prod          # -(0.5 u^2)_x

        Eh = np.exp(-nu * k2 * dt / 2.0)
        E1 = np.exp(-nu * k2 * dt)

        n_steps = int(round(t_max / dt))
        save_every = max(1, n_steps // (n_save - 1))
        t_saves = [0.0]
        U_saves = [u.copy()]

        u_hat = np.fft.fft(u)
        for step in range(1, n_steps + 1):
            k1 = nonlinear(u_hat)
            k2n = nonlinear(Eh * (u_hat + 0.5 * dt * k1))
            k3 = nonlinear(Eh * u_hat + 0.5 * dt * k2n)
            k4 = nonlinear(E1 * u_hat + dt * Eh * k3)
            u_hat = E1 * u_hat + (dt / 6.0) * (E1 * k1 + 2.0 * Eh * (k2n + k3) + k4)

            if step % save_every == 0 or step == n_steps:
                t_saves.append(step * dt)
                U_saves.append(np.real(np.fft.ifft(u_hat)))

        # append x = +1 (equals x = -1 by periodicity; BC value ~ 0) for interpolation
        x_full = np.concatenate([x, [1.0]])
        U = np.stack(U_saves, axis=0)                  # (n_t, N)
        U_full = np.concatenate([U, U[:, 0:1]], axis=1)  # (n_t, N+1)
        t_grid = np.array(t_saves)
        return t_grid, x_full, U_full
