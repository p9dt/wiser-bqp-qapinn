"""Sanity tests: the reference solution must (nearly) satisfy its own PDE, and
metric helpers must behave."""
from __future__ import annotations

import math

import torch

from src.metrics import relative_l2
from src.pdes.burgers import BurgersEquation
from src.pdes.heat import HeatEquation


def test_heat_reference_satisfies_pde():
    pde = HeatEquation()
    # analytic solution -> residual essentially zero (autodiff through closed form)
    def model(coords):
        return pde.reference(coords)

    coords = pde.sample_collocation(2000)
    res = pde.residual(model, coords)
    assert res.detach().abs().max().item() < 1e-4


def test_heat_ic_matches():
    pde = HeatEquation()
    x = torch.linspace(0, 1, 50).reshape(-1, 1)
    coords = torch.cat([x, torch.zeros_like(x)], dim=1)
    assert torch.allclose(pde.reference(coords), pde.initial_condition(x), atol=1e-6)


def test_burgers_reference_bc_and_ic():
    pde = BurgersEquation()
    # IC: u(x,0) = -sin(pi x)
    x = torch.linspace(-1, 1, 50).reshape(-1, 1)
    coords0 = torch.cat([x, torch.zeros_like(x)], dim=1)
    ref0 = pde.reference(coords0)
    assert relative_l2(ref0, -torch.sin(math.pi * x)) < 5e-3

    # BC: u(+-1, t) ~ 0
    t = torch.linspace(0, 1, 30).reshape(-1, 1)
    left = torch.cat([-torch.ones_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t), t], dim=1)
    assert pde.reference(left).abs().max().item() < 1e-3
    assert pde.reference(right).abs().max().item() < 1e-3


def test_relative_l2_zero_when_equal():
    a = torch.randn(100, 1)
    assert relative_l2(a, a) < 1e-6
