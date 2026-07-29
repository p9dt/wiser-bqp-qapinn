"""Training loop: Adam warm-up followed by an optional LBFGS refinement.

Works unchanged for both the classical MLP PINN and the hybrid QAPINN — the
only thing that varies is the model passed in.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn

from ..metrics import evaluate_on_grid
from ..pdes.base import PDE
from .losses import LossWeights, pinn_loss
from .sampling import PointSet, draw_points


@dataclass
class TrainConfig:
    n_collocation: int = 5000
    n_supervised: int = 400
    adam_epochs: int = 5000
    adam_lr: float = 1e-3
    lbfgs_epochs: int = 0          # number of LBFGS closure steps (0 disables)
    resample_every: int = 0        # 0 => fixed point set for the whole Adam run
    w_pde: float = 1.0
    w_data: float = 1.0
    log_every: int = 500
    eval_every: int = 1000
    eval_nx: int = 128
    eval_nt: int = 100


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    train_seconds: float = 0.0       # wall clock — unreliable if the machine sleeps
    process_seconds: float = 0.0     # CPU time — use this to size compute budgets


def train(
    model: nn.Module,
    pde: PDE,
    cfg: TrainConfig,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> TrainResult:
    weights = LossWeights(pde=cfg.w_pde, data=cfg.w_data)
    points = draw_points(pde, cfg.n_collocation, cfg.n_supervised, generator)
    result = TrainResult()
    start = time.time()
    start_cpu = time.process_time()

    def record(epoch: int, logs: dict) -> None:
        entry = {"epoch": epoch, **logs}
        if cfg.eval_every and (epoch % cfg.eval_every == 0):
            entry.update(evaluate_on_grid(model, pde, cfg.eval_nx, cfg.eval_nt))
        result.history.append(entry)
        if verbose and (epoch % cfg.log_every == 0):
            msg = " | ".join(
                f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in entry.items()
            )
            print(msg)

    # -- Adam phase -------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    for epoch in range(1, cfg.adam_epochs + 1):
        if cfg.resample_every and epoch % cfg.resample_every == 0:
            points = draw_points(pde, cfg.n_collocation, cfg.n_supervised, generator)
        opt.zero_grad()
        loss, logs = pinn_loss(
            model, pde, points.collocation,
            points.sup_coords, points.sup_target, weights,
        )
        loss.backward()
        opt.step()
        if epoch % cfg.log_every == 0 or epoch == 1:
            record(epoch, logs)

    # -- LBFGS phase ------------------------------------------------------
    if cfg.lbfgs_epochs > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=cfg.lbfgs_epochs,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        state = {"logs": {}}

        def closure():
            lbfgs.zero_grad()
            loss, logs = pinn_loss(
                model, pde, points.collocation,
                points.sup_coords, points.sup_target, weights,
            )
            loss.backward()
            state["logs"] = logs
            return loss

        lbfgs.step(closure)
        record(cfg.adam_epochs + cfg.lbfgs_epochs, state["logs"])

    result.train_seconds = time.time() - start
    result.process_seconds = time.process_time() - start_cpu
    result.final = evaluate_on_grid(model, pde, cfg.eval_nx, cfg.eval_nt)
    result.final["train_seconds"] = result.train_seconds
    result.final["process_seconds"] = result.process_seconds
    return result
