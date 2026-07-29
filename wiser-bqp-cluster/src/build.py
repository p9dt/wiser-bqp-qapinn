"""Factory functions that turn a config dict into PDE / model objects."""
from __future__ import annotations

from torch import nn

from .pdes.base import PDE
from .pdes.burgers import BurgersEquation
from .pdes.heat import HeatEquation
from .models.mlp import MLP
from .training.trainer import TrainConfig


def build_pde(cfg: dict) -> PDE:
    name = cfg["name"].lower()
    if name == "burgers":
        kwargs = {k: cfg[k] for k in ("nu", "t_max") if k in cfg}
        return BurgersEquation(**kwargs)
    if name == "heat":
        kwargs = {}
        if "alpha" in cfg:
            kwargs["alpha"] = cfg["alpha"]
        if "t_max" in cfg:
            kwargs["t_max"] = cfg["t_max"]
        if "modes" in cfg:
            kwargs["modes"] = [tuple(m) for m in cfg["modes"]]
        return HeatEquation(**kwargs)
    raise ValueError(f"Unknown PDE: {name!r}")


def build_model(cfg: dict, pde: PDE) -> nn.Module:
    mtype = cfg.get("type", "mlp").lower()
    if mtype == "mlp":
        return MLP(
            in_dim=pde.input_dim,
            out_dim=pde.output_dim,
            hidden=cfg.get("hidden", 20),
            depth=cfg.get("depth", 4),
            lower=pde.lower.tolist(),
            upper=pde.upper.tolist(),
            activation=cfg.get("activation", "tanh"),
        )
    if mtype == "qapinn":
        # imported lazily so the classical baseline has no PennyLane dependency
        from .models.qapinn import build_qapinn
        return build_qapinn(cfg, pde)
    raise ValueError(f"Unknown model type: {mtype!r}")


def build_train_config(cfg: dict) -> TrainConfig:
    known = TrainConfig.__dataclass_fields__.keys()
    return TrainConfig(**{k: v for k, v in cfg.items() if k in known})
