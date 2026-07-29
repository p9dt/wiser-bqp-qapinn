"""Classical PINN: a plain fully-connected tanh network.

The network maps coordinates ``[x, t]`` to the scalar field ``u``. Inputs are
affinely normalized to ``[-1, 1]`` using the PDE domain bounds, which markedly
stabilizes PINN training. The architecture is deliberately simple so that the
*only* structural difference against a QAPINN is the first hidden layer.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class InputNormalizer(nn.Module):
    """Affine map from ``[lower, upper]`` to ``[-1, 1]`` per input dimension."""

    def __init__(self, lower: Sequence[float], upper: Sequence[float]) -> None:
        super().__init__()
        lo = torch.tensor(lower, dtype=torch.float32)
        hi = torch.tensor(upper, dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("span", (hi - lo))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.lo) / self.span - 1.0


class MLP(nn.Module):
    """Fully-connected PINN with tanh activations."""

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: int = 20,
        depth: int = 4,
        lower: Sequence[float] | None = None,
        upper: Sequence[float] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.normalize = (
            InputNormalizer(lower, upper) if lower is not None else nn.Identity()
        )
        act = {"tanh": nn.Tanh, "sin": _Sine, "relu": nn.ReLU}[activation]

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(act())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(coords))


class _Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
