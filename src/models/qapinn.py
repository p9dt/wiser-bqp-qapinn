"""QAPINN: a PINN whose first hidden layer is a variational quantum circuit.

Classical PINN tail is preserved, so the *only* structural change vs. :class:`MLP`
is the first layer:

    MLP     :  norm -> [Linear(2,H) -> act] -> [Linear(H,H) -> act]*(depth-1) -> Linear(H,1)
    QAPINN  :  norm -> [QuantumLayer(2->Q)] -> [Linear(Q,H) -> act]*(depth-1)  -> Linear(H,1)

where ``Q`` = ``n_qubits`` (expectation readout) or ``2**n_qubits`` (probability
readout). Because ``Q`` is typically small (3-5), the downstream weight matrices
shrink, which is the parameter-reduction effect reported in ref [3].
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..pdes.base import PDE
from .mlp import InputNormalizer, _Sine
from .quantum_layer import QuantumConfig, QuantumLayer


class QAPINN(nn.Module):
    def __init__(
        self,
        qcfg: QuantumConfig,
        hidden: int = 20,
        depth: int = 4,
        out_dim: int = 1,
        lower: Sequence[float] | None = None,
        upper: Sequence[float] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.normalize = (
            InputNormalizer(lower, upper) if lower is not None else nn.Identity()
        )
        self.quantum = QuantumLayer(qcfg)
        act = {"tanh": nn.Tanh, "sin": _Sine, "relu": nn.ReLU}[activation]

        # classical tail: (depth-1) hidden blocks then output, matching MLP's tail
        layers: list[nn.Module] = []
        prev = qcfg.out_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(prev, hidden))
            layers.append(act())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.tail = nn.Sequential(*layers)
        self._init_tail()

    def _init_tail(self) -> None:
        for m in self.tail:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        z = self.normalize(coords)
        q = self.quantum(z)
        return self.tail(q)


def build_qapinn(cfg: dict, pde: PDE) -> QAPINN:
    """Build a QAPINN from a model-config dict (see configs/*_qapinn.yaml)."""
    q = cfg.get("quantum", {})
    qcfg = QuantumConfig(
        n_qubits=q.get("n_qubits", 4),
        n_layers=q.get("n_layers", 2),
        in_dim=pde.input_dim,
        encoding=q.get("encoding", "reupload"),
        entanglement=q.get("entanglement", "ring"),
        measurement=q.get("measurement", "expectation"),
        trainable_scaling=q.get("trainable_scaling", True),
        scaling_init=q.get("scaling_init", 1.0),
    )
    return QAPINN(
        qcfg,
        hidden=cfg.get("hidden", 20),
        depth=cfg.get("depth", 4),
        out_dim=pde.output_dim,
        lower=pde.lower.tolist(),
        upper=pde.upper.tolist(),
        activation=cfg.get("activation", "tanh"),
    )
