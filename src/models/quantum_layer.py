"""Variational quantum circuit as a ``torch.nn.Module`` (the "quantum layer").

This is the component that replaces a PINN's first hidden layer. It exposes the
design dials the challenge asks us to study:

* ``n_qubits``            - width of the quantum register (3/4/5 in ref [3]);
* ``n_layers``            - ansatz depth / number of data re-uploads;
* ``encoding``            - ``"angle"`` (upload data once) or ``"reupload"``
                            (re-upload the inputs every layer -> richer Fourier
                            spectrum, cf. ref [4]);
* ``entanglement``        - ``"none" | "linear" | "ring" | "all_to_all"``;
* ``measurement``         - ``"expectation"`` (n_qubits Pauli-Z expvals) or
                            ``"probs"`` (2**n_qubits basis probabilities);
* ``trainable_scaling``   - whether the input->angle frequency factors are learned.

The circuit is deterministic (state-vector ``default.qubit``, ``diff_method``
= ``backprop``) so it is fully differentiable in PyTorch, including the second
derivatives PINN residuals need.

Fourier view (ref [4]): with data re-uploading, the map ``(x, t) -> <Z>`` is a
truncated multivariate Fourier series whose accessible frequencies are set by
the encoding (gate generators + number of re-uploads) and whose coefficients are
set by the trainable gates and the measurement operator. That is the object the
Phase-4 analysis dissects.
"""
from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml
import torch
from torch import nn


@dataclass
class QuantumConfig:
    n_qubits: int = 4
    n_layers: int = 2
    in_dim: int = 2
    encoding: str = "reupload"          # "angle" | "reupload"
    entanglement: str = "ring"          # "none" | "linear" | "ring" | "all_to_all"
    measurement: str = "expectation"    # "expectation" | "probs"
    trainable_scaling: bool = True
    scaling_init: float = 1.0

    @property
    def out_dim(self) -> int:
        return self.n_qubits if self.measurement == "expectation" else 2 ** self.n_qubits


def _entangle(entanglement: str, n_qubits: int) -> None:
    if entanglement == "none" or n_qubits < 2:
        return
    if entanglement == "linear":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    elif entanglement == "ring":
        pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    elif entanglement == "all_to_all":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:  # pragma: no cover - guarded by builder
        raise ValueError(f"Unknown entanglement: {entanglement!r}")
    for i, j in pairs:
        qml.CNOT(wires=[i, j])


def build_qnode(cfg: QuantumConfig):
    """Construct the differentiable QNode and its weight-shape spec."""
    dev = qml.device("default.qubit", wires=cfg.n_qubits)
    n_uploads = cfg.n_layers if cfg.encoding == "reupload" else 1

    def circuit(inputs, weights, scaling):
        # inputs: (..., in_dim); weights: (n_layers, n_qubits, 2); scaling: (n_uploads, n_qubits)
        for layer in range(cfg.n_layers):
            # -- data encoding (re-uploaded each layer if requested) --------
            if cfg.encoding == "reupload" or layer == 0:
                up = layer if cfg.encoding == "reupload" else 0
                for q in range(cfg.n_qubits):
                    feat = inputs[..., q % cfg.in_dim]
                    qml.RY(scaling[up, q] * feat, wires=q)
            # -- trainable single-qubit rotations ---------------------------
            for q in range(cfg.n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            # -- entanglement ----------------------------------------------
            _entangle(cfg.entanglement, cfg.n_qubits)

        if cfg.measurement == "expectation":
            return [qml.expval(qml.PauliZ(q)) for q in range(cfg.n_qubits)]
        return qml.probs(wires=range(cfg.n_qubits))

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
    weight_shapes = {
        "weights": (cfg.n_layers, cfg.n_qubits, 2),
        "scaling": (n_uploads, cfg.n_qubits),
    }
    return qnode, weight_shapes


class QuantumLayer(nn.Module):
    """VQC wrapped as a Torch layer mapping ``(B, in_dim) -> (B, out_dim)``."""

    def __init__(self, cfg: QuantumConfig) -> None:
        super().__init__()
        self.cfg = cfg
        qnode, weight_shapes = build_qnode(cfg)

        init = {
            "weights": _small_uniform,
            "scaling": (lambda t: nn.init.constant_(t, cfg.scaling_init)),
        }
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=init)
        if not cfg.trainable_scaling:
            self.qlayer.scaling.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qlayer(x)
        if out.dim() == 1:  # single (unbatched) sample
            out = out.unsqueeze(0)
        return out


def _small_uniform(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return tensor.uniform_(-0.1, 0.1)
