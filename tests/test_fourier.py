"""The empirical Fourier spectrum of the quantum layer must be band-limited at
exactly the frequency the encoding predicts (ref [4])."""
from __future__ import annotations

import pytest
import torch

from src.models.quantum_layer import QuantumConfig, QuantumLayer
from src.xai.fourier import (
    effective_spectrum,
    empirical_spectrum,
    max_frequency,
)


@pytest.mark.parametrize(
    "n_qubits,n_layers,in_dim,encoding,expected_K",
    [
        (1, 1, 1, "angle", 1),
        (1, 3, 1, "reupload", 3),
        (2, 1, 1, "angle", 2),
        (4, 2, 2, "reupload", 4),  # feature 0 lives on qubits {0,2} -> 2 * 2 layers
    ],
)
def test_spectrum_bandlimited_at_theoretical_K(
    n_qubits, n_layers, in_dim, encoding, expected_K
):
    torch.manual_seed(0)
    cfg = QuantumConfig(
        n_qubits=n_qubits, n_layers=n_layers, in_dim=in_dim,
        encoding=encoding, entanglement="linear",
    )
    assert max_frequency(cfg, 0) == expected_K

    layer = QuantumLayer(cfg)
    spec = empirical_spectrum(
        layer, feature_index=0, in_dim=in_dim,
        base_freq=cfg.scaling_init, n_samples=256,
    )
    eff = effective_spectrum(spec, threshold_ratio=1e-2)
    # no spectral weight should appear above the predicted bandwidth ...
    assert max(eff) <= expected_K
    # ... and the top frequency should actually be reachable (present)
    assert max(eff) == expected_K
