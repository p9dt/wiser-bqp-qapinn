"""Fourier-spectrum analysis of the quantum layer (ref [4], Schuld et al.).

Central fact: a variational quantum model that encodes a scalar feature ``z`` via
Pauli rotations produces an output that is a **truncated Fourier series**

    f(z) = sum_{n=-K..K} c_n * exp(i n s z),

where the accessible frequency set ``{-K, ..., K}`` (the *spectrum*) is fixed by
the data-encoding — specifically ``K`` = number of times the feature is uploaded
(gates that use it), and ``s`` is the per-gate input-scaling — while the Fourier
coefficients ``c_n`` are set by the trainable gates and the measurement operator.

This module lets us (a) predict ``K`` from a :class:`QuantumConfig`, and
(b) empirically recover the spectrum of any layer/model by sweeping one input over
a full period and taking a DFT. Agreement between the two is the quantitative
"why" behind the quantum layer: it swaps the classical activation basis for a
band-limited Fourier basis whose bandwidth we can dial with qubits / re-uploads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from ..models.quantum_layer import QuantumConfig

Fn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------
# Theoretical spectrum from the encoding
# --------------------------------------------------------------------------
def max_frequency(cfg: QuantumConfig, feature_index: int) -> int:
    """Predicted spectrum bound K for ``feature_index`` (uniform scaling).

    Each encoding gate ``RY(s * z)`` has a Pauli generator with eigenvalue gap 1,
    contributing +/-1 to the integer spectrum. Summed over all gates that upload
    this feature, the reachable integer frequencies are ``{-K, ..., K}`` with K
    equal to the number of such gates.
    """
    n_upload_layers = cfg.n_layers if cfg.encoding == "reupload" else 1
    qubits_for_feature = sum(
        1 for q in range(cfg.n_qubits) if q % cfg.in_dim == feature_index
    )
    return n_upload_layers * qubits_for_feature


def theoretical_frequencies(cfg: QuantumConfig, feature_index: int) -> list[int]:
    K = max_frequency(cfg, feature_index)
    return list(range(-K, K + 1))


# --------------------------------------------------------------------------
# Empirical spectrum by DFT over one full period
# --------------------------------------------------------------------------
@dataclass
class Spectrum:
    freqs: torch.Tensor      # (F,) integer harmonic indices n
    magnitude: torch.Tensor  # (F, C) |c_n| per output channel
    coeffs: torch.Tensor     # (F, C) complex c_n
    z: torch.Tensor          # (N,) sweep locations
    values: torch.Tensor     # (N, C) raw f(z) samples


@torch.no_grad()
def empirical_spectrum(
    fn: Fn,
    feature_index: int,
    in_dim: int = 2,
    base_freq: float = 1.0,
    fixed: float = 0.0,
    n_samples: int = 512,
) -> Spectrum:
    """Recover the Fourier spectrum of ``fn`` along one input feature.

    ``fn`` maps ``(N, in_dim)`` angle inputs to ``(N, C)`` outputs. The feature is
    swept over exactly one fundamental period ``2*pi / base_freq`` so integer
    harmonic ``n`` lands cleanly in DFT bin ``n``.
    """
    period = 2.0 * math.pi / base_freq
    z = torch.linspace(0.0, period, n_samples + 1)[:-1]  # drop endpoint (periodic)
    coords = torch.full((n_samples, in_dim), float(fixed))
    coords[:, feature_index] = z
    values = fn(coords)
    if values.dim() == 1:
        values = values.unsqueeze(1)

    coeffs = torch.fft.rfft(values, dim=0) / n_samples   # (F, C)
    magnitude = coeffs.abs()
    freqs = torch.arange(coeffs.shape[0])
    return Spectrum(freqs, magnitude, coeffs, z, values)


def spectral_bandwidth(spec: Spectrum, threshold: float = 1e-3) -> int:
    """Highest harmonic whose (channel-max) magnitude exceeds ``threshold``."""
    per_freq = spec.magnitude.max(dim=1).values
    significant = torch.nonzero(per_freq > threshold).flatten()
    return int(significant.max().item()) if significant.numel() else 0


def effective_spectrum(spec: Spectrum, threshold_ratio: float = 1e-2) -> list[int]:
    """Integer frequencies carrying non-negligible weight (relative to the peak)."""
    per_freq = spec.magnitude.max(dim=1).values
    peak = per_freq.max().item()
    if peak == 0:
        return []
    keep = torch.nonzero(per_freq > threshold_ratio * peak).flatten()
    return [int(k) for k in keep.tolist()]
