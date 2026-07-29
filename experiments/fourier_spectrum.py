"""Headline explainability figure: how the encoding sets the quantum layer's
Fourier spectrum (ref [4]).

Sweeps a family of QuantumConfigs, recovers each one's empirical spectrum along
the x-feature, and plots magnitude vs harmonic with the theoretical bandwidth K
marked. Demonstrates that re-uploading and qubit count dial the representable
bandwidth, independent of training.

    python -m experiments.fourier_spectrum
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.models.quantum_layer import QuantumConfig, QuantumLayer  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402
from src.xai.fourier import (  # noqa: E402
    effective_spectrum,
    empirical_spectrum,
    max_frequency,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "fourier"

CONFIGS = [
    ("angle, 1 layer", QuantumConfig(n_qubits=4, n_layers=1, in_dim=2,
                                     encoding="angle", entanglement="ring")),
    ("reupload, 2 layers", QuantumConfig(n_qubits=4, n_layers=2, in_dim=2,
                                         encoding="reupload", entanglement="ring")),
    ("reupload, 3 layers", QuantumConfig(n_qubits=4, n_layers=3, in_dim=2,
                                         encoding="reupload", entanglement="ring")),
    ("reupload, 6 qubits, 2 layers", QuantumConfig(n_qubits=6, n_layers=2, in_dim=2,
                                                   encoding="reupload", entanglement="ring")),
]


def main() -> None:
    set_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4), sharey=True)
    for ax, (label, cfg) in zip(axes, CONFIGS):
        layer = QuantumLayer(cfg)
        K = max_frequency(cfg, feature_index=0)
        spec = empirical_spectrum(layer, feature_index=0, in_dim=2,
                                  base_freq=cfg.scaling_init, n_samples=256)
        mag = spec.magnitude.max(dim=1).values  # channel-max per harmonic
        freqs = spec.freqs

        show = freqs <= K + 4
        ax.stem(freqs[show].numpy(), mag[show].numpy(), basefmt=" ")
        ax.axvline(K + 0.5, color="crimson", ls="--", label=f"theory K={K}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("harmonic n")
        ax.legend(fontsize=8)
        eff = effective_spectrum(spec)
        print(f"{label:30s} theory K={K:2d}  empirical spectrum={eff}")
    axes[0].set_ylabel("|Fourier coefficient|")
    fig.suptitle("Quantum-layer Fourier spectrum vs. data encoding (feature x)")
    fig.tight_layout()
    out = RESULTS / "spectrum_vs_encoding.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
