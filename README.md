# QAPINN — Explainability of the Quantum Layer in Physics-Informed Neural Networks

WISER × BQP Global Quantum+AI 2026 Challenge.

**Goal.** Explain *when, why, and how* replacing a PINN's first hidden layer with a
Variational Quantum Circuit (VQC) — producing a **Quantum-Assisted PINN (QAPINN)** — changes
the learning dynamics of physics-informed neural networks, backed by both experiments and
mathematics, and distill a methodology for designing a problem-specific quantum circuit.

The mathematical spine is Schuld et al., *Effect of data encoding on the expressive power of
variational quantum-machine-learning models* (Phys. Rev. A) — a VQC output is a **truncated
Fourier series** whose accessible frequencies are set by the data encoding and whose
coefficients are set by the trainable gates + measurement operator.

## Layout

```
src/
  pdes/       PDE definitions: residual, IC/BC, reference solution (burgers, heat)
  models/     mlp (classical PINN), quantum_layer (VQC), qapinn (hybrid)
  training/   sampling, losses (PDE residual + IC/BC), trainer (Adam -> LBFGS)
  xai/        fourier, barren, landscape, neuron_analysis, attribution
  utils/      seeds, config loader, plotting
  metrics.py  rel-L2, PDE residual, param count, timing
configs/      YAML experiment configs
experiments/  runnable study scripts
tests/        unit tests (residual sanity, metric helpers)
results/      figures, logs, checkpoints (gitignored)
report/       technical report + slides
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

```bash
# Classical PINN baseline (Burgers or Heat)
python -m experiments.run_baseline --config configs/burgers_pinn.yaml
python -m experiments.run_baseline --config configs/heat_pinn.yaml

# Run unit tests
pytest -q
```

## Reproducibility

Every experiment reads a YAML config and a global seed (`src/utils/seeds.py`).
Configs live in `configs/`; results are written under `results/<run_name>/`.

## Compute

Developed CPU-only (torch CPU build, PennyLane `default.qubit`). Qubit counts kept modest
(3–6) accordingly.

> **AI-tool disclosure (per challenge rules).** AI coding assistants were used as a
> scaffolding aid. All mathematical derivations, design decisions, and results are authored,
> understood, and defended by the team. See `report/` for the disclosure section.
