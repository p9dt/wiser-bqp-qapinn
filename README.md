# QAPINN — Explainability of the Quantum Layer in Physics-Informed Neural Networks

WISER × BQP Global Quantum+AI 2026 Challenge.

This README is written so that someone who has never seen this repository — a teammate,
a grader, an interviewer — can read it top to bottom and understand what the project does,
why it's built this way, how to run every piece of it, and where it currently stands.
If you only read one section, read **§1** and **§4**; they explain the idea the rest of
the repo is built to test.

---

## 1. The big picture

A **Physics-Informed Neural Network (PINN)** is a neural network trained not by fitting
labeled data, but by penalizing how badly it *violates a differential equation*. You feed
it random points in space and time, evaluate the equation's residual through the network
using automatic differentiation, and push that residual to zero — plus a small amount of
supervised loss to pin down the initial and boundary conditions. The network ends up
approximating the PDE's solution without ever being shown the true solution directly.

This project asks a narrower, more mechanistic question: **what happens if you take a
standard PINN and replace only its very first layer with a small quantum circuit** — a
Variational Quantum Circuit (VQC) — **instead of a classical `Linear` layer?** Call the
result a **Quantum-Assisted PINN (QAPINN)**. Every other layer stays classical and
identical to the plain PINN, so any difference in behavior can be attributed to that one
swap.

The project doesn't just measure *whether* this changes accuracy — it tries to explain
**when, why, and how**, and to turn that explanation into a recipe: given a new PDE, how
big should the quantum circuit be, and what should its dials be set to?

### Why this is answerable at all: the Fourier view of a VQC

The theoretical anchor is a result from Schuld et al. (*Effect of data encoding on the
expressive power of variational quantum-machine-learning models*, Phys. Rev. A): if a
quantum circuit encodes an input feature `z` through Pauli rotation gates like `RY(s·z)`,
then the circuit's output, as a function of `z`, is exactly a **truncated Fourier series**:

```
f(z) = sum_{n=-K..K}  c_n * exp(i * n * s * z)
```

Two things are worth separating here. The *set of frequencies* `{-K, ..., K}` that the
circuit can possibly produce is fixed the moment you choose the circuit's structure — how
many qubits encode this feature, and how many times the feature is re-uploaded (fed into
the circuit again in a later layer). The *coefficients* `c_n` — how much weight each of
those frequencies actually gets — are what training adjusts, via the trainable rotation
gates and the choice of measurement operator.

The practical consequence: **no amount of training can add a frequency the circuit
structure doesn't already support.** If a PDE's true solution needs frequency content
the VQC's `K` doesn't cover, the QAPINN is mathematically capped below the classical
PINN's achievable accuracy, however long you train it. If the solution's frequency
content fits comfortably inside the VQC's `K`, replacing the first layer costs you little
and may even help (see §8's actual numbers). This is the mechanism this repo is built to
demonstrate, measure, and predict from — not just observe as a black-box accuracy delta.

`src/sweeps.py` gives the exact bandwidth formula used throughout the repo:

```
K = (n_qubits / in_dim) * n_uploads,     n_uploads = n_layers if encoding == "reupload" else 1
```

For example, 4 qubits with 2 layers of `reupload` encoding on a 2D input `(x, t)` gives
`K = (4/2) * 2 = 4` — the circuit can represent frequencies `-4` through `+4` in each
input feature.

---

## 2. Why these two PDEs (Heat and Burgers), and not others

The two benchmarks are chosen to sit at opposite ends of "how much does the network need
to represent," which is exactly the axis the Fourier theory in §1 makes predictions about.

**Heat equation** (`src/pdes/heat.py`) — linear, and given here with an initial condition
that is deliberately a finite sum of sine modes (`b_k · sin(kπx)`, default modes at
`k=1` and `k=4`). Because sine modes are eigenfunctions of the Laplacian under these
boundary conditions, the *exact* solution is known in closed form — no numerical solver,
no discretization error, and critically, the solution's frequency content is exactly
known in advance (only `k=1` and `k=4` are present). This makes Heat the "clean-room"
experiment: any accuracy gap between the QAPINN and a classical PINN can be checked
directly against the circuit's predicted bandwidth `K`, with nothing else confounding
the comparison.

**Burgers' equation** (`src/pdes/burgers.py`) — nonlinear (`u·u_x` advection term), and
with the small viscosity used here (`ν = 0.01/π`) it develops a sharp shock: a steep,
broadband, high-frequency feature that is much harder to represent than Heat's smooth
decay. There's no closed-form solution, so the repo builds and caches a high-accuracy
reference by numerically solving the PDE on a fine grid with a Fourier pseudo-spectral
method (integrating-factor RK4 time-stepping, see `BurgersEquation._spectral_solve`).
Burgers is also explicitly called out in the code as "the ref [3] benchmark" — i.e. the
same problem used by the paper this project is positioned against — so results here are
directly comparable to a published baseline.

**Why not other PDEs** (wave equation, Navier–Stokes, Schrödinger, Allen–Cahn, etc.):
these two are the standard, most widely used PINN benchmarks since Raissi, Perdikaris &
Karniadakis (2019), so results are checkable against a large existing literature. More
importantly, they isolate one variable at a time — Heat has no numerical-reference noise
and a known target spectrum; Burgers adds nonlinearity and a hard, realistic feature
without changing anything else about the setup. Harder PDEs would add solver complexity
on top of an already CPU-bound quantum simulator (§10) without adding new evidence about
the *mechanism* under test.

---

## 3. Repository layout

```
src/
  pdes/          PDE definitions — domain, residual, IC/BC sampling, reference solution
    base.py        abstract PDE interface + autograd gradient helper + collocation sampling
    heat.py         1D heat equation; exact analytical solution (sum of decaying sine modes)
    burgers.py       1D viscous Burgers' equation; cached Fourier pseudo-spectral reference

  models/        the neural network architectures
    mlp.py          classical PINN (tanh / relu / sin); includes proper SIREN init for "sin"
    quantum_layer.py the VQC as a differentiable torch.nn.Module — the piece that replaces
                     a PINN's first layer. Exposes every design dial: n_qubits, n_layers
                     (re-upload depth), encoding, entanglement, measurement, trainable_scaling
    qapinn.py        wires QuantumLayer as the first layer, then the same classical tail
                     structure as MLP — the *only* structural difference vs. MLP

  training/      the training loop
    sampling.py      draws collocation (interior) + supervised (IC/BC) point sets
    losses.py        PDE-residual loss + supervised loss, combined with configurable weights
    trainer.py        Adam warm-up -> optional L-BFGS refinement; logs history, wall-clock,
                     and CPU time (wall-clock is flagged unreliable if the machine sleeps)

  xai/           explainability tooling — the "why" behind the numbers
    fourier.py       predicts a VQC's accessible frequency band from its config, and
                     empirically recovers a layer's real spectrum via a swept-input DFT
    capacity.py      Hu et al.'s spectral-complexity metric over a model's classical
                     Linear layers — tests whether QAPINN's smaller parameter count
                     actually means lower model complexity, or not
    (planned, not yet built: barren-plateau variance, loss-landscape slices, attribution)

  utils/         seeds.py (global RNG seeding), config.py (YAML loader with dict/attribute
                 access), plotting.py (solution heatmaps + training-curve figures)

  metrics.py     relative-L2 error, max-abs error, PDE-residual RMS — all evaluated on a
                 regular grid against each PDE's reference solution

  build.py       turns a config dict into a runnable PDE object + model object + TrainConfig
                 — the one place that translates YAML into Python objects

  sweeps.py      defines the 18-task Heat "K-sweep": a ladder of (n_qubits, encoding,
                 n_layers) combinations engineered to hit exact bandwidths K=1,2,3,4,5,8,
                 each at 3 random seeds

configs/        one YAML file per experiment (see §7 for the anatomy of a config)

experiments/    runnable top-level scripts (see §6 for exact commands)
  run_baseline.py     train one model from one config; saves summary/history/checkpoint/plots
  run_sweep.py        run one indexed task of the 18-task K-sweep (parallelizable via SLURM)
  aggregate_sweep.py  after all 18 sweep tasks finish: writes ksweep.csv + ksweep_elbow.png
  fourier_spectrum.py produces the headline "spectrum vs. encoding" figure

tests/          unit tests — PDE reference sanity (residual ~0, IC/BC match), and a
                parametrized proof that a VQC's empirical spectrum never exceeds its
                theoretical bandwidth K

scripts/
  paramshakti_array.sbatch   SLURM job-array script: runs all 18 K-sweep tasks in parallel

results/        all experiment outputs (gitignored) — per-run summary.json / history.json /
                model.pt / plots, the cached Burgers reference solution, and sweep aggregates

report/         technical report + slides (not yet written — see §9)
```

---

## 4. How the QAPINN differs from a classical PINN, concretely

```
MLP     :  normalize -> [Linear(2, H) -> act] -> [Linear(H, H) -> act] * (depth-1) -> Linear(H, 1)
QAPINN  :  normalize -> [QuantumLayer(2 -> Q)]  -> [Linear(Q, H) -> act] * (depth-1) -> Linear(H, 1)
```

`Q` is `n_qubits` under expectation-value readout, or `2**n_qubits` under probability
readout (see `measurement` below). Because `Q` is small (typically 3–5), the downstream
`Linear(Q, H)` shrinks compared to `Linear(2, H)`'s implicit width — this is the
parameter-reduction effect the project measures directly (§8).

The `QuantumLayer` (`src/models/quantum_layer.py`) has five design dials, each of which
is a real experimental variable in this project, not just an implementation detail:

| Dial | Options | What it controls |
|---|---|---|
| `n_qubits` | integer (3–6 kept modest here) | width of the register; more qubits touching a feature raises its bandwidth `K` |
| `n_layers` | integer | ansatz depth / number of data re-uploads |
| `encoding` | `angle` (upload data once) or `reupload` (re-upload every layer) | re-uploading is what lets a small circuit reach a richer Fourier spectrum |
| `entanglement` | `none`, `linear`, `ring`, `all_to_all` | whether per-qubit frequencies get mixed via CNOTs |
| `measurement` | `expectation` (n_qubits Pauli-Z values) or `probs` (2**n_qubits basis probabilities) | cheap low-dimensional readout vs. exponentially richer but costlier readout |
| `trainable_scaling` | bool | whether the input→angle frequency multiplier is learned, "retuning" which harmonics land where |

The circuit runs on PennyLane's `default.qubit` state-vector simulator with
`diff_method="backprop"`, so it is fully differentiable in PyTorch — including the
second derivatives a PDE residual needs (e.g. `u_xx`).

---

## 5. Setup

```bash
python -m venv .venv
# Windows PowerShell:      .venv\Scripts\Activate.ps1
# macOS / Linux:           source .venv/bin/activate
pip install -r requirements.txt
```

Requirements (`requirements.txt`): `numpy`, `scipy`, `matplotlib`, `pyyaml` (core);
`torch>=2.2` (ML); `pennylane>=0.38` (quantum ML); `pytest>=8.0` (testing).

Everything here is developed and tested **CPU-only** — see §10 for what that means for
runtimes and for GPU support.

---

## 6. Running experiments — commands you'll actually use

**Run the test suite** (fast, ~7 seconds, should always pass before you trust anything else):
```bash
pytest -q
```

**Train a single baseline** from any config in `configs/`:
```bash
python -m experiments.run_baseline --config configs/heat_pinn.yaml
python -m experiments.run_baseline --config configs/heat_qapinn.yaml
python -m experiments.run_baseline --config configs/burgers_pinn.yaml
python -m experiments.run_baseline --config configs/burgers_qapinn.yaml
```
This writes `results/<run_name>/summary.json`, `history.json`, `model.pt` (the trained
checkpoint), plus `solution.png` and `history.png`.

**Qubit-count and viscosity variants** (config-only, same command as above, just point
at a different file — see §7's table for the full list):
```bash
python -m experiments.run_baseline --config configs/burgers_qapinn_q3.yaml
python -m experiments.run_baseline --config configs/burgers_qapinn_q5.yaml
python -m experiments.run_baseline --config configs/burgers_nu0.05_q4.yaml
```

**SIREN control** — a classical network using a sine activation with correct SIREN
initialization, for testing whether a purely classical periodic layer can match the
QAPINN at equal bandwidth:
```bash
python -m experiments.run_baseline --config configs/heat_siren.yaml
python -m experiments.run_baseline --config configs/burgers_siren.yaml
```

**The Heat K-sweep** (18 tasks: bandwidths K∈{1,2,3,4,5,8} × 3 seeds — the single
highest-value remaining experiment, fully coded, not yet executed):
```bash
# run all 18 tasks serially (any machine):
for i in $(seq 0 17); do python -m experiments.run_sweep --index $i; done
# PowerShell equivalent:
#   0..17 | % { python -m experiments.run_sweep --index $_ }

# or in parallel on a SLURM cluster (see scripts/paramshakti_array.sbatch):
sbatch scripts/paramshakti_array.sbatch

# after all 18 finish, aggregate into a CSV + the headline elbow figure:
python -m experiments.aggregate_sweep
# -> results/sweeps/heat_ksweep/ksweep.csv
# -> results/sweeps/heat_ksweep/ksweep_elbow.png
```
The prediction being tested: error sits near a theoretical floor below `K=4` (the point
where the circuit still can't reach the solution's `k=4` mode), drops sharply at `K=4`,
and may tick up slightly beyond it. A clean "cliff" landing exactly at `K=4` is the
causal signature that bandwidth — not just "being quantum" — explains the accuracy
differences seen elsewhere in this project.

**Fourier-spectrum demonstration** — the headline explainability figure, showing
empirically (no training required) that qubit count and re-upload depth directly control
the quantum layer's bandwidth, exactly as §1's theory predicts:
```bash
python -m experiments.fourier_spectrum
# -> results/fourier/spectrum_vs_encoding.png
```

**Spectral-complexity ("capacity") metric** on a trained checkpoint — tests whether
QAPINN's fewer parameters actually correspond to a lower-complexity model, which is a
direct test of a claim in the literature this project is checking:
```bash
python -m src.xai.capacity configs/heat_pinn.yaml   results/heat_pinn/model.pt
python -m src.xai.capacity configs/heat_qapinn.yaml results/heat_qapinn_q4/model.pt
```
> Note: this needs a `model.pt` checkpoint to exist for the run in question. As of this
> writing, the five original baseline runs listed in §8 predate checkpoint-saving being
> added to `run_baseline.py` and don't have one yet — re-run them to get a checkpoint
> before using this on those specific runs.

---

## 7. Anatomy of a config file

Every experiment is driven entirely by a YAML file under `configs/`; nothing is
hardcoded in the training scripts. A config has three blocks:

```yaml
run_name: heat_qapinn_q4        # results land in results/<run_name>/
seed: 1234                      # global seed (src/utils/seeds.py) — everything is reproducible

pde:
  name: heat                    # "heat" or "burgers"
  alpha: 0.05                   # heat: diffusion coefficient
  modes: [[1, 1.0], [4, 0.5]]   # heat: [(wavenumber k, coefficient b_k), ...] — the exact target spectrum
  t_max: 1.0                    # time horizon

model:
  type: qapinn                  # "mlp" (classical) or "qapinn" (quantum first layer)
  hidden: 20                    # classical tail width
  depth: 4                      # classical tail depth
  activation: tanh              # "tanh" | "sin" (SIREN) | "relu"
  quantum:                      # only used when type: qapinn — see §4's dial table
    n_qubits: 4
    n_layers: 2
    encoding: reupload
    entanglement: ring
    measurement: expectation
    trainable_scaling: true

training:
  n_collocation: 2000           # interior PDE-residual points per epoch (or per resample)
  n_supervised: 300             # IC/BC points
  adam_epochs: 3000             # Adam warm-up length
  adam_lr: 0.005
  lbfgs_epochs: 0               # L-BFGS refinement steps after Adam (0 disables)
  resample_every: 0             # 0 = fixed point set for the whole Adam run
  w_pde: 1.0                    # loss weight on the PDE residual
  w_data: 1.0                   # loss weight on IC/BC supervision
  log_every: 250
  eval_every: 500
```

### Every config currently in `configs/`

| File | PDE | Model | Purpose |
|---|---|---|---|
| `heat_pinn.yaml` | heat | classical MLP | baseline |
| `heat_qapinn.yaml` | heat | QAPINN, 4 qubits | main quantum comparison |
| `heat_qapinn_probs.yaml` | heat | QAPINN, 4 qubits, `probs` readout | richer-readout variant |
| `heat_siren.yaml` *(new)* | heat | classical MLP, sine (SIREN init) | matched-bandwidth classical control |
| `burgers_pinn.yaml` | burgers | classical MLP | baseline |
| `burgers_qapinn.yaml` | burgers | QAPINN, 4 qubits | main quantum comparison |
| `burgers_qapinn_q3.yaml` *(new)* | burgers | QAPINN, 3 qubits | qubit-threshold sweep |
| `burgers_qapinn_q5.yaml` *(new)* | burgers | QAPINN, 5 qubits | ref [3]'s predicted shock threshold |
| `burgers_siren.yaml` *(new)* | burgers | classical MLP, sine (SIREN init) | matched-bandwidth classical control |
| `burgers_nu{0.00318,0.05,0.1}_q{3,4,5}.yaml` *(new, 9 files)* | burgers | QAPINN | ν × qubit-count grid — tests whether required qubits track shock width |

---

## 8. Results so far

All runs below used the fixed training recipe in their own config (see `configs/`);
`rel_l2` is relative-L2 error against each PDE's reference solution over a regular grid,
`max_abs` is the worst-point absolute error (for Burgers, this is effectively the
shock-region error).

| Run | Model | Params | rel-L2 | max-abs | Notes |
|---|---|---|---|---|---|
| `heat_pinn` | classical MLP | 1341 | 0.0013 | 0.0067 | |
| `heat_qapinn_q4` | QAPINN, 4 qubits | 985 | 0.0119 | 0.0581 | ~9x worse than classical — Heat's needed bandwidth (k=4) sits right at this circuit's limit |
| `heat_qapinn_q4_probs` | QAPINN, 4 qubits, `probs` readout | 1225 | 0.0063 | 0.0495 | closer to classical — richer readout exposes more effective spectrum |
| `burgers_pinn` | classical MLP | 1341 | 0.0756 | 0.6297 | likely under-trained vs. published results (see §10) |
| `burgers_qapinn_q4` | QAPINN, 4 qubits | 985 | 0.0695 | 0.4674 | slightly *better* than classical on both metrics, at fewer params |

**Read this table carefully, not just for the headline numbers:** on the smooth Heat
problem, adding the quantum layer clearly *hurt* accuracy (its bandwidth ceiling binds).
On the shock-forming Burgers problem, it *helped* slightly. That split is exactly what
§1's theory predicts and is the central empirical hook of the whole project — it's why
the K-sweep (§6) and the qubit/ν sweeps (§7's new configs) exist: to nail down, with a
real controlled experiment, whether that split really is caused by bandwidth matching
the target spectrum, rather than some other confound.

---

## 9. Project status

Legend: ✅ done and run · 🟡 code ready, not yet executed · 🔴 not yet built.

- ✅ Core architecture: PDEs, MLP, quantum layer, QAPINN, trainer, metrics, plotting.
- ✅ Fourier explainability module (`src/xai/fourier.py`) + demonstration figure.
- ✅ Spectral-complexity metric (`src/xai/capacity.py`).
- ✅ SIREN-correct initialization for the sine-activation classical control.
- ✅ Five baseline runs (Heat/Burgers × classical/QAPINN, plus a `probs`-readout variant) — see §8.
- ✅ Unit tests (8/8 passing): PDE reference sanity, quantum-layer spectrum bound.
- 🟡 **Heat K-sweep** (18 tasks) — fully coded (`src/sweeps.py`, `run_sweep.py`,
  `aggregate_sweep.py`, SLURM script), not yet executed. Highest-priority remaining item.
- 🟡 3-qubit / 5-qubit Burgers variants and the 9-cell ν-sweep — configs exist, not yet run.
- 🟡 SIREN control configs exist, not yet run against QAPINN for the 3-way comparison.
- 🔴 Barren-plateau, loss-landscape, and attribution XAI modules — specified, not implemented.
- 🔴 GPU support — the repo has no device handling; everything runs on CPU today.
- 🔴 `report/` (technical report + slides) — not yet started.

---

## 10. Known caveats — read before citing any number above

- **`heat_qapinn_q4`'s reported `train_seconds` (≈10.8 hours) looks like a
  machine-sleep artifact, not real compute time.** `TrainResult` records both wall-clock
  (`train_seconds`) and CPU time (`process_seconds`) precisely because wall-clock is
  unreliable if the machine sleeps mid-run — and that run is one of two (along with
  `heat_pinn`) missing a `process_seconds` value entirely, meaning it predates that
  field being added. Don't use it as a timing benchmark.
- **No `model.pt` checkpoints exist yet for the five runs in §8.** `run_baseline.py`
  does save one today, but these particular runs predate that addition. Re-run them
  before using `src/xai/capacity.py` on them.
- **The Burgers classical-PINN baseline is likely under-trained.** `burgers_pinn`'s
  rel-L2 (0.0756) is roughly 80x worse than Raissi's canonical result (9.4e-4) for a
  similar architecture, which used more L-BFGS refinement. Raise `lbfgs_epochs` in
  `configs/burgers_pinn.yaml` before making any headline PINN-vs-QAPINN accuracy claim.
- **Ref [3]'s data-encoding scheme is unspecified in its paper**, so its effective
  bandwidth `K` isn't directly known — it has to be inferred spectrally with
  `src/xai/fourier.py`, or clarified with the challenge organizers.

---

## 11. Compute notes

Developed and validated **CPU-only** (`torch` CPU build, PennyLane `default.qubit`),
which is why qubit counts are kept modest (3–6) — state-vector simulation cost grows
quickly with qubit count, and every PDE residual needs the circuit differentiated twice
(for second-order derivatives like `u_xx`). This is also the main runtime bottleneck in
the whole project: the Burgers QAPINN baseline took roughly 200x longer to train than
its classical counterpart for the same recipe, purely from re-simulating the circuit at
every optimizer step.

There is currently no GPU code path. Adding one means moving the model and sampled
points onto a CUDA device in `src/training/trainer.py`; the quantum layer follows
automatically since it runs through PennyLane's torch interface, but at 3–6 qubits a GPU
buys little there — the real win would be on the classical tail + autodiff for larger
problems. For a heavier quantum simulation specifically, `pennylane-lightning[gpu]` is
an option, but it only supports `diff_method="adjoint"`, which does **not** provide the
second-order input derivatives a PDE residual needs — so it's only usable for
forward-only analysis (e.g. large-K Fourier scans), not for QAPINN training itself.

For large sweeps, `scripts/paramshakti_array.sbatch` submits the 18-task K-sweep as a
SLURM job array so all 18 run in parallel instead of serially.

---

## 12. Reproducibility

Every experiment reads a YAML config and a global seed (`src/utils/seeds.py`, which
seeds Python's `random`, NumPy, and PyTorch, and requests deterministic cuDNN
algorithms). Configs live in `configs/`; results are written under `results/<run_name>/`
and are gitignored. Re-running the same config with the same seed on the same machine
should reproduce the same numbers up to floating-point nondeterminism.

---

## 13. AI-tool disclosure (per challenge rules)

AI coding assistants were used as a scaffolding aid — for boilerplate, refactoring, and
drafting this documentation. All mathematical derivations, design decisions, and results
are authored, understood, and defended by the team. See `report/` (once written) for the
full disclosure section.
