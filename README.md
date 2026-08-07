# QAPINN: Explainability of the Quantum Layer in Physics-Informed Neural Networks

WISER × BQP Global Quantum+AI 2026 Challenge.

Team: 
Shravan Kumar Sharma - ssharma5@nd.edu  — University of Notre Dame — United States

Mayank Sharma  — ep23bt009@iitdh.ac.in — Indian Institute of Technology Dharwad — India 

Satyabrat Sahu — satyabratsahu71@gmail.com — Guru Gobind Singh Indraprastha University, Delhi — India



This README is written so that someone who has never seen this repository (a teammate,
a grader, an interviewer) can read it top to bottom and understand what the project does,
why it is built this way, how to run every piece of it, and where it currently stands.
If you only read two sections, read **§1** and **§8**. Section 1 explains the idea the
rest of the repo is built to test, and section 8 tells you whether that idea held up.

---

## 1. The big picture

A **Physics-Informed Neural Network (PINN)** is a neural network trained without labeled
data. Instead of fitting example inputs to example outputs, you feed it random points in
space and time, evaluate a differential equation's residual through the network using
automatic differentiation, and push that residual toward zero. You add a small supervised
loss on top, just enough to pin down the initial and boundary conditions. The network ends
up approximating the PDE's solution without ever being shown the true solution directly.

This project asks a narrower, more mechanistic question: what happens if you take a
standard PINN and replace only its very first layer with a small quantum circuit, a
Variational Quantum Circuit (VQC), instead of a classical `Linear` layer? We call the
result a **Quantum-Assisted PINN (QAPINN)**. Every other layer stays classical and
identical to the plain PINN, so any difference in behavior can be attributed to that one
swap and nothing else.

We did not just want to measure whether this changes accuracy. We wanted to explain when,
why, and how, and turn that explanation into something usable: given a new PDE, how big
should the quantum circuit be, and what should its dials be set to? That question has an
answer now, and it is not the one we expected going in. Section 8 has the full story.

### Why this is answerable at all: the Fourier view of a VQC

The theoretical anchor is a result from Schuld and colleagues (*Effect of data encoding
on the expressive power of variational quantum-machine-learning models*, Phys. Rev. A,
2021). If a quantum circuit encodes an input feature `z` through Pauli rotation gates
like `RY(s·z)`, the circuit's output, as a function of `z`, is exactly a **truncated
Fourier series**:

```
f(z) = sum_{n=-K..K}  c_n * exp(i * n * s * z)
```

Two things are worth separating here. The set of frequencies `{-K, ..., K}` the circuit
can possibly produce is fixed the moment you choose the circuit's structure: how many
qubits encode this feature, and how many times the feature is re-uploaded (fed into the
circuit again in a later layer). The coefficients `c_n`, how much weight each of those
frequencies actually gets, are what training adjusts, through the trainable rotation
gates and the choice of measurement operator.

The practical consequence is a hard ceiling: no amount of training can add a frequency
the circuit structure does not already support. If a PDE's true solution needs frequency
content the VQC's `K` does not cover, the QAPINN is mathematically capped below the
classical PINN's achievable accuracy, however long you train it. If the solution's
frequency content fits comfortably inside the VQC's `K`, replacing the first layer should
cost little. That is the theory. Section 8 tells you how much of it survived contact with
real training runs and real seed variance.

`src/sweeps.py` gives the exact bandwidth formula used throughout the repo:

```
K = (n_qubits / in_dim) * n_uploads,     n_uploads = n_layers if encoding == "reupload" else 1
```

For example, 4 qubits with 2 layers of `reupload` encoding on a 2D input `(x, t)` gives
`K = (4/2) * 2 = 4`, meaning the circuit can represent frequencies `-4` through `+4` in
each input feature.

---

## 2. Why these two PDEs, Heat and Burgers, and not others

The two benchmarks sit at opposite ends of "how much does the network need to represent,"
which is exactly the axis the Fourier theory in §1 makes predictions about.

**Heat equation** (`src/pdes/heat.py`) is linear, and given here with an initial condition
that is deliberately a finite sum of sine modes (`b_k · sin(kπx)`, default modes at `k=1`
and `k=4`). Sine modes are eigenfunctions of the Laplacian under these boundary
conditions, so the exact solution is known in closed form. There is no numerical solver
and no discretization error, and critically, the solution's frequency content is known in
advance (only `k=1` and `k=4` are present). This makes Heat the clean-room experiment: any
accuracy gap between the QAPINN and a classical PINN can be checked directly against the
circuit's predicted bandwidth `K`, with nothing else confounding the comparison.

**Burgers' equation** (`src/pdes/burgers.py`) is nonlinear (it has a `u·u_x` advection
term), and with the small viscosity used here (`ν = 0.01/π`) it develops a sharp shock: a
steep, broadband, high-frequency feature that is much harder to represent than Heat's
smooth decay. There is no closed-form solution, so the repo builds and caches a
high-accuracy reference by numerically solving the PDE on a fine grid with a Fourier
pseudo-spectral method (integrating-factor RK4 time-stepping, see
`BurgersEquation._spectral_solve`).

**Why not other PDEs** (wave equation, Navier-Stokes, Schrodinger, Allen-Cahn, and so on):
Heat and Burgers are the standard, most widely used PINN benchmarks since Raissi,
Perdikaris and Karniadakis (2019), so results are checkable against a large existing
literature. More importantly, they isolate one variable at a time. Heat has no numerical
reference noise and a known target spectrum. Burgers adds nonlinearity and a hard,
realistic feature without changing anything else about the setup. Harder PDEs would add
solver complexity on top of an already CPU-bound quantum simulator (§11) without adding
new evidence about the mechanism under test.

---

## 3. Repository layout

```
src/
  pdes/          PDE definitions: domain, residual, IC/BC sampling, reference solution
    base.py        abstract PDE interface, autograd gradient helper, collocation sampling
    heat.py         1D heat equation; exact analytical solution (sum of decaying sine modes)
    burgers.py       1D viscous Burgers' equation; cached Fourier pseudo-spectral reference

  models/        the neural network architectures
    mlp.py          classical PINN (tanh / relu / sin); includes proper SIREN init for "sin"
    quantum_layer.py the VQC as a differentiable torch.nn.Module, the piece that replaces
                     a PINN's first layer. Exposes every design dial: n_qubits, n_layers
                     (re-upload depth), encoding, entanglement, measurement, trainable_scaling
    qapinn.py        wires QuantumLayer as the first layer, then the same classical tail
                     structure as MLP; this is the only structural difference vs. MLP

  training/      the training loop
    sampling.py      draws collocation (interior) and supervised (IC/BC) point sets
    losses.py        PDE-residual loss plus supervised loss, combined with configurable weights
    trainer.py        Adam warm-up, then optional L-BFGS refinement; logs history, wall-clock,
                     and CPU time (wall-clock is flagged unreliable if the machine sleeps)

  xai/           explainability tooling, the "why" behind the numbers
    fourier.py       predicts a VQC's accessible frequency band from its config, and
                     empirically recovers a layer's real spectrum via a swept-input DFT
    capacity.py      Hu and colleagues' spectral-complexity metric over a model's classical
                     Linear layers; tests whether QAPINN's smaller parameter count actually
                     means lower model complexity, or not

  utils/         seeds.py (global RNG seeding), config.py (YAML loader with dict/attribute
                 access), plotting.py (solution heatmaps and training-curve figures)

  metrics.py     relative-L2 error, max-abs error, PDE-residual RMS, all evaluated on a
                 regular grid against each PDE's reference solution

  build.py       turns a config dict into a runnable PDE object, model object, and
                 TrainConfig; the one place that translates YAML into Python objects

  sweeps.py      defines the 18-task Heat K-sweep: a ladder of (n_qubits, encoding,
                 n_layers) combinations engineered to hit exact bandwidths K = 1, 2, 3, 4,
                 5, 8, each at 3 random seeds

configs/        one YAML file per experiment (see §7 for the anatomy of a config)

experiments/    runnable top-level scripts (see §6 for exact commands)
  run_baseline.py     train one model from one config; saves summary/history/checkpoint/plots
  run_sweep.py        run one indexed task of the 18-task K-sweep (parallelizable via SLURM)
  aggregate_sweep.py  after all 18 sweep tasks finish: writes ksweep.csv and ksweep_elbow.png
  fourier_spectrum.py produces the headline "spectrum vs. encoding" figure

tests/          unit tests: PDE reference sanity (residual near 0, IC/BC match), and a
                parametrized proof that a VQC's empirical spectrum never exceeds its
                theoretical bandwidth K

scripts/
  paramshakti_array.sbatch   SLURM job-array script: runs all 18 K-sweep tasks in parallel

results/        all experiment outputs (gitignored): per-run summary.json, history.json,
                model.pt, plots, the cached Burgers reference solution, and sweep aggregates

report/         technical report and slides
```

---

## 4. How the QAPINN differs from a classical PINN, concretely

```
MLP     :  normalize -> [Linear(2, H) -> act] -> [Linear(H, H) -> act] * (depth-1) -> Linear(H, 1)
QAPINN  :  normalize -> [QuantumLayer(2 -> Q)]  -> [Linear(Q, H) -> act] * (depth-1) -> Linear(H, 1)
```

`Q` is `n_qubits` under expectation-value readout, or `2**n_qubits` under probability
readout (see `measurement` below). Because `Q` is small (typically 3 to 5), the downstream
`Linear(Q, H)` shrinks compared to `Linear(2, H)`'s implicit width. That is the
parameter-reduction effect this project measures directly in §8.

The `QuantumLayer` (`src/models/quantum_layer.py`) has five design dials, and each one is
a real experimental variable in this project, not just an implementation detail.

| Dial | Options | What it controls |
|---|---|---|
| `n_qubits` | integer (3 to 6 kept modest here) | width of the register; more qubits touching a feature raises its bandwidth `K` |
| `n_layers` | integer | ansatz depth and number of data re-uploads |
| `encoding` | `angle` (upload data once) or `reupload` (re-upload every layer) | re-uploading is what lets a small circuit reach a richer Fourier spectrum |
| `entanglement` | `none`, `linear`, `ring`, `all_to_all` | whether per-qubit frequencies get mixed via CNOTs |
| `measurement` | `expectation` (n_qubits Pauli-Z values) or `probs` (2**n_qubits basis probabilities) | cheap, low-dimensional readout vs. exponentially richer but costlier readout |
| `trainable_scaling` | bool | whether the input to angle frequency multiplier is learned, retuning which harmonics land where |

The circuit runs on PennyLane's `default.qubit` state-vector simulator with
`diff_method="backprop"`, so it is fully differentiable in PyTorch, including the
second derivatives a PDE residual needs (for example `u_xx`).

### Architecture diagrams

<p align="center">
  <img src="Assets/Architecture/01-layer-swap-and-circuit.png" width="850" alt="Diagram comparing the classical PINN's first layer against the QAPINN's quantum layer, plus the underlying 4-qubit circuit">
</p>
<p align="center"><sub>The swap in full. Left: the classical PINN's first layer, <code>Linear(2 → 64)</code> with a tanh activation, 1,341 parameters total. Right: the same network with only that first layer replaced by <code>QuantumLayer(2 → 4)</code>, an expectation-value readout over a 4-qubit register, 985 parameters total. The circuit diagram at the bottom shows exactly what runs on those four qubits: two rounds of angle encoding (<code>R<sub>y</sub>(x)</code>, <code>R<sub>y</sub>(t)</code>), trainable single-qubit rotations, a ring of CNOTs for entanglement, and a final Pauli-Z measurement. Two encoding rounds on a 4-qubit register gives bandwidth K = 4, the number that Section 8.3 tests directly.</sub></p>

<p align="center">
  <img src="Assets/Architecture/02-full-pipeline.png" width="850" alt="End-to-end pipeline diagram showing input normalization, the swapped first layer, the shared classical tail, and the PDE-residual loss">
</p>
<p align="center"><sub>The same swap, traced end to end. An <code>(x, t)</code> pair is normalized to <code>[-1, 1]</code>, passed through whichever first layer is active (a 60-parameter classical matrix or a 24-parameter quantum circuit), then through a classical tail that is byte-identical either way, producing <code>u(x, t)</code>. Autograd differentiates back through the whole graph, including the circuit, to get the derivatives the PDE residual needs. That second-order backward pass through a state-vector simulation is the source of the roughly 20x training-time gap reported in Section 8.1.</sub></p>

---

## 5. Setup

```bash
python -m venv .venv
# Windows PowerShell:      .venv\Scripts\Activate.ps1
# macOS / Linux:           source .venv/bin/activate
pip install -r requirements.txt
```

Requirements (`requirements.txt`): `numpy`, `scipy`, `matplotlib`, `pyyaml` for the core;
`torch>=2.2` for the machine learning; `pennylane>=0.38` for the quantum layer;
`pytest>=8.0` for testing.

Everything here was developed and tested on CPU. See §11 for a device flag you can set to
opt into a GPU, and for an honest account of when that actually helps.

---

## 6. Running experiments: commands you'll actually use

**Run the test suite** (fast, about 7 seconds, should always pass before you trust
anything else):
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

**Qubit-count and viscosity variants** (config-only, same command as above, just point at
a different file; see §7's table for the full list):
```bash
python -m experiments.run_baseline --config configs/burgers_qapinn_q3.yaml
python -m experiments.run_baseline --config configs/burgers_qapinn_q5.yaml
python -m experiments.run_baseline --config configs/burgers_nu0.05_q4.yaml
```

**SIREN control**, a classical network using a sine activation with correct SIREN
initialization, for testing whether a purely classical periodic layer can match the
QAPINN at equal bandwidth:
```bash
python -m experiments.run_baseline --config configs/heat_siren.yaml
python -m experiments.run_baseline --config configs/burgers_siren.yaml
```

**The Heat K-sweep** (18 tasks: bandwidths K in {1, 2, 3, 4, 5, 8} times 3 seeds):
```bash
# run all 18 tasks serially (any machine):
for i in $(seq 0 17); do python -m experiments.run_sweep --index $i; done
# PowerShell equivalent:
#   0..17 | % { python -m experiments.run_sweep --index $_ }

# or in parallel on a SLURM cluster (see scripts/paramshakti_array.sbatch):
sbatch scripts/paramshakti_array.sbatch

# after all 18 finish, aggregate into a CSV and the headline elbow figure:
python -m experiments.aggregate_sweep
# -> results/sweeps/heat_ksweep/ksweep.csv
# -> results/sweeps/heat_ksweep/ksweep_elbow.png
```
This sweep, once isolated to a fixed register width, is what produced the clean elbow at
K = 4 reported in §8. If you rerun it, that is the shape you should expect to see.

**Fourier-spectrum demonstration**, the headline explainability figure, showing
empirically, with no training required, that qubit count and re-upload depth directly
control the quantum layer's bandwidth, exactly as §1's theory predicts:
```bash
python -m experiments.fourier_spectrum
# -> results/fourier/spectrum_vs_encoding.png
```

**Spectral-complexity ("capacity") metric** on a trained checkpoint, testing whether
QAPINN's fewer parameters actually correspond to a lower-complexity model:
```bash
python -m src.xai.capacity configs/heat_pinn.yaml   results/heat_pinn/model.pt
python -m src.xai.capacity configs/heat_qapinn.yaml results/heat_qapinn_q4/model.pt
```
This needs a `model.pt` checkpoint to exist for the run in question. `run_baseline.py`
saves one automatically; if you have an older results directory from before checkpoint
saving was added, re-run the config once to get one.

---

## 7. Anatomy of a config file

Every experiment is driven entirely by a YAML file under `configs/`. Nothing is hardcoded
in the training scripts. A config has three blocks.

```yaml
run_name: heat_qapinn_q4        # results land in results/<run_name>/
seed: 1234                      # global seed (src/utils/seeds.py); everything is reproducible

pde:
  name: heat                    # "heat" or "burgers"
  alpha: 0.05                   # heat: diffusion coefficient
  modes: [[1, 1.0], [4, 0.5]]   # heat: [(wavenumber k, coefficient b_k), ...], the exact target spectrum
  t_max: 1.0                    # time horizon

model:
  type: qapinn                  # "mlp" (classical) or "qapinn" (quantum first layer)
  hidden: 20                    # classical tail width
  depth: 4                      # classical tail depth
  activation: tanh              # "tanh" | "sin" (SIREN) | "relu"
  quantum:                      # only used when type: qapinn, see §4's dial table
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
  resample_every: 0             # 0 means a fixed point set for the whole Adam run
  w_pde: 1.0                    # loss weight on the PDE residual
  w_data: 1.0                   # loss weight on IC/BC supervision
  log_every: 250
  eval_every: 500
  device: auto                  # "auto" | "cpu" | "cuda"; see §11
```

### Every config currently in `configs/`

| File | PDE | Model | Purpose |
|---|---|---|---|
| `heat_pinn.yaml` | heat | classical MLP | baseline |
| `heat_qapinn.yaml` | heat | QAPINN, 4 qubits | main quantum comparison |
| `heat_qapinn_probs.yaml` | heat | QAPINN, 4 qubits, `probs` readout | richer-readout variant |
| `heat_siren.yaml` | heat | classical MLP, sine (SIREN init) | matched-bandwidth classical control |
| `burgers_pinn.yaml` | burgers | classical MLP | baseline |
| `burgers_qapinn.yaml` | burgers | QAPINN, 4 qubits | main quantum comparison |
| `burgers_qapinn_q3.yaml` | burgers | QAPINN, 3 qubits | qubit-threshold sweep |
| `burgers_qapinn_q5.yaml` | burgers | QAPINN, 5 qubits | qubit-threshold sweep |
| `burgers_siren.yaml` | burgers | classical MLP, sine (SIREN init) | matched-bandwidth classical control |
| `burgers_nu{0.00318,0.05,0.1}_q{3,4,5}.yaml` (9 files) | burgers | QAPINN | ν times qubit-count grid, tests whether required qubits track shock width |

---

## 8. Results: the full comparison

Everything below comes from the final rerun, not the early single-seed numbers we started
with. Every model in this table used the same training recipe per PDE: 5,000 L-BFGS
refinement steps, three seeds (1234, 2025, 7), one machine. That matters, because our
first pass at this comparison gave a misleading answer, and the story of catching that is
as important as the numbers themselves. See §8.4.

<p align="center">
  <img src="Assets/Architecture/03-experiment-design.png" width="850" alt="Flowchart of the five experiments run on Heat and Burgers, branching from a head-to-head comparison into a K-sweep, a K-ladder, and a SIREN control">
</p>
<p align="center"><sub>The five experiments behind every number below. E1 and E2 ask the basic question, does the quantum layer help, on Heat and Burgers respectively. E3 asks whether any effect is really about the quantum circuit or just about having a periodic activation, by comparing against the SIREN control. E4 and E5 test whether the bandwidth formula from Section 1 actually predicts behavior: an 18-run sweep on Heat, and a three-point Q3/Q4/Q5 ladder on Burgers. Every cell uses three seeds, one machine, and one training recipe per PDE. Two of the five verdicts shown here are revisions of what we first reported; see §8.4 for why.</sub></p>

### 8.1 The headline scoreboard

| Metric | Classical PINN | SIREN | QAPINN (4 qubits) |
|---|---|---|---|
| Heat, relative L2 error | 0.000243 | 0.000231 | 0.000487 |
| Burgers, relative L2 error | 0.006268 | 0.014832 | 0.019406 |
| Seed-to-seed variance (CV) | 0.7% to 11% | 21% to 36% | 24% to 82% |
| Training time, Heat | 2.7 min | 2.2 min | 60 min |
| Training time, Burgers | 4.2 min | 4.2 min | 81 min |
| Trainable parameters | 1,341 | 1,341 | 985 |

Read plainly: on Heat, the quantum layer is statistically tied with the classical
baseline. On Burgers, the classical baseline wins by a clear margin. In both cases the
quantum layer trains far slower and lands with far more run-to-run variance.

### 8.2 Is that difference actually significant, or just noise?

We ran a two-sample t-test between the classical and QAPINN seed distributions on each
PDE.

- **Heat: t = 1.05.** This does not clear significance. The 0.000243 vs. 0.000487 gap you
  see in the table above is real in this sample, but with only three seeds per arm and
  this much seed variance, we cannot say the quantum layer is actually worse here. Call
  it a tie.
- **Burgers: t is between 4.8 and 4.82** across the two ways we computed it. This does
  clear significance. Classical wins on Burgers by roughly 3.1x, and it is a repeatable
  result, not a lucky seed.

So the one-sentence summary of the whole benchmark is: tied on the easy problem, beaten on
the hard one, and about 20x more expensive to train either way.

### 8.3 The Heat K-sweep confirms the bandwidth theory

Section 1 predicted a specific shape: error should sit near a floor while bandwidth `K` is
below the target's true content, drop sharply once `K` reaches the target's spectrum, and
possibly creep back up past that point from extra, unused capacity.

Isolated to a fixed 4-qubit register (which removes a register-width confound present in
the original 18-task ladder), that is exactly what we saw:

| Bandwidth K | Heat relative L2 error |
|---|---|
| K = 2 | 0.024 |
| K = 4 (the target's true content) | 0.012 |
| K = 8 | 0.016 |

The elbow lands exactly at K = 4, which is where the Heat solution's real spectral content
(modes k = 1 and k = 4) sits. We also swept the untrained circuit's output through a DFT
and confirmed its energy is exactly zero past K, for any configuration. The bandwidth
ceiling is a structural property of the circuit, not something that only shows up after
training.

<p align="center">
  <img src="Assets/Architecture/04-bandwidth-verification.png" width="850" alt="Bar chart of Fourier coefficient magnitude versus harmonic number for three encoding configurations, showing a hard cutoff at K">
</p>
<p align="center"><sub>Direct evidence for the bandwidth ceiling, measured on an untrained circuit so there is no training involved at all. Each bar group is the magnitude of one Fourier coefficient, swept via a 256-point DFT, for three encodings with predicted bandwidths K = 2, K = 4, and K = 6. In every configuration, magnitude past the predicted K is exactly zero, not small, exactly zero, confirming the cutoff is structural rather than approximate. Amplitude also decays as the harmonic number rises, meaning the top usable mode is representable but weak, which is why Section 8.7's recipe recommends leaving margin inside K rather than sitting exactly at the boundary.</sub></p>

One caveat worth stating plainly: register width matters on its own, independent of K. In
that same sweep, K = 5 built from a 2-qubit register scored roughly five times worse than
K = 4 built from a 4-qubit register. Bandwidth and register width are not interchangeable,
even though the formula in §1 treats them as one number.

On Burgers, we ran the equivalent ladder across Q3, Q4, and Q5 registers. The results were
statistically indistinguishable across seeds, meaning no register width reliably beat any
other. A shock-forming, broadband target is not well summarized by a single bandwidth
number the way Heat's clean two-mode spectrum is.

### 8.4 The result we almost reported

Our very first pass at the Burgers comparison showed the QAPINN beating the classical
baseline by about 7.8%. We had flagged an under-trained classical baseline as limitation
L1 in an earlier draft of our internal report, before we had actually tested for it. Once
we went back and checked, that is exactly what had happened.

The original classical run used only 500 L-BFGS steps and landed at 0.0756 relative L2
error. Training that same architecture properly, with 5,000 L-BFGS steps, brought it down
to 0.0063: a 12x improvement, and enough on its own to erase and reverse the apparent
quantum advantage. The "quantum win" in our first pass was a training-budget artifact on
the classical side, not a real effect from the quantum layer. All numbers reported in
§8.1 through §8.3 use the properly trained baseline.

<p align="center">
  <img src="Assets/Architecture/05-baseline-correction.png" width="850" alt="Log-scale training loss curves for the classical PINN, SIREN, and QAPINN on Burgers, showing the classical curve overtaking the others once L-BFGS runs to completion">
</p>
<p align="center"><sub>The correction, visualized. All three models start on nearly identical curves through the Adam warm-up. Once L-BFGS refinement begins (the dashed line), the classical PINN keeps descending for the full 5,000 steps and ends almost an order of magnitude lower than the point where our original run stopped it at step 500, while every model was still descending. Cutting training short does not penalize every model equally: it penalizes whichever one has the most headroom left to give up, which in this case was the classical baseline, and that asymmetry is exactly how the apparent quantum win appeared.</sub></p>

We are including this not to bury it, but because catching your own baseline's training
budget before publishing a comparison is exactly the kind of check a benchmark like this
needs, and we think it is worth showing the work.

### 8.5 Explainability: capacity and complexity

We computed a spectral-complexity bound (following Hu and colleagues, and the
Bartlett-Mendelson framework) directly from each model's trained weights, using
`src/xai/capacity.py`. The QAPINN's bound came out at roughly 3.2 times lower than the
classical PINN's, alongside its 27% smaller parameter count (985 vs. 1,341). The quantum
layer is a genuinely more compact function class, not just a smaller number of weights
that behaves the same as a larger one.

That lower capacity is not automatically a good thing. It tracks closely with the higher
seed-to-seed variance in §8.1: a smaller hypothesis class is also a harder one to land
consistently, which is likely part of why QAPINN's variance is 2 to 7 times higher than
the classical baseline's across both PDEs.

### 8.6 What this benchmark does not claim

- **Simulator only.** Every circuit here ran on PennyLane's state-vector simulator. We
  have no results yet from physical quantum hardware, and none of the noise that comes
  with it.
- **The original 18-task K-ladder confounds qubit count with bandwidth.** Only the
  fixed-4-qubit slice in §8.3 isolates K cleanly. Treat the raw 18-task numbers as
  suggestive, not conclusive, unless you re-slice them the same way.
- **We mostly swept expectation-value readout.** A single probability-readout run
  (`heat_qapinn_probs`) halved Heat's error at an earlier, smaller training budget, which
  is a large enough effect to be worth a real sweep of its own. We have not done that
  sweep yet.
- **Narrow scope.** Two 1D PDEs, 2 to 6 qubits, three seeds. This is not evidence about
  higher-dimensional PDEs, larger quantum registers, or physical hardware noise.
- **No barren-plateau measurement.** We did not measure how gradient variance scales with
  qubit count, which is a known risk once VQCs grow past the modest sizes used here.

### 8.7 The practical recipe this project produced

If you are deciding whether to add a quantum layer to a PINN for a new PDE, our results
point to a concrete checklist rather than a yes or no answer:

1. Ask first whether you need a quantum layer at all. A classical control like SIREN, at
   the same bandwidth budget, may get you the same result at a fraction of the training
   cost. It did on both PDEs here.
2. Fourier-analyze your target before choosing hardware. Know its true wavenumbers, the
   way Heat's are known to be exactly k = 1 and k = 4.
3. Check that the modes fit inside K with margin, not exactly at the boundary. K = 4
   matching a target's exact ceiling is the worst place to sit if you care about variance,
   since you are relying on the circuit's full available capacity with none to spare.
4. You can buy K cheaply with data re-uploads, but do not forget that register width
   governs capacity independently, as shown in §8.3.

Our own conclusion, stated plainly: a quantum layer is a constraint you have to earn. On
these two problems, at this scale, it was not earned. Compute
`K = (n_qubits / in_dim) * n_uploads`, Fourier-analyze your PDE, check that the modes fit
inside K with margin, and then, going by the evidence here, use the classical network
anyway.

---

## 9. What this project actually delivers

Strip away the file list and the config tables, and this repository makes four concrete
contributions, each one backed by a run you can reproduce with the commands in §6.

**A falsifiable theory, tested rather than assumed.** Section 1's bandwidth formula is not
just cited, it is measured twice: once structurally, by sweeping an untrained circuit's
output through a DFT and confirming its energy is exactly zero past K for every
configuration we tried, and once empirically, by isolating a fixed 4-qubit register and
watching Heat's error form a clean elbow that lands exactly at K = 4 (§8.3). The theory
made a specific, checkable prediction, and the prediction held.

**A full statistical comparison, not a single-seed anecdote.** Every headline number in
§8.1 comes from three seeds per arm, per PDE, with a two-sample t-test behind the "tied"
and "wins" language in §8.2. That is what let us catch our own mistake in §8.4: a
single-seed run made the quantum layer look like it was winning on Burgers, and a proper
multi-seed comparison against a properly trained baseline showed that result was never
real.

**An explainability layer, not just an accuracy table.** `src/xai/capacity.py` answers a
question most quantum-classical comparisons skip entirely: does a smaller parameter count
actually mean a lower-complexity model, or just fewer numbers that behave the same way? We
measured it directly (§8.5), and it also explains why QAPINN's variance runs higher than
the classical baseline's.

**A decision procedure, not just a verdict.** The point of this project was never "quantum
wins" or "quantum loses" on two toy PDEs. It is the four-step checklist in §8.7: Fourier
analyze your target, compute K, check the modes fit inside it with margin, and only then
decide whether the quantum layer is worth its training cost. That procedure is what
someone could actually take and apply to a PDE we never tested.

---

## 10. What we'd check next, if we kept going

These are not blockers on the current results, but they are the natural next experiments
given what §8 found.

- A real probability-readout sweep, since the one data point we have suggests it might
  meaningfully change the picture.
- A K-margin sweep on Heat: does sitting comfortably inside K (say K = 6 for a target that
  needs K = 4) reduce the seed-to-seed variance we saw at the exact boundary?
- Running at least one configuration on physical hardware, or a noisy simulator, to see
  how much of the classical-vs-quantum training-time gap and variance gap comes from ideal
  simulation specifically.

---

## 11. Compute notes

Every model in §8 was trained on CPU (`torch` CPU build, PennyLane `default.qubit`), which
is why qubit counts are kept modest (3 to 6). State-vector simulation cost grows quickly
with qubit count, and every PDE residual needs the circuit differentiated twice, for
second-order derivatives like `u_xx`. This is also the main reason QAPINN training takes
so much longer than classical training in §8.1: roughly 20x longer on both PDEs, purely
from re-simulating the circuit at every optimizer step.

`src/training/trainer.py` now supports a `device` field (`auto`, `cpu`, or `cuda`), and
`auto` will use a GPU if one is available. Worth being honest about what this does and
does not help: at the qubit counts used throughout this project, the quantum layer itself
gains little or nothing from a GPU, since a 3 to 6 qubit statevector is tiny (16 to 64
amplitudes) and the per-gate kernel-launch and host-device sync overhead dominates at that
scale. The real win from a GPU would be on the classical tail and its autodiff, for larger
problems than the ones benchmarked here. Because of that, `src/sweeps.py` deliberately
pins the K-sweep to CPU even when a GPU is present, since every K-sweep task is a 2 or
4-qubit QAPINN that would run slower, not faster, on a GPU once you account for the
transfer overhead.

For a heavier quantum simulation specifically, `pennylane-lightning[gpu]` is an option,
but it only supports `diff_method="adjoint"`, which does not provide the second-order
input derivatives a PDE residual needs. That makes it usable for forward-only analysis
(for example, large-K Fourier scans) but not for QAPINN training itself.

For large sweeps, `scripts/paramshakti_array.sbatch` submits the 18-task K-sweep as a
SLURM job array so all 18 run in parallel instead of serially.

---

## 12. Reproducibility

Every experiment reads a YAML config and a global seed (`src/utils/seeds.py`, which seeds
Python's `random`, NumPy, and PyTorch, and requests deterministic cuDNN algorithms).
Configs live in `configs/`. Results are written under `results/<run_name>/` and are
gitignored. Re-running the same config with the same seed on the same machine should
reproduce the same numbers up to floating-point nondeterminism. The results in §8 come
from `results/rerun/`, using the exact recipe described at the top of that section: one
machine, one training recipe per PDE, 5,000 L-BFGS steps, seeds 1234, 2025, and 7.

---

## 13. AI-tool disclosure

AI coding assistants were used as a scaffolding aid: for boilerplate, refactoring, and
drafting this documentation. All mathematical derivations, design decisions, and results
are authored, understood, and defended by the team. See `report/` for the full disclosure
section in the technical report.
