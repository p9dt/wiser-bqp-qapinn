# NEXT STEPS — WISER × BQP QAPINN

Ordered by **evidential value per unit compute**. Deadline **2026-08-07**.
Legend: ✅ code ready (just run) · 🟡 config-only (no new code) · 🔴 needs new code.

Every step below runs with a plain `python -m ...` command. **Nothing requires a
cluster** — the heavy runs (K-sweep, Burgers variants) are just slower on a laptop.
Pick a lane in *Where to run* and use the same commands.

---

## Where to run

The code is written for `torch` + PennyLane `default.qubit` and defaults to **CPU**.
Three lanes, in increasing speed:

### A. Local laptop (CPU) — works today, no changes
```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows; or `source .venv/bin/activate`
pip install -r requirements.txt
python -m experiments.run_baseline --config configs/heat_qapinn.yaml   # ~30 min Heat, hours for Burgers
```
Good enough for Heat, the K-sweep (run tasks one at a time), the capacity metric,
and aggregation. Burgers (8000 Adam + 500 L-BFGS) is slow but *does* finish.

### B. Dedicated GPU — needs a small one-time patch
The repo has **no device handling yet** (grep: only CPU). To use a GPU you must
move the model and the sampled points to `cuda`. Two edits:

1. `src/training/trainer.py`, top of `train(...)`, after `points = draw_points(...)`:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   points = points.to(device)        # add a .to() on PointSet, or move its tensors
   ```
   (`PointSet` in `src/training/sampling.py` holds a few tensors — give it a
   `.to(device)` that moves each, or move them inline.)
2. The QAPINN's `default.qubit` runs through the torch interface, so it follows the
   tensors onto the GPU automatically — **but** at 3–6 qubits the quantum sim is
   tiny and GPU gives little there. The real GPU win is the **classical PINN tail +
   autodiff** (Burgers, SIREN, big collocation sets). For a heavier quantum sim use
   PennyLane-Lightning GPU instead:
   ```bash
   pip install pennylane-lightning[gpu]        # needs CUDA toolkit
   # then in src/models/quantum_layer.py line ~68:
   #   dev = qml.device("lightning.gpu", wires=cfg.n_qubits)
   #   qnode = qml.QNode(circuit, dev, interface="torch", diff_method="adjoint")
   ```
   > ⚠️ `lightning.gpu` uses `diff_method="adjoint"`, which does **not** give the
   > 2nd-order input derivatives a PINN residual needs. Keep `default.qubit` +
   > `backprop` for QAPINN training; use `lightning.gpu` only for forward-heavy
   > analysis (e.g. large-K Fourier scans). For QAPINN *training* on GPU, stick with
   > `default.qubit` on cuda tensors (edit 1 above).

Verify the GPU is used: `python -c "import torch; print(torch.cuda.is_available())"`.

### C. SLURM cluster (e.g. ParamShakti) — for the full parallel sweep
The K-sweep ships as an 18-task job array (`scripts/paramshakti_array.sbatch`),
so all 18 run at once instead of serially.
> **🚨 Golden rule:** never submit from `/home` — copy the tree to `$SCRATCH` and
> submit from there, or the job is auto-cancelled.
```bash
cp -r <this-repo> $SCRATCH/wiser-bqp-cluster
cd $SCRATCH/wiser-bqp-cluster
sed -i "s#cd .*wiser-bqp-cluster#cd $SCRATCH/wiser-bqp-cluster#" scripts/paramshakti_array.sbatch
grep -E 'cd |--array' scripts/paramshakti_array.sbatch      # cd line must be $SCRATCH/...
sbatch --array=0 scripts/paramshakti_array.sbatch           # single-task smoke test first
sbatch scripts/paramshakti_array.sbatch                     # then the full 18 (array 0-17)
```

---

## 1. Heat K-sweep ✅ CODE READY

**Highest-value item, fully coded.** Sweep the quantum-layer bandwidth
`K ∈ {1,2,3,4,5,8}` × 3 seeds (18 tasks) and look for the **elbow** — error near
the Parseval floor (~0.156) below K=4, a sharp drop at K=4, a slight rise above.
Files: `src/sweeps.py` · `experiments/run_sweep.py` · `experiments/aggregate_sweep.py`.

**Run all 18 (any lane):**
```bash
# Local (A/B) — serial loop over the 18 task indices:
for i in $(seq 0 17); do python -m experiments.run_sweep --index $i; done
#   (PowerShell:  0..17 | % { python -m experiments.run_sweep --index $_ } )

# Cluster (C) — parallel array, see "Where to run" above.
```
**Aggregate (any lane, after all 18 finish):**
```bash
python -m experiments.aggregate_sweep
# -> results/sweeps/heat_ksweep/ksweep.csv  and  ksweep_elbow.png
```
**Done when:** `ksweep_elbow.png` exists and the K=4 point sits well below the
Parseval floor line. Drop the figure straight into the report.

---

## 2. 3-qubit & 5-qubit QAPINN variants 🟡 CONFIG-ONLY

5 qubits is the predicted Burgers-shock threshold (ref [3] Fig. 2). No new code —
copy `configs/burgers_qapinn.yaml`, change `n_qubits` only.

Create `configs/burgers_qapinn_q3.yaml` and `_q5.yaml`, identical to
`burgers_qapinn.yaml` except:
```yaml
run_name: burgers_qapinn_q5   # (or _q3)
model:
  quantum:
    n_qubits: 5               # (or 3)
```
```bash
python -m experiments.run_baseline --config configs/burgers_qapinn_q5.yaml
python -m experiments.run_baseline --config configs/burgers_qapinn_q3.yaml
```
Burgers is the slow one (8000 Adam + 500 L-BFGS) — this is where a GPU (lane B) or
the cluster pays off most. **Done when:** you have a params/rel-L2/max-abs row for
3q/4q/5q showing max-abs (the shock error) dropping sharply at 5q.

---

## 3. ν-sweep on Burgers 🟡 CONFIG-ONLY — the cleanest single test

Larger `ν` ⟹ thicker shock ⟹ narrower spectrum ⟹ fewer qubits suffice. If the
required qubit count tracks the spectral width, the bandwidth mechanism is
confirmed. `BurgersEquation` already takes `nu`; `run_baseline` is config-driven —
**no new code**.

One config per (ν, n_qubits) cell. Minimal grid: ν ∈ {0.01/π ≈ 0.00318, 0.05, 0.1}
× n_qubits ∈ {3,4,5}. Template (`configs/burgers_nu0.05_q4.yaml`):
```yaml
run_name: burgers_nu0.05_q4
seed: 1234
pde: {name: burgers, nu: 0.05, t_max: 1.0}
model:
  type: qapinn
  hidden: 20
  depth: 4
  activation: tanh
  quantum: {n_qubits: 4, n_layers: 2, encoding: reupload, entanglement: ring,
            measurement: expectation, trainable_scaling: true}
training:
  n_collocation: 8000
  n_supervised: 400
  adam_epochs: 8000
  adam_lr: 0.001
  lbfgs_epochs: 500
  resample_every: 0
  w_pde: 1.0
  w_data: 1.0
  log_every: 1000
  eval_every: 2000
```
> ⚠️ Each new `nu` regenerates the reference cache
> `results/reference/burgers_nu*.npz` on first run (one-time few-min cost per ν).

Run each with `run_baseline`. **Done when:** the qubit count needed to resolve the
shock *shrinks* as ν grows.

*(Optional: `experiments/run_nu_sweep.py` globbing `configs/burgers_nu*.yaml` and
tabulating — not required, the configs run as-is.)*

---

## 4. SIREN / basis control 🔴 SMALL CODE CHANGE — thesis-critical

Does a **classical periodic layer at equal bandwidth** match the QAPINN? If yes,
the effect is periodicity, not "quantum".

**Already works:** `MLP(activation="sin")` runs today — set `activation: sin` in a
PINN config for an immediate sine-activation baseline.

**Missing for a *fair* SIREN:** proper init (first layer scaled by ω₀≈30, hidden
layers uniform-by-fan-in). Add in `src/models/mlp.py`:
```python
# _Sine that stores omega0 and returns sin(omega0 * x); use omega0=30 on layer 1, 1.0 after.
# In _init_weights, for the sine path (Sitzmann et al. 2020, SIREN §3.2):
#   first Linear:  weight ~ U(-1/fan_in, 1/fan_in)
#   hidden Linear: weight ~ U(-sqrt(6/fan_in)/omega0, +sqrt(6/fan_in)/omega0)
```
Then run the comparison at matched bandwidth on Heat (and Burgers):
vanilla PINN (tanh) · SIREN (sine) · QAPINN. **Done when:** you can state whether
SIREN matches QAPINN at equal K — either result is publishable.

*(GAAF-PINN — ref [3]'s SOTA baseline, one trainable scalar per layer — is a
further optional control; add only if time allows.)*

---

## 5. Hu's capacity metric 🔴 ONE NEW FILE — no retraining, runs on CPU

Deliverable 2, and a direct falsification of ref [3]'s "fewer params ⟹ less
overfitting". Computes ref [1]'s spectral-complexity bound on **existing
checkpoints** — a `torch.linalg` one-liner over the classical Linear tail (the VQC
layer has no weight matrix, so it's naturally skipped).
```
C = ∏_l ‖W^l‖₂ · ( Σ_l ( ‖W^l‖_{2,1} / ‖W^l‖₂ )^{2/3} )^{3/2}
```
Add `src/xai/capacity.py`:
```python
"""Ref [1] (Hu et al.) spectral-complexity metric over a model's Linear tail."""
import torch
from torch import nn

def spectral_complexity(model: nn.Module) -> dict:
    Ms, Ns = [], []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach()
            s = torch.linalg.matrix_norm(W, ord=2)      # ‖W‖₂
            n21 = W.norm(p=2, dim=0).sum()              # ‖W‖_{2,1}
            Ms.append(s); Ns.append(n21 / s)
    prod = torch.stack(Ms).prod()
    ratio = (torch.stack(Ns) ** (2/3)).sum() ** (3/2)
    return {"prod_spectral": prod.item(), "ratio_term": ratio.item(),
            "complexity": (prod * ratio).item()}

if __name__ == "__main__":
    import sys
    from src.build import build_model, build_pde
    from src.utils.config import load_config
    cfg = load_config(sys.argv[1])
    pde = build_pde(cfg["pde"]); model = build_model(cfg["model"], pde)
    model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))
    print(spectral_complexity(model))
```
```bash
python -m src.xai.capacity configs/heat_pinn.yaml   results/heat_pinn/model.pt
python -m src.xai.capacity configs/heat_qapinn.yaml results/heat_qapinn_q4/model.pt
```
**Done when:** if the QAPINN has *fewer params but comparable/higher* complexity,
ref [3]'s stated mechanism is falsified — the report's argument.

---

## 6. Remaining XAI (barren-plateau, loss-landscape, attribution) 🔴 NEEDS CODE

Lower priority — after 1–5. Each a new `src/xai/*.py`:
- **barren-plateau:** variance of `∂loss/∂θ` for quantum params vs qubits/depth
  (random inits, no full training) → `src/xai/barren.py`.
- **loss-landscape:** filter-normalized 2D slice around the trained minimum →
  `src/xai/landscape.py`.
- **attribution:** input-gradient / integrated-gradients of `u` w.r.t. `(x,t)` →
  `src/xai/attribution.py`.

Specs only; implement if time permits.

---

## Loose ends to resolve or state honestly in the report
- **Ref [3]'s encoding is unspecified, no code released** ⟹ its `K` is
  undetermined. Infer it spectrally with `src/xai/fourier.py`, or **email BQP**.
- **Burgers PINN baseline is likely under-trained** (7.6e-2 vs Raissi's 9.4e-4 for
  the same architecture — he used L-BFGS throughout). Raise `lbfgs_epochs` in
  `configs/burgers_pinn.yaml` before any headline PINN-vs-QAPINN accuracy claim.

## Suggested order before the deadline
1. Start the **K-sweep** (§1) — longest, kick it off first (serial locally, or on a
   GPU / the cluster if you get access).
2. In parallel (all CPU, cheap): **capacity metric** (§5), **3q/5q + ν configs**
   (§2, §3).
3. **SIREN init + control** (§4) — settles the thesis.
4. Aggregate, then **report/slides**. XAI extras (§6) only if time.
