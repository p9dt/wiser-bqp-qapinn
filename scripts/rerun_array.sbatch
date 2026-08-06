#!/bin/bash
#SBATCH --job-name=qapinn-rerun
#SBATCH --array=1-12                 # tier 1 = 12 jobs; see TIER note below
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/rerun_%A_%a.out
#SBATCH --error=logs/rerun_%A_%a.err

# ---------------------------------------------------------------------------
# Fair-comparison rerun. Every config in configs/rerun/ uses ONE training
# recipe per PDE, so classical and quantum get identical budgets:
#
#   heat    : 8000 collocation, 6000 Adam @ lr 1e-3, 5000 L-BFGS
#   burgers : 8000 collocation, 8000 Adam @ lr 1e-3, 5000 L-BFGS
#
# The old results used lbfgs_epochs=500 (and 0 for the K-sweep), which cut
# training off while it was still improving -- that is what produced the
# misleading classical/quantum gaps. Results land in results/rerun/ so nothing
# existing is overwritten.
#
# TIER: pick the manifest line range you want.
#   tier 1 only (12 jobs, headline comparisons) : grep '/T1_'
#   tiers 1+2   (24 jobs, + controls & ranking) : grep -E '/T[12]_'
#   everything  (48 jobs)                       : cat
# Set --array to match the count.
# ---------------------------------------------------------------------------

set -euo pipefail

# NOTE: run from scratch, never from /home.
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/rerun

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

MANIFEST=$(mktemp)
grep '/T1_' configs/rerun/ALL.txt > "$MANIFEST"     # <-- edit for tiers 1+2 or all

CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
rm -f "$MANIFEST"

if [[ -z "${CFG}" ]]; then
  echo "no config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} -> ${CFG}"
python -u -m experiments.run_baseline --config "${CFG}"
echo "[$(date -Is)] done"
