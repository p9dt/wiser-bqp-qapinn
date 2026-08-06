#!/bin/bash
#SBATCH --job-name=qapinn-rerun
#SBATCH --array=1-24
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/rerun_%A_%a.out
#SBATCH --error=logs/rerun_%A_%a.err

# ---------------------------------------------------------------------------
# Fair-comparison rerun. Submit as-is, no edits needed:
#
#     sbatch scripts/rerun_array.sbatch
#
# Runs every config listed in configs/rerun/ALL.txt (24 jobs = 8 runs x 3 seeds).
# Each PDE uses ONE training recipe, so classical and quantum get identical
# budgets:
#
#   heat    : 8000 collocation, 6000 Adam @ lr 1e-3, 5000 L-BFGS
#   burgers : 8000 collocation, 8000 Adam @ lr 1e-3, 5000 L-BFGS
#
# The old results used lbfgs_epochs=500 (and 0 for the K-sweep), which cut
# training off while it was still improving. That penalised whichever model had
# the most headroom left, which produced misleading gaps in both directions.
#
# Results land in results/rerun/ so nothing existing is overwritten.
# Longest single job is burgers_q5 at roughly 6.3 h.
#
# The T1_/T2_ filename prefixes are priority, not separate batches: if the queue
# only lets part of this through, the T1_ jobs (the classical-vs-quantum
# comparisons on both PDEs) are the ones that must land. T2_ adds the SIREN
# controls and the Q3/Q5 bandwidth ranking.
# ---------------------------------------------------------------------------

set -euo pipefail

# NOTE: run from scratch, never from /home.
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/rerun

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

MANIFEST=configs/rerun/ALL.txt
N=$(grep -c . "${MANIFEST}")
if [[ "${SLURM_ARRAY_TASK_ID}" -gt "${N}" ]]; then
  echo "array index ${SLURM_ARRAY_TASK_ID} exceeds ${N} configs in ${MANIFEST}" >&2
  echo "set --array=1-${N}" >&2
  exit 1
fi

CFG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}/${N} -> ${CFG}"
python -u -m experiments.run_baseline --config "${CFG}"
echo "[$(date -Is)] done"
