#!/bin/bash
# Heat K-sweep as a SLURM job array: 18 tasks (6 bandwidths x 3 seeds), one per element.
# Submit from ~/wiser-bqp-cluster:  sbatch scripts/paramshakti_array.sbatch
#SBATCH -J wiser-ksweep
#SBATCH -A aniketec
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o ksweep_%A_%a.out
#SBATCH --array=0-17

module purge
unset PYTHONPATH
source ~/miniforge3/bin/activate
conda activate wiser
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export OMP_NUM_THREADS=4

cd ~/wiser-bqp-cluster
python -m experiments.run_sweep --index $SLURM_ARRAY_TASK_ID
