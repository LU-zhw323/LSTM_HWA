#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-11

# UASGE: sbatch job.sh
# Queue Status: squeue -u zhw323

echo "This is job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}" >> output-${SLURM_ARRAY_TASK_ID}.txt


ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 hdf5/1.10.7

conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit

export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"

python lstm_hwa.py --task_id ${SLURM_ARRAY_TASK_ID} >> output-${SLURM_ARRAY_TASK_ID}.txt