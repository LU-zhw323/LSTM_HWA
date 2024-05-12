#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-500
#SBATCH --output=./output/std_log/myjob-%A-%a.out

# UASGE: sbatch inf.sh
# Queue Status: squeue -u zhw323
# Remeber to change the number of job in the array 
FIXED_JOB_ID=$SLURM_JOB_ID

MODEL_TYPES=("FP" "HWA")

MODEL_TYPE=${MODEL_TYPES[0]}

ADJUSTED_JOB_ID=$((SLURM_JOB_ID - SLURM_ARRAY_TASK_ID))

ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 hdf5/1.10.7

conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit

export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"

python lstm_inf_nested.py --task_id ${SLURM_ARRAY_TASK_ID} --model_type ${MODEL_TYPE}
STATUS=$?

if [ $STATUS -eq 0 ]; then
    rm "./output/std_log/myjob-${ADJUSTED_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"
fi
