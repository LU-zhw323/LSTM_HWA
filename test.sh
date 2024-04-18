#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-30

# UASGE: sbatch test.sh
# Queue Status: squeue -u zhw323
# Remeber to change the number of job in the array 
TASK_TYPES=("inference_program_noise" "inference_read_noise" "drift" "gmax")
TASK_TYPE=${TASK_TYPES[0]}

OUTPUT_FILE="output-${TASK_TYPE}-${SLURM_ARRAY_TASK_ID}.txt"

echo "This is job ${SLURM_ARRAY_TASK_ID} of type ${TASK_TYPE}" >> ${OUTPUT_FILE}


ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 hdf5/1.10.7

conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit

export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"

python lstm_inference.py --task_id ${SLURM_ARRAY_TASK_ID} --task_type ${TASK_TYPE} >> ${OUTPUT_FILE}