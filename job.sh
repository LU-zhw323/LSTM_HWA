#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-120
#SBATCH --output=./output/std_log/myjob-%A-%a.out
#SBATCH --error=./output/err_log/myjob-%A-%a.err

# UASGE: sbatch job.sh
# Queue Status: squeue -u zhw323
TASK_TYPES=("inference_program_noise" "inference_read_noise" "drift" "gmax")

TYPE_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 30 ))
TASK_TYPE=${TASK_TYPES[$TYPE_INDEX]}
LOCAL_TASK_ID=$(( (SLURM_ARRAY_TASK_ID - 1) % 30 + 1 ))

# Create the directory for the task type if it does not already exist
mkdir -p ./output/${TASK_TYPE}
OUTPUT_FILE="./output/${TASK_TYPE}/output-${TASK_TYPE}-${LOCAL_TASK_ID}.txt"

echo "This is job ${LOCAL_TASK_ID} of type ${TASK_TYPE}, global job ${SLURM_ARRAY_TASK_ID}" >> ${OUTPUT_FILE}


ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 hdf5/1.10.7

conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit

export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"

python lstm_inference.py --task_id ${LOCAL_TASK_ID} --task_type ${TASK_TYPE} >> ${OUTPUT_FILE}
STATUS=$?


if [ $STATUS -eq 0 ]; then
    rm "./output/std_log/myjob-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"
    rm "./output/err_log/myjob-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.err"
else
    mv "./output/std_log/myjob-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out" "./output/std_log/myjob-${TASK_TYPE}-${LOCAL_TASK_ID}.out"
    mv "./output/err_log/myjob-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.err" "./output/err_log/myjob-${TASK_TYPE}-${LOCAL_TASK_ID}.err"
fi