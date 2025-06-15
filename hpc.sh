#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:2
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-816
#SBATCH --output=./output/std_log/myjob-%A-%a.out

# UASGE: sbatch inf.sh
# Queue Status: squeue -u zhw323
# Total task:1111, per 101 as a group
# Remeber to change the number of job in the array 

T_LABELS=("second" "hour" "day" "week" "year")

T_LABEL=${T_LABELS[0]}





ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 hdf5/1.10.7

conda activate /share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit

export PATH="/share/ceph/hawk/nil422_proj/shared/shared-aihwkitgpu/conda-env-aihwkit/bin:$PATH"

python hwa_inference_slurm.py --job_id ${SLURM_ARRAY_TASK_ID} --t_inference ${T_LABEL}