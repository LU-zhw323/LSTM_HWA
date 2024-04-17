#!/bin/bash


#SBATCH -c 6
#SBATCH -p hawkgpu
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhw323@lehigh.edu
#SBATCH --array=1-2

# UASGE: sbatch hpc.sh
# Queue Status: squeue -u zhw323
# Remeber to change the number of job in the array 


echo "This is job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}" >> output-${SLURM_ARRAY_TASK_ID}.txt


ml anaconda3 cuda/11.6.0 mvapich2/2.3.4 py-mpi4py/3.0.3 hdf5/1.10.7



conda activate /share/ceph/hawk/nil422_proj/zhw323/aihwkitgpu/cenv

# Change the name of the pyton script below
python job_test.py --task_id ${SLURM_ARRAY_TASK_ID} >> output-${SLURM_ARRAY_TASK_ID}.txt 