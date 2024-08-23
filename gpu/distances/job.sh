#!/bin/bash

# SBATCH --job-name=colt
# SBATCH --output=z-%A.out
#SBATCH --output=y_D_varEff_z4_%a_%A.out
#SBATCH --array=0-68
#SBATCH --partition=sched_mit_mki_r8,sched_mit_mki_preempt_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
# SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu

## Module setup
. /etc/profile.d/modules.sh
module load gcc cuda/11.3 openmpi hdf5
export HDF5_USE_FILE_LOCKING=FALSE

sim=${group}/${run}
sim_dir="/orcd/data/mvogelsb/004/Thesan-Zooms/${sim}/output"

#for i in {94..141}; do
#for i in {20..188}; do
# for i in {0..188}; do
#for i in 189; do
#    mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS ./main_gpu_mpi $sim_dir $i
#done

mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS ./main_gpu_mpi $sim_dir $SLURM_ARRAY_TASK_ID

#echo "Done with ${group}/${run}"
