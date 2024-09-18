#!/bin/bash

# SBATCH --job-name=colt
# SBATCH --output=z-%A.out
# SBATCH --output=y_F_z8_%a_%A.out
# SBATCH --array=0-188
# SBATCH --array=0-32
#SBATCH --array=49
# SBATCH --array=163-188
# SBATCH --array=88-179
# SBATCH --array=189
#SBATCH --partition=sched_mit_mki_r8,sched_mit_mki_preempt_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
# SBATCH --ntasks-per-node=32
# SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=32
# SBATCH --gpus-per-node=1
#SBATCH --gpus-per-node=4
# SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
# SBATCH --time=168:00:00
# SBATCH --time=48:00:00
#SBATCH --time=18:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
#SBATCH --exclude=node2000
# SBATCH --nodelist=node2000

## Module setup
. /etc/profile.d/modules.sh
module load gcc cuda/11.3 openmpi hdf5
export HDF5_USE_FILE_LOCKING=FALSE

sim=${group}/${run}
sim_dir="/orcd/data/mvogelsb/004/Thesan-Zooms/${sim}/output"

#for i in {94..141}; do
#for i in {20..188}; do
#for i in 189; do
# for i in {0..189}; do
#    mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS ./main_gpu_mpi $sim_dir $i
# done

mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS ./main_gpu_mpi $sim_dir $SLURM_ARRAY_TASK_ID

# Serial: mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS ./main_gpu $sim_dir $SLURM_ARRAY_TASK_ID

#echo "Done with ${group}/${run}"
