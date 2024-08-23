#!/bin/bash

# SBATCH --job-name=colt
# SBATCH --output=halo-%A.out
#SBATCH --output=LyC-%a-%A.out
#SBATCH --array=0-20
# SBATCH --array=188
#SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mki_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=16
# SBATCH --cpus-per-task=32
#SBATCH --cpus-per-task=64
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu

## Module setup
. /etc/profile.d/modules.sh
source ~/.colt

## OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

call="mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl ^openib --mca psm2 ucx -np $SLURM_NTASKS --bind-to none"
# colt="${HOME}/colt/colt"
# colt="${HOME}/colt/colt-proj"
colt="${HOME}/colt/colt-flows"
sim=${group}/${run}

runs() {
    for suffix in "$@"; do
        config_file="config-${suffix}.yaml"
        echo "Running simulation with ${config_file} ..."
        start=$SLURM_ARRAY_TASK_ID
        stride=$SLURM_ARRAY_TASK_COUNT
        for ((i=start; i<=188; i+=stride)); do
            echo "Running snapshot ${i} ..."
            $call $colt $config_file $i
        done
    done
}

# runs halo-proj-RHD
runs halos-ion-eq-RHD

echo "Done with ${group}/${run}"
