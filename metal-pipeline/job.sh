#!/bin/bash

#SBATCH --job-name=colt
#SBATCH --output=colt-%a-%A.out
# SBATCH --array=20-188
#SBATCH --array=188
#SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu
#SBATCH --exclude=node1412,node1413,node1415,node1421,node1426,node1454,node1455,node1447

## Module setup
. /etc/profile.d/modules.sh
source ~/.colt

## OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

call="mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl ^openib --mca psm2 ucx -np $SLURM_NTASKS --bind-to none"
colt="${HOME}/colt/colt"

run() {
    for suffix in "$@"; do
        config_file="config-${suffix}.yaml"
        #echo "Running simulation with ${config_file} ..."
        $call $colt $config_file $SLURM_ARRAY_TASK_ID
    done
}

run ion-eq-pre7 ion-eq-pre8 ion-eq
run ion-eq-MCRT ion-eq-RHD
run proj Ha Lya
# run Ha-RHD Lya-RHD
# run OII-3727-3730 OIII-5008 M1500

