#!/bin/bash

# SBATCH --job-name=colt
#SBATCH --output=halo-%A.out
#SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mki_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu

## Module setup
. /etc/profile.d/modules.sh
export HDF5_USE_FILE_LOCKING=FALSE

sim=${group}/${run}

for i in {0..188}; do
    echo "Running halo ${i} ..."
    python halos.py $sim $i # Write halos to file
done

echo "Done with ${group}/${run}"
