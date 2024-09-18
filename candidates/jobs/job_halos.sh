#!/bin/bash

# SBATCH --job-name=colt
#SBATCH --output=halo-%A.out
# SBATCH --output=y_A_z4_%a_%A.out
# SBATCH --array=0-67
#SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mki_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# SBATCH --cpus-per-task=16
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=48:00:00
# SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu

## Module setup
. /etc/profile.d/modules.sh
export HDF5_USE_FILE_LOCKING=FALSE

sim=${group}/${run}

# for i in {0..188}; do
for i in {0..189}; do
# for i in {0..25}; do
# for i in 186; do
    #echo "Running halo ${i} ..."
    python candidates.py $sim $i # Write halos to file
    python halos.py $sim $i # Write halos to file
done

#python candidates.py $sim $SLURM_ARRAY_TASK_ID # Write halos to file
#python halos.py $sim $SLURM_ARRAY_TASK_ID # Write halos to file

echo "Done with ${group}/${run}"
