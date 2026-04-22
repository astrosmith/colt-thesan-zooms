#!/bin/bash

#SBATCH --job-name=extract
#SBATCH --output=y-%A.out
# SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=ou_mki,mit_normal,sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=mit_normal,sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=ou_mki,mit_normal,sched_mit_mvogelsb,sched_mit_mki
# ,newnodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=32
# SBATCH --constraint=centos7
# SBATCH --mem-per-cpu=3000 # 3GB of memory per CPU
# SBATCH --mem-per-cpu=4000 # 4GB of memory per CPU
# SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
# SBATCH --mem-per-cpu=15500 # 15.5GB of memory per CPU
# SBATCH --mem=0 # All of memory
#
#SBATCH --partition=mit_normal_gpu,mit_preemptable,ou_mki_preempt,sched_mit_mki_r8,sched_mit_mki_preempt_r8
#SBATCH --gpus-per-node=1
#SBATCH --mem=1000000 # 1000GB of memory
#
#SBATCH --export=ALL
# SBATCH --time=96:00:00
# SBATCH --time=48:00:00
#SBATCH --time=24:00:00
# SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
# SBATCH --dependency=afterok:
# SBATCH --dependency=afterany:

py="/home/arsmith/miniforge3/bin/python3 -u"

sim=${group}/${run}

# for snap in {0..189}; do
#     $py arepo_to_dm.py $sim $snap  # Arepo to COLT initial conditions
# done
# time $py group_centering.py $sim  # Group centering for movies
# time $py group_radius.py $sim  # Group centering for movies
# time $py extract.py $sim  # Extract data
# time $py extract_states.py $sim  # Extract data (states)
# time $py extract_split.py $sim  # Extract data (split)
# time $py Ha_recentering.py $sim  # Ha recentering

## long
# run g2/z4  # A
# run g39/z4  # B long
# run g205/z4  # C+ long
# run g578/z4  # D+ long
# run g1163/z4  # E+
# run g5760/z8  # F+ long
# run g10304/z8  # G+
# run g137030/z16  # J+
# run g500531/z16  # K+
# run g519761/z16  # L+
# run g2274036/z16  # M+

## int
# run g5760/z4  # F+
# run g10304/z4  # G+
# run g33206/z8  # H+
# run g33206/z4  # H+
# run g37591/z8  # I+
# run g37591/z4  # I+
# run g137030/z8  # J+
# run g137030/z4  # J+
# run g500531/z8  # K+
# run g500531/z4  # K+
# run g519761/z8  # L+
# run g519761/z4  # L+
# run g2274036/z8  # M+
# run g2274036/z4  # M+
# run g5229300/z16  # N+
# run g5229300/z8  # N+
# run g5229300/z4  # N+

