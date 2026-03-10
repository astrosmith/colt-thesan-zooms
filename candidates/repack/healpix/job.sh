#!/bin/bash

#SBATCH --job-name=maps
#SBATCH --output=y-%A.out
# SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=ou_mki,mit_normal,sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=mit_normal,sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
#SBATCH --partition=ou_mki,mit_normal,sched_mit_mvogelsb,sched_mit_mki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
# SBATCH --time=96:00:00
#SBATCH --time=48:00:00
# SBATCH --time=24:00:00
# SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
# SBATCH --dependency=afterok:
# SBATCH --dependency=afterany:

py="/home/arsmith/miniforge3/bin/python3 -u"

time $py plot-maps.py
