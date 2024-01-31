#!/bin/bash

#SBATCH --job-name=extract
#SBATCH --output=extract-%x.%j.out
#SBATCH --partition=sched_mit_mki,sched_mit_mvogelsb,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu
# SBATCH --exclude=node1412

time python pipeline.py

