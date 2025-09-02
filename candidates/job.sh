#!/bin/bash

#SBATCH --job-name=extract
#SBATCH --output=y-%A.out
#SBATCH --partition=mit_normal,sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=mit_normal,sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=newnodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
# SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=3000 # 3GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu

run() {
    time python -u group_centering.py $1  # Group centering for movies
    time python -u group_radius.py $1  # Group centering for movies
    time python -u extract.py $1  # Extract data
    time python -u extract_split.py $1  # Extract data (split)
}

## long
# run g2/z4
# run g39/z4
# run g205/z4
# run g578/z4
# run g1163/z4
# run g5760/z8
# run g10304/z8
# run g137030/z16
# run g500531/z16
# run g519761/z16
#run g2274036/z16

## int
# run g5760/z4
# run g10304/z4
# run g33206/z8
# run g33206/z4
# run g37591/z8
# run g37591/z4
# run g137030/z8
# run g137030/z4
# run g500531/z8
run g500531/z4
# run g519761/z8
# run g519761/z4
# run g2274036/z8
# run g2274036/z4
# run g5229300/z16
# run g5229300/z8
# run g5229300/z4

