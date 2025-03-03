#!/bin/bash

#SBATCH --job-name=extract
#SBATCH --output=y-%A.out
#SBATCH --partition=newnodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=3000 # 3GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu

## long
#time python extract.py g2/z4
#time python extract.py g39/z4
#time python extract.py g205/z4
#time python extract.py g578/z4
#time python extract.py g1163/z4
#time python extract.py g5760/z8
#time python extract.py g10304/z8
#time python extract.py g137030/z16
#time python extract.py g500531/z16
#time python extract.py g519761/z16
#time python extract.py g2274036/z16

## int
#time python extract.py g5760/z4
#time python extract.py g10304/z4
time python extract.py g33206/z8
time python extract.py g33206/z4
time python extract.py g37591/z8
time python extract.py g37591/z4
time python extract.py g137030/z8
time python extract.py g137030/z4
time python extract.py g500531/z8
time python extract.py g500531/z4
time python extract.py g519761/z8
time python extract.py g519761/z4
time python extract.py g2274036/z8
time python extract.py g2274036/z4
time python extract.py g5229300/z16
time python extract.py g5229300/z8
#time python extract.py g5229300/z8

