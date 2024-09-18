#!/bin/bash

# Define job sets with individual job, group, and runs
job_sets=(
# COLT (long circulators)
# "job='A'; group='g2'; runs=('z4')"              # 0 - 67
# "job='B'; group='g39'; runs=('z4')"             # 0 - 188
# "job='C'; group='g205'; runs=('z4')"            # 0 - 188
# "job='D'; group='g578'; runs=('z4')"            # 0 - 188
# "job='D'; group='g578'; runs=('noESF_z4')"      # 88 - 179
# "job='D'; group='g578'; runs=('varEff_z4')"     # 0 - 68
# "job='E'; group='g1163'; runs=('noESF_z4')"     # 0 - 188
# "job='E'; group='g1163'; runs=('z4')"           # 0 - 188
# "job='F'; group='g5760'; runs=('z8')"           # 0 - 188
# "job='G'; group='g10304'; runs=('z8')"          # 0 - 188
# "job='H'; group='g33206'; runs=('z16')"         # 0 - 8
# "job='J'; group='g137030'; runs=('z16')"        # 0 - 188
# "job='K'; group='g500531'; runs=('z16')"        # 0 - 188
# "job='L'; group='g519761'; runs=('z16')"        # 0 - 188
# "job='M'; group='g2274036'; runs=('z16')"       # 0 - 189
#
# COLT (int circulators)
# "job='F'; group='g5760'; runs=('z4')"            # 0 - 188
# "job='F'; group='g5760'; runs=('noESF_z4')"      # 0 - 188
# "job='F'; group='g5760'; runs=('noRlim_z4')"     # 0 - 188
# "job='F'; group='g5760'; runs=('varEff_z4')"     # 0 - 188
# "job='G'; group='g10304'; runs=('noESF_z4')"     # 0 - 188
# "job='G'; group='g10304'; runs=('z4')"           # 0 - 188
# "job='H'; group='g33206'; runs=('uvb_z8')"       # 0 - 188
# "job='H'; group='g33206'; runs=('noExt_z8')"     # 0 - 188
# "job='H'; group='g33206'; runs=('noRlim_z8')"    # 0 - 188
# "job='H'; group='g33206'; runs=('noESF_z8')"     # 0 - 188
# "job='H'; group='g33206'; runs=('varEff_z8')"    # 0 - 188
# "job='H'; group='g33206'; runs=('z8')"           # 0 - 188
# "job='H'; group='g33206'; runs=('noESF_z4')"     # 0 - 188
# "job='H'; group='g33206'; runs=('noRlim_z4')"    # 0 - 189
# "job='H'; group='g33206'; runs=('varEff_z4')"    # 0 - 189
# "job='H'; group='g33206'; runs=('z4')"           # 0 - 189
# "job='I'; group='g37591'; runs=('uvb_z8')"       # 0 - 188
# "job='I'; group='g37591'; runs=('noExt_z8')"     # 0 - 188
# "job='I'; group='g37591'; runs=('z8')"           # 0 - 188
# "job='I'; group='g37591'; runs=('z4')"           # 0 - 188
# "job='J'; group='g137030'; runs=('uvb_z8')"      # 0 - 188
# "job='J'; group='g137030'; runs=('noExt_z8')"    # 0 - 188
# "job='J'; group='g137030'; runs=('noRlim_z8')"   # 0 - 188
# "job='J'; group='g137030'; runs=('noESF_z8')"    # 0 - 188
# "job='J'; group='g137030'; runs=('varEff_z8')"   # 0 - 188
# "job='J'; group='g137030'; runs=('z8')"          # 0 - 188
# "job='J'; group='g137030'; runs=('noESF_z4')"    # 0 - 188
# "job='J'; group='g137030'; runs=('noRlim_z4')"   # 0 - 189
# "job='J'; group='g137030'; runs=('varEff_z4')"   # 0 - 188
# "job='J'; group='g137030'; runs=('z4')"          # 0 - 188
# "job='K'; group='g500531'; runs=('uvb_z8')"      # 0 - 189
# "job='K'; group='g500531'; runs=('noExt_z8')"    # 0 - 188
# "job='K'; group='g500531'; runs=('z8')"          # 0 - 188
# "job='K'; group='g500531'; runs=('z4')"          # 0 - 189
# "job='L'; group='g519761'; runs=('uvb_z8')"      # 0 - 189
# "job='L'; group='g519761'; runs=('noExt_z8')"    # 0 - 189
# "job='L'; group='g519761'; runs=('z8')"          # 0 - 189
# "job='L'; group='g519761'; runs=('z4')"          # 0 - 189
# "job='M'; group='g2274036'; runs=('uvb_z8')"     # 0 - 188
# "job='M'; group='g2274036'; runs=('noExt_z8')"   # 0 - 189
# "job='M'; group='g2274036'; runs=('z8')"         # 0 - 189
# "job='M'; group='g2274036'; runs=('z4')"         # 0 - 189
# "job='N'; group='g5229300'; runs=('z16')"        # 0 - 189
# "job='N'; group='g5229300'; runs=('uvb_z8')"     # 0 - 189
# "job='N'; group='g5229300'; runs=('noExt_z8')"   # 0 - 189
# "job='N'; group='g5229300'; runs=('z8')"         # 0 - 189
# "job='N'; group='g5229300'; runs=('z4')"         # 0 - 189
)

copy_dir=$PWD
zoom_dir=/orcd/data/mvogelsb/004/Thesan-Zooms
base_dir=${zoom_dir}-COLT

# Loop through job sets
for job_set in "${job_sets[@]}"; do
  eval "$job_set"  # Evaluate the job set to extract job, group, and runs

  for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    export group run
    # sbatch --job-name=${job}_${run} --output=x_${job}_${run}_%A.out job.sh
    sbatch --job-name=${job}_${run} --output=y_${job}_${run}_%a_%A.out job.sh
  done
done
