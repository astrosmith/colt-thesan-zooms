#!/bin/bash

# Define job sets with individual job, group, and runs
job_sets=(
# COLT (long circulators)
# "job='A'; group='g2'; runs=('z4'); first_snap=0; last_snap=67"                 # 0-67   ->  0-67
# "job='B'; group='g39'; runs=('z4'); first_snap=0; last_snap=188"               # 0-188  ->  0-188
# "job='C'; group='g205'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188  [2e9]
# "job='D'; group='g578'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188  [2e9]
## "job='E'; group='g1163'; runs=('z4'); first_snap=0; last_snap=188"             # 0-188  ->  0-188  [2e9]
# "job='F'; group='g5760'; runs=('z8'); first_snap=0; last_snap=188"             # 0-188  ->  0-188  [1e9]
## "job='G'; group='g10304'; runs=('z8'); first_snap=0; last_snap=188"            # 0-188  ->  0-188  [1e9]
## "job='J'; group='g137030'; runs=('z16'); first_snap=5; last_snap=188"          # 0-188  ->  5-188  [1e9]
## "job='K'; group='g500531'; runs=('z16'); first_snap=9; last_snap=188"          # 0-188  ->  9-188  [8e8]
## "job='L'; group='g519761'; runs=('z16'); first_snap=39; last_snap=188"         # 0-188  ->  39-188 [6e8]
## "job='M'; group='g2274036'; runs=('z16'); first_snap=9; last_snap=188"         # 0-189  ->  9-188  [4e8]
#
# COLT (int circulators)
# "job='F'; group='g5760'; runs=('z4'); first_snap=5; last_snap=188"             # 0-188  ->  5-188  [8e8]
# "job='G'; group='g10304'; runs=('z4'); first_snap=0; last_snap=188"            # 0-188  ->  0-188  [8e8]
# "job='H'; group='g33206'; runs=('z8'); first_snap=4; last_snap=188"            # 0-188  ->  4-188  [4e8]
# "job='H'; group='g33206'; runs=('z4'); first_snap=6; last_snap=189"            # 0-189  ->  6-189  [4e8]
# "job='I'; group='g37591'; runs=('z8'); first_snap=6; last_snap=188"            # 0-188  ->  6-188  [4e8]
# "job='I'; group='g37591'; runs=('z4'); first_snap=11; last_snap=188"           # 0-188  ->  11-188 [4e8]
# "job='J'; group='g137030'; runs=('z8'); first_snap=2; last_snap=188"           # 0-188  ->  2-188  [2e8]
# "job='J'; group='g137030'; runs=('z4'); first_snap=10; last_snap=188"          # 0-188  ->  10-188 [2e8]
# "job='K'; group='g500531'; runs=('z8'); first_snap=10; last_snap=188"          # 0-188  ->  10-188 [1e8]
# "job='K'; group='g500531'; runs=('z4'); first_snap=16; last_snap=189"          # 0-189  ->  16-189 [1e8]
# "job='L'; group='g519761'; runs=('z8'); first_snap=46; last_snap=189"          # 0-189  ->  46-189 [1e8]
# "job='L'; group='g519761'; runs=('z4'); first_snap=51; last_snap=189"          # 0-189  ->  51-189 [1e8]
# "job='M'; group='g2274036'; runs=('z8'); first_snap=27; last_snap=189"         # 0-189  ->  27-189 [1e8]
# "job='M'; group='g2274036'; runs=('z4'); first_snap=52; last_snap=189"         # 0-189  ->  52-189 [1e8]
# "job='N'; group='g5229300'; runs=('z16'); first_snap=72; last_snap=189"        # 0-189  ->  72-189 [1e8]
# "job='N'; group='g5229300'; runs=('z8'); first_snap=83; last_snap=189"         # 0-189  ->  83-189 [1e8]
# "job='N'; group='g5229300'; runs=('z4'); first_snap=87; last_snap=189"         # 0-189  ->  87-189 [1e8]
)

# Loop through job sets
for job_set in "${job_sets[@]}"; do
  eval "$job_set"  # Evaluate the job set to extract job, group, and runs

  # Loop through the run directories
  for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    export group run
    sbatch -J ${job}_${run} job.sh
  done
done

