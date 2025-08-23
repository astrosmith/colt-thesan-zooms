#!/bin/bash

# Python list of simulations to run
# sims = {
#   'g39': ['z4'],
#   'g205': ['z4'],
#   'g578': ['z4'],
#   'g1163': ['z4', 'noESF_z4'],
#   'g5760': ['z4', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'z8'],
#   'g10304': ['z4', 'noESF_z4', 'z8'],
#   'g33206': ['z4', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'z8', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'noExt_z8', 'uvb_z8'],
#   'g37591': ['z4', 'z8', 'uvb_z8'],
#   'g137030': ['z4', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'z8', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'noExt_z8', 'uvb_z8'],
#   'g500531': ['z4', 'z8', 'uvb_z8'],
#   'g519761': ['z4', 'z8', 'uvb_z8'],
#   'g2274036': ['z4', 'z8', 'uvb_z8'],
#   'g5229300': ['z4', 'z8', 'uvb_z8'],
# }

# Fiducial test case
# job='F'; group='g5760'; runs=('z4')

# Bash list of simulations to run
# job='B'; group='g39'; runs=('z4')
# job='C'; group='g205'; runs=('z4')
# job='D'; group='g578'; runs=('z4')
# job='E'; group='g1163'; runs=('z4' 'noESF_z4')
# job='F'; group='g5760'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8')
# job='G'; group='g10304'; runs=('z4' 'noESF_z4' 'z8')
# job='H'; group='g33206'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uvb_z8')
# job='I'; group='g37591'; runs=('z4' 'z8' 'noExt_z8' 'uvb_z8')
# job='J'; group='g137030'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uvb_z8')
# job='K'; group='g500531'; runs=('z4' 'z8' 'noExt_z8' 'uvb_z8')
# job='L'; group='g519761'; runs=('z4' 'z8' 'noExt_z8' 'uvb_z8')
# job='M'; group='g2274036'; runs=('z4' 'z8' 'noExt_z8' 'uvb_z8')
# job='N'; group='g5229300'; runs=('z4' 'z8' 'noExt_z8' 'uvb_z8')

# Define job sets with individual job, group, and runs
job_sets=(
# COLT (long circulators)
#"job='D'; group='g578'; runs=('noESF_z4'); first_snap=88; last_snap=179"       # 88-179 ->  88-179 [2e9]
#"job='D'; group='g578'; runs=('varEff_z4'); first_snap=0; last_snap=68"        # 0-68   ->  0-68
#"job='E'; group='g1163'; runs=('noESF_z4'); first_snap=0; last_snap=188"       # 0-188  ->  0-188  [2e9]
#"job='H'; group='g33206'; runs=('z16'); first_snap=0; last_snap=8"             # 0-8    ->  0-8
# Fiducial:
#X"job='A'; group='g2'; runs=('z4'); first_snap=0; last_snap=67"                 # 0-67   ->  0-67
#X"job='B'; group='g39'; runs=('z4'); first_snap=0; last_snap=188"               # 0-188  ->  0-188
# "job='C'; group='g205'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188  [2e9] long circ 128-188
# "job='D'; group='g578'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188  [2e9] long circ 170-188
# "job='E'; group='g1163'; runs=('z4'); first_snap=0; last_snap=188"             # 0-188  ->  0-188  [2e9]
# "job='F'; group='g5760'; runs=('z8'); first_snap=0; last_snap=188"             # 0-188  ->  0-188  [1e9] long circ 155-188
# "job='G'; group='g10304'; runs=('z8'); first_snap=0; last_snap=188"            # 0-188  ->  0-188  [1e9]
# "job='J'; group='g137030'; runs=('z16'); first_snap=5; last_snap=188"          # 0-188  ->  5-188  [1e9]
# "job='K'; group='g500531'; runs=('z16'); first_snap=9; last_snap=188"          # 0-188  ->  9-188  [8e8]
# "job='L'; group='g519761'; runs=('z16'); first_snap=39; last_snap=188"         # 0-188  ->  39-188 [6e8]
# "job='M'; group='g2274036'; runs=('z16'); first_snap=9; last_snap=188"         # 0-189  ->  9-188  [4e8]
#
# COLT (int circulators)
#"job='F'; group='g5760'; runs=('noESF_z4'); first_snap=5; last_snap=188"       # 0-188  ->  5-188  [8e8]
#"job='G'; group='g10304'; runs=('noESF_z4'); first_snap=0; last_snap=188"      # 0-188  ->  0-188  [8e8]
#"job='H'; group='g33206'; runs=('noESF_z8'); first_snap=4; last_snap=188"      # 0-188  ->  4-188  [4e8]
#"job='H'; group='g33206'; runs=('noESF_z4'); first_snap=6; last_snap=188"      # 0-188  ->  6-188  [4e8]
#"job='J'; group='g137030'; runs=('noESF_z8'); first_snap=2; last_snap=188"     # 0-188  ->  2-188  [2e8]
#"job='J'; group='g137030'; runs=('noESF_z4'); first_snap=10; last_snap=188"    # 0-188  ->  10-188 [2e8]
#
# "job='F'; group='g5760'; runs=('noRlim_z4'); first_snap=5; last_snap=188"      # 0-188  ->  5-188
# "job='F'; group='g5760'; runs=('varEff_z4'); first_snap=5; last_snap=188"      # 0-188  ->  5-188
# "job='H'; group='g33206'; runs=('uvb_z8'); first_snap=4; last_snap=188"        # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('noExt_z8'); first_snap=4; last_snap=188"      # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('noRlim_z8'); first_snap=4; last_snap=188"     # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('varEff_z8'); first_snap=3; last_snap=188"     # 0-188  ->  3-188
# "job='H'; group='g33206'; runs=('noRlim_z4'); first_snap=6; last_snap=189"     # 0-189  ->  6-189
# "job='H'; group='g33206'; runs=('varEff_z4'); first_snap=6; last_snap=189"     # 0-189  ->  6-189
# "job='I'; group='g37591'; runs=('uvb_z8'); first_snap=7; last_snap=188"        # 0-188  ->  7-188
# "job='I'; group='g37591'; runs=('noExt_z8'); first_snap=6; last_snap=188"      # 0-188  ->  6-188
# "job='J'; group='g137030'; runs=('uvb_z8'); first_snap=3; last_snap=188"       # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noExt_z8'); first_snap=3; last_snap=188"     # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noRlim_z8'); first_snap=2; last_snap=188"    # 0-188  ->  2-188
# "job='J'; group='g137030'; runs=('varEff_z8'); first_snap=3; last_snap=188"    # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noRlim_z4'); first_snap=10; last_snap=189"   # 0-189  ->  10-189
# "job='J'; group='g137030'; runs=('varEff_z4'); first_snap=10; last_snap=188"   # 0-188  ->  10-188
# "job='K'; group='g500531'; runs=('uvb_z8'); first_snap=8; last_snap=189"       # 0-189  ->  8-189
# "job='K'; group='g500531'; runs=('noExt_z8'); first_snap=8; last_snap=188"     # 0-188  ->  8-188
# "job='L'; group='g519761'; runs=('uvb_z8'); first_snap=23; last_snap=189"      # 0-189  ->  23-189
# "job='L'; group='g519761'; runs=('noExt_z8'); first_snap=46; last_snap=189"    # 0-189  ->  46-189
# "job='M'; group='g2274036'; runs=('uvb_z8'); first_snap=22; last_snap=182"     # 0-189  ->  22-182
# "job='M'; group='g2274036'; runs=('noExt_z8'); first_snap=28; last_snap=189"   # 0-189  ->  28-189
# "job='N'; group='g5229300'; runs=('uvb_z8'); first_snap=0; last_snap=0"        # 0-189  ->  0-0
# "job='N'; group='g5229300'; runs=('noExt_z8'); first_snap=0; last_snap=0"      # 0-189  ->  0-0
# Fiducial:
# "job='F'; group='g5760'; runs=('z4'); first_snap=5; last_snap=188"             # 0-188  ->  5-188  [8e8]
"job='G'; group='g10304'; runs=('z4'); first_snap=0; last_snap=188"            # 0-188  ->  0-188  [8e8]
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

copy_dir=$PWD
zoom_dir=/orcd/data/mvogelsb/004/Thesan-Zooms
base_dir=${zoom_dir}-COLT
#base_dir=/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT

# Loop through job sets
for job_set in "${job_sets[@]}"; do
  eval "$job_set"  # Evaluate the job set to extract job, group, and runs

  # Loop through the run directories
  for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    cd $copy_dir  # Reset path
    post_dir=${zoom_dir}/${group}/${run}/postprocessing
    colt_dir=${base_dir}/${group}/${run}

    cp config-tree*.yaml ${colt_dir}/.
    cp job-tree.sh ${colt_dir}/.

    cd ${colt_dir}
    pwd

    export group run
    sbatch -J ${job}_${run} job-tree.sh
  done
done

