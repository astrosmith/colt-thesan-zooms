#!/bin/bash

# Python list of simulations to run
# sims = {
#   'g39': ['z4'],
#   'g205': ['z4'],
#   'g578': ['z4'],
#   'g1163': ['z4', 'noESF_z4'],
#   'g5760': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4'],
#   'g10304': ['z4', 'z8', 'noESF_z4'],
#   'g33206': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'noExt_z8', 'uniformUVB_z8'],
#   'g37591': ['z4', 'z8', 'uniformUVB_z8'],
#   'g137030': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'uniformUVB_z8'],
#   'g500531': ['z4', 'z8', 'uniformUVB_z8'],
#   'g519761': ['z4', 'z8', 'uniformUVB_z8'],
#   'g2274036': ['z4', 'z8', 'uniformUVB_z8'],
#   'g5229300': ['z4', 'z8', 'uniformUVB_z8'],
# }

# Bash list of simulations to run
# group='g39'; runs=('z4')
# group='g205'; runs=('z4')
# group='g578'; runs=('z4')
# group='g1163'; runs=('z4' 'noESF_z4')
# group='g5760'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4')
# group='g10304'; runs=('z4' 'noESF_z4')
# group='g33206'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uniformUVB_z8')
# group='g37591'; runs=('z4' 'z8' 'uniformUVB_z8')
# group='g137030'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'uniformUVB_z8')
# group='g500531'; runs=('z4' 'z8' 'uniformUVB_z8')
# group='g519761'; runs=('z4' 'z8' 'uniformUVB_z8')
# group='g2274036'; runs=('z4' 'z8' 'uniformUVB_z8')
# group='g5229300'; runs=('z4' 'z8' 'uniformUVB_z8')
group='g5229300'; runs=('z4')

# Fiducial test case
# group='g5760'; runs=('z4')

copy_dir=$PWD
base_dir=/orcd/data/mvogelsb/004/Thesan-Zooms-COLT

for run in "${runs[@]}"; do
    echo "Group ${group}, Run ${run}"
    cd $copy_dir  # Reset path
    colt_dir=${base_dir}/${group}/${run}
    mkdir -p ${colt_dir}/ics
    ls ${colt_dir}/ics

    cp config*.yaml ${colt_dir}/.
    cp job.sh ${colt_dir}/.

    cd ${colt_dir}
    pwd

    export group run
    sbatch -J ${group}_${run} job.sh
done

