#!/bin/bash

# Python list of simulations to run
# sims = {
#   'g39': ['z4'],
#   'g205': ['z4'],
#   'g578': ['z4'],
#   'g1163': ['z4', 'noESF_z4'],
#   'g5760': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4'],
#   'g10304': ['z4', 'z8', 'noESF_z4'],
#   'g33206': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'noExt_z8', 'uvb_z8'],
#   'g37591': ['z4', 'z8', 'uvb_z8'],
#   'g137030': ['z4', 'z8', 'varEff_z4', 'noRlim_z4', 'noESF_z4', 'varEff_z8', 'noESF_z8', 'noRlim_z8', 'uvb_z8'],
#   'g500531': ['z4', 'z8', 'uvb_z8'],
#   'g519761': ['z4', 'z8', 'uvb_z8'],
#   'g2274036': ['z4', 'z8', 'uvb_z8'],
#   'g5229300': ['z4', 'z8', 'uvb_z8'],
# }

# Bash list of simulations to run
# job='B'; group='g39'; runs=('z4')
# job='C'; group='g205'; runs=('z4')
# job='D'; group='g578'; runs=('z4')
# job='E'; group='g1163'; runs=('z4' 'noESF_z4')
# job='F'; group='g5760'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4')
# job='G'; group='g10304'; runs=('z4' 'noESF_z4')
# job='H'; group='g33206'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uvb_z8')
# job='I'; group='g37591'; runs=('z4' 'z8' 'uvb_z8')
# job='J'; group='g137030'; runs=('z4' 'z8' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'uvb_z8')
# job='K'; group='g500531'; runs=('z4' 'z8' 'uvb_z8')
# job='L'; group='g519761'; runs=('z4' 'z8' 'uvb_z8')
# job='M'; group='g2274036'; runs=('z4' 'z8' 'uvb_z8')
# job='N'; group='g5229300'; runs=('z4' 'z8' 'uvb_z8')

# Fiducial test case
# job='F'; group='g5760'; runs=('z4')

copy_dir=$PWD
zoom_dir=/orcd/data/mvogelsb/004/Thesan-Zooms
base_dir=${zoom_dir}-COLT

for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    cd $copy_dir  # Reset path
    post_dir=${zoom_dir}/${group}/${run}/postprocessing
    mkdir -p ${post_dir}/distances
    mkdir -p ${post_dir}/candidates
    colt_dir=${base_dir}/${group}/${run}
    mkdir -p ${colt_dir}/ics
    ls ${colt_dir}/ics

    cp config*.yaml ${colt_dir}/.
    cp ../distances.py ${colt_dir}/.
    cp ../candidates.py ${colt_dir}/.
    cp ../arepo_to_colt.py ${colt_dir}/.
    cp ../remove_lowres.py ${colt_dir}/.
    cp job.sh ${colt_dir}/.

    cd ${colt_dir}
    pwd

    export group run
    sbatch -J ${job}_${run} job.sh
done

