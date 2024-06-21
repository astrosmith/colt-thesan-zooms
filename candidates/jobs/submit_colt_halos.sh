#!/bin/bash

# Bash list of simulations to run
# job='F'; group='g5760'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4')
# job='G'; group='g10304'; runs=('z4' 'noESF_z4')
# job='H'; group='g33206'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uvb_z8')
# job='I'; group='g37591'; runs=('z4' 'z8' 'uvb_z8')
# job='J'; group='g137030'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'uvb_z8')
# job='K'; group='g500531'; runs=('z4' 'z8' 'uvb_z8')
# job='L'; group='g519761'; runs=('z4' 'z8' 'uvb_z8')
# job='M'; group='g2274036'; runs=('z4' 'z8' 'uvb_z8')
# job='N'; group='g5229300'; runs=('z4' 'z8' 'uvb_z8')

# job='F'; group='g5760'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4')
# job='G'; group='g10304'; runs=('z4' 'noESF_z4')
# job='H'; group='g33206'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'noExt_z8' 'uvb_z8')
# job='I'; group='g37591'; runs=('z4' 'z8' 'uvb_z8')
job='J'; group='g137030'; runs=('z4' 'varEff_z4' 'noRlim_z4' 'noESF_z4' 'z8' 'varEff_z8' 'noESF_z8' 'noRlim_z8' 'uvb_z8')
# job='K'; group='g500531'; runs=('z4' 'z8' 'uvb_z8')
# job='L'; group='g519761'; runs=('z4' 'z8' 'uvb_z8')
# job='M'; group='g2274036'; runs=('z4' 'z8' 'uvb_z8')
# job='N'; group='g5229300'; runs=('z4' 'z8' 'uvb_z8')

copy_dir=$PWD
zoom_dir=/orcd/data/mvogelsb/004/Thesan-Zooms
base_dir=${zoom_dir}-COLT

for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    cd $copy_dir  # Reset path
    colt_dir=${base_dir}/${group}/${run}
    cp config-halo-proj-RHD.yaml ${colt_dir}/.
    cp config-halos-ion-eq-RHD.yaml ${colt_dir}/.
    cp job_colt_halos.sh ${colt_dir}/.
    cd ${colt_dir}
    pwd

    export group run
    sbatch -J ${job}_${run} job_colt_halos.sh
done
