#!/bin/bash

#sims = {
#  'g39': ['z4'],
#  'g205': ['z4'],
#  'g578': ['z4'],
#  'g1163': ['z4'],
#  'g5760': ['noESF', 'noRlim', 'varEff', 'z4', 'z8'],
#  'g10304': ['z4'],
#  'g33206': ['noESF', 'noESF_z8', 'noRlim', 'noRlim_z8', 'varEff', 'varEff_z8', 'z4', 'z8'],
#  'g37591': ['z4'],
#  'g137030': ['z4'],
#  'g500531': ['z4', 'z8'],
#  'g519761': ['z4'],
#  'g2274036': ['z4'],
#  'g5229300': ['z4', 'z8'],
#}

group='g5760'
#runs=('noESF' 'noRlim' 'varEff' 'z4' 'z8')
runs=('z4')
#runs=('z8')

#group='g33206'
#runs=('noESF' 'noESF_z8' 'noRlim' 'noRlim_z8' 'varEff' 'varEff_z8' 'z4' 'z8')
#runs=('noESF' 'noRlim' 'varEff' 'z4' 'z8')

copy_dir=$PWD
base_dir=/net/hstor001.ib/data2/group/mvogelsb/004/Thesan-Zooms-COLT

for run in "${runs[@]}"; do
    echo "Group ${group}, Run ${run}"
    cd $copy_dir
    colt_dir=${base_dir}/${group}/${run}
    ls ${colt_dir}/ics

    cp config*.yaml ${colt_dir}/.
    cp job.sh ${colt_dir}/.

    cd ${colt_dir}
    pwd

    sbatch job.sh
done

