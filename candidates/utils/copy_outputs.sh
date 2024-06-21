#!/bin/bash

# Define the base directory
zoom_dir=/orcd/data/mvogelsb/004/Thesan-Zooms
colt_dir=${zoom_dir}-COLT

# List of simulations to run
declare -A simulations
simulations=(
    ["g39"]="z4"
    ["g205"]="z4"
    ["g578"]="z4"
    ["g1163"]="z4 noESF_z4"
    ["g5760"]="z4 varEff_z4 noRlim_z4 noESF_z4 z8"
    ["g10304"]="z4 noESF_z4 z8"
    ["g33206"]="z4 varEff_z4 noRlim_z4 noESF_z4 z8 varEff_z8 noESF_z8 noRlim_z8 noExt_z8 uvb_z8"
    ["g37591"]="z4 z8 uvb_z8"
    ["g137030"]="z4 varEff_z4 noRlim_z4 noESF_z4 z8 varEff_z8 noESF_z8 noRlim_z8 uvb_z8"
    ["g500531"]="z4 z8 uvb_z8"
    ["g519761"]="z4 z8 uvb_z8"
    ["g2274036"]="z4 z8 uvb_z8"
    ["g5229300"]="z4 z8 uvb_z8"
)

# Create directories and move files
for group in "${!simulations[@]}"; do
    runs=${simulations[$group]}
    for run in $runs; do
        # Create the target directory
        target_dir=${zoom_dir}/${group}/${run}/postprocessing/proj
        mkdir -p $target_dir
        
        # Move the files
        echo "mv ${colt_dir}/${group}/${run}/output/halo_proj_RHD* $target_dir"
        mv ${colt_dir}/${group}/${run}/output/halo_proj_RHD* $target_dir
    done
done

echo "Directories created and files moved successfully."

