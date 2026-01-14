#!/bin/bash

# SBATCH --job-name=colt
#SBATCH --output=z-%a-%A.out
# SBATCH --array=0-188
# SBATCH --array=0-94 # B
# SBATCH --array=0-125 # C
# SBATCH --array=0-166 # D
# SBATCH --array=8-188 # E
# SBATCH --array=4-154 # F
# SBATCH --array=0-188 # G
# SBATCH --array=5-188 # J
# SBATCH --array=13-188 # K
# SBATCH --array=39-188 # L
# SBATCH --array=57-188 # M
# SBATCH --array=13-188 # K (including no stars)
# SBATCH --array=9-188 # M (including no stars)
# SBATCH --array=95-188 # B long
# SBATCH --array=126-188 # C long
# SBATCH --array=167-188 # D long
# SBATCH --array=155-188 # F long
# SBATCH --array=5-188  # F/z4
# SBATCH --array=0-188  # G/z4
# SBATCH --array=4-188  # H/z8
# SBATCH --array=6-188  # H/z4
# SBATCH --array=6-188  # I/z8
# SBATCH --array=11-188 # I/z4
# SBATCH --array=2-188  # J/z8
# SBATCH --array=10-188 # J/z4
# SBATCH --array=10-188 # K/z8
# SBATCH --array=16-188 # K/z4
# SBATCH --array=46-188 # L/z8
# SBATCH --array=51-188 # L/z4
# SBATCH --array=27-188 # M/z8
# SBATCH --array=52-188 # M/z4
# SBATCH --array=72-188 # N/z16
# SBATCH --array=83-188 # N/z8
# SBATCH --array=87-188 # N/z4
# SBATCH --array=0-188
# SBATCH --array=0-189
# SBATCH --array=
# SBATCH --partition=newnodes,mit_quicktest,sched_any,sched_engaging_default,mit_normal,mit_data_transfer,mit_preemptable
#SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal,ou_mki_preempt,sched_mit_mki_preempt,mit_preemptable
# SBATCH --partition=ou_mki,sched_mit_mki,mit_normal,ou_mki_preempt,sched_mit_mki_preempt,mit_preemptable
# SBATCH --partition=mit_quicktest
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal
# SBATCH --partition=sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mki_r8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# SBATCH --cpus-per-task=32
# SBATCH --cpus-per-task=16
# SBATCH --cpus-per-task=8
# SBATCH --constraint=centos7
#SBATCH --constraint=rocky8
# SBATCH --mem=0 # 6GB of memory per CPU
#SBATCH --mem-per-cpu=4000 # 4GB of memory per CPU
#SBATCH --export=ALL
# SBATCH --time=96:00:00
# SBATCH --time=48:00:00
#SBATCH --time=24:00:00
# SBATCH --time=4:00:00
# SBATCH --time=6:00:00
# SBATCH --time=0:15:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
#SBATCH --exclude=node1620
# SBATCH --exclude=node1400,node1406,node1414,node1418,node1454,node1456
# node1400,node1406,node1407,node1408,node1409,node1420,node1422,node1423,node1424,node1427,node1436,node1438,node1439,node1441,node1444,node1450
# ###SBATCH --exclude=node1412,node1413,node1415,node1421,node1426,node1454,node1455,node1447
# SBATCH --dependency=afterok:12345678

## Module setup
. /etc/profile.d/modules.sh
source ~/.colt

## OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

call="mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl ^openib --mca psm2 ucx -np $SLURM_NTASKS --bind-to none"
colt="${HOME}/colt/colt"
# colt="${HOME}/colt/colt-long"
sim=${group}/${run}

setup_runs() {
    echo "Setting up simulation ${sim} ..."
    ## time python distances.py $sim $SLURM_ARRAY_TASK_ID         # Low/high-res distances
    ## time python candidates.py $sim $SLURM_ARRAY_TASK_ID        # Uncontaminated candidates
    ## time python halos.py $sim $SLURM_ARRAY_TASK_ID             # Write halos to file
    ## time python arepo_to_colt.py $sim $SLURM_ARRAY_TASK_ID     # Arepo to COLT initial conditions
    ## $call $colt config-connect.yaml $SLURM_ARRAY_TASK_ID       # Run COLT to get connections (without circulators)
    ## time python remove_lowres.py $sim $SLURM_ARRAY_TASK_ID     # Remove exterior low-res particles
    ## $call $colt config-connect-circ.yaml $SLURM_ARRAY_TASK_ID  # Rerun for updated connections (with circulators)
    # for snap in {0..189}; do
    #     # echo "Running halo ${snap} ..."
    #     python candidates.py $sim $snap
    #     python halos.py $sim $snap
    # done
}

run() {
    for suffix in "$@"; do
        config_file="config-${suffix}.yaml"
        #echo "Running simulation with ${config_file} ..."
        #$call $colt $config_file $SLURM_ARRAY_TASK_ID
        # $call $colt-flows $config_file $SLURM_ARRAY_TASK_ID
        # $call $colt-ion $config_file $SLURM_ARRAY_TASK_ID
        $call $colt-line $config_file $SLURM_ARRAY_TASK_ID
        # $call $colt-ion-teq $config_file $SLURM_ARRAY_TASK_ID
        # $call $colt-line-teq $config_file $SLURM_ARRAY_TASK_ID
    done
}

# $call $colt-line config-halo-Ha.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OII-3727-3730.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OIII-5008.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq.yaml $SLURM_ARRAY_TASK_ID

# $call $colt-tree-proj config-proj-rho.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-proj config-proj-all.yaml $SLURM_ARRAY_TASK_ID

# $call $colt-tree-flows config-M1500.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-optical.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-ion-eq-RHD.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-ion-eq.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-Ha-RHD.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-Ha.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-Ha-cont.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-Hb.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-OIII-5008.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-OII-3727-3730.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-flows config-SiII-1190.yaml $SLURM_ARRAY_TASK_ID

# # TODO remove min_HI_bin_cdf # $call $colt-tree-flows config-ion-eq-full.yaml $SLURM_ARRAY_TASK_ID

# $call $colt-tree-Lya config-Lya-cont.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-Lya config-Lya.yaml $SLURM_ARRAY_TASK_ID
