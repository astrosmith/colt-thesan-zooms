#!/bin/bash

# SBATCH --job-name=colt
# SBATCH --output=colt-%a-%A.out
# SBATCH --output=ion-%a-%A.out
# SBATCH --output=M1500-%a-%A.out
# SBATCH --output=opt-%a-%A.out
# SBATCH --output=eq-%a-%A.out
# SBATCH --output=teq-%a-%A.out
# SBATCH --output=Ha-%a-%A.out
# SBATCH --output=dm-%a-%A.out
# SBATCH --output=proj-%a-%A.out
#SBATCH --output=z-%a-%A.out
# SBATCH --output=halos-%A.out
# long
# SBATCH --array=0-67   # A [2e9] (g2/z4)
# SBATCH --array=0-188  # B [2e9] (g39/z4)
# SBATCH --array=0-188  # C [2e9] (g205/z4)
# SBATCH --array=0-188  # D [2e9] (g578/z4)
# SBATCH --array=0-188  # E [2e9] (g1163/z4)
# SBATCH --array=0-188  # F [1e9] (g5760/z8)
# SBATCH --array=0-188  # G [1e9] (g10304/z8)
# SBATCH --array=5-188  # J [1e9] (g137030/z16)
# SBATCH --array=9-188  # K [8e8] (g500531/z16)
# SBATCH --array=39-188 # L [6e8] (g519761/z16)
# SBATCH --array=9-188  # M [4e8] (g2274036/z16)
# int
# SBATCH --array=5-188   # F [8e8] (g5760/z4)
# SBATCH --array=0-188   # G [8e8] (g10304/z4)
# SBATCH --array=4-188   # H [4e8] (g33206/z8)
# SBATCH --array=6-189   # H [4e8] (g33206/z4)
# SBATCH --array=6-188   # I [4e8] (g37591/z8)
# SBATCH --array=11-188  # I [4e8] (g37591/z4)
# SBATCH --array=2-188   # J [2e8] (g137030/z8)
# SBATCH --array=10-188  # J [2e8] (g137030/z4)
# SBATCH --array=10-188  # K [1e8] (g500531/z8)
# SBATCH --array=16-189  # K [1e8] (g500531/z4)
# SBATCH --array=46-189  # L [1e8] (g519761/z8)
# SBATCH --array=51-189  # L [1e8] (g519761/z4)
# SBATCH --array=27-189  # M [1e8] (g2274036/z8)
# SBATCH --array=52-189  # M [1e8] (g2274036/z4)
# SBATCH --array=72-189  # N [1e8] (g5229300/z16)
# SBATCH --array=83-189  # N [1e8] (g5229300/z8)
# SBATCH --array=87-189  # N [1e8] (g5229300/z4)
#SBATCH --array=188
#SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal,ou_mki_preempt,sched_mit_mki_preempt,mit_preemptable
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal
# SBATCH --partition=mit_normal,mit_preemptable
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal
# SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
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
#SBATCH --mem=0 # 6GB of memory per CPU
# SBATCH --mem-per-cpu=17500 # 17.5GB of memory per CPU
# SBATCH --mem-per-cpu=15500 # 15.5GB of memory per CPU
# SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
# SBATCH --mem-per-cpu=5000 # 5GB of memory per CPU
# SBATCH --mem-per-cpu=4000 # 4GB of memory per CPU
# SBATCH --mem-per-cpu=3000 # 3GB of memory per CPU
#SBATCH --export=ALL
# SBATCH --time=168:00:00
# SBATCH --time=96:00:00
# SBATCH --time=72:00:00
#SBATCH --time=48:00:00
# SBATCH --time=24:00:00
# SBATCH --time=12:00:00
# SBATCH --time=6:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
#SBATCH --exclude=node1620,node1431,node1432,node1453,node1454,node3909,node3910,node3911,node3908
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
# colt="${HOME}/colt/colt"
colt="${HOME}/colt/colt-long"
sim=${group}/${run}

setup_runs() {
    py="/home/arsmith/miniforge3/bin/python3 -u"
    echo "Setting up simulation ${sim} ..."
    ## time $py distances.py $sim $SLURM_ARRAY_TASK_ID         # Low/high-res distances
    ## time $py candidates.py $sim $SLURM_ARRAY_TASK_ID        # Uncontaminated candidates
    ## time $py halos.py $sim $SLURM_ARRAY_TASK_ID             # Write halos to file
    ## time $py arepo_to_colt.py $sim $SLURM_ARRAY_TASK_ID     # Arepo to COLT initial conditions
    # time $py arepo_to_dm.py $sim $SLURM_ARRAY_TASK_ID       # Arepo to COLT initial conditions
    ## $call $colt config-connect.yaml $SLURM_ARRAY_TASK_ID       # Run COLT to get connections (without circulators)
    ## time $py remove_lowres.py $sim $SLURM_ARRAY_TASK_ID     # Remove exterior low-res particles
    ## $call $colt config-connect-circ.yaml $SLURM_ARRAY_TASK_ID  # Rerun for updated connections (with circulators)
    # for snap in {0..189}; do
    #     # echo "Running halo ${snap} ..."
    #     $py candidates.py $sim $snap
    #     $py halos.py $sim $snap
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

# setup_runs

# Global runs
# $call $colt-ion config-ion-eq-pre7.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-ion config-ion-eq-pre8.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-ion config-ion-eq-pre.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-ion config-ion-eq-pre0.yaml $SLURM_ARRAY_TASK_ID

# Halo runs
# $call $colt-M1500 config-halo-M1500.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-M1500 config-halo-M1500_fdust_10.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-M1500 config-halo-M1500_fdust_20.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-M1500 config-halo-M1500_fdust_40.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-Ha.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OII-3727-3730.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OIII-5008.yaml $SLURM_ARRAY_TASK_ID
# Halo runs (quick intrinsic emission only)
# $call $colt-line config-halo-Ha-int.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OII-3727-3730-int.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-line config-halo-OIII-5008-int.yaml $SLURM_ARRAY_TASK_ID

$call $colt-flows config-halo-ion-eq-RHD.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-RHD-40.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-RHD-20.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-RHD-10.yaml $SLURM_ARRAY_TASK_ID

# $call $colt-flows config-halo-ion-eq.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-40.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-20.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-flows config-halo-ion-eq-10.yaml $SLURM_ARRAY_TASK_ID

# Temperature equilibrium runs
#$call $colt-ion-teq config-ion-eq-pre7-teq.yaml $SLURM_ARRAY_TASK_ID
#$call $colt-ion-teq config-ion-eq-pre8-teq.yaml $SLURM_ARRAY_TASK_ID

# run ion-eq
# run ion-eq-MCRT ion-eq-RHD
# run proj Ha Lya
# run halo-ion-eq-RHD
#run halo-M1500
#run halo-optical
#run halo-Ha-RHD
# run Lya-RHD
# run halo-Ha halo-OII-3727-3730 halo-OIII-5008
# run halo-OII-3727-3730
# run halo-Ha halo-OIII-5008
# run OII-3727-3730 OIII-5008 M1500
#run ion-eq-pre7-teq ion-eq-pre8-teq
# run halo-Ha-teq halo-OIII-5008-teq

