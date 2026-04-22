#!/bin/bash

# SBATCH --job-name=colt
#SBATCH --output=movie-%a-%A.out
# SBATCH --array=?-133 # B long
# SBATCH --array=103-133 # C long
# SBATCH --array=123-133 # D long
# SBATCH --array=119-133 # F long
# SBATCH --array=0-399
#SBATCH --array=151-267
# SBATCH --array=0-133
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,mit_normal,ou_mki_preempt,sched_mit_mki_preempt,mit_preemptable
# SBATCH --partition=sched_mit_mvogelsb,sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=sched_mit_mki,sched_mit_mki_preempt
# SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki,ou_mki_preempt,sched_mit_mki_preempt
#SBATCH --partition=ou_mki,sched_mit_mvogelsb,sched_mit_mki
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
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
# SBATCH --mem-per-cpu=3000 # 3GB of memory per CPU
#SBATCH --export=ALL
# SBATCH --time=168:00:00
# SBATCH --time=48:00:00
#SBATCH --time=24:00:00
# SBATCH --time=12:00:00
# SBATCH --time=6:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=arsmith@mit.edu
# SBATCH --exclude=node1447,node1445,node1454,node1456
# SBATCH --exclude=node1400,node1406,node1414,node1418,node1454,node1456
# node1400,node1406,node1407,node1408,node1409,node1420,node1422,node1423,node1424,node1427,node1436,node1438,node1439,node1441,node1444,node1450
# ###SBATCH --exclude=node1412,node1413,node1415,node1421,node1426,node1454,node1455,node1447
# SBATCH --dependency=afterany:5634629
# SBATCH --dependency=afterok:5634629

## Module setup
. /etc/profile.d/modules.sh
source ~/.colt

## OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

call="mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl ^openib --mca psm2 ucx -np $SLURM_NTASKS --bind-to none"
colt="${HOME}/colt/colt"
# colt="${HOME}/colt/colt-long"
#
# colt="${HOME}/colt/colt-flows"
# colt="${HOME}/colt/colt-long-flows"
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

# $call $colt-tree-proj config-movie-proj-rho.yaml $SLURM_ARRAY_TASK_ID
# $call $colt-tree-proj config-movie-proj-all.yaml $SLURM_ARRAY_TASK_ID

configs=(
    "config-movie-proj-rho.yaml"
    # "config-movie-proj-rho2.yaml"
    "config-movie-proj-all.yaml"
)

for config in "${configs[@]}"; do
    $call $colt-tree-proj $config $SLURM_ARRAY_TASK_ID
done

# for config in "${configs[@]}"; do
#     if [ $SLURM_ARRAY_TASK_ID -gt 0 ]; then
#         for sub in 0 1 2 3 4 5 6 7 8 9; do
#             $call $colt-tree-proj $config $SLURM_ARRAY_TASK_ID$sub
#         done
#     else
#         for sub in 0 1 2 3 4 5 6 7 8 9; do
#             $call $colt-tree-proj $config $sub
#         done
#         # Remainder files
#         # for sub in 0 1 2 3 4 5; do
#         for sub in 0 1 2 3 4 5 6 7 8; do
#             $call $colt-tree-proj $config 134$sub
#         done
#     fi
# done

# for config in "${configs[@]}"; do
#     for sub in 0 1 2 3 4 5 6 7 8 9; do
#         snap=$SLURM_ARRAY_TASK_ID$sub
#         # if [ $snap -ge 11111111 ]; then # B long
#         if [ $snap -ge 1034 ]; then # C long
#         # if [ $snap -ge 1239 ]; then # D long
#         # if [ $snap -ge 1197 ]; then # F long
#             $call $colt-long-tree-proj $config $snap
#         fi
#     done
# done

# $call $colt-long-tree-proj config-movie-proj-360.yaml 482

# for config in "${configs[@]}"; do
#     # Remainder files
#     # for sub in 0 1 2 3 4 5; do
#     for sub in 0 1 2 3 4 5 6 7 8; do
#         $call $colt-long-tree-proj $config 134$sub
#     done
# done
