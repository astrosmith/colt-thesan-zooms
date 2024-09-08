#!/bin/bash

# module load gcc cuda/11.3 openmpi hdf5

# Set flags
serial=false
mpi=false
gpu=false
gpu_mpi=false

# Override flags
# serial=true
# mpi=true
# gpu=true
gpu_mpi=true

# Build executable
compile() {
    rm -f $1_serial $1_mpi $1_gpu $1_gpu_mpi
    # Serial
    if [ "$serial" = true ]; then
        CFLAGS="-std=c++14"
        nvcc $CFLAGS $1.cu -o $1_serial -lhdf5 -lm
    fi
    # MPI
    if [ "$mpi" = true ]; then
        CFLAGS="-std=c++14 -DMPI"
        # Compile the CUDA source file to an object file
        nvcc $CFLAGS -c $1.cu -o $1.o -I/home/software/cuda/11.3/targets/x86_64-linux/include
        # Link the object file with mpicxx
        mpicxx $CFLAGS -o $1_mpi $1.o -L/home/software/cuda/11.3/targets/x86_64-linux/lib -lhdf5 -lm -lcudart
        rm -f $1.o
    fi
    # GPU (Serial)
    if [ "$gpu" = true ]; then
        CFLAGS="-std=c++14 -arch=compute_80 -code=sm_80 -DGPU"
        nvcc $CFLAGS $1.cu -o $1_gpu -lhdf5 -lm -lcudart
    fi
    # GPU (MPI)
    if [ "$gpu_mpi" = true ]; then
        CFLAGS="-std=c++14 -DGPU -DMPI"
        GFLAGS="-arch=compute_80 -code=sm_80"
        # Compile the CUDA source file to an object file
        nvcc $CFLAGS $GFLAGS -c $1.cu -o $1.o -I/home/software/cuda/11.3/targets/x86_64-linux/include
        # Link the object file with mpicxx
        mpicxx $CFLAGS -o $1_gpu_mpi $1.o -L/home/software/cuda/11.3/targets/x86_64-linux/lib -lhdf5 -lm -lcudart
        rm -f $1.o
    fi
}

compile main

run() {
    sim_dir="/orcd/data/mvogelsb/004/Thesan-Zooms/$1/output"
    # for i in {0..189}; do
    for i in 0; do
        if [ "$serial" = true ]; then
            echo "Running serial: $sim_dir $i"
            ./main_serial $sim_dir $i
        fi
        if [ "$mpi" = true ]; then
            echo "Running mpi: $sim_dir $i"
            mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np 16 ./main_mpi $sim_dir $i
        fi
        if [ "$gpu" = true ]; then
            echo "Running gpu: $sim_dir $i"
            ./main_gpu $sim_dir $i
            echo "Finished gpu: $sim_dir $i"
        fi
        if [ "$gpu_mpi" = true ]; then
            echo "Running gpu_mpi: $sim_dir $i"
            #mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np 16 ./main_gpu_mpi $sim_dir $i
            mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np 16 ./main_gpu_mpi . $sim_dir $i
        fi
    done
    echo "Done with $1"
}

run g2274036/z8

# Define job sets with individual job, group, and runs
job_sets=(
#  "job='A'; group='g2'; runs=('z4')"
#  "job='B'; group='g39'; runs=('z4')"
#  "job='C'; group='g205'; runs=('z4')"
#  "job='D'; group='g578'; runs=('z4')"
#  "job='E'; group='g1163'; runs=('z4' 'noESF_z4')"
#  "job='F'; group='g5760'; runs=('z4')"
#  "job='F'; group='g5760'; runs=('varEff_z4')"
#  "job='F'; group='g5760'; runs=('noRlim_z4')"
#  "job='F'; group='g5760'; runs=('noESF_z4')"
#  "job='F'; group='g5760'; runs=('z8')"
#  "job='G'; group='g10304'; runs=('z8')"
#  "job='G'; group='g10304'; runs=('z4' 'noESF_z4')"
#  "job='H'; group='g33206'; runs=('z8')"
#  "job='H'; group='g33206'; runs=('varEff_z8')"
#  "job='H'; group='g33206'; runs=('noESF_z8')"
#  "job='H'; group='g33206'; runs=('noRlim_z8')"
#  "job='H'; group='g33206'; runs=('noExt_z8')"
#  "job='H'; group='g33206'; runs=('uvb_z8')"
#  "job='H'; group='g33206'; runs=('varEff_z4')"
#  "job='H'; group='g33206'; runs=('z4')"
#  "job='H'; group='g33206'; runs=('noRlim_z4' 'noESF_z4')"
#  "job='I'; group='g37591'; runs=('z4' 'z8')"
#  "job='I'; group='g37591'; runs=('noExt_z8' 'uvb_z8')"
#  "job='J'; group='g137030'; runs=('z8')"
#  "job='J'; group='g137030'; runs=('varEff_z8')"
#  "job='J'; group='g137030'; runs=('noESF_z8')"
#  "job='J'; group='g137030'; runs=('noRlim_z8')"
#  "job='J'; group='g137030'; runs=('uvb_z8')"
#  "job='J'; group='g137030'; runs=('noExt_z8')"
#  "job='J'; group='g137030'; runs=('varEff_z4')"
#  "job='J'; group='g137030'; runs=('z4')"
#  "job='J'; group='g137030'; runs=('noRlim_z4' 'noESF_z4')"
#  "job='K'; group='g500531'; runs=('z4')"
#  "job='K'; group='g500531'; runs=('z8')"
#  "job='K'; group='g500531'; runs=('noExt_z8' 'uvb_z8')"
#  "job='L'; group='g519761'; runs=('z4' 'z8')"
#  "job='L'; group='g519761'; runs=('noExt_z8' 'uvb_z8')"
#  "job='M'; group='g2274036'; runs=('z4' 'z8')"
#  "job='M'; group='g2274036'; runs=('noExt_z8' 'uvb_z8')"
#  "job='N'; group='g5229300'; runs=('z4')"
#  "job='N'; group='g5229300'; runs=('z8')"
#  "job='N'; group='g5229300'; runs=('noExt_z8' 'uvb_z8')"
)

# Loop through job sets
for job_set in "${job_sets[@]}"; do
  eval "$job_set"  # Evaluate the job set to extract job, group, and runs

  for run in "${runs[@]}"; do
    echo "Job ${job}, Group ${group}, Run ${run}"
    run $group/$run
  done
done

