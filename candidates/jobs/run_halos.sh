#!/bin/bash

## Module setup
. /etc/profile.d/modules.sh
export HDF5_USE_FILE_LOCKING=FALSE

# sim=g2/z4
# sim=g39/z4
# sim=g205/z4
# sim=g578/z4
# sim=g1163/z4
# sim=g5760/z8
# sim=g5760/z4
# sim=g10304/z8
# sim=g10304/z4
# sim=g137030/z16
# sim=g137030/z8
# sim=g137030/z4
# sim=g500531/z16
# sim=g500531/z8
# sim=g500531/z4
# sim=g519761/z16
# sim=g519761/z8
# sim=g519761/z4
# sim=g2274036/z16
# sim=g2274036/z8
# sim=g2274036/z4
# sim=g5229300/z16
# sim=g5229300/z8
sim=g5229300/z4

# for i in {0..188}; do
#for i in {0..8}; do
#for i in 188; do
for i in 1 2 3 4 5; do
    #echo "Running halo ${i} ..."
    python ../candidates.py $sim $i # Write halos to file
    python ../halos.py $sim $i # Write halos to file
done

echo "Done with ${sim}"
