## Thesan-Zooms COLT Pipeline

This pipeline is designed to catalog particle distances, generate uncontaminated candidates, convert Arepo to COLT initial conditions, provide the COLT setup cutting out low resolution boundaries, and then perform some MCRT calculations.

## Pipeline Setup Instructions

To run the setup pipeline, follow these steps (automated using `bash submit.sh` to in turn call `sbatch job.sh`):

1. Catalog low/high-res particle distances for every group and subhalo:
  ```
  python distances.py [sim] [snap]
  ```

2. Generate uncontaminated halo candidates, also saving condensed group catalogs:
  ```
  python candidates.py [sim] [snap]
  ```

3. Convert Arepo snapshots to COLT initial conditions:
  ```
  python arepo_to_colt.py [sim] [snap]
  ```

4. Run COLT to get connections (without circulators at first):
  ```
  colt config-connect.yaml [snap]
  ```

5. Remove exterior low-resolution particles leaving a buffer of neighbors of edge cells:
  ```
  python remove_lowres.py [sim] [snap]
  ```

6. Rerun COLT to obtain updated connections (this time saving face circulators too):
  ```
  colt config-connect-circ.yaml [snap]
  ```

Make sure to replace `[sim]` and `[snap]` with the appropriate simulation and snapshot values (see `job.sh`).

## Halo Projections Instructions

To run projections of each halo, follow these steps:

1. Make the candidate halo files for COLT to target halos (automated using `bash submit_halos.sh` to in turn call `sbatch job_halos.sh`):
  ```
  python halos.py [sim] [snap]
  ```

2. Run COLT to get a file with projections of every halo candidate (automated using `bash submit_colt_halos.sh` to in turn call `sbatch job_colt_halos.sh`):
  ```
  colt config-halo-proj-RHD.yaml [snap]
  ```
