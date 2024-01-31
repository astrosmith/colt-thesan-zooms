# COLT Pipelines for THESAN-Zooms
The Cosmic Lyman-alpha Transfer (`colt`) code is a Monte Carlo radiative transfer
(MCRT) solver for post-processing hydrodynamical simulations. Please consult the
online [documentation](https://colt.readthedocs.io/en/latest/) for more details.

## Paths
Please edit machine specific pipeline paths and options as necessary. Some of this
could be streamlined in the future, but for now it is set up for Engaging at MIT.

## Getting Started
The directory `metal-pipeline` focuses on central halos. Setting up COLT requires three steps:
1. `extract_halo.py`: Selects a region within which the calculations will be performed.
2. `combine_files.py`: Combines all extraction files into a single Arepo compatible file.
3. `arepo_to_colt.py`: Converts an Arepo file into a COLT initial conditions file.

These are run (with subfiles done in parallel) with the `pipelines.py` script,
which can be submitted via `sbatch job-extract.sh`. The actual `colt` runs can
start whenever these initial condidtions files exist, and can be started with
`bash submit.sh`, which copies the `config-*.yaml` files to run directories and
then submits from there with `sbatch job.sh` (using the latest parameter files).


