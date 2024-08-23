#!/bin/bash

# Directory containing the simulation files
zooms_dir="/orcd/data/mvogelsb/004/Thesan-Zooms"

# Directory containing the output files
# field="offsets"; out_pre="${zooms_dir}"; out_post="postprocessing/${field}"
field="distances"; out_pre="${zooms_dir}"; out_post="postprocessing/${field}"
# field="mag"; out_pre="${zooms_dir}"; out_post="postprocessing/${field}"
# field="candidates"; out_pre="${zooms_dir}"; out_post="postprocessing/${field}"
# field="halo"; out_pre="${zooms_dir}-COLT"; out_post="ics"
#field="colt"; out_pre="${zooms_dir}-COLT"; out_post="ics"

echo "Searching for missing ${field} files..."
# Capture the group directories
cd $zooms_dir
groups=($(ls -d g*))
# echo "Found ${#groups[@]} group directories"
# echo "${groups[@]}"

# Set the maximum number of groups and runs
max_groups=200
max_runs=200

# Loop through the group directories
group_counter=0
for group in "${groups[@]}"; do
  # Check if the group counter exceeds the maximum number of groups
  if [ $group_counter -ge $max_groups ]; then
    break # Exit the loop
  fi
  ((group_counter++)) # Increment the group counter

  # Capture the run directories
  group_dir="${zooms_dir}/${group}"
  cd $group_dir
  runs=($(ls -d * 2>/dev/null))
  if [ ${#runs[@]} -eq 0 ]; then
    continue # Skip the group if no snapdir directories are found
  fi
  # echo "Found ${#runs[@]} run directories in ${group}"
  # echo "${runs[@]}"

  # Loop through the run directories
  run_counter=0
  for run in "${runs[@]}"; do
    # Check if the run counter exceeds the maximum number of runs
    if [ $run_counter -ge $max_runs ]; then
      break # Exit the loop
    fi
    if [ $group == "g500531" ]; then
      if [ $run == "z16_noESF" ] ||
         [ $run == "z16_c10" ] ||
         [ $run == "z16_gcc" ] ||
         [ $run == "z16_limitedEnrich" ]; then
        continue # Skip the run
      fi
    fi
    if [ $group == "g2" ] && [ $run == "z4" ]; then
      continue # Skip the run
    fi
    if [ $group == "g33206" ] && [ $run == "z16" ]; then
      continue # Skip the run
    fi
    if [ $group == "g137030" ] && [ $run == "z16" ]; then
      continue # Skip the run
    fi
    if [ $group == "g578" ]; then
      if [ $run == "noESF_z4" ] ||
         [ $run == "varEff_z4" ]; then
        continue # Skip the run
      fi
    fi
    ((run_counter++)) # Increment the run counter

    # Capture the snapshot numbers
    run_dir="${group_dir}/${run}"/output
    cd $run_dir
    out_dir="${out_pre}/${group}/${run}/${out_post}"
    snap_dirs=($(ls -d snapdir_* 2>/dev/null))
    if [ ${#snap_dirs[@]} -eq 0 ]; then
      continue # Skip the run if no snapdir directories are found
    fi
    padded_snaps=()
    for dir in "${snap_dirs[@]}"; do
      padded_snaps+=("${dir#snapdir_}")
    done
    # echo "Found ${#padded_snaps[@]} snapshots in ${run}"

    # Check if the output directory exists
    if [ -d "${out_dir}" ]; then
      # Make sure there is an output file for each snapshot
      missing_snaps=()
      for padded_snap in "${padded_snaps[@]}"; do
        # Check if the output file exists
        if [ -f "snapdir_${padded_snap}/snapshot_${padded_snap}.0.hdf5" ] && \
           [ ! -f "${out_dir}/${field}_${padded_snap}.hdf5" ]; then
          snap=$(echo $padded_snap | sed 's/^0*//')
          missing_snaps+=("${snap}")
        fi
      done
      if [ ${#missing_snaps[@]} -gt 0 ]; then
          echo "${out_dir} is missing ${missing_snaps[@]}"
      fi
    else
      echo "Missing ${out_dir}"
    fi
  done
done

# # Loop through job sets
# for job_set in "${job_sets[@]}"; do
# # for job_set in "${job_sets_189[@]}"; do
#   eval "$job_set"# Evaluate the job set to extract job, group, and runs

#   for run in "${runs[@]}"; do
#     # echo "Job ${job}, Group ${group}, Run ${run}"
#     directory="${zooms_dir}/${group}/${run}/postprocessing/${field}"
#     # directory="${zooms_dir}/${group}/${run}/ics"
#     for i in {0..188}; do
#     # for i in 0; do
#     #for i in 188; do
#     # for i in 189; do
#       # Format the number with leading zeros
#       filename=$(printf "${field}_%03d.hdf5" $i)

#       # Check if the file exists
#       if [ ! -f "$directory/$filename" ]; then
#         # echo "job='${job}'; group='${group}'; run='${run}'"
#         echo "\"job='${job}'; group='${group}'; runs=('${run}')\""
#         # echo -n ",$i"
#       fi
#     done
#     # echo ""
#   done
# done

