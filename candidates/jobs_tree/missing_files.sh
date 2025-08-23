#!/bin/bash

# Directory containing the simulation files
zoom_dir="/orcd/data/mvogelsb/004/Thesan-Zooms"
colt_dir="${zoom_dir}-COLT"
#colt_dir="/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT"

# Directory containing the output files
field="proj_rho"; out_pre="${colt_dir}"; out_post="output_tree"; check="proj_rho"

#out_pre="${zoom_dir}"; out_post="postprocessing/colt/${field}"
#out_post+="_fdust_10"

echo "Searching for missing ${field} files..."

# Define job sets with individual job, group, and runs
job_sets=(
# COLT (long circulators)
# "job='D'; group='g578'; runs=('noESF_z4'); first_snap=88; last_snap=179"       # 88-179 ->  88-179
# "job='D'; group='g578'; runs=('varEff_z4'); first_snap=0; last_snap=68"        # 0-68   ->  0-68
# "job='E'; group='g1163'; runs=('noESF_z4'); first_snap=0; last_snap=188"       # 0-188  ->  0-188
# "job='H'; group='g33206'; runs=('z16'); first_snap=0; last_snap=8"             # 0-8    ->  0-8
# Fiducial:
#"job='A'; group='g2'; runs=('z4'); first_snap=40; last_snap=67"                 # 0-67   ->  0-67
#"job='A'; group='g2'; runs=('z4'); first_snap=0; last_snap=67"                 # 0-67   ->  0-67
#"job='B'; group='g39'; runs=('z4'); first_snap=0; last_snap=188"               # 0-188  ->  0-188
#"job='C'; group='g205'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188
#"job='D'; group='g578'; runs=('z4'); first_snap=0; last_snap=188"              # 0-188  ->  0-188
#"job='E'; group='g1163'; runs=('z4'); first_snap=0; last_snap=188"             # 0-188  ->  0-188
#"job='F'; group='g5760'; runs=('z8'); first_snap=0; last_snap=188"             # 0-188  ->  0-188
#"job='G'; group='g10304'; runs=('z8'); first_snap=0; last_snap=188"            # 0-188  ->  0-188
#"job='J'; group='g137030'; runs=('z16'); first_snap=5; last_snap=188"          # 0-188  ->  5-188
#"job='K'; group='g500531'; runs=('z16'); first_snap=9; last_snap=188"          # 0-188  ->  9-188
#"job='L'; group='g519761'; runs=('z16'); first_snap=39; last_snap=188"         # 0-188  ->  39-188
#"job='M'; group='g2274036'; runs=('z16'); first_snap=9; last_snap=188"         # 0-189  ->  9-188
#
# COLT (int circulators)
#"job='F'; group='g5760'; runs=('noESF_z4'); first_snap=5; last_snap=188"       # 0-188  ->  5-188
#"job='G'; group='g10304'; runs=('noESF_z4'); first_snap=0; last_snap=188"      # 0-188  ->  0-188
#"job='H'; group='g33206'; runs=('noESF_z8'); first_snap=4; last_snap=188"      # 0-188  ->  4-188
#"job='H'; group='g33206'; runs=('noESF_z4'); first_snap=6; last_snap=188"      # 0-188  ->  6-188
#"job='J'; group='g137030'; runs=('noESF_z8'); first_snap=2; last_snap=188"     # 0-188  ->  2-188
#"job='J'; group='g137030'; runs=('noESF_z4'); first_snap=10; last_snap=188"    # 0-188  ->  10-188
#
# "job='F'; group='g5760'; runs=('noRlim_z4'); first_snap=5; last_snap=188"      # 0-188  ->  5-188
# "job='F'; group='g5760'; runs=('varEff_z4'); first_snap=5; last_snap=188"      # 0-188  ->  5-188
# "job='H'; group='g33206'; runs=('uvb_z8'); first_snap=4; last_snap=188"        # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('noExt_z8'); first_snap=4; last_snap=188"      # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('noRlim_z8'); first_snap=4; last_snap=188"     # 0-188  ->  4-188
# "job='H'; group='g33206'; runs=('varEff_z8'); first_snap=3; last_snap=188"     # 0-188  ->  3-188
# "job='H'; group='g33206'; runs=('noRlim_z4'); first_snap=6; last_snap=189"     # 0-189  ->  6-189
# "job='H'; group='g33206'; runs=('varEff_z4'); first_snap=6; last_snap=189"     # 0-189  ->  6-189
# "job='I'; group='g37591'; runs=('uvb_z8'); first_snap=7; last_snap=188"        # 0-188  ->  7-188
# "job='I'; group='g37591'; runs=('noExt_z8'); first_snap=6; last_snap=188"      # 0-188  ->  6-188
# "job='J'; group='g137030'; runs=('uvb_z8'); first_snap=3; last_snap=188"       # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noExt_z8'); first_snap=3; last_snap=188"     # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noRlim_z8'); first_snap=2; last_snap=188"    # 0-188  ->  2-188
# "job='J'; group='g137030'; runs=('varEff_z8'); first_snap=3; last_snap=188"    # 0-188  ->  3-188
# "job='J'; group='g137030'; runs=('noRlim_z4'); first_snap=10; last_snap=189"   # 0-189  ->  10-189
# "job='J'; group='g137030'; runs=('varEff_z4'); first_snap=10; last_snap=188"   # 0-188  ->  10-188
# "job='K'; group='g500531'; runs=('uvb_z8'); first_snap=8; last_snap=189"       # 0-189  ->  8-189
# "job='K'; group='g500531'; runs=('noExt_z8'); first_snap=8; last_snap=188"     # 0-188  ->  8-188
# "job='L'; group='g519761'; runs=('uvb_z8'); first_snap=23; last_snap=189"      # 0-189  ->  23-189
# "job='L'; group='g519761'; runs=('noExt_z8'); first_snap=46; last_snap=189"    # 0-189  ->  46-189
# "job='M'; group='g2274036'; runs=('uvb_z8'); first_snap=22; last_snap=182"     # 0-189  ->  22-182
# "job='M'; group='g2274036'; runs=('noExt_z8'); first_snap=28; last_snap=189"   # 0-189  ->  28-189
# "job='N'; group='g5229300'; runs=('uvb_z8'); first_snap=0; last_snap=0"        # 0-189  ->  0-0
# "job='N'; group='g5229300'; runs=('noExt_z8'); first_snap=0; last_snap=0"      # 0-189  ->  0-0
# Fiducial:
"job='F'; group='g5760'; runs=('z4'); first_snap=5; last_snap=188"             # 0-188  ->  5-188
"job='G'; group='g10304'; runs=('z4'); first_snap=0; last_snap=188"            # 0-188  ->  0-188
"job='H'; group='g33206'; runs=('z8'); first_snap=4; last_snap=188"            # 0-188  ->  4-188
"job='H'; group='g33206'; runs=('z4'); first_snap=6; last_snap=189"            # 0-189  ->  6-189
"job='I'; group='g37591'; runs=('z8'); first_snap=6; last_snap=188"            # 0-188  ->  6-188
"job='I'; group='g37591'; runs=('z4'); first_snap=11; last_snap=188"           # 0-188  ->  11-188
"job='J'; group='g137030'; runs=('z8'); first_snap=2; last_snap=188"           # 0-188  ->  2-188
"job='J'; group='g137030'; runs=('z4'); first_snap=10; last_snap=188"          # 0-188  ->  10-188
"job='K'; group='g500531'; runs=('z8'); first_snap=10; last_snap=188"          # 0-188  ->  10-188
"job='K'; group='g500531'; runs=('z4'); first_snap=16; last_snap=189"          # 0-189  ->  16-189
"job='L'; group='g519761'; runs=('z8'); first_snap=46; last_snap=189"          # 0-189  ->  46-189
"job='L'; group='g519761'; runs=('z4'); first_snap=51; last_snap=189"          # 0-189  ->  51-189
"job='M'; group='g2274036'; runs=('z8'); first_snap=27; last_snap=189"         # 0-189  ->  27-189
"job='M'; group='g2274036'; runs=('z4'); first_snap=52; last_snap=189"         # 0-189  ->  52-189
"job='N'; group='g5229300'; runs=('z16'); first_snap=72; last_snap=189"        # 0-189  ->  72-189
"job='N'; group='g5229300'; runs=('z8'); first_snap=83; last_snap=189"         # 0-189  ->  83-189
"job='N'; group='g5229300'; runs=('z4'); first_snap=87; last_snap=189"         # 0-189  ->  87-189
)

# Loop through job sets
for job_set in "${job_sets[@]}"; do
  eval "$job_set"  # Evaluate the job set to extract job, group, and runs

  # Loop through the run directories
  for run in "${runs[@]}"; do
    out_dir="${out_pre}/${group}/${run}/${out_post}"
    # Check if the output directory exists
    if [ -d "${out_dir}" ]; then
      # Make sure there is an output file for each snapshot
      missing_snaps=()
      # for snap in {0..189}; do
      for snap in $(seq ${first_snap} ${last_snap}); do
        # Check if the output file exists
        padded_snap=$(printf "%03d" $snap)
        out_file="${out_dir}/${field}_${padded_snap}.hdf5"
        if [ ! -f "${out_file}" ]; then
          missing_snaps+=("${snap}")
        else
          result=$(h5ls -r "${out_file}" | grep -q "${check}"; echo $?)
          #result=$(h5ls "${out_file}/stars" | grep -q "${check}"; echo $?)
          #result=$(h5ls "${out_file}/group" | grep -q "${check}"; echo $?)
          if [ "$result" -ne 0 ]; then
            missing_snaps+=("$snap")
          fi
        fi
      done
      if [ ${#missing_snaps[@]} -gt 0 ]; then
        printf "%s/%s: %s is missing %s\n" "${job}" "${run}" "${out_dir}" "$(IFS=','; echo "${missing_snaps[*]}")"
      fi
    else
      echo "Missing ${out_dir} (${job}/${run})"
    fi
  done
done

