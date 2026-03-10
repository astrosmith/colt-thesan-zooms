#!/bin/bash

# sim=g2; run=z4; start=0
# sim=g39; run=z4; start=0
# sim=g205; run=z4; start=0
# sim=g578; run=z4; start=0
# sim=g1163; run=z4; start=8
sim=g5760; run=z8; start=4
# sim=g10304; run=z8; start=0
# sim=g33206; run=z8; start=4
# sim=g37591; run=z8; start=22
# sim=g137030; run=z16; start=5
# sim=g500531; run=z16; start=13
# sim=g519761; run=z16; start=39
# sim=g2274036; run=z16; start=57
# sim=g5229300; run=z16; start=?

# field=ion-eq

style=maps
# style=images

movie () {
  # rm -f -- movies/$1_${field}_${style}.mp4
  # ffmpeg -start_number $2 -r 10 -i $1/${field}/${style}_%03d.png -pix_fmt yuv420p -vb 3000k -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" movies/$1_${field}_${style}.mp4
  # Maps
  movie_dir=/orcd/scratch/orcd/006/arsmith/${style}
  mkdir -p "${movie_dir}/movies"
  rm -f "${movie_dir}/movies/${1}_${style}.mp4"
  # 3840 x 2160 resolution (cropped)
  ffmpeg -start_number "$2" -framerate 8 -i "${movie_dir}/${1}/${style}_%03d.png" \
    -vf "scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,setsar=1,format=yuv420p" \
    -c:v libx264 -profile:v high -level 5.1 -preset slow -crf 16 \
    -movflags +faststart \
    "${movie_dir}/movies/${1}_${style}.mp4"
}

movie ${sim}_${run} ${start}
