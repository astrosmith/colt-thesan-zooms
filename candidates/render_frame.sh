#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12       # Number of CPU cores for parallel processing
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal
#SBATCH --job-name=frame_render
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.err
#SBATCH --mem=128G

field=proj_rho2_avg
group="g500531"
res="z16"
sim=${group}/${res}

#zoom_colt_dir=/orcd/data/mvogelsb/004/Thesan-Zooms-COLT
zoom_colt_dir=/orcd/data/mvogelsb/005/Lab/Thesan-Zooms-COLT
data_dir=${zoom_colt_dir}/${sim}

echo "Creating movie for field ${field} ..."
#echo "Creating movie for thumbnails..."

# python -u frame_render.py $sim $field $data_dir
# python -u thumbnail_render.py $sim $zoom_colt_dir
#python -u collect_thumbnail_data.py $sim $data_dir

# lowres=false

# if [ "$lowres" = true ]; then
#    echo "Low resolution mode enabled."
#    image_dir=${data_dir}/image_data_lowres_${snap}
#    echo "Image directory: ${image_dir}"
#    output_dir=${data_dir}/movie_${snap}/lowres/
#    mkdir -p $output_dir
#    rm -f ${output_dir}/movie_${snap}_${field}.mp4
#    # Create the movie using ffmpeg
#    ffmpeg -r 40 -i ${image_dir}/${field}/Image_%04d.png -pix_fmt yuv420p -crf 20 -vf "scale=360:360" ${output_dir}/movie_${snap}_${field}.mp4
# else
#    echo "High resolution mode enabled."
#    image_dir=~/work/image_data/${sim}
#    echo "Image directory: ${image_dir}"
#    output_dir=~/work/movie/${sim}
#    mkdir -p $output_dir
#    rm -f ${output_dir}/movie_${field}.mp4
#    # Create the movie using ffmpeg
#    ffmpeg -r 60 -i ${image_dir}/${field}/Image_%04d.png -pix_fmt yuv420p -crf 20 -vf "scale=1280:720" ${output_dir}/movie_${field}.mp4
# fi

image_dir=$HOME/work/image_data/$sim/Mosaic
output_dir=$HOME/work/movie/$sim
rm -f ${output_dir}/movie_mosaic_thumbnail.mp4
ffmpeg -r 60 -i ${image_dir}/Mosaic_%04d.png -pix_fmt yuv420p -crf 20 -vf "scale=1080:720" ${output_dir}/movie_mosaic_thumbnail.mp4
