#!/bin/bash

mp4_directory="mp4"
video_ids_file="somethings_aff_ids.csv"
jpgs_directory="jpg"

mkdir -p "$jpgs_directory"

total_videos=$(wc -l < "$video_ids_file")
processed_videos=0

exec 3< "$video_ids_file"

while IFS= read -r -u 3 video_id; do
  video_jpg_directory="$jpgs_directory/$video_id"
  mkdir -p "$video_jpg_directory"
  
  mp4_file="$mp4_directory/$video_id.mp4"
  
  # Check if the MP4 file exists
  if [ -f "$mp4_file" ]; then
    # Extract frames from the MP4 file and save them as JPGs in the corresponding directory
    ffmpeg -i "$mp4_file" "$video_jpg_directory/%04d.jpg" >/dev/null 2>&1
  else
    echo "MP4 file for video ID $video_id does not exist."
  fi

  processed_videos=$((processed_videos + 1))

  echo -ne "Processed: $processed_videos / $total_videos\r"
done

exec 3<&-

echo -ne "\n"
echo "Frame extraction completed."
