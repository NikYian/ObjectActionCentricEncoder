#!/bin/bash

input_dir="webm"
output_dir="mp4"
converted_count=0
skipping_count=0
error_count=0
total_files=$(find "$input_dir" -type f -name "*.webm" | wc -l)

find "$input_dir" -type f -name "*.webm" | while read -r webm_file; do
    relative_path="${webm_file#$input_dir/}"

    filename=$(basename "$relative_path")
    filename_no_ext="${filename%.*}"
    
    mp4_file="$output_dir/${filename_no_ext}.mp4"
    
    if [ -f "$mp4_file" ]; then
        # echo "Skipping $webm_file. File already converted."&& ((skipping_count++))
        ((skipping_count++))
    else
        # Convert the WebM file to MP4 format using ffmpeg with suppressed output
        if ffmpeg -i "$webm_file" "$mp4_file" >/dev/null 2>ffmpeg_error.log; then
         ((converted_count++))
        else 
        ((error_count++))

        fi 
    fi
    progress=$((converted_count * 100 / total_files))
    total=$((converted_count+skipping_count))
        
    echo -ne "Skipped:$skipping_count, Converted:$converted_count, errors:$error_count Progress: $total/$total_files\r"

done
total_mp4_files=$(find "$output_dir" -type f -name "*.mp4" | wc -l)

echo "Conversion completed. $converted_count new files. $total_mp4_files of $total_files files were successfully converted to MP4 format."
