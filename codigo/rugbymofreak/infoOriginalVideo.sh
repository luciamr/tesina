#!/bin/bash

data=/home/lucia/data/rugby


for dir in "${data}"/original_videos/*
do
  echo "dir: $(basename $dir)"
  for file in "$dir"/*
  do
    if [ "${file: -4}" == ".mp4" ] || [ "${file: -4}" == ".avi" ]
    then
      filename="$(echo $file | sed 's/.*\///')"
      IFS='_' read -a array <<< "$filename"
      if [ "${array[2]}" == 1 ]
      then
		#echo "file: $filename"
		output="$(avprobe "${file}" 2>&1)"
		video="Video:  $(echo $output | sed 's/.*Video: //' | cut -f1 -d$'\n')"
		echo $video
	  fi
    fi
  done
done  
