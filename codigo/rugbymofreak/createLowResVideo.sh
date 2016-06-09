#!/bin/bash

data=/home/lucia/data/rugby
frame_width=200

new_videos="${data}"/videos
mkdir "$new_videos"

for dir in "${data}"/original_videos/*
do
  echo "dir: $(basename $dir)"
  new_action="$new_videos/$(basename $dir)"
  mkdir "$new_action"
  for file in "$dir"/*
  do
    if [ "${file: -4}" == ".mp4" ] || [ "${file: -4}" == ".avi" ]
    then
	  filename="$(echo $file | sed 's/.*\///')"
      if [ ! -f "$new_action/$filename" ]
      then
	    echo "convert file: $filename" 
		avp_output="$(avprobe -show_streams "${file}")"
		width="$(echo $avp_output | sed 's/.*\width=//' | cut -f1 -d" ")"
		height="$(echo $avp_output | sed 's/.*height=//' | cut -f1 -d" ")"
		frame_height=$(( height * frame_width / width ))
		#avcon -an descarta el audio
		avconv -i "${file}" -an -s "$frame_width"x"$frame_height" -aspect 1:1 "$new_action/$filename"
	  fi
    fi
  done
done  
