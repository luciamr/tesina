#!/bin/bash

data=/home/lucia/data/rugby_cured

new_videos=/home/lucia/data/renamed_cured_videos
mkdir "$new_videos"

for dir in "${data}"/*
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
	    game="${filename:6:4}"
        clip="${filename:15:4}"
        new_filename=${game}_${(basename $dir)}_${clip}_video
        cp "$file" "$new_action/$new_filename" 
	  fi
    fi
  done
done  
