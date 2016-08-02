#!/bin/bash

data=/home/lucia/Documentos/data/videos

for dir in "${data}"/*
do
  echo "dir: $(basename $dir)"
  for file in "$dir"/*
  do
    filename=$(basename $file)
    if [ "${filename:0:5}" == "match" ]
      then
	new_filename=$(basename $dir)_${filename}
	mv "$file" "$dir/$new_filename" 
      fi
  done
done  
