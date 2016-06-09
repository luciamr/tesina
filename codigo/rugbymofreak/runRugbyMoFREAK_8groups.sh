#!/bin/bash

build=/home/lucia/Documents/repositorio/rugbymofreak/build
svm=/home/lucia/data/rugby/svm/*

cd $build


echo "Clusters: 5000"
rm $svm
./RugbyMoFREAK 8 5000 > corrida_8_5000.txt

sleep 15m

echo "Clusters: 10000"
rm $svm
./RugbyMoFREAK 8 10000 > corrida_8_10000.txt

#sleep 30 m

#echo "Clusters: 3000"
#rm $svm
#./RugbyMoFREAK 8 3000 > corrida_8_3000.txt

#sleep 15m

#echo "Clusters: 8000"
#rm $svm
#./RugbyMoFREAK 8 8000 > corrida_8_8000.txt






