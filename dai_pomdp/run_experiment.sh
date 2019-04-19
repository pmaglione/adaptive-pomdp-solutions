#!/bin/bash

#time learning = param 1, num items = param 2, data balance = param 3

echo 'Time Learning = ' $1
echo 'Num Items =     ' $2
echo 'Data Balance =  ' $3
echo 'Num States =    ' $4
echo 'Expert Cost =    ' $5

echo '---'

echo "Running: Parameters Creation" 
python RunSimulations.py 0 $1 $2 $3 $4 $5

#echo "Running: Ballot Generator"
#python BallotGenerator.py $2 $3

echo "Running: Simulations"
python RunSimulations.py 1 $1 $2 $3 $4 $5
