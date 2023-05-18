#!/bin/bash

# Define the CSV file path
csv_file="/pscratch/sd/k/ktub1999/main/DL4neurons2/testcell.csv"

# Define the second file path
second_file="/pscratch/sd/k/ktub1999/main/DL4neurons2/Params.csv"
i_cell=0
numSamples=500
count=1
# Outer loop to read from the CSV file
while IFS=',' read -r name mtype etype; do
    # echo "Outer loop: $col1, $col2, $col3"
    
    # Inner loop to read from the second file
    while IFS= read -r wideP; do
        sbatch BBP_sbatch_submit.sh $mtype $etype $i_cell $numSamples $count $wideP
        echo sbatch BBP_sbatch_submit.sh $mtype $etype $i_cell $numSamples $count $wideP
    done < "$second_file"
done < "$csv_file"
