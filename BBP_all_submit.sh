#!/bin/bash
INPUT=/pscratch/sd/k/ktub1999/main/DL4neurons2/testcell.csv
INPUT=/global/homes/k/ktub1999/mainDL4/DL4neurons2/testcell.csv
INPUT=/pscratch/sd/k/ktub1999/main/DL4neurons2/testcell.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
count=1
numSamples=700
export PYTHONPATH=""
while read name mtype etype
do
    if [[ $count -gt 0 ]]; then
            i_cell=0
            while [ $i_cell -ne 2 ]
            do
                sbatch BBP_sbatch_submit.sh $mtype $etype $i_cell $numSamples $count
                i_cell=$(($i_cell+1))
            done
    fi
    count=$((count+1))
    echo $count

done < $INPUT
wait
IFS=$OLDIFS