#!/bin/bash
INPUT=/global/homes/k/ktub1999/mainDL4/DL4neurons2/testcell.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
count=0
while read name mtype etype
do
        if [[ $count -gt 0 ]]; then

            sbatch AllClones.sh $mtype $etype
        fi
        count=$((count+1))
        echo $count

done < $INPUT
wait
IFS=$OLDIFS