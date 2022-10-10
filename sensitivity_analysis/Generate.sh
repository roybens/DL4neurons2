#!/bin/bash
INPUT=/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
nsamp=10/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/Sensitivity Plots
count=0
while read name mtype etype
do
    if [[ $count -gt 0 ]]; then
        i_cell=1
        while [ $i_cell -ne 6 ]
        do
        # echo "Name : $name"
        #Data Generation
        # args=" $mtype $etype $nsamp"
        # line=" -n 1  shifter python3 -u generate_analysis_data_copy.py $args"
        # echo $line
            srun -k  -n 1  shifter python3 -u analyze_sensitivity_copy.py $mtype $etype $i_cell&
            i_cell=$(($i_cell+1))
        done   
    fi
    count=$((count+1))
    echo $count
done < $INPUT
wait
IFS=$OLDIFS