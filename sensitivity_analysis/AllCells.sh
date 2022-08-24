#!/bin/bash
INPUT=/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
nsamp=10
while read name mtype etype
do
	echo "Name : $name"
    #Data Generation
    args=" $mtype $etype $nsamp"
    $cmd="srun -k  -n 128  shifter python3 -u /global/homes/k/ktub1999/mainDL4/DL4neurons2/generate_analysis_data_copy.py $args"
    echp $cmd
    #Data Analysis
    #args2="-k  -n 1 shifter python3 -m pdb  analyze_sensitivity_copy.py $mtype $etype"
    #srun args2
done < $INPUT
IFS=$OLDIFS