#!/bin/bash
# INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/testcell.csv
# INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/testcell.csv
INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/testcell.csv
OLDIFS=$IFS
IFS=','
echo "$1"
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
[ ! -f $1 ] && { echo "$1 file not found"; exit 99; }
count=1
numSamples=250
export PYTHONPATH=""
rm -rf ./x86_64
# shifter --image=balewski/ubu20-neuron8:v5  nrnivmodl ./newDevelopingmechanisms
# shifter --image=balewski/ubu20-neuron8:v5  nrnivmodl ./newAdultmechanisms
shifter --image=balewski/ubu20-neuron8:v5  nrnivmodl ./Neuron_Model_HH/mechanisms
# Make a copy of run.py to where we are running
while read name mtype etype
do
    if [[ $count -gt 0 ]]; then
            i_cell=0
            while [ $i_cell -ne 1 ]
            do
                
                output=$(sbatch M1_sbatch_submit.sh $mtype $etype $i_cell $numSamples $count $1)
                # output=$(sbatch M1_sbatch_submit.sh $mtype $etype $i_cell $numSamples $count)
                # sbatch BBP_Def_Exp.sh $mtype $etype $i_cell 10 $count
                job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
                echo  "$job_id"
                i_cell=$(($i_cell+1))
            done
    fi
    #count=$((count+1))
    echo $count

done < $INPUT
wait
IFS=$OLDIFS