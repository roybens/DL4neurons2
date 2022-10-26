#!/bin/bash -l
#SBATCH -N 5
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -J ALL_CLONES
#SBATCH -C knl
#SBATCH --output /global/homes/k/ktub1999/mainDL4/DL4neurons2/logs/Ktub/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v3
#SBATCH --array 1-1 #a

echo "Mtype: $1";
echo "EType: $2";
nsamp=10
Mtype=$1
Etype=$2
i_cell=0
while [ $i_cell -ne 5 ]
do
    echo "srun -k  -n 128 --exclusive shifter python3 -u generate_analysis_data_copy.py  $Mtype $Etype $i_cell $nsamp"
    srun -k  -n 128  shifter python3 -u generate_analysis_data_copy.py  $Mtype $Etype $i_cell $nsamp&
    i_cell=$(($i_cell+1))
done

wait