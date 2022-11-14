#!/bin/bash -l
#SBATCH -N 5
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -J ALL_CLONES
#SBATCH --output /global/homes/k/ktub1999/mainDL4/DL4neurons2/logs/Ktub/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

echo "Mtype: $1";
echo "EType: $2";
nsamp=10
Mtype1=$1
Etype1=$2
# Mtype2=$3
# Etype2=$4
# Mtype3=$5
# Etype3=$6
# Mtype2=$3
# Etype2=$4
i_cell=0
#srun -k  -n 640  shifter python3 -u analysis_data_copy_2.py  $Mtype $Etype $i_cell $nsamp&
while [ $i_cell -ne 5 ]
do
    echo "srun -k  -n 128 --exclusive shifter python3 -u generate_analysis_data_copy.py  $Mtype $Etype $i_cell $nsamp"
    srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype1 $Etype1 $i_cell $nsamp&
    # srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype2 $Etype2 $i_cell $nsamp&
    i_cell=$(($i_cell+1))
done

wait

# while [ $i_cell -ne 6 ]
# do
#     echo "srun -k  -n 128 --exclusive shifter python3 -u generate_analysis_data_copy.py  $Mtype $Etype $i_cell $nsamp"
#     srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype2 $Etype2 $i_cell $nsamp&
#     # srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype2 $Etype2 $i_cell $nsamp&
#     i_cell=$(($i_cell+1))
# done

# wait

# while [ $i_cell -ne 6 ]
# do
#     echo "srun -k  -n 128 --exclusive shifter python3 -u generate_analysis_data_copy.py  $Mtype $Etype $i_cell $nsamp"
#     srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype3 $Etype3 $i_cell $nsamp&
#     # srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype2 $Etype2 $i_cell $nsamp&
#     i_cell=$(($i_cell+1))
# done

# wait