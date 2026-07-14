#!/bin/bash -l
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 4:30:00
#SBATCH -J ALL_CLONES
#SBATCH --output sensitivity_analysis/logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --mail-user=swdougherty@ucdavis.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --array 1-1 #a

echo "Mtype: $1";
echo "EType: $2";
nsamp=100
Mtype1=$1
Etype1=$2
start_bound=$3
end_bound=$4
# Mtype2=$3
# Etype2=$4
# Mtype3=$5
# Etype3=$6
# Mtype2=$3
# Etype2=$4
nregion=$5
param_csv_path=$6
stim_csv_path='/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/stims/5k50kInterChaoticB.csv'
# nregion=1
i_cell=0
model='developing_model'
#srun -k  -n 640  shifter python3 -u analysis_data_copy_2.py  $Mtype $Etype $i_cell $nsamp&
while [ $i_cell -ne 1 ]
do
    echo "srun -k  -n 128 --exclusive shifter python3 -u generate_analysis_data.py  $Mtype1 $Etype1 $i_cell $nsamp $start_bound $end_bound $stim_csv_path $param_csv_path"
    srun -k  -n 128  shifter python3 -u ./sensitivity_analysis/generate_analysis_data.py  $Mtype1 $Etype1 $i_cell $nsamp $nregion $model $start_bound $end_bound $stim_csv_path $param_csv_path
    # srun -k  -n 128  shifter python3 -u analysis_data_copy_2.py  $Mtype2 $Etype2 $i_cell $nsamp&
    i_cell=$(($i_cell+1))
done

# wait

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