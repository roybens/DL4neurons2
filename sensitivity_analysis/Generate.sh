#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -J ALL_CLONES
#SBATCH --output sensitivity_analysis/logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --mail-user=swdougherty@ucdavis.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --array 1-1 #a

# INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/testcellInhEtypes.csv
# INPUT=/global/homes/s/sdough/mainDL4/DL4neurons2/testInh1.csv
# INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/InhibitoryCell1.csv

INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/testcell.csv

# INPUT=/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/Inhibitory50Cells.csv

OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
nsamp=100
# nregions=1

MTYPE=$1
ETYPE=$2
start_bound=$3
end_bound=$4
nregions=$5
BASELINE_CSV=$6
stim_csv_path='5k50kInterChaoticB'
# 5k50kInterChaoticB
# 5k0chaotic5A
model_name='developing_model'
count=1
# while read name mtype etype
# do
    if [[ $count -gt 0 ]]; then
        i_cell=0
        while [ $i_cell -ne 1 ]
        do
        # echo "Name : $name"
        #Data Generation
        # args=" $mtype $etype $nsamp"
        # line=" -n 1  shifter python3 -u generate_analysis_data_copy.py $args"
        # echo $line
            # srun -k  -n 1  shifter python3 -m pdb sensitivity_analysis/analyze_sensitivity.py $mtype $etype $i_cell $nregions $model_name axon
            # srun -k  -n 1  shifter python3 -u ./sensitivity_analysis/analyze_sensitivity_using_scores.py $mtype $etype $i_cell $nregions $model_name $start_bound $end_bound soma $stim_csv_path&
            srun -k  -n 1  shifter python3 -u ./sensitivity_analysis/analyze_sensitivity_using_scores.py $MTYPE $ETYPE $i_cell $nregions $model_name $start_bound $end_bound soma $stim_csv_path $BASELINE_CSV&
            # srun -k  -n 1  shifter python3 -u ./sensitivity_analysis/analyze_sensitivity_using_scores.py $mtype $etype $i_cell $nregions $model_name $start_bound $end_bound axon&
            # srun -k  -n 1  shifter python3 -u ./sensitivity_analysis/analyze_sensitivity_using_scores.py $mtype $etype $i_cell $nregions $model_name $start_bound $end_bound api&
            # srun -k  -n 1  shifter python3 -u ./sensitivity_analysis/analyze_sensitivity_using_scores.py $mtype $etype $i_cell $nregions $model_name $start_bound $end_bound dend&
            i_cell=$(($i_cell+1))
        done   
    fi
    count=$((count+1))
    echo $count
# done < $INPUT
wait
IFS=$OLDIFS