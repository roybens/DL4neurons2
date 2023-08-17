paths="
/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/9376390/temp_param_default.csv
/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/9376390/temp_param_default.csv
/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/L5_TTPC1cADpyr0/9376390/temp_param_min-1+1.csv
"

for path in $paths ; do
    sbatch BBP_sbatch_exp.sh $path
done