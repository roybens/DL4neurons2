#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -J DL4N_full_prod
#SBATCH -L cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

# Stuff for knl
export OMP_NUM_THREADS=1
module unload craype-hugepages2M
# cd /pscratch/sd/k/ktub1999/main/DL4neurons2
# All paths relative to this, prepend this for full path name
#WORKING_DIR=/global/cscratch1/sd/adisaran/DL4neurons
#OUT_DIR=/global/cfs/cdirs/m2043/adisaran/wrk/
# OUT_DIR=/global/homes/k/ktub1999/testRun/
OUT_DIR=/pscratch/sd/k/ktub1999/BBP_TEST1/
# simu run in the dir where  Slurm job was started
model='M1_TTPC_NA_HH'
# model='BBP'
rm -rf ./x86_64
if [ "$model" = "M1_TTPC_NA_HH" ]; then
    shifter nrnivmodl ./Neuron_Model_HH/mechanisms
else
    shifter nrnivmodl ./modfiles
fi
CELLS_FILE='testcell.csv'
START_CELL=0
NCELLS=1
END_CELL=$((${START_CELL}+${NCELLS}))
NSAMPLES=1
NRUNS=1
NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))

echo "CELLS_FILE" ${CELLS_FILE}
echo "START_CELL" ${START_CELL}
echo "NCELLS" ${NCELLS}
echo "END_CELL" ${END_CELL}

export THREADS_PER_NODE=1

# to prevent: H5-write error: unable to lock file, errno = 524
export HDF5_USE_FILE_LOCKING=FALSE

# Create all outdirs
echo "Making outdirs at" `date`
arrIdx=${SLURM_ARRAY_TASK_ID}
jobId=${SLURM_ARRAY_JOB_ID}_${arrIdx}
RUNDIR=${OUT_DIR}/runs2/${jobId}
mkdir -p $RUNDIR
for i in $(seq $((${START_CELL}+1)) ${END_CELL});
do
    line=$(head -$i ${CELLS_FILE} | tail -1)
    bbp_name=$(echo $line | awk -F "," '{print $1}')
    for k in 1
    do
        mkdir -p $RUNDIR/$bbp_name/c${k}
        chmod a+rx $RUNDIR/$bbp_name/c${k}
    done
done
##MPT
mType=L5_TTPC1
eType=cADpyr
# cell_name=$mType
# cell_name+=$eType
# cell_name+=$i_cell
# mkdir -p $RUNDIR/$cell_name/
# chmod a+rx $RUNDIR/$cell_name/

cp $(basename "$0") $RUNDIR
chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*
echo done
date

echo "Done making outdirs at" `date`

export stimname1=5k50kInterChaoticB
export stimname2=5k0chaotic_kevinB
export stimname3=5k0ramp
export stimname4=5k0chirp
export stimname5=5k0step_500

stimfile1=stims/${stimname1}.csv
stimfile2=stims/${stimname2}.csv
stimfile3=stims/${stimname3}.csv
stimfile4=stims/${stimname4}.csv
stimfile5=stims/${stimname5}.csv
echo


env | grep SLURM
echo


FILENAME=\{BBP_NAME\}-v3
echo "STIM FILE" $stimfile
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
numParamSets=10
REMOTE_CELLS_FILE='/pscratch/sd/k/ktub1999/main/DL4neurons2/testcell.csv'
PARAM_VALUE_FILE='/global/homes/k/ktub1999/plots/Exp_Data/temp_param.csv'
#sbcast ${CELLS_FILE} ${REMOTE_CELLS_FILE}
REMOTE_CELLS_FILE=${CELLS_FILE}
echo REMOTE_CELLS_FILE $REMOTE_CELLS_FILE
#( sleep 180; echo `hostname` ; date; free -g; top ibn1) >&L1&
#( sleep 260; echo `hostname` ; date; free -g; top ibn1) >&L2&
#( sleep 400; echo `hostname` ; date; free -g; top ibn1) >&L3&

param_lb='-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1.45'
param_ub='2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1.45'

array_string=$(printf "%s," "${param_lb[@]}")
array_string=${array_string%,} # Remove the trailing comma

param_array_string=$(printf '%s\n' "${param_array[@]}")
for j in $(seq 1 ${NRUNS});
do
    echo "Doing run $j of $NRUNS at" `date`
    for l in 1
    do
        adjustedval=$((l))
        OUT_DIR=$RUNDIR/\{BBP_NAME\}/c${adjustedval}/
        # OUT_DIR=$RUNDIR/$cell_name
        # FILE_NAME=${FILENAME}-\{NODEID\}-c${i_cell}.h5
        FILE_NAME=${FILENAME}-\{NODEID\}-$j-c${adjustedval}.h5
        OUTFILE=$OUT_DIR/$FILE_NAME
	
        args="--outfile $OUTFILE --stim-file ${stimfile1} ${stimfile2} ${stimfile3} ${stimfile4} ${stimfile5} --model $model --cell-i ${l} \
          --cori-csv ${REMOTE_CELLS_FILE} --num 128  --cori-start ${START_CELL} --cori-end ${END_CELL} \
          --trivial-parallel --print-every 5 --linear-params-inds 12 17 18 --stim-dc-offset 0 --stim-multiplier 1\
          --dt 0.1 --param-file /global/homes/k/ktub1999/mainDL4/DL4neurons2/M1IhChange.csv"
        echo "args" $args
        srun --input none -k -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) --ntasks-per-node ${THREADS_PER_NODE} shifter python3 -u run.py $args

    done
    # run.py sets permissions on the data files themselves (doing them here simultaneously takes forever)
    
    echo "Done run $j of $NRUNS at" `date`

done
echo ope-read $RUNDIR
chmod  a+r $RUNDIR/*/c*/*.h5


#salloc -q debug -C knl --image=balewski/ubu20-neuron8:v2 -t 30:00 -N 1