#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -J DL4N_full_prod
#SBATCH -L SCRATCH,cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

# Stuff for knl
export OMP_NUM_THREADS=1
module unload craype-hugepages2M

# All paths relative to this, prepend this for full path name
#WORKING_DIR=/global/cscratch1/sd/adisaran/DL4neurons
#OUT_DIR=/global/cfs/cdirs/m2043/adisaran/wrk/
# OUT_DIR=/global/homes/k/ktub1999/testRun/
OUT_DIR=/pscratch/sd/k/ktub1999/BBP_TEST1/
# simu run in the dir where  Slurm job was started

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

cp BBP_sbatch_def.sh $RUNDIR
chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*
echo done
date

echo "Done making outdirs at" `date`

export stimname1=5k0chaotic5A_0.25x
export stimname2=5k0chaotic5A_0.50x
export stimname3=5k0chaotic5A_0.75x
export stimname4=5k0chaotic5A_1.00x
export stimname5=5k0chaotic5A_1.33x
export stimname6=5k0chaotic5A_2.00x
export stimname7=5k0chaotic5A_4.00x
export stimname8=5k0step_200_0.25x
export stimname9=5k0step_200_0.50x
export stimname10=5k0step_200_0.75x
export stimname11=5k0step_200_1.00x
export stimname12=5k0step_200_1.33x
export stimname13=5k0step_200_2.00x
export stimname14=5k0step_200_4.00x
export stimname15=5k0ramp_0.25x
export stimname16=5k0ramp_0.50x
export stimname17=5k0ramp_0.75x
export stimname18=5k0ramp_1.00x
export stimname19=5k0ramp_1.33x
export stimname20=5k0ramp_2.00x
export stimname21=5k0ramp_4.00x
export stimname22=5k0chirp_0.25x
export stimname23=5k0chirp_0.50x
export stimname24=5k0chirp_0.75x
export stimname25=5k0chirp_1.00x
export stimname26=5k0chirp_1.33x
export stimname27=5k0chirp_2.00x
export stimname28=5k0chirp_4.00x
export stimname29=5k0step_500_0.25x
export stimname30=5k0step_500_0.50x
export stimname31=5k0step_500_0.75x
export stimname32=5k0step_500_1.00x
export stimname33=5k0step_500_1.33x
export stimname34=5k0step_500_2.00x
export stimname35=5k0step_500_4.00x
export stimname36=5k50kInterChaoticB_0.25x
export stimname37=5k50kInterChaoticB_0.50x
export stimname38=5k50kInterChaoticB_0.75x
export stimname39=5k50kInterChaoticB_1.00x
export stimname40=5k50kInterChaoticB_1.33x
export stimname41=5k50kInterChaoticB_2.00x
export stimname42=5k50kInterChaoticB_4.00x
export stimname43=5k0chaotic5B_0.25x
export stimname44=5k0chaotic5B_0.50x
export stimname45=5k0chaotic5B_0.75x
export stimname46=5k0chaotic5B_1.00x
export stimname47=5k0chaotic5B_1.33x
export stimname48=5k0chaotic5B_2.00x
export stimname49=5k0chaotic5B_4.00x


stimfile1=stims/scaled_stims/${stimname1}.csv
stimfile2=stims/scaled_stims/${stimname2}.csv
stimfile3=stims/scaled_stims/${stimname3}.csv
stimfile4=stims/scaled_stims/${stimname4}.csv
stimfile5=stims/scaled_stims/${stimname5}.csv
stimfile6=stims/scaled_stims/${stimname6}.csv
stimfile7=stims/scaled_stims/${stimname7}.csv
stimfile8=stims/scaled_stims/${stimname8}.csv
stimfile9=stims/scaled_stims/${stimname9}.csv
stimfile10=stims/scaled_stims/${stimname10}.csv
stimfile11=stims/scaled_stims/${stimname11}.csv
stimfile12=stims/scaled_stims/${stimname12}.csv
stimfile13=stims/scaled_stims/${stimname13}.csv
stimfile14=stims/scaled_stims/${stimname14}.csv
stimfile15=stims/scaled_stims/${stimname15}.csv
stimfile16=stims/scaled_stims/${stimname16}.csv
stimfile17=stims/scaled_stims/${stimname17}.csv
stimfile18=stims/scaled_stims/${stimname18}.csv
stimfile19=stims/scaled_stims/${stimname19}.csv
stimfile20=stims/scaled_stims/${stimname20}.csv
stimfile21=stims/scaled_stims/${stimname21}.csv
stimfile22=stims/scaled_stims/${stimname22}.csv
stimfile23=stims/scaled_stims/${stimname23}.csv
stimfile24=stims/scaled_stims/${stimname24}.csv
stimfile25=stims/scaled_stims/${stimname25}.csv
stimfile26=stims/scaled_stims/${stimname26}.csv
stimfile27=stims/scaled_stims/${stimname27}.csv
stimfile28=stims/scaled_stims/${stimname28}.csv
stimfile29=stims/scaled_stims/${stimname29}.csv
stimfile30=stims/scaled_stims/${stimname30}.csv
stimfile31=stims/scaled_stims/${stimname31}.csv
stimfile32=stims/scaled_stims/${stimname32}.csv
stimfile33=stims/scaled_stims/${stimname33}.csv
stimfile34=stims/scaled_stims/${stimname34}.csv
stimfile35=stims/scaled_stims/${stimname35}.csv
stimfile36=stims/scaled_stims/${stimname36}.csv
stimfile37=stims/scaled_stims/${stimname37}.csv
stimfile38=stims/scaled_stims/${stimname38}.csv
stimfile39=stims/scaled_stims/${stimname39}.csv
stimfile40=stims/scaled_stims/${stimname40}.csv
stimfile41=stims/scaled_stims/${stimname41}.csv
stimfile42=stims/scaled_stims/${stimname42}.csv
stimfile43=stims/scaled_stims/${stimname43}.csv
stimfile44=stims/scaled_stims/${stimname44}.csv
stimfile45=stims/scaled_stims/${stimname45}.csv
stimfile46=stims/scaled_stims/${stimname46}.csv
stimfile47=stims/scaled_stims/${stimname47}.csv
stimfile48=stims/scaled_stims/${stimname48}.csv
stimfile49=stims/scaled_stims/${stimname49}.csv


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
	
        args="--outfile $OUTFILE --stim-file ${stimfile1} ${stimfile2} ${stimfile3} ${stimfile4} ${stimfile5} ${stimfile6} ${stimfile7} ${stimfile8} ${stimfile9} ${stimfile10} ${stimfile11} ${stimfile12} ${stimfile13} ${stimfile14} ${stimfile15} ${stimfile16} ${stimfile17} ${stimfile18} ${stimfile19} ${stimfile20} ${stimfile21} ${stimfile22} ${stimfile23} ${stimfile24} ${stimfile25} ${stimfile26} ${stimfile27} ${stimfile28} ${stimfile29} ${stimfile30} ${stimfile31} ${stimfile32} ${stimfile33} ${stimfile34} ${stimfile35} ${stimfile36} ${stimfile37} ${stimfile38} ${stimfile39} ${stimfile40} ${stimfile41} ${stimfile42} ${stimfile43} ${stimfile44} ${stimfile45} ${stimfile46} ${stimfile47} ${stimfile48} ${stimfile49} \
         --model M1_TTPC_NA_HH --cell-i ${l} \
          --cori-csv ${REMOTE_CELLS_FILE} --num 11  --cori-start ${START_CELL} --cori-end ${END_CELL} \
          --trivial-parallel --print-every 5 --linear-params-inds 12 17 18 --stim-dc-offset 0 --stim-multiplier 1\
          --dt 0.1 --param-file /pscratch/sd/k/ktub1999/compare/bestFit_results/xander_param.csv"
        echo "args" $args
        srun --input none -k -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) --ntasks-per-node ${THREADS_PER_NODE} shifter python3 -u run.py $args

    done
    # run.py sets permissions on the data files themselves (doing them here simultaneously takes forever)
    
    echo "Done run $j of $NRUNS at" `date`

done
echo ope-read $RUNDIR
chmod  a+r $RUNDIR/*/c*/*.h5


#salloc -q debug -C knl --image=balewski/ubu20-neuron8:v2 -t 30:00 -N 1