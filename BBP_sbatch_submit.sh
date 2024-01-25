#!/bin/bash -l
#SBATCH -N 16
#SBATCH -t 11:30:00
#SBATCH -q regular
#SBATCH -J DL4N_full_prod
#SBATCH -L SCRATCH,cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

# Stuff for knl
# export OMP_NUM_THREADS=128
module unload craype-hugepages2M

# All paths relative to this, prepend this for full path name
#WORKING_DIR=/global/cscratch1/sd/adisaran/DL4neurons
#OUT_DIR=/global/cfs/cdirs/m2043/adisaran/wrk/
# OUT_DIR=/global/homes/k/ktub1999/testRun/
OUT_DIR=/pscratch/sd/k/ktub1999/Jan24PaperData/
# simu run in the dir where  Slurm job was started
model='BBP'
# rm -rf ./x86_64
# if [ "$model" = "M1_TTPC_NA_HH" ]; then
#     shifter nrnivmodl ./Neuron_Model_HH/mechanisms
# else
#     shifter nrnivmodl ./modfiles
# fi

CELLS_FILE='excitatorycells.csv'
START_CELL=0
NCELLS=2
END_CELL=$((${START_CELL}+${NCELLS}))
NSAMPLES=1
NRUNS=1
NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))

echo "CELLS_FILE" ${CELLS_FILE}
echo "START_CELL" ${START_CELL}
echo "NCELLS" ${NCELLS}
echo "END_CELL" ${END_CELL}

export THREADS_PER_NODE=128

# to prevent: H5-write error: unable to lock file, errno = 524
export HDF5_USE_FILE_LOCKING=FALSE

# Create all outdirs
echo "Making outdirs at" `date`
arrIdx=${SLURM_ARRAY_TASK_ID}
jobId=${SLURM_ARRAY_JOB_ID}_${arrIdx}
RUNDIR=${OUT_DIR}/runs2/${jobId}
mkdir -p $RUNDIR

mType=$1
eType=$2
i_cell=$3
numSamples=$4
cell_count=$5
wideP=$6
echo numSamples $numSamples

cell_name=$mType
cell_name+=$eType
cell_name+=$i_cell
mkdir -p $RUNDIR/$cell_name/
chmod a+rx $RUNDIR/$cell_name/

cp BBP_sbatch_submit.sh $RUNDIR
chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*
echo done
date

echo "Done making outdirs at" `date`


export stimname1=5k0chaotic5A
export stimname2=5k0step_200
export stimname3=5k0ramp
export stimname4=5k0chirp
export stimname5=5k0step_500
export stimname6=5k50kInterChaoticB
export stimname7=5k0chaotic5B
# export stimname1=5k50kInterChaoticB
# export stimname2=5k0chirpScaled
# export stimname3=5k0rampScaled
# export stimname4=5kChaoticRamp
# export stimname5=5k0chaotic5C

export stimname1=5k50kInterChaoticB

stimfile1=stims/${stimname1}.csv
stimfile2=stims/${stimname2}.csv
stimfile3=stims/${stimname3}.csv
stimfile4=stims/${stimname4}.csv
stimfile5=stims/${stimname5}.csv
# stimfile5=stims/${stimname5}.csv
stimfile6=stims/${stimname6}.csv
stimfile7=stims/${stimname7}.csv
echo
env | grep SLURM
echo


FILENAME=\{BBP_NAME\}-v3
echo "STIM FILE" $stimfile
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
# numParamSets=expr $numSamples /  $((SLURM_NNODES*THREADS_PER_NODE))
denom=$(expr ${SLURM_NNODES} \* $THREADS_PER_NODE)
echo denom "DENOM"
# numParamSets=$(expr $numSamples / $denom)
numParamSets=$numSamples
echo "numParamSets" $numParamSets


echo "numParamSets" $numParamSets
REMOTE_CELLS_FILE='/pscratch/sd/k/ktub1999/main/DL4neurons2/excitatorycells.csv'
PARAM_VALUE_FILE='/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/BaseTest.csv'
#sbcast ${CELLS_FILE} ${REMOTE_CELLS_FILE}
REMOTE_CELLS_FILE=${CELLS_FILE}
echo REMOTE_CELLS_FILE $REMOTE_CELLS_FILE
#( sleep 180; echo `hostname` ; date; free -g; top ibn1) >&L1&
#( sleep 260; echo `hostname` ; date; free -g; top ibn1) >&L2&
#( sleep 400; echo `hostname` ; date; free -g; top ibn1) >&L3&
param_lb='-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.5 -1 -1 -1 -1 0.5 -85'
# param_ub='2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1.45'
param_ub='1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 2.0 -65'

for j in $(seq 1 ${NRUNS});
do
    echo "Doing run $j of $NRUNS at" `date`
    for l in 1
    do
        adjustedval=$((l))
        OUT_DIR=$RUNDIR/$cell_name
        FILE_NAME=${FILENAME}-\{NODEID\}-c${i_cell}.h5
        OUTFILE=$OUT_DIR/$FILE_NAME
	
        args="--outfile $OUTFILE --stim-file ${stimfile1} --model $model \
          --m-type $mType --e-type $eType --cell-i $i_cell --num $numParamSets --cori-start ${START_CELL} --cori-end ${END_CELL} \
          --trivial-parallel --thread-number --print-every 100 --linear-params-inds 12 17 18 --exclude g_pas_axonal g_pas_somatic e_pas_all \
          --dt 0.1  --cell-count $cell_count  --default-base"
        echo "args" $args
        srun --input none -k -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) --ntasks-per-node ${THREADS_PER_NODE} shifter python3 -u run.py $args
        
        # --stim-dc-offset 0 --stim-multiplier 1
        #--exclude g_pas_axonal cm_axonal g_pas_somatic cm_somatic e_pas_all

        # --wide gNaTs2_tbar_NaTs2_t_apical gSKv3_1bar_SKv3_1_apical gImbar_Im_apical gIhbar_Ih_dend gNaTa_tbar_NaTa_t_axonal gK_Tstbar_K_Tst_axonal gNap_Et2bar_Nap_Et2_axonal \
        #   gSK_E2bar_SK_E2_axonal gCa_HVAbar_Ca_HVA_axonal gK_Pstbar_K_Pst_axonal gCa_LVAstbar_Ca_LVAst_axonal g_pas_axonal cm_axonal gSKv3_1bar_SKv3_1_somatic \
        #   gNaTs2_tbar_NaTs2_t_somatic gCa_LVAstbar_Ca_LVAst_somatic g_pas_somatic cm_somatic e_pas_all
        #For 16 Wide, Below
          #   --wide gSKv3_1bar_SKv3_1_apical gImbar_Im_apical gK_Tstbar_K_Tst_axonal \
          # gSK_E2bar_SK_E2_axonal gCa_HVAbar_Ca_HVA_axonal gCa_LVAstbar_Ca_LVAst_axonal g_pas_axonal cm_axonal gSKv3_1bar_SKv3_1_somatic \
          # g_pas_somatic cm_somatic
        


    done
    # run.py sets permissions on the data files themselves (doing them here simultaneously takes forever)
    
    echo "Done run $j of $NRUNS at" `date`

done
echo ope-read $RUNDIR
# chmod  a+r $RUNDIR/*/c*/*.h5
chmod  a+r $RUNDIR/$cell_name/*.h5

#salloc -q debug -C knl --image=balewski/ubu20-neuron8:v2 -t 30:00 -N 1