RUNSCRIPT=${1-sbatch.sh} # take first arg, default to 'sbatch.sh'
echo "Submitting: $RUNSCRIPT"
JOBID=$(sbatch --parsable $RUNSCRIPT)
cp $RUNSCRIPT runs/slurm/${JOBID}_run.sh
echo "Submitted batch job $JOBID"
