#!/bin/bash
mType=L5_BTC
eType=cAC
i_cell=0
# mType=L5_TTPC1
# eType=cADpyr
# i_cell=0

argsFilePredict="/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS_Inhibitory/39241917/predictionlOntraNewCell/predictConverted"
argsFileActual="/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS_Inhibitory/39241917/predictionlOntraNewCell/actualConverted"

output=$(sbatch BBP_sbatch_MultiParamFile.sh $mType $eType $i_cell $argsFilePredict)

predicted_job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
echo  Predicted:"$predicted_job_id"


predict_path="//pscratch/sd/k/ktub1999/OntrExcPredictedSims/runs2/${predicted_job_id}_1/${mType}${eType}${i_cell}/"

output=$(sbatch BBP_sbatch_MultiParamFile.sh $mType $eType $i_cell $argsFileActual)

actual_job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
echo  Actual:"$actual_job_id"
actual_path="//pscratch/sd/k/ktub1999/OntrExcPredictedSims/runs2/${actual_job_id}_1/${mType}${eType}${i_cell}/"

output=$(sbatch --dependency=afterany:$predicted_job_id:$actual_job_id ScoreFunctionHDF5_batchScript.sh $predict_path $actual_path "MSE_Score_PlotsResults" 0)
job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
echo  ScoreFunction:"$job_id"

output=$(sbatch --dependency=after:$predicted_job_id:$actual_job_id ScoreFunctionHDF5_batchScript.sh $predict_path $actual_path "MSE_Score_PlotsResults_Chirp" 1)
job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
echo  ScoreFunctionChirp:"$job_id"

