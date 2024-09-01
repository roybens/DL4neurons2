INPUT=/global/homes/k/ktub1999/mainDL4/DL4neurons2/testcellInhEtypes.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
count=1
while read name mtype etype
do
        if [[ $count -gt 0 ]]; then

            bash ./sensitivity_analysis/AllClones.sh $mtype $etype&
        fi
        count=$((count+1))
        echo $count

done < $INPUT
wait
IFS=$OLDIFS