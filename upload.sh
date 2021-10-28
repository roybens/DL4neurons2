#!/bin/bash -l

echo "copying to cori:/project/projectdirs/m2043/vbaratha/DL4neurons/generated_data/"
for f in "$@"
do
    scp $f cori:/project/projectdirs/m2043/vbaratha/DL4neurons/generated_data/
done

