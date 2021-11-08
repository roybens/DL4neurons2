# DL4neurons

Requires numpy >= 1.17

## Running BBP models:

Compile modfiles: `$ nrnivmodl modfiles/*.mod` (all the ones you need are included in this repo)

```
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --cell-i 0 --num 10 --outfile test.h5 --debug
```

This command should generate 10 traces and put them in the file test.h5

Also consider playing around with the `--stim-multiplier` option to run.py, and the `--cell-i <integer>` option which allows you to select up to 5 different morphologies (`--cell-i 0` through `4`)).

If you want to run on cells other than L5_TTPC1, see "Optional: obtain cell models" below


#### Optional: Obtain cell models

The repo contains 5 example cells of m-type L5_TTPC1, e-type cADpyr. If you don't do the following optional steps, you will only be able to run with these 5 cells.

Obtain cell models from https://bbp.epfl.ch/nmc-portal/web/guest/downloads (Use the link where it says "The complete set of neuron models is available here")

They will arrive as a zip file called hoc_combos_syn.1_0_10.allzips.tar, which you should save into a directory called hoc_templates alongside run.py

Then enter hoc_templates and untar/unzip everything:

```
$ ls
run.py  models.py  hoc_templates  [...]  hoc_combos_syn.1_0_10.allzips.tar
$ tar -xvf hoc_combos_syn.1_0_10.allzips.tar --directory hoc_templates
$ cd hoc_templates
$ unzip 'hoc_combos_syn.1_0_10.allzips/*.zip' # quotes are necessary!!
$ ls
L1_DAC_cNAC187_1	L1_DAC_cNAC187_2	L1_DAC_cNAC187_3	L1_DAC_cNAC187_4	L1_DAC_cNAC187_5 [...] hoc_combos_syn.1_0_10.allzips

## Cleanup
$ rm -r hoc_combos_syn.1_0_10.allzips
$ cd ..
$ rm hoc_combos_syn.1_0_10.allzips.tar
```

You should have the following structure:

```
DL4neurons/
   run.py
   models.py
   [...]
   hoc_templates/
       L1_DAC_cNAC187_1/
       L1_DAC_cNAC187_2/
       L1_DAC_cNAC187_3/
       [...]
       L6_UTPC_cADpyr231_1/
       L6_UTPC_cADpyr231_2/
```

Now you can use any m-type and e-type in the BBP model. 

A programmatically-accessible list of all m_types and e_types can be found in cells.json

## Running at scale on Cori w/ Shifter:

### Setup

The shifter image is `balewski/ubu18-py3-mpich:v2` which [BBP_sbatch.sh](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh) is configured to use

Set up as described above, then follow these instructions for installing NEURON within the Shifter image (from https://bitbucket.org/balewski/jannersc/src/master/dockerVaria/ubuntu/Readme.ubu18-py3-mpich-NEURON):

```
salloc --qos=interactive --image=balewski/ubu18-py3-mpich:v2 -N 1 -t 4:00:00 -C knl
shifter   bash

# this must produce 1
env|grep  SHIFTER_RUNTIME

mkdir $CSCRATCH/neuronBBP_build2
cd $CSCRATCH/neuronBBP_build2

wget https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.7/nrn-7.6.7.tar.gz
tar  xvzf  nrn-7.6.7.tar.gz
mv nrn-7.6  nrn
cd nrn
./configure --prefix=`pwd` --with-paranrn --without-iv --with-nrnpython  # it takes ~10 min on KNL, ~1 min on haswell
time make  # it takes 25 min on KNL, ~5 min on haswell
make install
chmod a+x bin/nrnivmodl bin/nrngui
exit # shifter
exit # salloc
```

Note that you may need to recompile the modfiles from within the shifter image (just do this before `exit`)

### Run

You will run by submitting BBP_sbatch.sh to Slurm - **please change the [email address on line 8](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh#L8) and [the working directory on line 19](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh#L19)**

Choose cells by index in allcells.csv (or generate your own csv listing cell names, m type, e type) by setting the variables in [lines 22-24 of BBP_sbatch.sh](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh#L22). When `--cori-csv` is passed to run.py (as is done in BBP_sbatch.sh), the cell that each thread will run is [chosen](https://github.com/VBaratham/DL4neurons/blob/master/run.py#L352) based on the value of `$SLURM_PROCID`.

`NSAMPLES` on line 26 gives the number of samples to be produced per thread **(the number of threads is the [number of nodes](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh#L3) times the [number of threads per node](https://github.com/VBaratham/DL4neurons/blob/2019_12_full_production/BBP_sbatch.sh#L35))**
`NRUNS` on line 26 gives the number of files *per thread* to break up the data into. For example, `NSAMPLES=100` and `NRUNS=20` will produce 20 files per thread, each containing 5 samples. If `NRUNS` is too large, you will produce too many files. If too small, you run the risk of losing data if the run crashes halfway through, and you lose the option to kill+restart a run that's hanging without waiting for a new compute reservation (see "NB" below)

NB: When Cori is under heavy load, I have noticed that one or more threads occasionally hangs, eating up reservation time without producing any results. Because logs are too expensive to write for jobs consisting of thousands of nodes, 1.) it is exceedingly difficult to figure out the cause of this bug, as it only appears at scale, and 2.) the only way I've been able to detect it is to look at the elapsed time of each job step (each run is executed as a separate job step): `sacct -j <jobid> --format JobID,Elapsed` and kill them when they're taking way too long - the next run will begin immediately after the current one is killed. For this reason, I usually configure the job to write at least 2x as many files as I think I'll need, since I may have to kill some steps that hang.


## Some use cases

### Running the same parameter set with different stimuli

First, generate the parameter set csv:

```
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 100 --param-file params.csv --create-params
```
(or you can create it by yourself, by hand, or whatever method you like. Run this code for an example of what it should look like)

You must put the stimulus into a csv file. See the "stims" directory in this repo for examples.

Then pass this params file along with the stimulus file to run.py:

```
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --outfile results_stim1.h5 --param-file params.csv --stim-file stims/chaotic_1.csv
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --outfile results_stim2.h5 --param-file params.csv --stim-file stims/some_other_stim.csv
```


## Update 2021 RBS
Using the precompiled Neuron version on /global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/
```
shifter --image=balewski/ubu18-py3-mpich:v2
PATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/bin:$PATH
PYTHONPATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/lib/python/
```
Then we need to compile the modfiles assuming the path is DL4Neurons2 root (this only needs to be done once)
```
cd modfiles
nrnivmodl
cp -r ./x86_64 ../
```
Then to check you have a working neuron with the compiled mechanisms you can write
```
nrngui
```
And neuron should start with a list of loaded mechanisms.
