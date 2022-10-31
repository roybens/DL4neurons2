from __future__ import print_function

import os
import stat
import json
import csv
import itertools
import logging as log
from argparse import ArgumentParser
from datetime import datetime
import neuron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from  toolbox.Util_H5io3 import write3_data_hdf5
import ruamel.yaml as yaml

#import yaml as yaml
# import yaml as yaml

from stimulus import stims, add_stims
import models

templates_dir = '/global/cfs/cdirs/m2043/hoc_templates/hoc_templates'

try:
    from mpi4py import MPI
    mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_tasks = comm.Get_size()
except:
    mpi = False
    comm = None
    rank = 0
    n_tasks = 1
    
from neuron import h, gui

VOLTS_SCALE = 150

MODELS_BY_NAME = models.MODELS_BY_NAME
stim_mul_range = [0.9999999999,1]
stim_offset_range = [-1e-5,0]


def _rangeify_linear(data, _range):
    return data * (_range[1] - _range[0]) + _range[0]

def _rangeify_exponential(data, _range):
    if tuple(_range) == (0, 0):
        return 0

    # return np.float_power(_range[1], data) * np.float_power(_range[0], 1-data)

    return np.exp(
        data * (np.log(_range[1]) - np.log(_range[0])) + np.log(_range[0])
    )

def get_model(model, log, m_type=None, e_type=None, cell_i=0, *params):
    if model != 'BBP':
        return MODELS_BY_NAME[model](*params, log=log)
    else:
        if m_type is None or e_type is None:
            raise ValueError('Must specify --m-type and --e-type when using BBP')
        
        if e_type == 'cADpyr':
            model = models.BBPExcV2(m_type, e_type, cell_i, *params, log=log)
        else:
            model = models.BBPInh(m_type, e_type, cell_i, *params, log=log)
            
        model.create_cell()
        return model

def clean_params(args, model):
    """convert to float, use defaults where requested

    Return: args.params, but with 'rand' replaced by float('inf') and
    'def' replaced by the default value of that param (from the
    model). Also return a list of booleans indicating whether 'def'
    was passed (this is needed to special case its value in the output
    to 1.1
    """
    defaults = model.DEFAULT_PARAMS
    if args.params:
        assert len(args.params) == len(defaults)
        defs = [param == 'def' for param in args.params]
        return [float(x if x != 'rand' else 'inf') if x != 'def' else default
#                 for (x, default) in zip(args.params, defaults)]
                for (x, default) in zip(args.params, defaults)], [False if param != 'def' else True for param in args.params]
    else:
        return [float('inf')] * len(defaults), [False] * len(defaults)


def report_random_params(args, params, model):
    param_names = model.PARAM_NAMES
    for param, name in zip(params, param_names):
        if param == float('inf'):
            log.debug("Using random values for '{}'".format(name))

    
def get_random_params(args,model,n=1):
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    ranges = model.PARAM_RANGES
    Default_paramsdf =pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv") 
    Default_params = Default_paramsdf[args.m_type+"_"+args.e_type+"_"+str(args.cell_i-1)].tolist()
    ranges=[]
    for params_single in Default_params:
        ranges.append((params_single*np.exp(int(-1)*np.log(10)),params_single*np.exp(int(+1)*np.log(10))))
    ranges = tuple(ranges)
    ndim = len(ranges)
    rand = np.random.rand(n, ndim)
    phys_rand = np.zeros(shape=rand.shape)
    params, defaulteds = clean_params(args, model)
    rangeify = _rangeify_linear if args.linear else _rangeify_exponential
    report_random_params(args, params, model)
    for i, (_range, param, varied, defaulted) in enumerate(zip(ranges, params, model.get_varied_params(), defaulteds)):
        # Default params swapped in by clean_params()

        # If that parameter is not variable (either not present, or equals zero)
        if not varied:
            rand[:, i] = 0.5 # so it gets set to 0 when writen to disk (see upar in save_h5())

        # If that parameter is in general allowed to vary (is present
        # and nonzero in the BBP model), but we asked for it to be
        # fixed by passing 'def' on the command line:
        if varied and defaulted:
            rand[:, i] = (-1.1 + 1) / 2.0 # so it gets set to -1.1

        # Put either the fixed or random params into phys_rand
        if param == float('inf'):
            #print(f'{i}-{model.PARAM_NAMES[i]}')
            if i in args.linear_params_inds:
                #print(f'{i}-{model.PARAM_NAMES[i]} is linear')
                phys_rand[:, i] = _rangeify_linear(rand[:, i], _range)
            else:
                phys_rand[:, i] = rangeify(rand[:, i], _range)
        else:
            phys_rand[:, i] = np.array([param] * n)
    phys_rand[np.isnan(phys_rand)] = 9e9
        
            
        
    return phys_rand, rand
def get_stim(args,idx):
    
    stim_fn = f'./{args.stim_file[idx]}'
    stim = np.genfromtxt(stim_fn, dtype=np.float32)
    if args.stim_multiplier:
        stim_mul = args.stim_multiplier
        print("Taking Sim from ARGS for MUL",args.stim_multiplier)    
    else:
        stim_mul = np.random.uniform(stim_mul_range[0],stim_mul_range[1])
    if args.stim_dc_offset:
        stim_offset = args.stim_dc_offset
        print("Taking Sim from ARGS for OFFSET",args.stim_dc_offset)
    else:
        stim_offset = np.random.uniform(stim_offset_range[0],stim_offset_range[1])
    stim = stim*stim_mul+stim_offset
    return stim,stim_mul,stim_offset
        
def get_mpi_idx(args, nsamples):
    if args.trivial_parallel:
        return 0, args.num
    elif args.node_parallel:
        params_per_task = (nsamples // 64) + 1
        task_i = rank % 64
    else:
        params_per_task = (nsamples // n_tasks) + 1
        task_i = rank
    start = params_per_task * task_i
    stop = min(params_per_task * (task_i + 1), nsamples)
    if args.num:
        stop = min(stop, args.num)
    log.debug("There are {} ranks, so each rank gets {} param sets".format(n_tasks, params_per_task))
    log.debug("This rank is processing param sets {} through {}".format(start, stop))

    return start, stop





def create_h5(args, nsamples,model):
    print(f'in crate h5 we have {nsamples} nsamples ')
    #log.info("Creating h5 file {}".format(args.outfile))
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    with h5py.File(args.outfile, 'w') as f:
        # write params
        ndim = len(model.PARAM_NAMES)
        f.create_dataset('phys_par', shape=(nsamples, ndim), dtype=np.float32)
        f.create_dataset('norm_par', shape=(nsamples, ndim), dtype=np.float32)
        #f.create_dataset('varParL', data=np.string_(model.PARAM_NAMES))
        #if args.model == 'BBP':
            #f.create_dataset('probeName', data=np.string_(model.get_probe_names())) ## MISSING CERTAIN PROBES



        # create stim, qa, and voltage datasets
        stim = get_stim(args,0)
        ntimepts = len(stim)
        if args.model == 'BBP':
            f.create_dataset('voltages', shape=(nsamples, ntimepts, model._n_rec_pts(),len(args.stim_file)), dtype=np.int16)
        else:
            f.create_dataset('voltages', shape=(nsamples, ntimepts), dtype=np.int16)
        f.create_dataset('stim_par', shape=(nsamples,2,len(args.stim_file)), dtype=np.int32)
        #f.create_dataset('stim', data=stim)
        #f.create_dataset('binQA', shape=(nsamples,), dtype=np.int32)
        
    #log.info("Done.")


def _normalize(args, data,model ,minmax=1):
    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    nsamples = data.shape[0]
    mins = np.array([tup[0] for tup in model.PARAM_RANGES])
    mins = np.tile(mins, (nsamples, 1)) # stacked to same shape as input
    ranges = np.array([_max - _min for (_min, _max) in model.PARAM_RANGES])
    ranges = np.tile(ranges, (nsamples, 1))

    return 2*minmax * ( (data - mins)/ranges ) - minmax

    
def save_h5(args, buf_vs, buf_stims, params, start, stop,force_serial=False, upar=None,stim_params = [],stim=None):
    #log.info("saving into h5 file {}".format(args.outfile))
    #print(f'printing soma? {model.get_probe_names()}')
    if (comm and n_tasks > 1) and not force_serial:
        log.debug("using parallel")
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        log.debug("using serial")
        kwargs = {}

    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    if not os.path.exists(args.outfile):
        create_h5(args, stop-start,model)
    
    with h5py.File(args.outfile, 'a', **kwargs) as f:
        log.debug("opened h5")
        log.debug(str(params))
        #f['voltages'][start:stop, ...] = (buf*VOLTS_SCALE).clip(-32767,32767).astype(np.int16)
        #f['binQA'][start:stop] = qa
        f['stim_par'][start:stop] = stim_params
        if not args.blind:
            f['phys_par'][start:stop, :] = params
            f['norm_par'][start:stop, :] = (upar*2 - 1) if upar is not None else _normalize(args, params,model)
        #log.info("saved h5")
    #log.info("closed h5")
    os.chmod(args.outfile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

def plot(args, data, stim):
    if args.plot is not None:
        ntimepts = len(stim)
        t_axis = np.linspace(0, ntimepts*h.dt, ntimepts)
        plt.figure(figsize=(10, 5))
        plt.xlabel('Time (ms)')
        if args.plot == [] or 'v' in args.plot:
            plt.plot(t_axis, data['v'][:ntimepts], label='V_m')
        if args.plot == [] or 'v_dend' in args.plot:
            plt.plot(t_axis, data['v_dend'][:ntimepts], label='v_dend')
        if args.plot == [] or 'stim' in args.plot:
            plt.plot(t_axis, stim[:ntimepts], label='stim')
        if args.plot == [] or 'ina' in args.plot:
            plt.plot(t_axis, data['ina'][:ntimepts] * 100, label='i_na*100')
        if args.plot == [] or 'ik' in args.plot:
            plt.plot(t_axis, data['ik'][:ntimepts] * 100, label='i_k*100')
        if args.plot == [] or 'ica' in args.plot:
            plt.plot(t_axis, data['ica'][:ntimepts] * 100, label='i_ca*100')
        if args.plot == [] or 'i_cap' in args.plot:
            plt.plot(t_axis, data['i_cap'][:ntimepts] * 100, label='i_cap*100')
        if args.plot == [] or 'i_leak' in args.plot:
            plt.plot(t_axis, data['i_leak'][:ntimepts] * 100, label='i_leak*100')

        if not args.no_legend:
            plt.legend()

        plt.show()


def lock_params(args, paramsets,model):
    # DEPRECATED. Create/use Latched model sublcasses (see HHBallStick7ParamLatched)
    assert len(args.locked_params) % 2 == 0

    #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    paramnames = model.PARAM_NAMES
    nsets = len(args.locked_params)//2
    
    targets = [args.locked_params[i*2] for i in range(nsets)]
    sources = [args.locked_params[i*2+1] for i in range(nsets)]

    for source, target in zip(sources, targets):
        source_i = paramnames.index(source)
        target_i = paramnames.index(target)
        paramsets[:, target_i] = paramsets[:, source_i]

def template_present(cellName,i_cell):
    
    i_cell = str(int(i_cell)+1)
    # template_cell = templates_dir+"/"+cellName
    cell_clones =  os.listdir(templates_dir)
    cell_clones =[x for x in cell_clones if cellName in x]
    cell_is=[]
    print(i_cell)
    for x in cell_clones:
        cell_is.append(x.split('_')[-1])
    if(i_cell not in cell_is):
        print("Template Doesnt Exist{}, Skipping".format(cellName))
        # print(template_cell)
        return False
    return True

def main(args):
    print(args)
    if args.trivial_parallel and args.outfile and '{NODEID}' in args.outfile:
        args.outfile = args.outfile.replace('{NODEID}', os.environ['SLURM_PROCID'])

    if (not args.outfile) and (not args.force) and (args.plot is None) and (not args.create_params):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")
    cellName = args.m_type+"_"+args.e_type
    # template_cell = templates_dir+"/"+cellName
    if(not template_present(cellName,args.cell_i)):
        return
    if args.cori_csv:
        cori_i = args.cori_start + int(os.environ.get('SLURM_PROCID')) % (args.cori_end - args.cori_start)
        if cori_i == 9:
            return
        with open(args.cori_csv, 'r') as infile:
            allcells = csv.reader(infile, delimiter=',')
            for i, row in enumerate(allcells):
                if i == cori_i:
                    log.debug(row)
                    bbp_name = row[0]
                    args.m_type = row[1]
                    args.e_type = row[2]
                    log.info("from rank {} running cell {}".format(rank, bbp_name))
                    print("from rank {} running cell {}".format(rank, bbp_name))
                    break

        # Get param string for holding some params fixed
        #paramuse = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1] \
        paramuse = [1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1] if args.e_type == 'cADpyr' else [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        args.params = [('inf' if use else 'def') for use in paramuse]

        if args.e_type in ('bIR', 'bAC'):
            paramuse[20] = 0
            log.info('Not varying negative parameters for e-type {}'.format(args.e_type))

    if args.outfile and '{BBP_NAME}' in args.outfile:
        args.outfile = args.outfile.replace('{BBP_NAME}', bbp_name)
        #args.metadata_file = args.metadata_file.replace('{BBP_NAME}', bbp_name)
    
    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)

    if args.create:
        if not args.num:
            raise ValueError("Must pass --num when creating h5 file")
        create_h5(args, args.num,model)
        exit()

    if args.create_params:
        np.savetxt(args.param_file, get_random_params(args,model,n=args.num)[0])
        # exit()

    if args.blind and not args.param_file:
        raise ValueError("Must pass --param-file with --blind")

    
    if args.metadata_only:
        write_metadata(args, model)
        exit()
    
    if args.param_file:
        all_paramsets = np.genfromtxt(args.param_file, dtype=np.float32)
        upar = None # TODO: save or generate unnormalized params when using --param-file
        start, stop = get_mpi_idx(args, len(all_paramsets))
        print("Reading from param_file")
        print(start,stop)
        if args.num and start > args.num:
            return
        if(args.num==1):
            paramsets=[]
            paramsets.append(all_paramsets)
        else:
            paramsets = all_paramsets[start:stop, :]
        paramsets = np.atleast_2d(paramsets)
        print("Param Size",paramsets.size)
        print("Reading from param_file")
        print(paramsets.shape)

    elif args.num:
        start, stop = get_mpi_idx(args, args.num)
        paramsets, upar = get_random_params(args, model,n=stop-start)
    elif args.params not in (None, [None]):
        #paramsets = np.atleast_2d(np.array(args.params))
        #NEED TO CHANGE THIS
        paramsets = np.atleast_2d(model.DEFAULT_PARAMS)
        upar = None
        start, stop = 0, 1
    
    else:
        log.info("Cell parameters not specified, running with default parameters")
        paramsets = np.atleast_2d(model.DEFAULT_PARAMS)
        upar = None
        start, stop = 0, 1
    stimL = []

    # MAIN LOOP   
    lock_params(args, paramsets,model)
    stim,stim_mul,stim_offset = get_stim(args,0)#only for gettign the length to create the buffer
    buf_vs = np.zeros(shape=(stop-start, len(stim), model._n_rec_pts(),len(args.stim_file)), dtype=np.float32)
    buf_stims = np.zeros(shape=(stop-start, 2,len(args.stim_file)), dtype=np.float32)
    print('dd',type(paramsets),paramsets.shape)
    model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)
    model.set_attachments(stim,len(stim),args.dt)
    counter_params =0
    
    for iSamp, params in enumerate(paramsets):
        
        if args.print_every and iSamp % args.print_every == 0:
            log.info("Processed {} samples".format(i))
        log.debug("About to run with params = {}".format(params))
        
        #print(f'printing buf {buf}, printing shape{buf.shape}')
        for stim_idx in range(len(args.stim_file)):
            stim,stim_mul,stim_offset = get_stim(args,stim_idx)
            #stimL.append(stim)
            buf_stims[iSamp,:,stim_idx]=np.array([stim_mul,stim_offset])
            #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)   
            #model = get_model(args.model, log, args.m_type, args.e_type, args.cell_i)   
            print("Conter Param",counter_params)
            counter_params+=1
            print("Simulating for ",params)    
            model._set_self_params(*params)
            model.init_parameters()
            print(model.param_dict())
            data = model.simulate(stim, args.dt)
            Data = data[list(data.keys())[0]]
            plt.plot(Data)
            print('aaa',buf_vs[0,:,:,stim_idx].shape,len(data.values()))
            for iProb,k in enumerate(data):
                wave=data[k][:-1]
                print('bbb',iProb,k,wave.shape)
                buf_vs[iSamp,:,iProb,stim_idx]=wave#change here!!!!
    plt.savefig("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/TESTING.png")
    plot(args, data, stim)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("COMPLETED ALL SIMULATIONS",dt_string)

    # Save to disk
    if args.outfile:
        myUpar= _normalize(args, paramsets,model).astype(np.float32)
        buf_vs=(buf_vs*VOLTS_SCALE).clip(-32767,32767).astype(np.int16)
        myUpar=myUpar*2 -1
        assert not np.isnan(buf_vs).any()
        assert not np.isnan(buf_stims).any()
        bbp_name = model.cell_kwargs['model_directory']
        outD={'volts':buf_vs,'stim_par':buf_stims,'phys_par':paramsets.astype(np.float32),'unit_par':myUpar}
        outF=args.outfile
        
        metadata = {
        'timeAxis': {'step': args.dt, 'unit': "(ms)"},
        'voltsScale': VOLTS_SCALE,
        'probeName': str(model.get_probe_names()),
        'bbpName': bbp_name,
        'parName': model.PARAM_NAMES,
        'linearParIdx': args.linear_params_inds,
        'stimParRange':[stim_mul_range,stim_offset_range],
        'physParRange':model.PARAM_RANGES,
        'jobId':os.environ['SLURM_ARRAY_JOB_ID'],
        'stimName': args.stim_file, 
        'neuronSimVer': neuron.__version__
    }
        write3_data_hdf5(outD,outF,metaD=metadata)

        
if __name__ == '__main__':
    parser = ArgumentParser()

    with open('cells.json') as infile:
        cells = json.load(infile)
        ALL_MTYPES = cells.keys()
        ALL_ETYPES = list(set(itertools.chain.from_iterable(mtype.keys() for mtype in cells.values())))

    parser.add_argument('--model', choices=MODELS_BY_NAME.keys(),
                        default='hh_ball_stick_7param')
    parser.add_argument('--m-type', choices=ALL_MTYPES, required=False, default=None)
    parser.add_argument('--e-type', choices=ALL_ETYPES, required=False, default=None)
    parser.add_argument('--cell-i', type=int, required=False, default=0)
    parser.add_argument('--cori-start', type=int, required=False, default=None, help='start cell')
    parser.add_argument('--cori-end', type=int, required=False, default=None, help='end cell')
    parser.add_argument('--cori-csv', type=str, required=False, default=None,
                        help='When running BBP on cori, use SLURM_PROCID to compute m-type and e-type from the given cells csv')
    
    parser.add_argument('--celsius', type=float, default=34)
    parser.add_argument('--dt', type=float, default=.025)

    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to')
    parser.add_argument('--metadata-file', type=str, required=False, default=None,
                        help='for BBP only')
    parser.add_argument('--metadata-only', action='store_true', default=False,
                        help='create metadata file then exit')
    parser.add_argument('--create', action='store_true', default=False,
                        help="create the file, store all stimuli, and then exit " \
                        + "(useful for writing to the file from multiple ranks)"
    )
    parser.add_argument(
        '--create-params', action='store_true', default=False,
        help="create the params file (--param-file) and exit. Must use with --num"
    )

    parser.add_argument(
        '--plot', nargs='*',
        choices=['v', 'stim', 'ina', 'ica', 'ik', 'i_leak', 'i_cap', 'v_dend'],
        default=None,
        help="--plot w/ no arguments: plot everything. --plot [args]: print the given variables"
    )
    parser.add_argument('--no-legend', action='store_true', default=False,
                        help="do not display the legend on the plot")
    
    parser.add_argument(
        '--force', action='store_true', default=False,
        help="make the script run even if you don't plot or save anything"
    )

    # CHOOSE PARAMETERS
    parser.add_argument(
        '--num', type=int, default=None,
        help="number of param values to choose. Will choose randomly. " + \
        "See --params. When multithreaded, this is the total number over all ranks, " + \
        "except when using --trivial-parallel"
    )
    parser.add_argument(
        '--trivial-parallel', action='store_true', default=False, required=False,
        help='each process runs all --num samples, with each rank writing output to a ' + \
        'separate file.'
    )
    parser.add_argument(
        '--node-parallel', action='store_true', default=False, required=False,
        help='each node runs --num samples over 64 processes. One output file per node'
    )
    parser.add_argument(
        '--params', type=str, nargs='+', default=None,
        help='When used with --num, fixes the value of some params. To indicate ' + \
        'that a param should not be held fixed, set it to "rand". ' + \
        'to use the default value, use "def"' + \
        'eg to use the default 1st param, random 2nd param, ' + \
        'and specific values 3.0 and 4.0 for the 3rd and 4th params, use "def inf 3.0 4.0"'
    )
    parser.add_argument('--param-file', '--params-file', type=str, default=None)
    parser.add_argument(
        '--blind', action='store_true', default=False,
        help='do not save parameter values in the output nwb. ' + \
        'You better have saved them using --param-file'
    )
    parser.add_argument(
        '--linear', action='store_true', default=False,
        help='when selecting random params, distribute them uniformly' + \
        'throughout the range, rather than exponentially'
    )
    parser.add_argument(
        '--linear-params-inds', type=int, nargs='*', default=[],required=False,
        help='When used with --num, indicates which parameters should be randomized linearly all the other would be randomized exponentially or linearly if --linear is true'
    )
    #CHOOSE PROBES
    parser.add_argument(
        '--axon-probes', type=float,nargs='*',
        help="specify the distances of axonal probe/s from the soma")
    
    parser.add_argument(
        '--dend-probes', type=float,nargs='*',
        help="specify the distances of axonaldend probe/s from the soma")
    

    # CHOOSE STIMULUS
    parser.add_argument(
        '--stim-file', type=str,nargs='+', default=os.path.join('stims', 'chaotic_2.csv'),
        help="csv to use as the stimulus")
    
    parser.add_argument(
        '--stim-dc-offset', type=float, default=0.0,
        help="apply a DC offset to the stimulus (shift it). Happens after --stim-multiplier"
    )
    parser.add_argument(
        '--stim-multiplier',  type=float, default=1,
        help="scale the stimulus amplitude by this factor. Happens before --stim-dc-offset"
    )
    parser.add_argument(
        '--stim-noise', action='store_true', default=False,
        help="add random multiplier and dc-offset to the stim in the range hard coded in this script this will override --stim-multiplier and stim-dc-offest"
    )
    parser.add_argument(
        '--tstart', type=int, default=None, required=False,
        help='when to start the recording'
    )
    parser.add_argument(
        '--tstop', type=int, default=None, required=False,
        help='when to stop the recording'
    )

    parser.add_argument('--print-every', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--locked-params', '--lock-params', type=str, nargs='+', default=[])
    
    args = parser.parse_args()

    if args.tstart or args.tstop:
        raise ValueError('--tstart and --tstop not yet implemented')

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)

    main(args)
