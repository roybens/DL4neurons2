# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:59:43 2019

@author: bensr
"""



import sys
from pathlib import Path
# sys.path.insert(1, '../DL4neurons2/')
parent_dir = Path(__file__).resolve().parent.parent

# Append the parent directory to sys.path
sys.path.append(str(parent_dir))

from run import get_model
import logging as log
import models
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import itertools
import pickle as pkl
import random
import pandas as pd
import glob
# stimfn = '/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/stims/5k0chaotic5A.csv'
stimfn = sys.argv[9]
stim_path = Path(stimfn)
stim_name = stim_path.stem

stim =  np.genfromtxt(stimfn, dtype=np.float32) 
# print("Stimulus diagnostics:")
# print("Stim length:", len(stim))
# print("Stim min:", np.min(stim))
# print("Stim max:", np.max(stim))


plt.subplots_adjust(hspace=0.3)
times = [0.025*i for i in range(len(stim))]
templates_dir = '/global/cfs/cdirs/m3513/M1_Hoc_template/HocTemplate'
Default_Parameters = pd.read_csv("/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv") 
Bounds = pd.read_csv("/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/sensitivity_analysis/Bounds.csv")
# Default_Parameters = Default_Parameters['New Base'].tolist()

def compute_global_bounds_with_log_perturbation(baseline_dir, log_range=1.0):
    """
    Computes global min/max bounds by aggregating per-neuron log perturbation ranges.

    For each baseline value p0:
        lower = p0 * 10^(-log_range)
        upper = p0 * 10^(+log_range)

    Then:
        global_min = min(all lowers)
        global_max = max(all uppers)
    """

    csv_files = sorted(glob.glob(os.path.join(baseline_dir, "*.csv")))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {baseline_dir}")

    print(f"[INFO] Found {len(csv_files)} baseline CSVs")

    all_lowers = []
    all_uppers = []

    for f in csv_files:
        df = pd.read_csv(f)
        vals = df["Values"].values.astype(float)

        lowers = vals * (10 ** (-log_range))
        uppers = vals * (10 ** (log_range))

        all_lowers.append(lowers)
        all_uppers.append(uppers)

    all_lowers = np.array(all_lowers)
    all_uppers = np.array(all_uppers)

    global_mins = np.min(all_lowers, axis=0)
    global_maxs = np.max(all_uppers, axis=0)

    param_names = pd.read_csv(csv_files[0])["Parameters"].tolist()

    print(f"[INFO] Computed global bounds using ±{log_range} log range")

    for i, name in enumerate(param_names):
        print(f"{name}: [{global_mins[i]:.6e}, {global_maxs[i]:.6e}]")

    return param_names, global_mins, global_maxs

def make_paramset(my_model,param_ind,nsamples):
    def_param_vals = my_model.DEFAULT_PARAMS
    param_set = np.array([def_param_vals]*nsamples)
    range_to_vary = my_model.PARAM_RANGES[param_ind]
    #vals_check = np.linspace(range_to_vary[0],range_to_vary[1],nsamples)
    vals_check=def_param_vals[param_ind]*np.exp(np.random.uniform(-1,1,size=nsamples)*np.log(10))
    param_set[:,param_ind] = vals_check
    return param_set

def make_paramset_regions(my_model,param_ind,nsamples,nregions,mtype,etype,i_cell, start_bound, end_bound, param_csv_path):
    # def_param_vals = my_model.DEFAULT_PARAMS
    # Mean_param_values=pd.read_csv("/global/homes/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams0.csv")
    # def_param_vals = Mean_param_values["Values"]
    if param_csv_path and os.path.exists(param_csv_path):
        print(f"Loading parameter values from: {param_csv_path}")
        df = pd.read_csv(param_csv_path)
        def_param_vals = df["Values"].values
    else:
        def_param_vals = my_model.DEFAULT_PARAMS
    a_value = 0
    b_value = 1

    param_name = my_model.PARAM_NAMES[param_ind]
    print(f"\nParameter: {param_name} (index {param_ind})")
    print(f"Default value: {def_param_vals[param_ind]:.6e}")
    
    if(False):    
        def_param_vals= Default_Parameters[mtype+"_"+etype+"_"+str(i_cell)].tolist()
        param_name = my_model.PARAM_NAMES[param_ind]
        bounds = Bounds.loc[Bounds['Parameter']==param_name]
        LB = bounds['LB']
        UB = bounds['UB']
        # TO CALCULATE THE VLAUE :Value = Base* e^(x*a*ln10)
        #a =(Upper Bound - LowerBound)/2
        a_value = (UB-LB)/2
        a_value = float(a_value.iloc[0])
    param_sets = []
    # range_to_vary = my_model.PARAM_RANGES[param_ind]
    region_width = (end_bound - start_bound) / nregions
    for curr_region in range(nregions):
        curr_lb = start_bound + (curr_region)*(region_width)
        curr_ub = start_bound + (curr_region+1)*(region_width)
        curr_param_set = np.array([def_param_vals]*nsamples)
        curr_vals_check=def_param_vals[param_ind]*np.exp(np.random.uniform(curr_lb,curr_ub,size=nsamples)*b_value*np.log(10))
        
        print(f"\nRegion {curr_region}: bounds [{curr_lb:.3f}, {curr_ub:.3f}]")
        print("Perturbed values:")
        for i, perturbed_val in enumerate(curr_vals_check):
            perturbation_factor = perturbed_val / def_param_vals[param_ind]
            random_val = np.log(perturbation_factor) / (b_value * np.log(10))
            print(f"  Sample {i}: {perturbed_val:.6e} (x{perturbation_factor:.3f}, random_val: {random_val:.3f})")

        curr_param_set[:,param_ind] = curr_vals_check
        param_sets.append(curr_param_set)
    return param_sets


# decentralzed version of make_paramset_regions that uses global bounds
# def make_paramset_regions(my_model, param_ind, nsamples, nregions,
#                           mtype, etype, i_cell,
#                           param_csv_path,
#                           global_mins, global_maxs):

#     # --- Load BASELINE parameters (per neuron, unchanged) ---
#     if param_csv_path and os.path.exists(param_csv_path):
#         print(f"Loading BASELINE parameter values from: {param_csv_path}")
#         df_base = pd.read_csv(param_csv_path)
#         baseline_param_vals = df_base["Values"].values
#     else:
#         baseline_param_vals = my_model.DEFAULT_PARAMS

#     param_name = my_model.PARAM_NAMES[param_ind]

#     global_min = global_mins[param_ind]
#     global_max = global_maxs[param_ind]

#     print(f"\nParameter: {param_name} (index {param_ind})")
#     print(f"Global bounds: [{global_min:.6e}, {global_max:.6e}]")

#     param_sets = []

#     # --- Define regions in absolute parameter space ---
#     region_width = (global_max - global_min) / nregions

#     for curr_region in range(nregions):

#         curr_lb = global_min + curr_region * region_width
#         curr_ub = global_min + (curr_region + 1) * region_width

#         print(f"\nRegion {curr_region}: bounds [{curr_lb:.6e}, {curr_ub:.6e}]")

#         # Copy baseline for all samples
#         curr_param_set = np.array([baseline_param_vals] * nsamples)

#         # Sample directly in absolute space
#         perturbed_vals = np.random.uniform(curr_lb, curr_ub, size=nsamples)

#         for i, val in enumerate(perturbed_vals):
#             print(f"  Sample {i}: {val:.6e}")

#         # Replace only the target parameter
#         curr_param_set[:, param_ind] = perturbed_vals

#         param_sets.append(curr_param_set)

#     return param_sets


def get_volts(mtype,etype,param_ind,nsamples):
    all_volts = []
    my_model = get_model('BBP',log,m_type=mtype,e_type=etype,cell_i=1) 
    param_set = make_paramset(my_model,param_ind,nsamples)
    #param_name = my_model.PARAM_NAMES[param_ind]
    for i in range(nsamples):
        print("working on param_ind" + str(param_ind) + " sample" + str(i))
        params = param_set[i]
        my_model = get_model('BBP',log,mtype,etype,1,*params)
        my_model.DEFAULT_PARAMS = False
        volts = my_model.simulate(stim,0.1)
        all_volts.append(volts)
    return all_volts
# def get_volts_regions(mtype,etype,i_cell,param_ind,nsamples,nregions, start_bound, end_bound, param_csv_path, global_mins, global_maxs):
def get_volts_regions(mtype,etype,i_cell,param_ind,nsamples,nregions, start_bound, end_bound, param_csv_path):
    all_volts = []
    my_model = get_model(model_name,log,m_type=mtype,e_type=etype,cell_i=int(i_cell)) 
    my_model.set_attachments(stim,len(stim),0.1)
    param_sets = make_paramset_regions(my_model,param_ind,nsamples,nregions,mtype,etype,i_cell, start_bound, end_bound, param_csv_path)
    # param_sets = make_paramset_regions(my_model,param_ind,nsamples,nregions,mtype,etype,i_cell, param_csv_path, global_mins, global_maxs)
    param_name = my_model.PARAM_NAMES[param_ind]
    for params_set in param_sets:
        region_volts = []
        for i in range(nsamples):
            curr_params = params_set[i]
            print("working on param_ind" + str(param_ind) + " sample" + str(i) )
            # my_model = get_model('BBP',log,mtype,etype,int(i_cell),*curr_params)
            # my_model.DEFAULT_PARAMS = False

            my_model._set_self_params(*curr_params)
            my_model.init_parameters()
            print(f"Parameter {param_ind} value in model: {getattr(my_model, my_model.PARAM_NAMES[param_ind])}")
            print(f"Should be: {curr_params[param_ind]}")

            curr_volts = my_model.simulate(stim,0.1)
            if param_ind == 0 and i == 0:
                if hasattr(curr_volts, "to_python"):
                    volts = np.array(curr_volts.to_python())
                else:
                    volts = np.array(curr_volts)

                print("Voltage diagnostics:")
                print("Max voltage:", np.max(volts))
                print("Min voltage:", np.min(volts))

                # spike_count = np.sum((volts[:-1] < 0) & (volts[1:] >= 0))
                # print("Spike count:", spike_count)

            region_volts.append(curr_volts)
        all_volts.append(region_volts)
    return all_volts

def get_rec_sec(def_volts,adjusted_param):
    probes = list(def_volts.keys())
    rec_sec=adjusted_param
    if 'soma' in adjusted_param:
        rec_sec = probes[0]
    if 'apic' in adjusted_param or 'dend' in adjusted_param:
        res = [i for i in probes if 'apic' in i or 'dend' in i]
        rec_sec = res[2]   
    if 'axon' in adjusted_param:
        res = [i for i in probes if 'axon' in i]
        rec_sec = res[2]  
    dot_ind = rec_sec.find('.')+1
    return rec_sec[dot_ind:],rec_sec[:dot_ind]
    
def check_param_sensitivity(all_volts,def_volts_probes,adjusted_param,m_type,e_type):
    fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
    fig.suptitle(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(def_volts_probes,adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    def_volts = def_volts_probes.get(prefix + def_rec_sec)
    ax1.plot(times,def_volts[:-1],'black')
    def_cum_sum = np.cumsum(np.abs(def_volts))*0.025
    cum_sum_errs = []
    plt.subplots_adjust(hspace=0.3)
    for curr_volts in all_volts:
        curr_rec_sec,prefix = get_rec_sec(curr_volts,adjusted_param)
        if (curr_rec_sec != def_rec_sec):
            print("curr_rec_sec is " + curr_rec_sec + 'and def rec_sec is' + def_rec_sec )
        volts_to_plot = curr_volts.get(prefix +def_rec_sec)
        curr_cum_sum= np.cumsum(np.abs(volts_to_plot))*0.025
        cum_sum_err = curr_cum_sum - def_cum_sum
        err = def_volts - volts_to_plot
        ax1.plot(times,volts_to_plot[:-1])
        ax2.plot(times,err[:-1])
        ax3.plot(times,cum_sum_err[:-1])
        cum_sum_errs.append(cum_sum_err)
    fig.suptitle('m_type + e_type + adjusted_param')
    ax1.title.set_text('Volts')
    ax2.title.set_text('error')
    ax3.title.set_text('cum_sum_error')
    fig_name = m_type + e_type + adjusted_param +'.pdf'
    fig.savefig(fig_name)
    return cum_sum_errs
#analyze_volts([])
#with open('cells.json') as infile:
#        cells = json.load(infile)
#        ALL_MTYPES = cells.keys()
#        ALL_ETYPES = list(set(itertools.chain.from_iterable(mtype.keys() for mtype in cells.values())))

def main_for_all_range():
    NTHREADS = 128
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    nsamples = int(sys.argv[3])
    
    try:
        procid = int(os.environ['SLURM_PROCID'])
        print("in cori")
        
    except:
        print("not in cori")
        procid = 0   
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0)
    
    def_vals = my_model.DEFAULT_PARAMS
    pnames = [my_model.PARAM_NAMES[i] for i in range(len(def_vals)) if def_vals[i]>0]
    threads_per_param = int(NTHREADS/len(pnames))
    samples_per_thread = int(nsamples/threads_per_param)+1
    p_ind = procid%(len(pnames))
    adjusted_param = my_model.PARAM_NAMES[p_ind]
    print("working on " + adjusted_param + "will be sampled " + str(samples_per_thread*threads_per_param) )
    all_volts = get_volts(m_type,e_type,p_ind,samples_per_thread)
    pkl_fn=m_type + '_' + e_type + adjusted_param + '_' + str(procid) + '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_volts,output)
        
        
def main_for_divided_range():
    
    NTHREADS = 128
    print(sys.argv)
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    i_cell = sys.argv[3]
    nsamples = int(sys.argv[4])
    nregions = int(sys.argv[5])
    global model_name
    model_name = sys.argv[6]
    start_bound = float(sys.argv[7])
    end_bound = float(sys.argv[8])
    param_csv_path = sys.argv[10] if len(sys.argv) > 10 else None

    if param_csv_path:
        filename = os.path.basename(param_csv_path).replace('.csv', '')
        parts = filename.split('_')
        layer_mtype = '_'.join(parts[:-3])  # drop last 3 parts
    else:
        layer_mtype = "UNKNOWN"

    # files_loc = f'/pscratch/sd/s/sdough/sens_ana/developing_model_somarun/{m_type}_{e_type}_{i_cell}/'
    # base_number = param_csv_path.split('_')[-1].split('.')[0]
    files_loc = f'/pscratch/sd/s/sdough/sens_ana/Developing_Model_{layer_mtype}_{stim_name}_/{m_type}_{e_type}_{i_cell}/'
    # files_loc = f'/pscratch/sd/s/sdough/sens_ana/developing_model_{stim_name}_100_test/{m_type}_{e_type}_{i_cell}/'


    try:
        procid = int(os.environ['SLURM_PROCID'])
        print("in cori")
        
    except:
        print("not in cori")
        procid = 0   
    cellName = m_type+"_"+e_type
    # template_cell = templates_dir+"/"+cellName
    cell_clones =  os.listdir(templates_dir)
    cell_clones =[x for x in cell_clones if cellName in x]
    cell_is=[]
    for x in cell_clones:
        cell_is.append(x.split('_')[-1])
    if(str(int(i_cell)+1) not in cell_is):
        print(cell_is,str(int(i_cell)+1))
        print("Template Doesnt Exist{}, Skipping".format(cellName))
        return
    os.makedirs(files_loc,exist_ok=True)
    try:
        my_model = get_model(model_name,log,m_type=m_type,e_type=e_type,cell_i=int(i_cell))
        
        def_vals = my_model.DEFAULT_PARAMS
        pnames = [my_model.PARAM_NAMES[i] for i in range(len(def_vals)) ]#if def_vals[i]>0]

        print(f"\n{'='*80}")
        print(f"ALL DEFAULT PARAMETER VALUES for {m_type}_{e_type}:")
        print(f"{'='*80}")
        for i, (name, value) in enumerate(zip(my_model.PARAM_NAMES, def_vals)):
            print(f"Parameter {i}: {name} = {value:.6e}")

        threads_per_param = int(NTHREADS/len(pnames))
        if threads_per_param < 1:
            threads_per_param = 1 
        samples_per_thread = int(nsamples/threads_per_param)+1
        p_ind = procid%(len(pnames))
        adjusted_param = my_model.PARAM_NAMES[p_ind]
        print("working on " + adjusted_param + "will be sampled " + str(samples_per_thread*threads_per_param) )
        all_volts = get_volts_regions(m_type,e_type,i_cell,p_ind,samples_per_thread,nregions, start_bound, end_bound, param_csv_path)
        print("SAVING in pkl")
        pkl_fn =files_loc + str(nregions) + 'regions_' + m_type + '_' + e_type + adjusted_param + '_' + str(procid) + '.pkl'
        print(pkl_fn)
        with open(pkl_fn, 'wb') as output:
            pkl.dump(all_volts,output)
    except FileNotFoundError:
        print("FILE not found for ",i_cell)

# def main_for_divided_range():
    
#     NTHREADS = 128
#     print(sys.argv)
#     m_type = sys.argv[1]
#     e_type = sys.argv[2]
#     i_cell = sys.argv[3]
#     nsamples = int(sys.argv[4])
#     nregions = int(sys.argv[5])
#     global model_name
#     model_name = sys.argv[6]
#     start_bound = float(sys.argv[7])
#     end_bound = float(sys.argv[8])
#     param_csv_path = sys.argv[10] if len(sys.argv) > 10 else None
#     baseline_dir = "/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/baseline_params"

#     if param_csv_path:
#         filename = os.path.basename(param_csv_path).replace('.csv', '')
#         parts = filename.split('_')
#         layer_mtype = '_'.join(parts[:-3])
#     else:
#         layer_mtype = "UNKNOWN"

#     files_loc = f'/pscratch/sd/s/sdough/sens_ana/developing_model_{layer_mtype}_{stim_name}_/{m_type}_{e_type}_{i_cell}/'

#     try:
#         procid = int(os.environ['SLURM_PROCID'])
#         print("in cori")
        
#     except:
#         print("not in cori")
#         procid = 0   

#     cellName = m_type + "_" + e_type
#     cell_clones = os.listdir(templates_dir)
#     cell_clones = [x for x in cell_clones if cellName in x]

#     cell_is = []
#     for x in cell_clones:
#         cell_is.append(x.split('_')[-1])

#     if(str(int(i_cell)+1) not in cell_is):
#         print(cell_is, str(int(i_cell)+1))
#         print("Template Doesnt Exist{}, Skipping".format(cellName))
#         return

#     os.makedirs(files_loc, exist_ok=True)

#     try:
#         my_model = get_model(model_name, log, m_type=m_type, e_type=e_type, cell_i=int(i_cell))

#         def_vals = my_model.DEFAULT_PARAMS
#         pnames = [my_model.PARAM_NAMES[i] for i in range(len(def_vals))]

#         print(f"\n{'='*80}")
#         print(f"ALL DEFAULT PARAMETER VALUES for {m_type}_{e_type}:")
#         print(f"{'='*80}")
#         for i, (name, value) in enumerate(zip(my_model.PARAM_NAMES, def_vals)):
#             print(f"Parameter {i}: {name} = {value:.6e}")

#         # ---------------------------------------------------------
#         # 🔥 NEW: compute global bounds ONCE
#         # ---------------------------------------------------------
#         if param_csv_path is not None:
#             print("[INFO] Computing global parameter bounds...")
#             param_names, global_mins, global_maxs = compute_global_bounds_with_log_perturbation(
#                 os.path.dirname(param_csv_path)
#             )
#         else:
#             global_mins = None
#             global_maxs = None

#         threads_per_param = int(NTHREADS / len(pnames))
#         if threads_per_param < 1:
#             threads_per_param = 1 

#         samples_per_thread = int(nsamples / threads_per_param) + 1

#         p_ind = procid % (len(pnames))
#         adjusted_param = my_model.PARAM_NAMES[p_ind]

#         print("working on " + adjusted_param + "will be sampled " + str(samples_per_thread * threads_per_param))

#         # ---------------------------------------------------------
#         # 🔥 MODIFIED CALL: pass global bounds
#         # ---------------------------------------------------------
#         all_volts = get_volts_regions(
#             m_type,
#             e_type,
#             i_cell,
#             p_ind,
#             samples_per_thread,
#             nregions,
#             start_bound,
#             end_bound,
#             param_csv_path,
#             global_mins,
#             global_maxs
#         )

#         print("SAVING in pkl")
#         pkl_fn = files_loc + str(nregions) + 'regions_' + m_type + '_' + e_type + adjusted_param + '_' + str(procid) + '.pkl'

#         print(pkl_fn)
#         with open(pkl_fn, 'wb') as output:
#             pkl.dump(all_volts, output)

#     except FileNotFoundError:
#         print("FILE not found for ", i_cell)


log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG)
#main_for_all_range()
main_for_divided_range()