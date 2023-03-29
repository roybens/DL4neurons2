# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:59:43 2019

@author: bensr
"""

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
stimfn = './stims/chaotic_1.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.02*i for i in range(len(stim))]

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
def check_param_sensitivity(all_volts,adjusted_param):
    fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
    fig.suptitle(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(all_volts[0],adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    #ax1.plot(times,def_volts[:-1],'black')
    cum_sum_errs = []
    plt.subplots_adjust(hspace=0.3)
    for i in range(int(len(all_volts)/2)):
        volts1 = all_volts[2*i]
        volts2 = all_volts[2*i + 1]
        curr_rec_sec1,prefix1 = get_rec_sec(volts1,adjusted_param)
        curr_rec_sec2,prefix2 = get_rec_sec(volts2,adjusted_param)
        if (curr_rec_sec1 != curr_rec_sec2):
            print("curr_rec_sec is " + curr_rec_sec1 + 'and curr_rec_sec2  is' + curr_rec_sec2 )
        volts_to_plot1 = volts1.get(prefix1 +def_rec_sec)
        volts_to_plot2 = volts2.get(prefix2 +def_rec_sec)
        curr_cum_sum1= np.cumsum(np.abs(volts_to_plot1))*0.02
        curr_cum_sum2= np.cumsum(np.abs(volts_to_plot2))*0.02
        cum_sum_err = curr_cum_sum1 - curr_cum_sum2
        err = volts_to_plot1 - volts_to_plot2
        ax1.plot(times,volts_to_plot1[:-1])
        ax1.plot(times,volts_to_plot2[:-1])
        ax2.plot(times,err[:-1])
        ax3.plot(times,cum_sum_err[:-1])
        cum_sum_errs.append(cum_sum_err)
    ax1.title.set_text('Volts')
    ax2.title.set_text('error')
    ax3.title.set_text('cum_sum_error')
    fig_name = mtype + etype + adjusted_param +'.pdf'
    fig.savefig(fig_name)
    return cum_sum_errs


def check_param_sensitivityv2(all_volts,def_volts_probes,adjusted_param):
    fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
    def_volts_probes=all_volts[0]
    fig.suptitle(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(def_volts_probes,adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    
    def_volts = def_volts_probes.get(prefix + def_rec_sec)
    ax1.plot(times,def_volts[:-1],'black')
    def_cum_sum = np.cumsum(np.abs(def_volts))*0.02
    cum_sum_errs = []
    plt.subplots_adjust(hspace=0.3)
    
    for curr_volts in all_volts:
        
        curr_rec_sec,prefix = get_rec_sec(curr_volts,adjusted_param)
        if (curr_rec_sec != def_rec_sec):
            print("curr_rec_sec is " + curr_rec_sec + 'and def rec_sec is' + def_rec_sec )
        volts_to_plot = curr_volts.get(prefix +def_rec_sec)
        curr_cum_sum= np.cumsum(np.abs(volts_to_plot))*0.02
        cum_sum_err = curr_cum_sum - def_cum_sum
        err = def_volts - volts_to_plot
        ax1.plot(times,volts_to_plot[:-1])
        ax2.plot(times,err[:-1])
        ax3.plot(times,cum_sum_err[:-1])
        cum_sum_errs.append(cum_sum_err)
    ax1.title.set_text('Volts')
    ax2.title.set_text('error')
    ax3.title.set_text('cum_sum_error')
    fig_name = mtype + etype + adjusted_param +'.pdf'
    fig.savefig(fig_name)
    return cum_sum_errs

def test_sensitivity(mtype,etype,def_volts,nsamples):
    param_names = my_model.PARAM_NAMES
    all_ECDS ={}
    for i in range(len(param_names)):
        adjusted_param = param_names[i]
        
        pkl_fn=mtype + etype + adjusted_param + '.pkl'
        try:
            with open(pkl_fn, 'rb') as f:
                all_volts = pkl.load(f) 
        except:
            print("no file" + pkl_fn)
            continue
        curr_errs = check_param_sensitivity(all_volts,adjusted_param)
        curr_ECDs = [ls[-1] for ls in curr_errs]
        all_ECDS[adjusted_param]=curr_ECDs
    pkl_fn=mtype + etype + 'sensitivity.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_ECDS,output)
        pkl.dump(param_names,output)
    return all_ECDS



def analyze_ecds(ECDS,def_vals):
    ymx_axon = 700
    ymx_dend = 300
    ymx_soma = 600
    threshold_axon = 50
    threshold_soma = 70
    threshold_dend = 100
    param_names = list(ECDS.keys())
    pnames_soma = []
    pnames_axon = []
    pnames_dend = []
    means_axon = []
    STDs_axon = []
    means_soma = []
    STDs_soma = []
    means_dend = []
    STDs_dend = []
    params_sensitivity_dict = {}
    nsamples = 50
    param_inds = range(len(param_names))
    for i in param_inds:
        if def_vals[i] <= 0:
            continue
        curr_ecds = np.abs(ECDS[param_names[i]])
        nsamples = len(curr_ecds)
        curr_mean = np.mean(curr_ecds)
        curr_std = np.std(curr_ecds)/np.sqrt(len(curr_ecds))
        params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std]
        if 'soma' in param_names[i]:
            pnames_soma.append(param_names[i][0:8])
            means_soma.append(curr_mean)
            STDs_soma.append(curr_std)
        if 'apic' in param_names[i] or 'dend' in param_names[i]:
            pnames_dend.append(param_names[i][0:8])
            means_dend.append(curr_mean)
            STDs_dend.append(curr_std)
        if 'axon' in param_names[i]:
            pnames_axon.append(param_names[i][0:8])
            means_axon.append(curr_mean)
            STDs_axon.append(curr_std)
    pkl_fn='mean_std_sensitivity' + mtype + etype + str(nsamples) + '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(params_sensitivity_dict,output)
    fig, ((ax_soma,ax_dend),( ax_axon,ax4))= plt.subplots(2,2,figsize=(15,15))
    fig.suptitle('Sensitivity analysis')
    ax_axon.title.set_text('Axonal Parameters')
    means_axon = np.clip(means_axon,0,ymx_axon)
    STDs_axon = np.clip(STDs_axon,0,ymx_axon/2)
    ax_axon.errorbar(range(len(pnames_axon)), means_axon, STDs_axon, linestyle='None', marker='s')
    ax_axon.set_xticks(range(len(pnames_axon)))
    ax_axon.set_xticklabels(pnames_axon)
    ax_axon.axhline(threshold_axon,color='red')
    ax_axon.grid()
    ax_dend.title.set_text('Dendritic Parameters')
    means_dend = np.clip(means_dend,0,ymx_dend)
    STDs_dend = np.clip(STDs_dend,0,ymx_dend/2)
    ax_dend.errorbar(range(len(pnames_dend)), means_dend, STDs_dend, linestyle='None', marker='s')
    ax_dend.set_xticks(range(len(pnames_dend)))
    ax_dend.set_xticklabels(pnames_dend)
    ax_dend.axhline(threshold_dend,color='red')
    ax_dend.grid()
    ax_soma.grid()
    ax_soma.title.set_text('Somatic Parameters')
    means_soma = np.clip(means_soma,0,ymx_soma)
    STDs_soma = np.clip(STDs_soma,0,ymx_soma/2)
    ax_soma.errorbar(range(len(pnames_soma)), means_soma, STDs_soma, linestyle='None', marker='s')
    ax_soma.set_xticks(range(len(pnames_soma)))
    ax_soma.set_xticklabels(pnames_soma)
    ax_soma.axhline(threshold_soma,color='red')
    fig_name = mtype + etype  + str(nsamples) + 'Analysis.pdf'
    fig.savefig(fig_name)
    plt.show()
    return params_sensitivity_dict
    
nsamples=2
mtype = 'L6_TPC_L1'
etype ='cADpyr'
# m_type = 'L6_TPC_L1'
# e_type ='cADpyr'
my_model = get_model('BBP',log,m_type=mtype,e_type=etype,cell_i=1) 
#def_volts = my_model.simulate(stim,0.02)
def_volts = None

def_vals = my_model.DEFAULT_PARAMS 
ECDS = test_sensitivity(mtype,etype,def_volts,nsamples)
analyze_ecds(ECDS,def_vals)
    
    