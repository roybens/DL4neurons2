# -*- coding: utf-8 -*-
"""
Creaed on Wed Dec 11 21:59:43 2019

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
import ruamel.yaml as yaml
import sys
import csv

stimfn = './stims/step_500.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.025*i for i in range(len(stim))]
create_pdfs = True
def get_ml_results(short_name,pnames):
    #data_loc = '/project/m2043/ML4neuron2b/' +short_name +'/cellSpike.sum_pred.yaml
    data_loc = '/project/m2043/ML4neuron2b/' +short_name +'/cellSpike.sum_pred.yaml'
    with open(data_loc, 'r') as f:
        ml_preds = yaml.safe_load(f)
    all_preds = ml_preds['lossAudit']
    
    return all_preds
        
def shorten_param_names(rawMeta):
    mapD={'_apical':'_api', '_axonal':'_axn','_somatic':'_som','_dend':'_den'}
    inpL=rawMeta
    outL=[]
    print('M: shorten_param_names(), len=',len(inpL))
    for x in inpL:
        #print('0x=',x)
        for k in mapD:
            x=x.replace(k,mapD[k])
        x=x.replace('_','.')
        #print('1x=',x)
        outL.append(x)
    return outL

def get_rec_sec(def_volts,adjusted_param):
    probes = list(def_volts.keys())
    rec_sec=adjusted_param
    rec_sec = probes[0]
    # if 'soma' in adjusted_param:
    #     rec_sec = probes[0]
    # if 'apic' in adjusted_param or 'dend' in adjusted_param:
    #     res = [i for i in probes if 'apic' in i or 'dend' in i]
    #     rec_sec = res[2]   
    # if 'axon' in adjusted_param:
    #     res = [i for i in probes if 'axon' in i]
    #     rec_sec = res[2]  
    dot_ind = rec_sec.find('.')+1
    return rec_sec[dot_ind:],rec_sec[:dot_ind]  

  
def check_param_sensitivity(all_volts,adjusted_param,files_loc):
    print(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(all_volts[0],adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    #ax1.plot(times,def_volts[:-1],'black')
    volt_debug = []
    cum_sum_errs = []
    if(create_pdfs):
        fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
        fig.suptitle(adjusted_param)
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
        volt_debug.append(volts_to_plot1)
        volt_debug.append(volts_to_plot2)
        curr_cum_sum1= np.cumsum(np.abs(volts_to_plot1))*0.025
        curr_cum_sum2= np.cumsum(np.abs(volts_to_plot2))*0.025
        cum_sum_err = curr_cum_sum1 - curr_cum_sum2
        cum_sum_errs.append(cum_sum_err)
        if(create_pdfs):
            err = volts_to_plot1 - volts_to_plot2
            ax1.plot(times,volts_to_plot1[:-1])
            ax1.plot(times,volts_to_plot2[:-1])
            ax2.plot(times,err[:-1])
            ax3.plot(times,cum_sum_err[:-1])
    if(create_pdfs):   
        ax1.title.set_text('Volts')
        ax1.set_ylim(-200,+200)
        ax2.title.set_text('error')
        ax3.title.set_text('cum_sum_error')
        fig_name = adjusted_param +'.pdf'
        fig.savefig(files_loc + fig_name)
    volt_debug = np.array(volt_debug)
    return cum_sum_errs



def test_sensitivity(files_loc,my_model):
    old_param_names = my_model.PARAM_NAMES
    param_names = shorten_param_names(old_param_names)
    all_ECDS ={}
    all_fns = os.listdir(files_loc)
    for i in range(len(param_names)):
        adjusted_param = old_param_names[i]
        adjusted_param_new_name = param_names[i]
        param_files = [files_loc + fn for fn in all_fns if adjusted_param in fn]
        param_files = [ fn for fn in param_files if '.pkl' in fn]
        all_volts = []
        for fn in param_files:
            with open(fn, 'rb') as f:
                curr_volts = pkl.load(f)
                all_volts = all_volts + curr_volts
        if len(all_volts)>0:
            curr_errs = check_param_sensitivity(all_volts,adjusted_param,files_loc)
            curr_ECDs = [ls[-1] for ls in curr_errs]
            all_ECDS[adjusted_param_new_name]=curr_ECDs
    pkl_fn=files_loc + my_model.m_type + my_model.e_type + 'ECDs.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_ECDS,output)
        pkl.dump(param_names,output)
    return all_ECDS


def analyze_ecds(ECDS,def_vals,files_loc,ml_results):
    ymx = 1000
    threshold = 100
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
    ml_STDs_soma = []
    ml_STDs_axon = []
    ml_STDs_dend = []
    ml_STDs_raw_soma = []
    ml_STDs_raw_axon = []
    ml_STDs_raw_dend = []
    nsamples = 0
    params_sensitivity_dict = {}
    param_inds = range(len(param_names))
    for i in param_inds:
        if def_vals[i] <= 0:
            continue
        curr_ecds = ECDS[param_names[i]]
        nsamples = len(curr_ecds)
        curr_mean = np.mean(curr_ecds)
        curr_std = np.std(curr_ecds)
        if ml_results is not None:
            for l in ml_results:
                if str(l[0])== param_names[i]:
                    ml_std =  np.sqrt((1-(l[2]/0.6)**2))
                    ml_std_raw = l[2]
                    break
            params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std,ml_std,ml_std_raw]
        else:
            ml_std = []
            ml_std_raw=[]
            params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std]
        if 'som' in param_names[i]:
            pnames_soma.append(param_names[i])
            means_soma.append(curr_mean)
            STDs_soma.append(curr_std)
            ml_STDs_soma.append(ml_std)
            ml_STDs_raw_soma.append(ml_std_raw)
            
        if 'api' in param_names[i] or 'den' in param_names[i]:
            pnames_dend.append(param_names[i])
            means_dend.append(curr_mean)
            STDs_dend.append(curr_std)
            ml_STDs_dend.append(ml_std)
            ml_STDs_raw_dend.append(ml_std_raw)
        if 'axn' in param_names[i]:
            pnames_axon.append(param_names[i])
            means_axon.append(curr_mean)
            STDs_axon.append(curr_std)
            ml_STDs_axon.append(ml_std)
            ml_STDs_raw_axon.append(ml_std_raw)
    pkl_fn=files_loc + sys.argv[1] + sys.argv[2] +'mean_std_sensitivity'  +  '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(params_sensitivity_dict,output)
    fig, ((ax_soma,ax_dend),( ax_axon,ax4))= plt.subplots(2,2,figsize=(15,15))
    fig.suptitle('Sensitivity analysis mean/rms ' + sys.argv[1] + sys.argv[2])
    
    ax_axon.title.set_text('Axonal del_ecds')
    means_axon = np.array(means_axon)
    STDs_axon = np.array(STDs_axon)
    yaxis_axon = np.divide(means_axon,STDs_axon)
    #yaxis_axon = np.clip(yaxis_axon,0,1)
    ax_axon.plot(range(len(pnames_axon)),yaxis_axon,'o')
    ax_axon.set_xticks(range(len(pnames_axon)))
    ax_axon.set_xticklabels(pnames_axon,rotation=45)
    ax_axon.grid()
    #ax_axon.set_ylim([0,1])
    ax_axon.set_ylabel('avr/std')
    
    
    ax_soma.title.set_text('somaal del_ecds')
    means_soma = np.array(means_soma)
    STDs_soma = np.array(STDs_soma)
    yaxis_soma = np.divide(means_soma,STDs_soma)
    #yaxis_soma = np.clip(yaxis_soma,0,1)
    ax_soma.plot(range(len(pnames_soma)),yaxis_soma,'o')
    ax_soma.set_xticks(range(len(pnames_soma)))
    ax_soma.set_xticklabels(pnames_soma)
    ax_soma.grid()
    #ax_soma.set_ylim([0,1])
    ax_soma.set_ylabel('avr/std')
    
    ax_dend.title.set_text('dendal del_ecds')
    means_dend = np.array(means_dend)
    STDs_dend = np.array(STDs_dend)
    yaxis_dend = np.divide(means_dend,STDs_dend)
    #yaxis_dend = np.clip(yaxis_dend,0,1)
    ax_dend.plot(range(len(pnames_dend)),yaxis_dend,'o')
    ax_dend.set_xticks(range(len(pnames_dend)))
    ax_dend.set_xticklabels(pnames_dend)
    ax_dend.grid()
    #ax_dend.set_ylim([0,1])
    ax_dend.set_ylabel('avr/std')
    
    
    fig_name = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sampling_size.pdf'
    fig.savefig(files_loc + fig_name)
    
    fig1, ax = plt.subplots(1,figsize=(15,15))
    fig1.suptitle('Sensitivity analysis RMS ' + sys.argv[1] + sys.argv[2])
    all_STDs = np.concatenate((STDs_axon,STDs_soma,STDs_dend),axis=None) 
    all_STDs = np.clip(all_STDs,0,ymx)
    
    all_pnames = pnames_axon + pnames_soma + pnames_dend
    all_ml_STDs = np.concatenate((ml_STDs_axon,ml_STDs_soma,ml_STDs_dend),axis=None)
    all_ml_STDs = np.array(all_ml_STDs)*ymx
    
    
    ax.title.set_text('Parameters RMS clipped at ' + str(ymx))
    ax.plot(range(len(all_pnames)),all_STDs,'o')
    ax.plot(range(len(all_pnames)),all_ml_STDs,'x',color='red')
    ax.set_xticks(range(len(all_pnames)))
    ax.set_xticklabels(all_pnames,rotation=45)
    ax.axhline(threshold,color='red')
    ax.axvline(len(pnames_axon) + 0.5)
    ax.axvline(len(pnames_axon) +len(pnames_soma) + 0.5)
    ax.grid()
    
    fig_name1 = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sensitivity_threshold.pdf'
    fig1.savefig(files_loc + fig_name1)
    #plt.show()
    all_means = np.concatenate((means_axon,means_soma,means_dend),axis=None)
    raw_ml_stds = np.concatenate((ml_STDs_raw_axon,ml_STDs_raw_soma,ml_STDs_raw_dend),axis=None)
    excl_header = ['param_name', 'Mean ECD','STD ECD','Adj ML STD','Raw ML STD']
#    params_sensitivity_dict_csv = {}
#    params_sensitivity_dict_csv['param_name'] = all_pnames
#    params_sensitivity_dict_csv['Mean_ECD'] = all_means
#    params_sensitivity_dict_csv['STD_ECD'] = all_STDs
#    params_sensitivity_dict_csv['Adj_ML_STD'] = raw_ml_stds
    
    excl_fn=files_loc + 'sensitivity'  + sys.argv[1] + sys.argv[2] + '.csv'
    with open(excl_fn, 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(excl_header)
        for (pname,mean,std,ml_std_raw,ml_std) in  zip(all_pnames,all_means,all_STDs,all_ml_STDs,raw_ml_stds):
            writer.writerow([pname,mean,std,ml_std_raw,ml_std])
   
    return params_sensitivity_dict

def analyze_ecds_no_ML(ECDS,def_vals,files_loc):
    ymx = 1000
    threshold = 100
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
    nsamples = 0
    params_sensitivity_dict = {}
    param_inds = range(len(param_names))
    for i in param_inds:
        if def_vals[i] <= 0:
            continue
        curr_ecds = ECDS[param_names[i]]
        nsamples = len(curr_ecds)
        curr_mean = np.mean(curr_ecds)
        curr_std = np.std(curr_ecds)
        params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std]
        # if 'som' in param_names[i]:
        #     pnames_soma.append(param_names[i])
        #     means_soma.append(curr_mean)
        #     STDs_soma.append(curr_std)
            
        # if 'api' in param_names[i] or 'den' in param_names[i]:
        #     pnames_dend.append(param_names[i])
        #     means_dend.append(curr_mean)
        #     STDs_dend.append(curr_std)
        # if 'axn' in param_names[i]:
        if True:
            pnames_axon.append(param_names[i])
            means_axon.append(curr_mean)
            STDs_axon.append(curr_std)
    pkl_fn=files_loc + sys.argv[1] + sys.argv[2] +'mean_std_sensitivity'  +  '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(params_sensitivity_dict,output)
    fig, ((ax_soma,ax_dend),( ax_axon,ax4))= plt.subplots(2,2,figsize=(15,15))
    fig.suptitle('Sensitivity analysis mean/rms ' + sys.argv[1] + sys.argv[2])
    
    ax_axon.title.set_text('Axonal del_ecds')
    means_axon = np.array(means_axon)
    STDs_axon = np.array(STDs_axon)
    yaxis_axon = np.divide(means_axon,STDs_axon)
    #yaxis_axon = np.clip(yaxis_axon,0,1)
    ax_axon.plot(range(len(pnames_axon)),yaxis_axon,'o')
    ax_axon.set_xticks(range(len(pnames_axon)))
    ax_axon.set_xticklabels(pnames_axon,rotation=45)
    ax_axon.grid()
    #ax_axon.set_ylim([0,1])
    ax_axon.set_ylabel('avr/std')
    
    
    ax_soma.title.set_text('somaal del_ecds')
    means_soma = np.array(means_soma)
    STDs_soma = np.array(STDs_soma)
    yaxis_soma = np.divide(means_soma,STDs_soma)
    #yaxis_soma = np.clip(yaxis_soma,0,1)
    ax_soma.plot(range(len(pnames_soma)),yaxis_soma,'o')
    ax_soma.set_xticks(range(len(pnames_soma)))
    ax_soma.set_xticklabels(pnames_soma)
    ax_soma.grid()
    #ax_soma.set_ylim([0,1])
    ax_soma.set_ylabel('avr/std')
    
    ax_dend.title.set_text('dendal del_ecds')
    means_dend = np.array(means_dend)
    STDs_dend = np.array(STDs_dend)
    yaxis_dend = np.divide(means_dend,STDs_dend)
    #yaxis_dend = np.clip(yaxis_dend,0,1)
    ax_dend.plot(range(len(pnames_dend)),yaxis_dend,'o')
    ax_dend.set_xticks(range(len(pnames_dend)))
    ax_dend.set_xticklabels(pnames_dend)
    ax_dend.grid()
    #ax_dend.set_ylim([0,1])
    ax_dend.set_ylabel('avr/std')
    
    
    fig_name = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sampling_size.pdf'
    fig.savefig(files_loc + fig_name)
    
    fig1, ax = plt.subplots(1,figsize=(15,15))
    fig1.suptitle('Sensitivity analysis RMS ' + sys.argv[1] + sys.argv[2])
    all_STDs = np.concatenate((STDs_axon,STDs_soma,STDs_dend),axis=None) 
    all_STDs = np.clip(all_STDs,0,ymx)
    
    all_pnames = pnames_axon + pnames_soma + pnames_dend
    
    
    ax.title.set_text('Parameters RMS clipped at ' + str(ymx))
    ax.plot(range(len(all_pnames)),all_STDs,'o')
    ax.set_xticks(range(len(all_pnames)))
    ax.set_xticklabels(all_pnames,rotation=45)
    ax.axhline(threshold,color='red')
    ax.axvline(len(pnames_axon) + 0.5)
    ax.axvline(len(pnames_axon) +len(pnames_soma) + 0.5)
    ax.grid()
    
    fig_name1 = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sensitivity_threshold.pdf'
    fig1.savefig(files_loc + fig_name1)
    #plt.show()
    all_means = np.concatenate((means_axon,means_soma,means_dend),axis=None)
    excl_header = ['param_name', 'Mean ECD','STD ECD']
#    params_sensitivity_dict_csv = {}
#    params_sensitivity_dict_csv['param_name'] = all_pnames
#    params_sensitivity_dict_csv['Mean_ECD'] = all_means
#    params_sensitivity_dict_csv['STD_ECD'] = all_STDs
#    params_sensitivity_dict_csv['Adj_ML_STD'] = raw_ml_stds
    
    excl_fn=files_loc + 'sensitivity'  + sys.argv[1] + sys.argv[2] + '.csv'
    with open(excl_fn, 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(excl_header)
        for (pname,mean,std,) in  zip(all_pnames,all_means,all_STDs):
            writer.writerow([pname,mean,std])
   
    return params_sensitivity_dict

def main():
    short_name = None
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    try:
        short_name = sys.argv[3]
    except:
        print('no short name')
        short_name = None  

    files_loc = './sen_ana/'
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0)
    def_vals = my_model.DEFAULT_PARAMS
    
    ECDS = test_sensitivity(files_loc,my_model)
    #if short_name is not  None:
    if short_name is not None:
        ml_results = get_ml_results(short_name,list(ECDS.keys())) 
        analyze_ecds(ECDS,def_vals,files_loc,ml_results)
    else:
        analyze_ecds_no_ML(ECDS,def_vals,files_loc)
    
    
main()
