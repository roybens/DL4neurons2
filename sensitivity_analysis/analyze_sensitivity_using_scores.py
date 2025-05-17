# -*- coding: utf-8 -*-
"""

@author: bensr

Analyze the sensitivity pkl files created by generate_analysis_data for all the parameters (and all the samples).


"""
import sys
sys.path.insert(1, '../DL4neurons2/')
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
import efel

from sklearn.preprocessing import MinMaxScaler


stimfn = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5k50kInterChaoticB.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.1*i for i in range(len(stim))]
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

def get_rec_sec(def_volts,adjusted_param,probe='soma'):
    probes = list(def_volts.keys())
    rec_sec=adjusted_param
    rec_sec = probes[0]
    # print(probes)
    if 'soma' in probe:
        rec_sec = probes[0]
    if 'apic' in probe or 'dend' in probe:
        res = [i for i in probes if 'apic' in i or 'dend' in i]
        rec_sec = res[0]   
    if 'axon' in probe:
        res = [i for i in probes if 'axon' in i]
        rec_sec = res[0]  
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
        fig.suptitle(adjusted_param,size='20')
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
        cum_sum_err = np.abs(curr_cum_sum1 - curr_cum_sum2)
        cum_sum_errs.append(cum_sum_err)
        if(create_pdfs):
            err = volts_to_plot1 - volts_to_plot2
            err = [abs(pos) for pos in err]
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


def safe_mean(lis):
    if np.size(lis) == 0:
        return 0
    return np.mean(lis)

def weighted_score_compute(feature1,feature2):
    feature_scores = {}
    feature_names = []
    for f1_names in feature1.keys():
        if(f1_names != None):
            if(f1_names not in feature_names ):
                feature_names.append(f1_names)

    for f2_names in feature2.keys():
        if(f2_names !=  None):
            if(f2_names not in feature_names ):
                feature_names.append(f2_names)
        else:
            print(f2_names)
    for feature_name in feature_names:
        try:
            lis1 = feature1[feature_name]
        
        except KeyError:
            lis1 = np.array([0])
        
        try:
            lis2 = feature2[feature_name]
        except KeyError:
            lis2 = np.array([0])
        if(lis1 is None):
            lis1=np.array([0])
        if(lis2 is None):
            lis2=np.array([0])
        len1, len2 = len(lis1), len(lis2)
        if(type(lis1) is list or type(lis2) is list):
            print(lis1)
        if len1 > len2:
            lis2 = np.concatenate((lis2, np.zeros(len1 - len2)), axis=0)
        if len2 > len1:
            lis1 = np.concatenate((lis1, np.zeros(len2 - len1)), axis=0)
        feature_scores[feature_name] = np.sqrt(safe_mean((lis1 - lis2)**2))
    weighted_score = 0
    for feature,score in feature_scores.items():
        weighted_score+=score
    return weighted_score,feature_scores

def compute_features(voltage_trace, dt):
    features_names=[
        "mean_frequency",
    ]
    features_to_compute = [
        'mean_frequency', 'AP_amplitude', 'AHP_depth_abs_slow',
        'fast_AHP_change', 'AHP_slow_time',
        'spike_half_width', 'time_to_first_spike', 'inv_first_ISI', 'ISI_CV',
        'ISI_values','adaptation_index'
    ]

    trace1 = {}
    timsteps = len(voltage_trace)
    trace1['T'] = [x*dt for x in range(len(voltage_trace))]
    trace1['V'] = voltage_trace
    trace1['stim_start'] = [0]
    trace1['stim_end'] = [timsteps*dt]

    # traces = [trace1]
    
    feature_values_list = efel.get_feature_values([trace1], features_to_compute)
    # traces_results = efel.get_feature_values(traces,
    #                                        ['mean_frequency', 'AP_amplitude'])
    
    # for trace_results in feature_values_list:
    #     # trace_result is a dictionary, with as keys the requested features
    #     for feature_name, feature_values in trace_results.items():
    #         print("Feature %s has the following values: %s" % \
    #             (feature_name, ', '.join([str(x) for x in feature_values])))

    return feature_values_list[0]

def normalize_scores_per_region(feature_score_dict):
    total_mse_sum=[]
    scaler = MinMaxScaler()

    for feature,score in feature_score_dict.items():
        if(type(total_mse_sum)==list):
            total_mse_sum = np.zeros(len(score))
        mse_arr = np.array(score)
        mse_arr_max = np.nanmax(mse_arr)
        mse_arr[np.where(np.isnan(mse_arr))]=mse_arr_max
        mse_arr = scaler.fit_transform(np.array(mse_arr).reshape(-1,1)).squeeze()
        total_mse_sum += mse_arr
    return total_mse_sum
    



'''
The below function takes in the all Volts which is a 3d array.
The Difference is computed between the samples. For example if samples = 10, the resultant will be 5 as we are coming the cumulative diff between two samples.

'''
def check_param_sensitivity_regions(all_volts_regions,adjusted_param,files_loc,nregions,probe):
    cum_sum_errs_regions = []
    for curr_region in range(nregions): 
        print(f'{adjusted_param} region - {curr_region}')
        all_volts = all_volts_regions[curr_region]
        def_rec_sec,prefix = get_rec_sec(all_volts[0],adjusted_param,probe)
         #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
        #ax1.plot(times,def_volts[:-1],'black')
        volt_debug = []
        cum_sum_errs = {} #appending list to scores.
        if(create_pdfs):
            fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
            fig.suptitle(f'{adjusted_param} region - {curr_region}',size='20')
            plt.subplots_adjust(hspace=0.3)

        for i in range(int(len(all_volts)/2)):

            volts1 = all_volts[2*i]
            volts2 = all_volts[2*i + 1]

            curr_rec_sec1,prefix1 = get_rec_sec(volts1,adjusted_param,probe)
            curr_rec_sec2,prefix2 = get_rec_sec(volts2,adjusted_param,probe)
            if (curr_rec_sec1 != curr_rec_sec2):
                print("curr_rec_sec is " + curr_rec_sec1 + 'and curr_rec_sec2  is' + curr_rec_sec2 )
            volts_to_plot1 = volts1.get(prefix1 +def_rec_sec)
            volts_to_plot2 = volts2.get(prefix2 +def_rec_sec)
            volt_debug.append(volts_to_plot1)
            volt_debug.append(volts_to_plot2)
            dt = 0.1
            # curr_cum_sum1= np.cumsum(np.abs(volts_to_plot1))*0.025
            # curr_cum_sum2= np.cumsum(np.abs(volts_to_plot2))*0.025
            features_1 = compute_features(volts_to_plot1, dt)
            features_2 = compute_features(volts_to_plot2, dt)

            weighted_score,score_dictionary = weighted_score_compute(features_1,features_2)
    
            cum_sum_err = weighted_score
            # cum_sum_errs.append(cum_sum_err)
            for feature_name,feature_values in score_dictionary.items():
                if feature_name not in cum_sum_errs:
                    cum_sum_errs[feature_name] = []
                cum_sum_errs[feature_name].append(feature_values)

            # cum_sum_errs.append(score_dictionary)
            
            if(create_pdfs):
                err = volts_to_plot1 - volts_to_plot2
                err = [abs(pos) for pos in err]
                ax1.plot(times,volts_to_plot1[:-1])
                ax1.plot(times,volts_to_plot2[:-1])
                ax2.plot(times,err[:-1])
                # ax3.plot(cum_sum_err)
        if(create_pdfs):   
            ax1.title.set_text('Volts')
            ax1.set_ylim(-200,+200)
            ax2.title.set_text('error')
            ax3.title.set_text('cum_sum_error')
            fig_name = adjusted_param +'.pdf'
            fig.savefig(files_loc + fig_name)
        
        volt_debug = np.array(volt_debug)
        cum_sum_errs = normalize_scores_per_region(cum_sum_errs)
        cum_sum_errs_regions.append(cum_sum_errs)
    return cum_sum_errs_regions



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
            if os.stat(fn).st_size > 0:
                print(f'opening fn - {fn}')
                with open(fn, 'rb') as f:
                    curr_volts = pkl.load(f)
                    all_volts = all_volts + curr_volts
            else:
                print(f'{fn} is empty!')
        if len(all_volts)>0:
            curr_errs = check_param_sensitivity(all_volts,adjusted_param,files_loc)
            curr_ECDs = [ls[-1] for ls in curr_errs]
            all_ECDS[adjusted_param_new_name]=curr_ECDs
    pkl_fn=files_loc + my_model.m_type + my_model.e_type + 'ECDs.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_ECDS,output)
        pkl.dump(param_names,output)
    return all_ECDS

def test_sensitivity_regions(files_loc,my_model,nregions,probe):
    old_param_names = my_model.PARAM_NAMES
    param_names = shorten_param_names(old_param_names)
    all_ECDS_regions = []
    for i in range(nregions):
        all_ECDS = {}
        all_ECDS_regions.append(all_ECDS)
    all_fns = os.listdir(files_loc)
    for i in range(len(param_names)):
        adjusted_param = old_param_names[i]
        adjusted_param_new_name = param_names[i]
        param_files = [files_loc + fn for fn in all_fns if adjusted_param in fn]
        param_files = [ fn for fn in param_files if '.pkl' in fn]
        all_volts =  [[] for i in range(nregions)]
        for fn in param_files:
            if os.stat(fn).st_size > 0:
                print(f'opening fn - {fn}')
                with open(fn, 'rb') as f:
                    curr_volts = pkl.load(f)
                    #print(curr_volts)
                    for curr_region in range(nregions):
                        all_volts[curr_region] = all_volts[curr_region] + curr_volts[curr_region]
            else:
                print(f'{fn} is empty!')
        # if len(all_volts[3])>0:
        if True:
                curr_errs_regions = check_param_sensitivity_regions(all_volts,adjusted_param,files_loc,nregions,probe)
                for i in range(nregions):
                    all_ECDS = all_ECDS_regions[i]
                    curr_errs = curr_errs_regions[i]
                    # curr_ECDs = [ls[-1] for ls in curr_errs]
                    curr_ECDs = curr_errs
                    all_ECDS[adjusted_param_new_name]=curr_ECDs

    #all_ECDS are not being updated
    pkl_fn=files_loc + my_model.m_type + my_model.e_type +probe+ 'ECDs.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_ECDS_regions,output)
        pkl.dump(param_names,output)
    return all_ECDS_regions


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
        # if True:
        #     pnames_soma.append(param_names[i])
        #     means_soma.append(curr_mean)
        #     STDs_soma.append(curr_std)
        #     ml_STDs_soma.append(ml_std)
        #     ml_STDs_raw_soma.append(ml_std_raw)
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

def analyze_ecds_no_ML(ECDS,def_vals,files_loc,curr_region="",probe='soma'):
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
        # if def_vals[i] <= 0:
        #     continue
        curr_ecds = ECDS[param_names[i]]
        nsamples = len(curr_ecds)
        curr_mean = np.mean(curr_ecds)
        curr_std = np.std(curr_ecds)
        params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std]
        if True:
            pnames_soma.append(param_names[i])
            means_soma.append(curr_mean)
            STDs_soma.append(curr_std)
            
        # if 'som' in param_names[i]:
        #     pnames_soma.append(param_names[i])
        #     means_soma.append(curr_mean)
        #     STDs_soma.append(curr_std)
            
        # if 'api' in param_names[i] or 'den' in param_names[i]:
        #     pnames_dend.append(param_names[i])
        #     means_dend.append(curr_mean)
        #     STDs_dend.append(curr_std)
        # if 'axn' in param_names[i]:
        #     pnames_axon.append(param_names[i])
        #     means_axon.append(curr_mean)
        #     STDs_axon.append(curr_std)
    pkl_fn=files_loc + sys.argv[1] + sys.argv[2] +'mean_std_sensitivity'  + curr_region + probe +  '.pkl'
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
    #yaxis_soma = np.divide(means_soma,STDs_soma)
    yaxis_soma = means_soma
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
    
    
    fig_name = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sampling_size' + curr_region+probe +'.pdf'
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
    
    fig_name1 = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sensitivity_threshold' + curr_region+probe  + '.pdf'
    fig1.savefig(files_loc + fig_name1)
    #plt.show()
    all_means = np.concatenate((means_axon,means_soma,means_dend),axis=None)
    excl_header = ['param_name', 'Mean ECD','STD ECD']
#    params_sensitivity_dict_csv = {}
#    params_sensitivity_dict_csv['param_name'] = all_pnames
#    params_sensitivity_dict_csv['Mean_ECD'] = all_means
#    params_sensitivity_dict_csv['STD_ECD'] = all_STDs
#    params_sensitivity_dict_csv['Adj_ML_STD'] = raw_ml_stds
    
    excl_fn=files_loc + 'sensitivity' + curr_region + '_'  + sys.argv[1] + sys.argv[2] +probe + '.csv'
    with open(excl_fn, 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(excl_header)
        for (pname,mean,std,) in  zip(all_pnames,all_means,all_STDs):
            writer.writerow([pname,mean,std])
   
    return params_sensitivity_dict





# def main():
#     short_name = None
#     m_type = sys.argv[1]
#     e_type = sys.argv[2]
#     try:
#         short_name = sys.argv[3]
#     except:
#         print('no short name')
#         short_name = None    

    
#     files_loc = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana_selected2/' + m_type + '_' + e_type + '/'
#     my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0)
#     def_vals = my_model.DEFAULT_PARAMS
    
#     ECDS = test_sensitivity(files_loc,my_model)
#     #if short_name is not  None:
#     if short_name is not None:
#         ml_results = get_ml_results(short_name,list(ECDS.keys())) 
#         analyze_ecds(ECDS,def_vals,files_loc,ml_results)
#     else:
#         analyze_ecds_no_ML(ECDS,def_vals,files_loc)
def main_regions():
    
    short_name = None
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    i_cell = sys.argv[3]
    nregions = int(sys.argv[4])
    global model_name
    model_name = sys.argv[5]
    probe = sys.argv[6]
    print("DOUNG FOR Probe",probe)
    try:
        short_name = sys.argv[4]
    except:
        print('no short name')
        short_name = None    
    files_loc = '/global/cfs/cdirs/m2043/roybens/sens_ana/InhSens_50_scoreFuncNorm/' + m_type + '_' + e_type + '_' + i_cell + '/'
    #files_loc = './'
    if (len(os.listdir(files_loc))<4):
        print(f'{files_loc} has less than 4 files')
        with open("to_analyze.txt", "a") as myfile:
            myfile.write(f'abc {m_type} {e_type}')
        return;
       
    my_model = get_model(model_name,log,m_type=m_type,e_type=e_type,cell_i=int(i_cell))
    def_vals = my_model.DEFAULT_PARAMS
    
    ECDS = test_sensitivity_regions(files_loc,my_model,nregions,probe)
    for curr_region in range(nregions):
        curr_lb = -1 + (curr_region)*(2)
        curr_ub = -1 + (curr_region+1)*(2)
        curr_region_str = f'region_{curr_lb}_{curr_ub}'
        #print(ECDS[0])
        analyze_ecds_no_ML(ECDS[curr_region],def_vals,files_loc,curr_region_str,probe)
    print("Done for ",m_type,e_type,i_cell)

    
main_regions()
