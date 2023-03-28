# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:19:23 2019

@author: bensr
"""

import csv
import numpy as np
import os
import glob
etypes_dict = {}
etypes_list = []
def find_etype(st):
    for et in etypes_list:
        if st.find(et)>-1:
            print(et)
            return et
def get_fn_params(f):
    curr_dict = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            curr_dict[row[0]] = row[1]
    return curr_dict
def update_etype_dict(f):
    currst = str(f)
    etype = find_etype(currst)
    etype_dict = etypes_dict[etype]
    curr_dict = get_fn_params(f)
    pnames = list(curr_dict.keys())
    for k in pnames:
        if k in etype_dict:
            curr_list = etype_dict[k]
            curr_list.append(curr_dict[k])
        else:
            etype_dict[k] = [curr_dict[k]] 
    return etype_dict
def get_inhibit_dict():
    inhibit_etypes = etypes_list.copy()
    inhibit_etypes.remove('cAD')
    inhibit_dict = {}
    for iet in inhibit_etypes:
        curr_dict = etypes_dict[iet]
        pnames = list(curr_dict.keys())
        
        for pn in pnames:
            min_val = float(min(curr_dict[pn]))
            max_val = float(max(curr_dict[pn]))
            if pn in inhibit_dict:
                [all_min,all_max] = inhibit_dict[pn]
                new_min = float(min([all_min,min_val]))
                new_max = float(max([all_max,max_val]))
                inhibit_dict[pn] = [new_min,new_max]
            else:
                inhibit_dict[pn] = [min_val,max_val]
    return inhibit_dict
def get_excit_dict():
    curr_dict = etypes_dict['cAD']
    excit_dict = {}
    pnames = list(curr_dict.keys())
    for pn in pnames:
        min_val = float(min(curr_dict[pn]))
        max_val = float(max(curr_dict[pn]))
        excit_dict[pn] = [min_val,max_val]
    return excit_dict
for f in glob.glob('./etypes/*.csv'):
     currst = str(f)
     currst = currst[9:-4]
     etypes_dict[currst] = {}
     etypes_list.append(currst)
for f in glob.glob('./*.csv'):
    currst = str(f)
    #param_names = get_pnames(f)
    print(f)
    etype_dict = update_etype_dict(f)
inhibit_params = get_inhibit_dict()
excit_params = get_excit_dict()
with open('../params/bbp_inhibitory.csv', 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k in inhibit_params.keys():
            curr_val = inhibit_params[k]
            writer.writerow([k,str(curr_val[0]),str(curr_val[1])])
with open('../params/bbp_excitatory.csv', 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k in excit_params.keys():
            curr_val = excit_params[k]
            writer.writerow([k,str(curr_val[0]),str(curr_val[1])])

    
                
        
    
            
    
    
    
