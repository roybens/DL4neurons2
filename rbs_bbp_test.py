import os
import json
import numpy as np
from neuron import h
import logging as log
from run import get_model

def get_stim(stim_fn = 'chaotic_2',stim_factor = 1,stim_addition = 0):
    stim =  np.genfromtxt(stim_fn, dtype=np.float32)
    stim = stim * stim_factor
    stim = stim + stim_addition
    times = [0.025*i for i in range(len(stim))]
    return stim,times


with open('cells.json') as infile:
    cells = json.load(infile)

h.load_file('import3d.hoc')
templates_dir = 'hoc_templates'
m_type = 'L5_TTPC2'
e_type = 'cADpyr'
i = 0 
#mycell =BBPExcV2(m_type, e_type, i)
my_cell = get_model('BBP',log,m_type,e_type,i)
home_folder = '/neuron_wrk/'
stim_folder = f'{home_folder}/DL4neurons2/stims/'
stim_fn = f'{stim_folder}chaotic_2.csv'
stim,times = get_stim(stim_fn)
print(stim)
my_cell.simulate(stim)

