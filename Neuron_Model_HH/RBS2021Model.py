# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""

import numpy as np
from vm_plotter import *
from neuron import h
import json
from scipy.signal import find_peaks
h.load_file("runModel.hoc")
soma_ref = h.root.sec
soma = h.secname(sec=soma_ref)
sl = h.SectionList()
sl.wholetree(sec=soma_ref)
def init_settings(nav12=1,
                  nav16=1,
                  dend_nav12=1, 
                  soma_nav12=1, 
                  ais_nav12=1, 
                  dend_nav16=1, 
                  soma_nav16=1,
                  ais_nav16=1, 
                  axon_Kp=1,
                  axon_Kt =1,
                  axon_K=1,
                  soma_K=1,
                  dend_K=1,
                  gpas_all=1):

        self.h = h  # NEURON h
        h.load_file("runModel.hoc")
        self.soma_ref = h.root.sec
        self.soma = h.secname(sec=self.soma_ref)
        self.sl = h.SectionList()
        self.sl.wholetree(sec=self.soma_ref)

        h.dend_na12 = 0.026145/2
        h.dend_na16 = h.dend_na12
        h.dend_k = 0.004226 * soma_K


    h.soma_na12 = 0.983955/10 
    h.soma_na16 = h.soma_na12 
    h.soma_K = 0.303472 * soma_K

    h.ais_na16 = 4 
    h.ais_na12 = 4 
    h.ais_ca = 0.000990
    h.ais_KCa = 0.007104

    h.node_na = 2

    h.axon_KP = 0.973538 * axon_Kp
    h.axon_KT = 0.089259 * axon_Kt
    h.axon_K = 1.021945 * axon_K

    h.cell.axon[0].gCa_LVAstbar_Ca_LVAst = 0.001376286159287454
    
    #h.soma_na12 = h.soma_na12/2
    h.naked_axon_na = h.soma_na16/5
    h.navshift = -10
    h.myelin_na = h.naked_axon_na
    h.myelin_K = 0.303472
    h.myelin_scale = 10
    h.gpas_all = 3e-5 * gpas_all
    h.cm_all = 1
    
    
    h.dend_na12 = h.dend_na12 * nav12 * dend_nav12
    h.soma_na12 = h.soma_na12 * nav12 * soma_nav12
    h.ais_na12 = h.ais_na12 * nav12 * ais_nav12
    
    h.dend_na16 = h.dend_na16 * nav16 * dend_nav16
    h.soma_na16 = h.soma_na16 * nav16 * soma_nav16
    h.ais_na16 = h.ais_na16 * nav16 * ais_nav16
    h.working()

def update_mechs_props(dict_fn,mechs):
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                for p_name in param_dict.keys():
                    hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} = {param_dict[p_name]}'
                    print(hoc_cmd)
                    h(hoc_cmd)
                #in case we need to go per sec:
                  #  for seg in curr_sec:
                  #      hoc_cmd = f'{curr_name}.gbar_{channel}({seg.x}) *= {wt_mul}'
                  #      print(hoc_cmd)

def update_gbar(mechs,mltplr,gbar_name = 'gbar'):
    for curr_sec in sl:
        curr_name = h.secname(sec=curr_sec)
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                for seg in curr_sec:
                    hoc_cmd = f'{curr_name}.{gbar_name}_{curr_mech}({seg.x}) *= {mltplr}'
                    print(hoc_cmd)
                    par_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    h(hoc_cmd)
                    assigned_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    print(f'par_value before{par_value} and after {assigned_value}')
def multiply_param(mechs,p_name,multiplier):
    for curr_sec in sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} *= {multiplier}'
                print(hoc_cmd)
                h(hoc_cmd)
def offset_param(mechs,p_name,offset):
    for curr_sec in sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} += {offset}'
                print(hoc_cmd)
                h(hoc_cmd)
#remember to compensate for previous changes parameters are not being initialized all over again hence for multiplier use [0.5,4] to check X0.5,X2
def explore_param(mechs,p_name,values,multiplier = True):
    for curr_val in values:
        if multiplier:
            multiply_param(mechs,p_name,curr_val)
        else:
            offset_param(mechs,p_name,curr_val)
        init_settings()
        init_stim(amp=0.2)
        Vm, I, t, stim = run_model()
        plot_stim_volts_pair(Vm, 'Step Stim 200pA', file_path_to_save=f'./Plots/WT_200pA_{p_name}_{curr_val}',times=t)

    
def init_stim(sweep_len = 400, stim_start = 100, stim_dur = 200, amp = 0.3, dt = 0.1):
    # updates the stimulation params used by the model
    # time values are in ms
    # amp values are in nA
    
    h("st.del = " + str(stim_start))
    h("st.dur = " + str(stim_dur))
    h("st.amp = " + str(amp))
    h.tstop = sweep_len
    h.dt = dt


def get_fi_curve(s_amp,e_amp,nruns,wt_data=None,ax1=None):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    stim_length = int(600/dt)
    for curr_amp in x_axis:
        init_stim(amp = curr_amp)
        curr_volts,_,_,_ = run_model()
        curr_peaks,_ = find_peaks(curr_volts[:stim_length],height = -20)
        all_volts.append(curr_volts)
        npeaks.append(len(curr_peaks))
    print(npeaks)
    if ax1 is None:
        fig,ax1 = plt.subplots(1,1)
    ax1.plot(x_axis,npeaks,'black')
    ax1.set_title('FI Curve')
    ax1.set_xlabel('Stim [nA]')
    ax1.set_ylabel('nAPs for 500ms epoch')
    if wt_data is None:
        return npeaks
    else:
        ax1.plot(x_axis,npeaks,'blue')
        ax1.plot(x_axis,wt_data,'black')
    plt.show()


    
    
def run_model(start_Vm = -72,dt = 0.025):
    h.dt = dt
    #h.working()
    h.finitialize(start_Vm)
    timesteps = int(h.tstop/h.dt)
    
    Vm = np.zeros(timesteps)
    I = {}
    I['Na'] = np.zeros(timesteps)
    I['Ca'] = np.zeros(timesteps)
    I['K'] = np.zeros(timesteps)
    I['INa12'] = np.zeros(timesteps)
    stim = np.zeros(timesteps)
    t = np.zeros(timesteps)
    for i in range(timesteps):
        Vm[i] = h.cell.soma[0].v
        I['Na'][i] = h.cell.soma[0](0.5).ina
        I['Ca'][i] = h.cell.soma[0](0.5).ica
        I['K'][i] = h.cell.soma[0](0.5).ik
        #I['INa12'][i] = h.cell.soma[0].gna_na12
        stim[i] = h.st.amp 
        t[i] = i*h.dt / 1000
        h.fadvance()
    return Vm, I ,t, stim

        
## WT

fig,ficurveax = plt.subplots(1,1)
init_settings()
h.working()
mechs = ['na16']
update_gbar(mechs,2,gbar_name = 'gbar')
mechs = ['na16mut']
update_gbar(mechs,0,gbar_name = 'gbar')
init_stim(amp=0.5)
Vm, I, t, stim = run_model(dt = 0.1)
plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/WT_500pA',times=t)




#mechs = ['na16']
#update_gbar(mechs,0,gbar_name = 'gbar')
#mechs = ['na16mut']
#WT_fn = './params/na16WT.txt'
#update_mechs_props(WT_fn,mechs)


#update_gbar(mechs,200,gbar_name = 'gbar')


#mechs = ['na12','na12mut']
#WT_fn = './params/na12WT.txt'
#update_mechs_props(WT_fn,mechs)
#vshift_values = [-30,60]

#explore_param(mechs,'vShift',vshift_values,False)
#explore_param(mechs,'vShift_inact',vshift_values,False)
#mechs = ['na12','na12mut','na16','na16mut']
#update_gbar(mechs,10000,gbar_name = 'gbar')
#update_mechs_props(WT_fn,mechs)
#init_stim(amp=0.75)
#Vm, I, t, stim = run_model()
#plot_stim_volts_pair(Vm, 'Step Stim 200pA', file_path_to_save='./Plots/WTHMM',times=t)
#fig,ax = plt.subplots(1,1)
#ax.plot(t,I['INa12'],'black')
#ax.plot(t,I['Na'],'red')
#fig.savefig('./Plots/WTIna.pdf')




#init_stim(amp=0.5)
#Vm, I, t, stim = run_model()
#plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/WT_500pA',times=t)
#wtnpeaks = get_fi_curve(0.05, 0.55, 11,ax1=ficurveax)


# update_na16('./params/na16_mutv1.txt')
# init_stim(amp=0.2)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 200pA', file_path_to_save='./Plots/Mut_200pA',times=t,color_str='blue')
# init_stim(amp=0.5)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/Mut_500pA',times=t,color_str='blue')
# get_fi_curve(0.05, 0.55, 11,wt_data=wtnpeaks,ax1=ficurveax)
# fig.savefig('./Plots/FI_curves.pdf')



