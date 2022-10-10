from run import get_model
import logging as log
import models
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
from datetime import datetime

stim_files=['chaotic3.csv','step_200.csv','ramp.csv','chirp.csv','step_500.csv']
stimfn = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/'

NewBaseCsv = "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase/L6_TPC_L1 cADpyr 1.csv"
NewBase = pd.read_csv(NewBaseCsv)
BaseVal = NewBase["New Base"].tolist()
DefVal = NewBase["Old Base"].tolist()
mtype="L5_STPC"
etype="cADpyr"
itype =1
pdf = matplotlib.backends.backend_pdf.PdfPages("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/"+mtype+etype+str(itype)+".pdf")
for stim_file in stim_files:
    stim =  np.genfromtxt(stimfn+stim_file, dtype=np.float32) 
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Before Sim =", current_time)
    my_model = get_model('BBP',log,mtype,etype,itype,*BaseVal)
    my_model.DEFAULT_PARAMS = False
    volts = my_model.simulate(stim,0.025)
    now2 = datetime.now()

    current_time = now2.strftime("%H:%M:%S")
    print("After Sim =", current_time)
    diff= now2-now
    print("Difference=",diff.total_seconds())
    print(volts.keys())
    
    print(len(volts[list(volts.keys())[0]]))
    Data = volts[list(volts.keys())[0]]
    fig, axs = plt.subplots(2,sharex = False,sharey = False, gridspec_kw = {'height_ratios':[2,8]})
    df = pd.read_csv(stimfn+stim_file)
    axs[0].plot(df[df.columns[0]],label= 'I', color = 'pink')
    axs[1].set_title("New"+str(stim_file))
    axs[1].plot(Data,color ='red')
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("nA")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("nV")
    axs[0].legend()
    axs[1].legend()
    pdf.savefig(fig)
    my_model = get_model('BBP',log,mtype,etype,1,*DefVal)
    volts = my_model.simulate(stim,0.025)
    print(len(volts[list(volts.keys())[0]]))
    Data = volts[list(volts.keys())[0]]
    fig, axs = plt.subplots(2,sharex = False,sharey = False, gridspec_kw = {'height_ratios':[2,8]})
    axs[0].plot(df[df.columns[0]],label= 'I', color = 'pink')

    axs[1].set_title("Old"+str(stim_file))
    axs[1].plot(Data,color='green')
    pdf.savefig(fig)

#plt.savefig("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/"+mtype+etype+str(itype)+".png")
  
pdf.close()