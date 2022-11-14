from run import get_model
import logging as log
import models
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
from datetime import datetime

stim_files=['chaotic3.csv']
stimfn = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/'

NewBaseCsv = "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase/L6_TPC_L1 cADpyr 1.csv"
NewBase = pd.read_csv(NewBaseCsv)
BaseVal = NewBase["New Base"].tolist()
DefVal = NewBase["Old Base"].tolist()
mtype="L6_TPC_L1"
etype="cADpyr"
itype =1
cells = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
for cell in range(len(cells)):
    mtype =cells['mType'].iloc[cell]
    etype =cells['eType'].iloc[cell]
    for itype in range(0,5):
        pdf = matplotlib.backends.backend_pdf.PdfPages("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/"+mtype+etype+str(itype)+".pdf")
        for stim_file in stim_files:
            stim =  np.genfromtxt(stimfn+stim_file, dtype=np.float32) 
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            print("Before Sim =", current_time)
            all_paramsets = np.genfromtxt("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/BaseTest.csv", dtype=np.float32)
            all_paramsets = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams.csv")
            all_paramsets=list(all_paramsets["Values"])
            my_model = get_model('BBP',log,mtype,etype,itype)
            a = my_model.PARAM_RANGES
            my_model.set_attachments(stim,len(stim),0.025)
            for paramsCount in range(0,1):
                
                # my_model.DEFAULT_PARAMS = False
                my_model._set_self_params(*all_paramsets[:])
                # my_model._set_self_params()
                my_model.init_parameters()
                volts = my_model.simulate(stim,0.025)
                print(len(volts[list(volts.keys())[0]]))
                print("SECOND TIME")
                #volts = my_model.simulate(stim,0.025)
                # now2 = datetime.now()

                # current_time = now2.strftime("%H:%M:%S")
                # print("After Sim =", current_time)
                # diff= now2-now
                # print("Difference=",diff.total_seconds())
                # print(volts.keys())
                
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
            # my_model = get_model('BBP',log,mtype,etype,1,*DefVal)
            # my_model._set_self_params(*all_paramsets[1,:])
            # volts = my_model.simulate(stim,0.025)
            # print(len(volts[list(volts.keys())[0]]))
            # Data = volts[list(volts.keys())[0]]
            # fig, axs = plt.subplots(2,sharex = False,sharey = False, gridspec_kw = {'height_ratios':[2,8]})
            # axs[0].plot(df[df.columns[0]],label= 'I', color = 'pink')

            # axs[1].set_title("Old"+str(stim_file))
            # axs[1].plot(Data,color='green')
            # pdf.savefig(fig)
        

    #plt.savefig("/global/homes/k/ktub1999/mainDL4/DL4neurons2/NewBasePlots/"+mtype+etype+str(itype)+".png")
    
        pdf.close()