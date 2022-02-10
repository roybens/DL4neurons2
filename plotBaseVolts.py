#!/usr/bin/env python3
'''
plot BBP3 simulation: soma volts

'''

from pprint import pprint
from toolbox.Util_H5io3 import   read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
import sys,os

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='data1/',help="scored data location")

    parser.add_argument("--dataName",  default='L6_TPC_L1_cADpyr231_1-v3-0-1-c1.h5', help="BBP3 simu file")
    parser.add_argument("--metaData",  default='data1/bbp3_simu_feb9.meta.h5', help="meta-data for BBP3 simu")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='baseBBP3'
    args.metaData=None  # for Roy
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):  os.mkdir(args.outPath) 
    return args


#............................
#............................
#............................

class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.mL7=['*','^','x','h','o','x','D']
                
#...!...!..................
    def waveArray(self,simD,simMD,comD,plDD,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=5,1; yIn=10
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,yIn))
        jSamp=0
        jSoma=0
        
        
        waveA=simD['volts'][jSamp,:,jSoma]
        stimP=simD['stim_par'][jSamp]
        stimN=simMD['stimName']
        if comD!=None:
            stimA=simD['stim_wave']
        
        # prep time axis
        numTbin=waveA.shape[0]
        timeV =np.arange(numTbin,dtype=np.float32)*simMD['timeAxis']['step']
        
        spFac=[10,0.1,0.1,10,0.1]
        
        M=waveA.shape[-1]
        print('wdif1:',M,waveA.shape)        
        assert M<=nrow*ncol

        for n in range(M): 
            ax = self.plt.subplot(nrow,ncol,1+n)
            if comD!=None:
                ax.plot(timeV,stimA[n]*spFac[n], 'r',linewidth=0.5,label='stim')
            ax.plot(timeV,waveA[:,n], 'b',linewidth=0.7,label='soma')
            txt='mult=%.2f  offset=%.2f '%(stimP[0,n],stimP[1,n])
            ax.text(0.25,0.6,txt,transform=ax.transAxes,color='g')
            yLab='probe potential (mV)'
            xLab='time (ms) '
            ax.set(xlabel=xLab,ylabel=yLab)
            ax.grid()
            ax.legend(loc='upper right',title=stimN[n])
            if n==0: ax.set_title(simMD['jobId']+'  '+plDD['shortName'])
            #print('P: %s avrScore=%.1f '%(plDD['text1'],ssum/(idxR-idxL)))
            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

            

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    comD=None; comMD=None
    if args.metaData!=None:
        comD,comMD=read3_data_hdf5(args.metaData)
        
    inpF=os.path.join(args.dataPath,args.dataName)
    simD,simMD=read3_data_hdf5(inpF)
    print('M:sim meta-data');   pprint(simMD)
    # restore mV scale for volts
    simD['volts']=simD['volts']/float(simMD['voltsScale'])
    
    # print selected data
    j=0
    print('jSamp=%d  \nstim_par:'%j,simD['stim_par'][j])
    print('phys_par:',simD['phys_par'][j])
    print('unit_par:',simD['unit_par'][j])

    
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    #for x in [ 'units','shortName','bbpName']: plDD[x]=inpMD[x]
    #plDD['stimAmpl']=np.array(inpMD['stimAmpl'])

    plDD['shortName']=args.dataName
    #plDD['simVer']='NeuronSim_2019_1'
    
    if 1:  # wavforms as array, decimated
        #plDD['idxLR']=[8,24,2] # 1st range(n0,n1,step) ampl-index
        #plDD['idxLR']=[0,8,1] # low ampl
        
        #plDD['text1']='holdCurr=%.2f nA'%(inpMD['holdCurr'][ihc])
        #plDD['timeLR']=[10.,160.]  # (ms)  time range
        #plDD['timeLR']=[15.,40.]  # (ms)  time range 
        #plDD['amplLR']=[-100,70]  #  (mV) amplitude range
        
        plot.waveArray(simD,simMD,comD,plDD)


    plot.display_all(args.dataName,pdf=1)
