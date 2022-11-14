import numpy as np
from run import get_model
import logging as log
import math
import pandas as pd
import numpy as np
import os

df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
Bounds = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/Bounds.csv")

linear = ['epass_all', 'stim_ampl_fact', 'stim_hold_curr']

def isExponential(param_name):
    if(param_name in linear):
        return False
    return True

def isCellPresent(m_type,e_type,i_cell):
    templates_dir = '/global/cfs/cdirs/m2043/hoc_templates/hoc_templates'
    cellName = m_type+"_"+e_type
    cell_clones =  os.listdir(templates_dir)
    cell_clones =[x for x in cell_clones if cellName in x]
    cell_is=[]
    for x in cell_clones:
        cell_is.append(x.split('_')[-1])
    if(str(int(i_cell)+1) not in cell_is):
        print(cell_is,str(int(i_cell)+1))
        print("Template Doesnt Exist{}, Skipping".format(cellName+str(i_cell)))
        return False
    return True

def saveNewBase():
    Data={}
    for i in range(len(df)):
        m_type = df.iloc[i]['mType']
        e_type = df.iloc[i]['eType']
        if(i==0):
            model = get_model('BBP',log,m_type,e_type,0)
            Data["Parameters"]=param_names = model.PARAM_NAMES
        for i_cell in range(0,5):
            if(isCellPresent(m_type,e_type,i_cell)):
                model = get_model('BBP',log,m_type,e_type,i_cell)
                parameters = model.DEFAULT_PARAMS
                param_names = model.PARAM_NAMES
                NewBase=[]
                for j in range(len(param_names)):
                    row = Bounds.loc[Bounds['Parameter']==param_names[j]]
                    if(len(row)>=1):
                        L = row['LB'][:]
                        LB = np.exp(int(row['LB'])*np.log(10))*parameters[j]
                        UB = np.exp(int(row['UB'])*np.log(10))*parameters[j]
                        print(LB,UB,parameters[j],"BOUNDS")
                        if(isExponential(param_names)):
                            NewBase.append(math.sqrt(LB*UB))
                        else:
                            NewBase.append((LB+UB)/2)
                    else:
                        NewBase.append(parameters[j])
                Data[m_type+"_"+e_type+"_"+str(i_cell)]=NewBase
    Base = pd.DataFrame(Data)
    Base.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv",index=False)    

def changeBase():
    OldMeanBase = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv")
    oldParam=OldMeanBase["Values"]
    param = OldMeanBase["Parameters"]
    
    newParams = oldParam*math.sqrt(10)
    # Base = pd.DataFrame({"Parameters":param,"Values":newParams})
    # Base.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv",index=False)    
    np.savetxt("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/TEST.csv",list(oldParam))

# saveNewBase()
changeBase()