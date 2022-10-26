from run import get_model
import logging as log
import math
import pandas as pd
df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
Bounds = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/Bounds.csv")

linear = ['epass_all', 'stim_ampl_fact', 'stim_hold_curr']

def isExponential(param_name):
    if(param_name in linear):
        return False
    return True


def saveNewBase():
    Data={}
    for i in range(len(df)):
        m_type = df.iloc[j]['mType']
        e_type = df.iloc[j]['eType']
        if(i==0):
            model = get_model('BBP',log,m_type,e_type,0)
            Data["Parameters"]=param_names = model.PARAM_NAMES
        for i_cell in range(0,5):
            model = get_model('BBP',log,m_type,e_type,i_cell)
            parameters = model.DEFAULT_PARAMETERS
            param_names = model.PARAM_NAMES
            NewBase=[]
            for j in range(len(param_names)):
                row = Bounds.loc[Bounds['Parameter']==param_names]
                LB = int(row['LB'][0])
                UB = int(row['UB'][0])
                print(LB,UB,"BOUNDS")
                if(isExponential(param_names)):
                    NewBase.append(math.sqrt(LB*UB))
                else:
                    NewBase.append((LB+UB)/2)
            Data[m_type+"_"+e_type+"_"+str(i_cell)]=NewBase
    Base = pd.DataFrame(Data)
    Base.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/NewBase.csv")    

saveNewBase()