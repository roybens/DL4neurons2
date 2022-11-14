from run import get_model
import logging as log
import models
import pandas as pd
from scipy.stats import gmean

def geometric_mean(exp_params):
    mean_params=[]
    for i in range(len(exp_params)):
        mean_params.append(gmean(exp_params[i]))
        
        # prt = 1
        # for j in range(len(exp_params[i])):
        #     prt*=exp_params[i][j]
        # prt=prt**1/len(exp_params[i])
        # mean_params.append(prt)
    return mean_params
def mean(a):
    if(len(a)>0):
        avg=0
        for i in range(len(a)):
            avg+=a[i]
        avg/=len(a)
        return avg

def mean_all_default():
    exp_params =[]
    linear_params=[]
    for j in range(18):
        a=[]
        exp_params.append(a)
    AllCells = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
    for cells in range(len(AllCells)):
        m_type = AllCells["mType"].iloc[cells]
        e_type = AllCells["eType"].iloc[cells]
        for i_cell in range(0,1):
            model =  get_model('BBP',log,m_type,e_type,i_cell)
            def_param = model.DEFAULT_PARAMS
            param_names = model.PARAM_NAMES
            temp_exp=[]
            for p_index in range(len(param_names)):
                if(param_names[p_index]=="e_pas_all"):
                    linear_params.append(def_param[p_index])
                    print("LINERARRRRRRRRRRRRRRRRRRrr",def_param[p_index])
                elif(def_param[p_index]>=0):
                    exp_params[p_index].append(def_param[p_index])
            # exp_params.append(temp_exp)
    avg_param = geometric_mean(exp_params)
    print(param_names)
    print(avg_param)
    lin_param =mean(linear_params)
    print(lin_param)
    avg_param.append(lin_param)
    df = pd.DataFrame({"Prameters":param_names,"Values":avg_param})
    df.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/MeanParams.csv",index=False)
           

mean_all_default()