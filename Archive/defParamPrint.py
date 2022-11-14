#Writing for a single parameter. TODO Extend to multiple
#parameter  Region1 Region 2
#Cell 1     
#Cell 2
#           Average 1 Avg 2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
from run import get_model
import logging as log
import models
import pandas as pd

# pdf = "Param1.pdf"
# path="/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana3/"
# param_name = "gNaTs2.tbar.NaTs2.t.api"
# df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv")
# 2D Array 
# nRanges = 8
# meanAllCellsRegions =[]
# for i in range(8):
#     meanAllCellsRegions.append([])
# for i in range(len(df)):
#     m_type = df.iloc[i]['mType']
#     e_type = df.iloc[i]['eType']
#     currCellPath = path+""
#     for i_cell in range(1,6):

def printDefault():
    AllCells = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
    dic={}
    for cells in range(len(AllCells)):
        m_type = AllCells["mType"].iloc[cells]
        e_type = AllCells["eType"].iloc[cells]
        for i_cell in range(0,1):
            model =  get_model('BBP',log,m_type,e_type,i_cell)
            def_param = model.DEFAULT_PARAMS
            param_names = model.PARAM_NAMES
            for k in range(len(def_param)):
                if(param_names[k] not in dic.keys()):
                    dic[param_names[k]]=[def_param[k]]
                else:
                    dic[param_names[k]].append(def_param[k])
    print(dic)

printDefault()