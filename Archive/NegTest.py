import pandas as pd
from run import get_model
import logging as log


Cells = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv")

mTypes = Cells['mType']
eTypes = Cells['eType']
CellNames=[]



for i in range(len(mTypes)):
    try:
        my_model = get_model('BBP',log,mTypes[i],eTypes[i],cell_i=1)
        def_param_vals = my_model.DEFAULT_PARAMS
        for j in range(len(def_param_vals)-1):
            if(def_param_vals[j]<0):
                CellNames.append(Cells.iloc[i]['cellName'])
                break
    except FileNotFoundError:
        print(Cells.iloc[i])
print(CellNames) 
