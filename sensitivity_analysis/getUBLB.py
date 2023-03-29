import pandas as pd
from run import get_model
import logging as log
import numpy as np
import math

logicalBouds= pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/Sensitivity AllCells  - ICell wise 16_09.csv")
Cell = "L5_STPC cADpyr 1"
Cell2 = "L5_STPC-cADpyr-1"
my_model = get_model('BBP',log,'L5_STPC','cADpyr',cell_i=1)
def_param_vals = my_model.DEFAULT_PARAMS
LB = logicalBouds[Cell+"L"]
UB = logicalBouds[Cell+"U"]
print(def_param_vals)

# for i len(LB):
new_base_val = []
for i in range(17):
    l = def_param_vals[i]*np.exp(LB.iloc[i])
    u = def_param_vals[i]*np.exp(UB.iloc[i])
    new_base_val.append(math.sqrt(l*u))
new_base_val.append(def_param_vals[17])
new_base_val.append(def_param_vals[18])
print(new_base_val)
pf = pd.DataFrame( new_base_val)
pf.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/"+Cell2+".csv",index = False)

