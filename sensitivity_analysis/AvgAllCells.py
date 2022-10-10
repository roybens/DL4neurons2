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

pdf = "Param1.pdf"
path="/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana3/"
param_name = "gNaTs2.tbar.NaTs2.t.api"
df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv")
# 2D Array 
nRanges = 8
meanAllCellsRegions =[]
for i in range(8):
    meanAllCellsRegions.append([])
for i in range(len(df)):
    m_type = df.iloc[i]['mType']
    e_type = df.iloc[i]['eType']
    currCellPath = path+""
    for i_cell in range(1,6):