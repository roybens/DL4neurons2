from os import listdir
import os
import pandas as pd
from pathlib import Path
import pylab as P
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
num_param = 3
'''Generate Plots of Cells STD from csv frpm analyze_sensitivity'''


path = "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana_selected/"
pdf = matplotlib.backends.backend_pdf.PdfPages("STD.pdf")
df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/excitatorycells.csv")
nregions = 4
colour=['r','b','g','y']
flag = False
total_param = []
for i in range(num_param):
    total_param.append([])
for i in range(nregions):
    param_values=[]
    for j in range(num_param):
        param_values.append([])
    fig = P.figure()
    for j in range(len(df)):
        m_type = df.iloc[j]['mType']
        e_type = df.iloc[j]['eType']
        # filenames = listdir(path+m_type+'_'+e_type)
        search_dir = path+m_type+'_'+e_type
        if(os.path.exists(search_dir)):
            os.chdir(search_dir)
            files = filter(os.path.isfile, os.listdir(search_dir))
            files = [os.path.join(search_dir, f) for f in files] # add path to each file
            all_csv = [ filename for filename in files if str(filename).endswith( ".csv" ) ]

            all_csv.sort(key=lambda x: os.path.getmtime(x))
            if(len(all_csv)>i):
                curr_csv = all_csv[i]
                df_cell = pd.read_csv(curr_csv)
                mean_cell = list(df_cell["STD ECD"])
                if(flag==False):
                    flag=True
                    P_names = list(df_cell["param_name"])
                    P_names= [" "]+P_names
                if(len(mean_cell)>num_param):
                    print("Here")
                for k in range(len(mean_cell)):
                    param_values[k].append(mean_cell[k])
    for k in range(num_param):
        y = param_values[k]
        total_param[k] = total_param[k] + y 
        x = np.random.normal(1+k, 0.1, size=len(y))
        max_y = max(y)
        if(k ==0):
            P.plot(x, y,'.', alpha=0.3,color = colour[i],label = "Region"+str(i))
            #plt.g.plot(x, y, 'r.', alpha=0.3,color = colour[i],label = "Region"+str(i))

        else:
            P.plot(x, y,'.', alpha=0.3,color = colour[i])
            #fig.plot(x, y, 'r.', alpha=0.3,color = colour[i])
    P.axis([0,num_param+1,0,  max_y*1.5])
    x = range(0,num_param+1)
    P.xticks(x,P_names,rotation=90)

    pdf.savefig(fig,bbox_inches="tight")
x = range(0,num_param+1)
P.axis([0,num_param+1,0, 500 ])

P.legend(loc="lower right")
P.ylabel("STD")
P.xticks(x,P_names,rotation=90)
P.savefig("STD.png",bbox_inches="tight")
P.boxplot(total_param)
P.xticks(x,P_names,rotation=90)
P.savefig("STD_box.png",bbox_inches="tight")
pdf.savefig(fig)
pdf.close()