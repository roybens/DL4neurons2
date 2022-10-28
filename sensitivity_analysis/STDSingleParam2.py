from os import listdir
import os
import pandas as pd
from pathlib import Path
import pylab as P
import matplotlib
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
num_param = 1
'''Generate Plots of Cells STD from csv frpm analyze_sensitivity'''


path = "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana2/"

df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/testCell3.csv")
nregions = 1
para_df = pd.read_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sen_ana2/L6_TPC_L1_cADpyr_2/sensitivityregion_-0.5_0.0_L6_TPC_L1cADpyr.csv")
colour=['r','b','g','y']
flag = False
# total_param = []
Parameters = para_df['param_name']
Plot_DIR = "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/SensitivityPlots2"
CSV_DIR=    "/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/SensitivityCSVs/"
right_bound_all_cells = {}
left_bound_all_cells = {}

AllPlotsPDF = matplotlib.backends.backend_pdf.PdfPages("ALLSTD.pdf")
for param in range(len(Parameters)):
    ALL_MEAN_PARAMS={}
    ALL_STD_PARAMS={}
    Mean_Params=[]
    STD_Params=[]    
    os.chdir(Plot_DIR)
    pdf = matplotlib.backends.backend_pdf.PdfPages(str(Parameters[param])+".pdf")
    # for i in range(num_param):
    #     total_param.append([])
    # for i in range(nregions):
    #     param_values=[]
    #     for j in range(num_param):
    #         param_values.append([])
    if(True):
        
        for j in range(len(df)):
            m_type = df.iloc[j]['mType']
            e_type = df.iloc[j]['eType']
            # filenames = listdir(path+m_type+'_'+e_type)
            fig, axs = plt.subplots(1, 5)
            fig.set_figwidth(18)
            for i_cell in range(0,5):
                
                
                search_dir = path+m_type+'_'+e_type+'_'+str(i_cell)
                if(os.path.exists(search_dir)):
                    # fig = P.figure()
                    os.chdir(search_dir)
                    files = filter(os.path.isfile, os.listdir(search_dir))
                    files = [os.path.join(search_dir, f) for f in files] # add path to each file
                    all_csv = [ filename for filename in files if str(filename).endswith( ".csv" ) ]
                    all_csv.sort(key=lambda x: os.path.getmtime(x))
                    # all_csv.sort()
                    param_values_single =[]
                    param_values_single_mean =[]
                    num=0
                    for res in all_csv:
                        
                        if(len(all_csv)>0):
                            curr_csv = res
                            # inum = res.rfind('/')+1
                            # num = res[inum]
                            # inum+=1
                            # while(res[inum].isdigit()):
                            #     num+=res[inum]
                            #     inum+=1

                            
                            # if(not num.isdigit()):
                            #     print("HEPLPPPPPPPPPPPPPPP")
                            #     print(num)
                            # num = int(num)
                            
                            df_cell = pd.read_csv(curr_csv)
                            if(len(df_cell["Mean ECD"])>param):
                                std_cell = df_cell["STD ECD"].iloc[param]
                                mean_cell = df_cell["Mean ECD"].iloc[param]
                            # if(flag==False):
                            #     flag=True
                            #     P_names = list(df_cell["param_name"])
                            #     P_names= [" "]+P_names
                            # # if(len(mean_cell)>num_param):
                            # #     print("Here")

                            param_values_single.append(std_cell)
                            param_values_single_mean.append(mean_cell)
                           
                            #param_values_single.append(mean_cell)
                            # for k in range(len(mean_cell)):
                            #     param_values[k].append(mean_cell[k])
                    

                    # dict1 = OrderedDict(sorted(param_values_single.items()))
                    # dict2 = OrderedDict(sorted(param_values_single_mean.items()))
                    
                    if(len(param_values_single)>0):
                        if(m_type+'_'+e_type+'_'+str(i_cell) not in right_bound_all_cells.keys()):
                                right_bound_all_cells[m_type+'_'+e_type+'_'+str(i_cell)]=[param_values_single[-1]]
                        else:
                                right_bound_all_cells[m_type+'_'+e_type+'_'+str(i_cell)].append(param_values_single[-1])
                        if(m_type+'_'+e_type+'_'+str(i_cell) not in left_bound_all_cells.keys()):
                                left_bound_all_cells[m_type+'_'+e_type+'_'+str(i_cell)]=[param_values_single[0]]
                        else:
                                left_bound_all_cells[m_type+'_'+e_type+'_'+str(i_cell)].append(param_values_single[0])


                    y = param_values_single
                    STD_Params.append(param_values_single)
                    Mean_Params.append(param_values_single_mean)
                    # x = range(0,len(all_csv))
                    # for x1 in x:
                    #     x1 /=2
                    # x = [x1/100-1 for x1 in range(0,200,25)]
                    x =[x1/10 for x1 in range(-15,25,5)]
                    # print(x)
                    #P.xticks(x,P_names,rotation=90)
                    #P.axis([0,len(all_csv)+1,0, max(y)+1 ])
                    # if(len(y)>8):
                    #     y=y[8:]
                    if(len(y)>0):
                        axs[i_cell].set_title(m_type+" "+e_type+" "+str(i_cell) + str(Parameters[param]))
                        axs[i_cell].plot(x, y, alpha=0.3,color = 'y',label = m_type+" "+e_type+" "+str(i_cell))
                    
                    y = param_values_single_mean
                    # if(len(y)>8):
                    #     y=y[8:]
                    # print(m_type,e_type,i_cell,"More than 8")
                    if(len(y)>0):
                        axs[i_cell].plot(x, y, alpha=0.3,color = 'r',label = m_type+" "+e_type+" "+str(i_cell))
                        axs[i_cell].legend(["STD", "Mean"], loc ="lower right")
            pdf.savefig(fig,bbox_inches="tight")
            AllPlotsPDF.savefig(fig,bbox_inches="tight")
            plt.clf()
        # for k in range(num_param):
        #     y = param_values[k]
        #     total_param[k] = total_param[k] + y 
        #     x = np.random.normal(1+k, 0.1, size=len(y))
        #     max_y = max(y)
        #     if(k ==0):
        #         P.plot(x, y,'.', alpha=0.3,color = colour[i],label = "Region"+str(i))
        #         #plt.g.plot(x, y, 'r.', alpha=0.3,color = colour[i],label = "Region"+str(i))

        #     else:
        #         P.plot(x, y,'.', alpha=0.3,color = colour[i])
        #         #fig.plot(x, y, 'r.', alpha=0.3,color = colour[i])
        # P.axis([0,num_param+1,0,  max_y*1.5])
    # ALL_MEAN_PARAMS[str(Parameters[param])]=Mean_Params
    # ALL_STD_PARAMS[str(Parameters[param])]=STD_Params
    #     x = range(0,num_param+1)
    #     P.xticks(x,P_names,rotation=90)

    #     pdf.savefig(fig,bbox_inches="tight")
    # x = range(0,num_param+1)
    # P.axis([0,num_param+1,0, 500 ])

    # P.legend(loc="lower right")
    # P.ylabel("STD")
    # P.xticks(x,P_names,rotation=90)
    # P.savefig("STD.png",bbox_inches="tight")
    # P.boxplot(total_param)
    # P.xticks(x,P_names,rotation=90)
    # P.savefig("STD_box.png",bbox_inches="tight")
    #pdf.savefig(fig)
    pdf.close()
    
    Mean_param = pd.DataFrame.from_dict(Mean_Params)
    STD_param = pd.DataFrame.from_dict(STD_Params)
    Mean_param.to_csv(CSV_DIR+Parameters[param]+"Mean.csv",index=False)
    STD_param.to_csv(CSV_DIR+Parameters[param]+"STD.csv",index=False)
left_bound_all_cells = pd.DataFrame.from_dict(left_bound_all_cells)
left_bound_all_cells.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/left_bound_all_cells.csv")
right_bound_all_cells = pd.DataFrame.from_dict(right_bound_all_cells)
right_bound_all_cells.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/left_bound_all_cells.csv")
AllPlotsPDF.close()