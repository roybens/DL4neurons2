import pandas as pd
import numpy as np
import os
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSVpath ="/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/SensitivityCSVs"
nregions = 8
def plotMean():
    pdf = matplotlib.backends.backend_pdf.PdfPages("Mean_all_STD"+".pdf")
    files = os.listdir(CSVpath)  
    filesNames = [ filename for filename in files if str(filename).endswith( ".csv" ) ]
    filesNames = [ filename for filename in filesNames if "STD" in str(filename)]

    files = [os.path.join(CSVpath, f) for f in files] # add path to each file
    # print(files)
    allCSV = [ filename for filename in files if str(filename).endswith( ".csv" ) ]
    allCSV = [ filename for filename in allCSV if "STD" in str(filename) ]
    
    count =0
    x=[x1-1 for x1 in range(0,nregions)]
    for currCSV in allCSV:
        fig = plt.figure()
        df = pd.read_csv(currCSV)
        mean= df.mean()
        STD =df.std()
        plt.title(filesNames[count])
        plt.plot(x,mean)
        plt.errorbar(x, mean,
             yerr = STD,
             fmt ='o')
        pdf.savefig(fig,bbox_inches="tight")
        count+=1

    pdf.close()    


plotMean()