import pandas as pd
import os

CSVpath ="/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/SensitivityCSVs"
def generateBounds():
    Bound ={}
    Bound["Parameter"]=[]
    Bound["UB"]=[]
    Bound["LB"]=[]
    files = os.listdir(CSVpath) # filter(os.path.isfile, os.listdir(CSVpath))
    filesNames = [ filename for filename in files if str(filename).endswith( ".csv" ) ]
    filesNames = [ filename for filename in filesNames if "STD" in str(filename)]
    files = [os.path.join(CSVpath, f) for f in files] # add path to each file
    # print(files)
    allCSV = [ filename for filename in files if str(filename).endswith( ".csv" ) ]
    allCSV = [ filename for filename in allCSV if "STD" in str(filename) ]
    maxVals =[]
    count=0
    thresholdSTD=10
    for currCSV in allCSV:
        df = pd.read_csv(currCSV)
        
        name=df.max(numeric_only=True).tolist()
        UB = -1
        nmax = max(name)
        for n in range(len(name)):
            if(name[n]==nmax):
                UB=min(n-1+2,7)
        LB =-1

        for n in range(len(name)):
            if(name[n]<thresholdSTD):
                LB=n-1
            else:
                break

        Bound["Parameter"].append(filesNames[count])
        Bound["UB"].append(UB)
        Bound["LB"].append(LB)

        name.insert(0,filesNames[count])
        count+=1
        maxVals.append(name)
    df = pd.DataFrame(maxVals)
    df.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/MaxCells.csv",index=False)
    df = pd.DataFrame(Bound)
    df.to_csv("/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/Bounds.csv",index=False)
generateBounds()