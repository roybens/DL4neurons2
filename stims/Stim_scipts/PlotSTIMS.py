import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


stims1 = ['chaotic_50khz.csv','step_200_50khz.csv','ramp_50khz.csv','chirp_50khz.csv','step_500_50khz.csv']
stims1 = ['chaotic_50khz.csv']

stims2 = ['chaotic4.csv','step_200.csv','ramp.csv','chirp.csv','step_500.csv']
pdf = matplotlib.backends.backend_pdf.PdfPages("UpdatedSTIM.pdf") 
x = np.linspace(0,5001,5000,endpoint=False)

x1 = np.linspace(0,20001,20000,endpoint=False)

for stim in range(len(stims1)):
    # file = open("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/4k50kInter"+stims1[stim],"r")
    # file = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/Exp50k/4k50kInter"+stims1[stim],"r")
    file = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/Exp50k/chaotic_50khz.csv","r")
    
    data = list(csv.reader(file, delimiter=","))
    file.close()
    y = [float(row[0]) for row in data]
    # if(stim!=0 and stim!=3):
    #     for y1 in range(len(y)):
    #         y[y1]=y[y1]/1000

    file = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5k0chaotic5B.csv","r")
    data2 = list(csv.reader(file, delimiter=","))
    file.close()
    y2 = [float(row[0]) for row in data2]
    # y22 = [0]*1000
    # y22.extend(y2)
    # y2=y22
    fig = plt.figure()
    plt.plot(x1,y,'b')
    plt.title("Chaotic50HZ")
    pdf.savefig(fig)
    fig = plt.figure()
    plt.clf()
    plt.plot(x,y2,'r')
    plt.title("ChaoticB")
    pdf.savefig(fig)
    plt.title("Compare") 
    plt.plot(x1,y,'b')
    pdf.savefig(fig) 
        
pdf.close()

plt.savefig("UpdatedSTIM.png")


