import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


stims1 = ['chaotic_50khz.csv','step_200_50khz.csv','ramp_50khz.csv','chirp_50khz.csv','step_500_50khz.csv']
stims2 = ['chaotic4.csv','step_200.csv','ramp.csv','chirp.csv','step_500.csv']
pdf = matplotlib.backends.backend_pdf.PdfPages("/pscratch/sd/k/ktub1999/main/DL4neurons2/UpdatedSTIM.pdf") 
x = np.linspace(0,4001,4000,endpoint=False)

for stim in range(len(stims1)):
    file = open("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/4k50kInter"+stims1[stim],"r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    y = [float(row[0]) for row in data]
    if(stim!=0 and stim!=3):
        for y1 in range(len(y)):
            y[y1]=y[y1]/1000

    file = open("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/"+stims2[stim],"r")
    data2 = list(csv.reader(file, delimiter=","))
    file.close()
    y2 = [float(row[0]) for row in data2]

    fig = plt.figure()
    plt.plot(x,y,'b')
    plt.title("From 50K")
    pdf.savefig(fig)
    fig = plt.figure()
    plt.clf()
    plt.plot(x,y2,'r')
    plt.title("Currently used")
    pdf.savefig(fig)
    plt.title("Compare") 
    plt.plot(x,y,'b')
    pdf.savefig(fig) 
        
pdf.close()

plt.savefig("/pscratch/sd/k/ktub1999/main/DL4neurons2/UpdatedSTIM.png")


