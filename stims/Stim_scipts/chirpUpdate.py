import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

def UpdateChaotic():
    pdf = matplotlib.backends.backend_pdf.PdfPages("/global/homes/k/ktub1999/mainDL4/DL4neurons2/UpdatedChaotic.pdf") 
    file = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/Updatedchaotic3.csv","r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    yold = [float(row[0]) for row in data]
    y = [float(row[0])*1.5 for row in data]
    fig = plt.figure()
    plt.clf()
    plt.plot(yold)
    pdf.savefig(fig)
    plt.clf()
    plt.plot(y)
    pdf.savefig(fig)
    myfile = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/Updatedchaotic4.csv","w")
    for yU in y:
        myfile.write(str(yU)+"\n")
    pdf.close()

UpdateChaotic()
