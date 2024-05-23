import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf

def read_waveform_from_csv(file_path):
    waveform = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            value = float(row[0])  # Assuming the waveform values are in the first column
            waveform.append(value)
    return waveform

def add_waveforms(waveform1, waveform2):
    max_limit = 6 #max(max(waveform1), max(waveform2))
    result = []
    for i in range(len(waveform1)):
        value = waveform1[i] + waveform2[i]
        if value > max_limit:
            value = max_limit
        result.append(value)
    return result

# Example usage
ramp_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/ramp.csv'
chaotic_file = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5k50kInterChaoticB.csv'
pdf = matplotlib.backends.backend_pdf.PdfPages("ChaoticRAMP.pdf") 

ramp = read_waveform_from_csv(ramp_file)
chaotic = read_waveform_from_csv(chaotic_file)
ramp2=[0]*1000
ramp2.extend(ramp)
ramp2=[r*8 for r in ramp2]
ramp_chaotic = add_waveforms(ramp2, chaotic)
fig = plt.figure()
x=np.linspace(0,5001,5000,endpoint=False)
plt.plot(x,ramp2,'b')
plt.title("Chaotic+RAMP")

plt.plot(x,chaotic,'b')
pdf.savefig(fig)

fig = plt.figure()
plt.plot(x,ramp_chaotic,'b')
plt.title("ChaoticRAMP")
pdf.savefig(fig)
pdf.close()
myfile = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5kChaoticRamp.csv","w")
for yU in ramp_chaotic:
        myfile.write(str(yU)+"\n")
