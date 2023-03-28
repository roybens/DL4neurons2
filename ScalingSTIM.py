import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

stims = ['chaotic3.csv','step_200.csv','ramp.csv','chirp.csv','step_500.csv']
pdf = matplotlib.backends.backend_pdf.PdfPages("/global/homes/k/ktub1999/mainDL4/DL4neurons2/UpdatedSTIM.pdf") 
x = np.linspace(0,4001,4000,endpoint=False)

for stim in stims:
    file = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/"+stim,"r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    y = [float(row[0]) for row in data]

    yUpdated = resample_by_interpolation(y, 4000, 1600)
    xUpdated = np.linspace(0,1601,1600,endpoint=False)
    fig = plt.figure()
    plt.plot(x,y,'b')
    pdf.savefig(fig)
    fig = plt.figure()
    plt.clf()
    plt.plot(xUpdated,yUpdated,'r')
    pdf.savefig(fig)
    myfile = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/Updated"+stim,"w")
    for yU in yUpdated:
        myfile.write(str(yU)+"\n")
        
pdf.close()

plt.savefig("/global/homes/k/ktub1999/mainDL4/DL4neurons2/UpdatedSTIM.png")


