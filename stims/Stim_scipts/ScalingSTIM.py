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

stims = ['chaotic_50khz.csv','step_200_50khz.csv','ramp_50khz.csv','chirp_50khz.csv','step_500_50khz.csv']
stims = ['chaotic_50khz.csv']
pdf = matplotlib.backends.backend_pdf.PdfPages("UpdatedSTIM.pdf") 
x = np.linspace(0,20001,20000,endpoint=False)

for stim in stims:
    
    # file = open("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/Exp50k/"+stim,"r")
    file = open("/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/Stims/cahotic_50khz.csv","r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    y = [float(row[0]) for row in data]
    yUpdated = resample_by_interpolation(y, 20000, 4000)
    xUpdated = np.linspace(0,4000,4000,endpoint=False)
    fig = plt.figure()
    plt.plot(x,y,'b')
    pdf.savefig(fig)
    fig = plt.figure()
    plt.clf()
    plt.plot(xUpdated,yUpdated,'r')
    pdf.savefig(fig)
    yRes=[0]*1000
    yRes=[*yRes,*yUpdated]
    plt.clf()
    xRes = np.linspace(0,5000,5000)
    plt.plot(xRes,yRes,'r')
    pdf.savefig(fig)
    # myfile = open("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/Exp50k/4k50kInter"+stim,"w")
    myfile = open("/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5k50kInterChaoticB.csv","w")
    for yU in yRes:
        myfile.write(str(yU)+"\n")
        
pdf.close()


# plt.savefig("/pscratch/sd/k/ktub1999/main/DL4neurons2/UpdatedSTIM.png")
# plt.savefig("/pscratch/sd/k/ktub1999/main/DL4neurons2/UpdatedSTIM.png")

