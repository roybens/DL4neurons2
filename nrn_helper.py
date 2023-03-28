import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

tick_major = 6
tick_minor = 4
plt.rcParams["xtick.major.size"] = tick_major
plt.rcParams["xtick.minor.size"] = tick_minor
plt.rcParams["ytick.major.size"] = tick_major
plt.rcParams["ytick.minor.size"] = tick_minor

font_small = 12
font_medium = 13
font_large = 14
plt.rc('font', size=font_small)          # controls default text sizes
plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_small)    # legend fontsize
plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

stim_folder = './stims/'
def plot_stim(stim_fn,dt=0.1):
    stim = np.genfromtxt(stim_fn, dtype=np.float32)
    time_pts = np.ones(len(stim))*dt
    time_pts = np.cumsum(time_pts)
    fig, ax= plt.subplots(1,figsize=(5,5))
    ax.plot(time_pts,stim)
    ax.set_title(f'{stim_fn}')
    fig.savefig(f'{stim_fn}.pdf')
    plt.show()
    return fig,ax

plot_stim(f'{stim_folder}chaotic_2.csv')
plot_stim(f'{stim_folder}step.csv')
plot_stim(f'{stim_folder}sine.csv')
plot_stim(f'{stim_folder}ramp.csv')
plot_stim(f'{stim_folder}chirp16a.csv')



