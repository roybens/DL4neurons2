# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:18:05 2020

@author: bensr
"""


from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from optimize_na_ga import *


def plot_exp_vs_sim_vclamp(experiment,variant):
    real_data_map = read_all_raw_data()
    real_data = real_data_map[exp][mutant]
    gen_all_curves_given_params(mutant, exp, params, real_data, save=save)
    return


def gen_all_curves_given_params(mutant, exp, params, target_data, save=False, file_name=None):
    plt.close()
    fig, axs = plt.subplots(2, figsize=(6,10))
    fig.suptitle("Mutant: {} \n Experiment: {}".format(mutant, exp))
    change_params(params)
    sim_data = gen_sim_data()
    data = [target_data, sim_data]
    names = ["experimental", "simulated"]
    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Fraction In/activated')
    axs[0].set_title("Inactivation and Activation Curves")
    for i in range(len(data)):
        data_pts = data[i]["inact"]
        sweeps = data[i]["inact sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        axs[0].scatter(sweeps, data_pts)
        axs[0].plot(even_xs, curve, label=names[i]+" inactivation")
    for i in range(len(data)):
        data_pts = data[i]["act"]
        sweeps = data[i]["act sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        axs[0].scatter(sweeps, data_pts)
        axs[0].plot(even_xs, curve, label=names[i]+" activation")
    axs[0].legend()
    axs[1].set_xlabel('Log(Time)')
    axs[1].set_ylabel('Fractional Recovery')
    axs[1].set_title("Recovery from Inactivation")
    for i in range(len(data)):
        data_pts = data[i]["recov"]
        times = data[i]["recov times"]
        axs[1].scatter(np.log(times), data_pts, label=names[i])
    axs[1].legend()
    fig.text(.5, .92, "\n Target tau: {}, Sim tau: {}".format(target_data['tau0'], sim_data['tau0']), ha='center')
    plt.show()
    if save:
        if file_name is None:
            file_name = "{}_{}_plots".format(exp, mutant).replace(" ", "_")
        fig.savefig("./my_curves/"+file_name+'.eps')
        fig.savefig("./my_curves/"+file_name+'.pdf')
        
opt_na_pipeline("M1879 T and R1626Q", "NaV12 adult WT")