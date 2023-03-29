""" 
Tools for comparing predicted/actual voltage traces
"""
import logging as log
import itertools
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import h5py
import numpy as np

from models import MODELS_BY_NAME
from compute_similarity import Similarity


def histogram(exp_exp_similarity, exp_pred_similarity, sim_pred_similarity):
    with h5py.File(exp_exp_similarity) as infile:
        exp_exp_sim = infile['similarity'][:]

    with h5py.File(exp_pred_similarity) as infile:
        exp_pred_sim = infile['similarity'][:]

    with h5py.File(sim_pred_similarity) as infile:
        sim_pred_sim = infile['similarity'][:]

    # plt.hist([exp_exp_sim, exp_pred_sim, sim_pred_sim], density=True, bins=20, label=['exp/exp', 'exp/pred', 'sim/pred'])
    plt.hist(exp_exp_sim, density=True, bins=80, alpha=0.4, label='exp/exp', edgecolor='k', color='blue')
    plt.hist(exp_pred_sim, density=True, bins=30, alpha=0.4, label='exp/pred', edgecolor='k')
    plt.hist(sim_pred_sim, density=True, bins=80, alpha=0.4, label='sim/pred', edgecolor='k', color='orange')

    expexp_mean = np.average(exp_exp_sim)

    plt.plot([expexp_mean, expexp_mean], plt.ylim(), color='blue', alpha=0.6, label='exp/exp mean')
    # plt.text(expexp_mean*1.01, .9*plt.ylim()[1], '{0:.2f}'.format(expexp_mean), color='blue')
    plt.text(
        0.005, 0.95,
        '{}% below'.format(int(100 * sum(sim_pred_sim < expexp_mean) / len(sim_pred_sim))),
        transform=plt.gca().transAxes, color='orange'
    )
    
    plt.xlabel('ISI distance')
    plt.legend()
    plt.show()


def traces(sim, is_sim, circle_params, true_vs, pred_vs, trace_axs, colors, exp_exp_similarity=None):
    x_axis = np.arange(0, .02*9000, .02)
    for circle_par, true_v, pred_v, ax, (col_pred, col_true) in zip(circle_params, true_vs, pred_vs, trace_axs, colors):
        trace_similarity = sim._similarity(true_v, pred_v)

        if is_sim:
            true_label = 'Sim. (true params)'
            pred_label = 'Sim. (pred. params)'
        else:
            true_label = 'Experiment'
            pred_label = 'Sim. (pred. params)'
            
        ax.plot(x_axis, pred_v, linewidth=0.5, label=pred_label, color=col_pred)
        ax.plot(x_axis, true_v, linewidth=0.5, label=true_label, color=col_true)
        
        if exp_exp_similarity:
            txt = "rel. sim = {0:.2f}\n".format((trace_similarity - expexp_mean)/expexp_std)
        else:
            txt = ''
        txt += "sim. = {0:.2f}".format(trace_similarity)
            
        ax.text(1.02, 0.6, txt, transform=ax.transAxes, fontsize=8)
        ax.legend(bbox_to_anchor=(0.95, 0.6), loc=2, prop={'size': 6})

    trace_axs[0].set_xticklabels([])
    trace_axs[1].set_xticklabels([])
    trace_axs[-1].set_xlabel("Time (ms)")
    trace_axs[-1].set_ylabel("V_m")


def similarity_heatmap(similarity_file, param_x=0, param_y=1, exp_exp_similarity=None, sweepfile=None, nbins=20, phys=True, plot_params='pred', range_x=None, range_y=None, vminmax=3):
    """
    Compute heatmaps of similarity between actual/predicted traces.
    if exp_exp_similarity is specified, we normalize to its mean/stdev
    if phys=False, plot w/ unit-normalized params on axes
    sweepfile: if passing an experimental similarity file, pass in the sweep file so it can get traces
    """
    with h5py.File(similarity_file, 'r') as infile:
        modelname = infile['similarity'].attrs['modelname']
        sim = Similarity(modelname, 'stims/chirp23a.csv')
        param_ranges = MODELS_BY_NAME[modelname].PARAM_RANGES
        param_names = MODELS_BY_NAME[modelname].PARAM_NAMES
        nsamples = infile['similarity'].shape[0]

        if exp_exp_similarity:
            with h5py.File(exp_exp_similarity, 'r') as expexp_infile:
                expexp_mean = np.average(expexp_infile['similarity'])
                expexp_std = np.std(expexp_infile['similarity'])
        else:
            expexp_mean, expexp_std = 0, 1

        # DETERMINE IF EXPERIMENT OR SIMULATION
        if 'unitTruth2D' in infile or 'unitTruth3D' in infile:
            is_expt, is_sim = False, True
        else:
            is_expt, is_sim = True, False

        if plot_params == 'truth':
            assert is_sim, "Can only access truth in simulation"

        if phys:
            if plot_params == 'truth':
                range_x = range_x or param_ranges[param_x]
                range_y = range_y or param_ranges[param_y]
            else:
                range_x = range_x or [val*1.05 for val in param_ranges[param_x]]
                range_y = range_y or [val*1.05 for val in param_ranges[param_y]]
            truth_paramskey = 'physTruth2D' if 'physTruth2D' in infile else 'physTruth3D'
            pred_paramskey = 'physPred2D' if 'physPred2D' in infile else 'physPred3D'
        else:
            if plot_params == 'truth':
                range_x = range_x or (-1, 1)
                range_y = range_y or (-1, 1)
            else:
                range_x = range_x or (-1.05, 1.05)
                range_y = range_y or (-1.05, 1.05)
            truth_paramskey = 'unitTruth2D' if 'unitTruth2D' in infile else 'unitTruth3D'
            pred_paramskey = 'unitPred2D' if 'unitPred2D' in infile else 'unitPred3D'

        plot_paramskey = truth_paramskey if plot_params == 'truth' else pred_paramskey

        all_bins_x = np.linspace(*range_x, nbins+1)
        all_bins_y = np.linspace(*range_y, nbins+1)

        bins_x = np.digitize(infile[plot_paramskey][:, param_x], all_bins_x) - 1
        bins_y = np.digitize(infile[plot_paramskey][:, param_y], all_bins_y) - 1

        valid = (bins_x >= 0) & (bins_y >= 0) & (bins_x < nbins) & (bins_y < nbins)

        binned_similarities = {(binx, biny): [] for binx, biny
                               in itertools.product(range(nbins), repeat=2)}
        similarity = (infile['similarity'][:] - expexp_mean) / expexp_std
        for bin_x, bin_y, simil in zip(bins_x[valid], bins_y[valid], similarity[valid]):
            binned_similarities[(bin_x, bin_y)].append(simil)

        binned_averaged_similarity = np.zeros(shape=(nbins, nbins))
        for (bin_x, bin_y), similarities in binned_similarities.items():
            # binned_averaged_similarity[bin_x, bin_y] = np.min(similarities) if similarities else np.nan
            binned_averaged_similarity[bin_x, bin_y] = np.average(similarities)

        if plot_params == 'truth' and is_expt:
            raise ValueError("Cannot plot truth params for experimental sweep")


        # indices for random traces
        trace_i = np.random.choice(range(nsamples), 3)
        # trace_i[2] = 981
        trace_i = sorted(trace_i)

        # Grab 3 true traces
        if is_sim:
            true_params = infile[truth_paramskey][trace_i, :]
            true_vs = [sim._data_for(*true_p, unit=not phys) for true_p in true_params]
        else:
            true_params = None
            true_vs = infile['sweep2D'][trace_i, :]

        # Grab 3 random sets of predicted params and compute traces for them
        pred_params = infile[pred_paramskey][trace_i, :]
        pred_vs = [sim._data_for(*pred_p, unit=not phys) for pred_p in pred_params]

        # Grab coordinates for circles
        if plot_params == 'truth':
            assert is_sim
            circle_params = true_params
        else:
            circle_params = pred_params

    # Display with correct axis labeling
    plt.clf()
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.5)
    heatmap_ax = plt.subplot(gs[:3, :3])
    trace_axs = [plt.subplot(gs[i, 3:]) for i in range(3)]

    cmap = plt.cm.RdGy
    cmap.set_bad('magenta')
    heatmap_ax.set_facecolor('magenta')
    vminmax = vminmax or np.nanpercentile(np.abs(binned_averaged_similarity), 98) / 0.98
    im = heatmap_ax.pcolormesh(all_bins_x, all_bins_y, binned_averaged_similarity.T,
                               cmap=cmap, vmin=-vminmax, vmax=vminmax)
    plt.colorbar(im, ax=heatmap_ax)
    heatmap_ax.set_xlabel('{} param: {}'.format(plot_params, param_names[param_x]))
    heatmap_ax.set_ylabel('{} param: {}'.format(plot_params, param_names[param_y]))

    # Plot traces
    x_axis = np.arange(0, .02*9000, .02)
    colors = (('k', 'grey',), ('green', 'lime'), ('blue', 'cyan'))
    traces(sim, is_sim, circle_params, true_vs, pred_vs, trace_axs, colors)
    for circle_par, true_v, pred_v, ax, (col_pred, col_true) in zip(circle_params, true_vs, pred_vs, trace_axs, colors):
        # trace_similarity = sim._similarity(true_v, pred_v)

        # if is_sim:
        #     true_label = 'Sim. (true params)'
        #     pred_label = 'Sim. (pred. params)'
        # else:
        #     true_label = 'Experiment'
        #     pred_label = 'Sim. (pred. params)'
            
        # ax.plot(x_axis, pred_v, linewidth=0.5, label=pred_label, color=col_pred)
        # ax.plot(x_axis, true_v, linewidth=0.5, label=true_label, color=col_true)
        
        # if exp_exp_similarity:
        #     txt = "rel. sim = {0:.2f}\n".format((trace_similarity - expexp_mean)/expexp_std)
        # else:
        #     txt = ''
        # txt += "sim. = {0:.2f}".format(trace_similarity)
            
        # ax.text(1.02, 0.6, txt, transform=ax.transAxes, fontsize=8)
        # ax.legend(bbox_to_anchor=(0.95, 0.6), loc=2, prop={'size': 6})

        col_dot = col_true if plot_params == 'truth' else col_pred
        # TODO: The circles need to be drawn in axis coordinates, not data coords
        # dot = Circle((circle_par[param_x], circle_par[param_y]), radius=0.04, color=col_dot)
        # heatmap_ax.add_patch(dot, transform=heatmap_ax.transAxes)
        heatmap_ax.scatter(circle_par[param_x], circle_par[param_y], color=col_dot)

    trace_axs[0].set_xticklabels([])
    trace_axs[1].set_xticklabels([])
    trace_axs[-1].set_xlabel("Time (ms)")
    trace_axs[-1].set_ylabel("V_m")


    if '--save' in sys.argv:
        plt.savefig('similarity/{}_avg_similarity_{}_vs_{}.png'.format(modelname, param_names[param_x], param_names[param_y]))
    plt.show()


if __name__ == '__main__':
    # histogram('041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', '041019A_1-ML203b_izhi_4pv6c.mpred_ExpPredSimilarity.h5', 'cellRegr.sim.pred_SimPredSimilarity.h5')

    
    # pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # # pairs = [(0, 1),]
    # for x, y, in pairs:
    #     similarity_heatmap('cellRegr.sim.pred_SimPredSimilarity.h5', exp_exp_similarity='041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', param_x=x, param_y=y, phys=False, plot_params='pred')
    #     similarity_heatmap('cellRegr.sim.pred_SimPredSimilarity.h5', exp_exp_similarity='041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', param_x=x, param_y=y, phys=True, plot_params='pred')

        
    # similarity_heatmap('041019A_1-ML203b_izhi_4pv6c.mpred_ExpPredSimilarity.h5', exp_exp_similarity='041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', param_x=2, param_y=3, phys=False, plot_params='pred', range_x=(1.05, 1.15), range_y=(.6, 1))

    # similarity_heatmap('041019A_1-ML203b_izhi_4pv6c.mpred_ExpPredSimilarity.h5', exp_exp_similarity='041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', param_x=1, param_y=2, phys=False, plot_params='pred', range_x=(-1, -.75), range_y=(1.05, 1.15))

    # similarity_heatmap('041019A_1-ML203b_izhi_4pv6c.mpred_ExpPredSimilarity.h5', exp_exp_similarity='041019A_1-ML203b_izhi_4pv6c.mpred_ExpExpSimilarity.h5', param_x=0, param_y=1, phys=False, plot_params='pred', range_x=(0.95, 1.15), range_y=(-.95, -.75))

    # similarity_heatmap('/data/izhi/hh_ballstick_7pv3-ML693-hh_ballstick_7pv3/0/cellRegr.sim.pred.h5')
    similarity_heatmap('cellRegr.sim.pred_SimPredSimilarity.h5')
