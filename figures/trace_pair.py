import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import numpy as np
import h5py

from compute_similarity import Similarity

def _qa(trace, thresh=30):
    crossings = np.diff((trace > thresh).astype('int'))
    return np.sum(crossings == 1) # It has positive and negative ones for falling/rising edges

def draw_scale(ax, center=(0, 0), x_len=1, y_len=1, x_unit=None, y_unit=None):
    x, y = center
    ax.plot([0+x, 0+x], [0+y, y_len+y], color='k')
    ax.plot([0+x, x_len+x], [0+y, 0+y], color='k')

    if x_unit is not None:
        ax.text(x, y-29, '{} {}'.format(x_len, x_unit), fontsize=6)

    if y_unit is not None:
        ax.text(x-7, y, '{} {}'.format(y_len, y_unit), rotation=90, fontsize=6, verticalalignment='bottom')


if __name__ == '__main__':

    FILE_i = 0

    NTRACES = 2
    NMODELS = 3

    fig = plt.figure(figsize=(NTRACES*4, NMODELS))

    heights = [0.2, 1] * NMODELS
    top_gs = gridspec.GridSpec(NMODELS*2, 2, figure=fig, hspace=0.5, height_ratios=heights, wspace=0)
    # top_gs = gridspec.GridSpec(1, NMODELS, figure=fig, hspace=0.5)

    simpredfiles = [
        h5py.File(x, 'r') for x in (
            '/data/izhi/izhi_4pv6c-ML693-izhi_4pv6c/{}/cellRegr.sim.pred.h5'.format(FILE_i),
            # '/data/izhi/hh_ballstick_4pEv3-ML693-hh_ballstick_4pEv3/{}/cellRegr.sim.pred.h5'.format(FILE_i),
            # '/data/izhi/hh_ballstick_4pHv3-ML693-hh_ballstick_4pHv3/{}/cellRegr.sim.pred.h5'.format(FILE_i),
            '/data/izhi/hh_ballstick_7pv3-ML693-hh_ballstick_7pv3/{}/cellRegr.sim.pred.h5'.format(FILE_i),
        )
    ]
    models = [
        'izhi',
        # 'hh_ball_stick_4param_easy',
        # 'hh_ball_stick_4param_hard',
        'hh_ball_stick_7param_latched',
    ]
    modelnames = ['Izhikevich',
                  # 'Hodgkin-Huxley 4 param (easy)', 'Hodgkin-Huxley 4 param (hard)',
                  'Hodgkin-Huxley 7 param']
    # dist_cutoffs = [(.0001, .0013), (.0008, .0015), (.003, .05)]
    # dist_cutoffs = [(1, 20), (2.0, 7.5), (2.0, 7.5), (5, 5)]
    # dist_cutoffs = [(.01, .064), (.01, .104), (.01, .059), (.03, .188)] # mse params
    dist_cutoffs = [(750, 1400),
                    # (2427, 3121), (2406, 3202),
                    (2324, 3267)] # mse traces
 
    # good_traces = [1, 7, 6, 2]
    # bad_traces = [23, 25, 25, 25]
 
    good_traces = [97,
                   # 6, 146,
                   291] 
    bad_traces = [38,
                  # 20, 22,
                  110]

    for model_i, (model, pname, infile, (d1, d2), good_tr_i, bad_tr_i) in enumerate(zip(models, modelnames, simpredfiles, dist_cutoffs, good_traces, bad_traces)):
        trace_gs = gridspec.GridSpecFromSubplotSpec(1, NTRACES, subplot_spec=top_gs[model_i*2+1, :], wspace=0)
        print(model)
        sim = Similarity(model, 'stims/chirp23a.csv')
        t_axis = np.arange(0, 0.02*9000, 0.02)

        ax_i = -1      # within each model, how many have we plotted
        trace_i = -1   # within each file, how many traces have we checked
        prev_ax = None

        mindist, maxdist = 999, -1
        min_i, max_i = None, None
        nsamples = infile['trace2D'].shape[0]
        
        # for trace_i in range(NTRACES):
        
        # while ax_i < NTRACES-1:
        #     trace_i += 1
        #     if trace_i > nsamples-1:
        #         break

        for trace_i, col in zip((good_tr_i, bad_tr_i), (('lime', 'green'),('cyan', 'blue'))):
            ax_i += 1
            
            v_truth = infile['trace2D'][trace_i, ...].squeeze()
            unit_pred = infile['unitPred2D'][trace_i, ...]
            unit_truth = infile['unitTruth2D'][trace_i, ...]
            phys_par = infile['physPred2D'][trace_i, ...]
            v_pred = sim._data_for(*phys_par)

            if not _qa(v_truth) > 1 or not _qa(v_pred) > 1:
                print("ONE FAILED QA")
                continue

            # dist = sim._similarity(v_truth, v_pred)
            # dist = np.sqrt(sum( (x-y)**2 for (x, y) in zip(unit_pred, unit_truth) ))
            dist = np.sqrt(sum( (x-y)**2 for (x, y) in zip(v_pred, v_truth) ))
            # print(dist)
            if dist < mindist:
                mindist = dist
                min_i = trace_i
            elif dist > maxdist:
                maxdist = dist
                max_i = trace_i
                
            # if dist < d1:
            #     print("got one")
            #     ax_i += 1
            # elif dist > d2:
            # if dist > d2:
            #     ax_i += 1
            # else:
            #     continue

            ax = plt.subplot(trace_gs[:, ax_i], sharex=prev_ax)#, sharey=prev_ax)
            ax.axis('off')
            
            ax.plot(t_axis, v_truth, color='k', linewidth=0.5, label='True')
            # ax.plot(t_axis, v_pred, color='red', linewidth=0.5, label='Predicted', linestyle='--')

            # ax.text(80, 0, "trace #{}".format(trace_i))

            # if model_i == 2:
            if ax_i == 0 and model_i == 0:
                draw_scale(ax, center=(160, -30), x_len=10, y_len=40, x_unit='ms', y_unit='mV')

            # if trace_i == NTRACES-1:
            #     ax.set_xlabel('Time (ms)')
            # else:
            #     ax.set_xticklabels([])

            if ax_i == 0:
                # ax.set_title(pname)
                title_gs = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=top_gs[2*model_i, :])
                title_ax = plt.subplot(title_gs[:])
                title_ax.axis('off')
                title_ax.text(0.5, 0, pname, horizontalalignment='center',
                              verticalalignment='center', transform=title_ax.transAxes)

            if ax_i == NTRACES-1 and model_i == 0:
                ax.legend(bbox_to_anchor=(0.8, 0.9), loc=2, prop={'size': 6}, frameon=False)

            if model_i == NMODELS-1:
                ax.text(0.5, -0.3, "Similar" if ax_i == 0 else 'Dissimilar',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

            prev_ax = ax

            
        # good_v_truth = infile['trace2D'][good_i, ...]
        # good_phys_par = infile['physPred2D'][good_i, ...]
        # good_v_pred = sim._data_for(*good_phys_par)

        # bad_v_truth = infile['trace2D'][bad_i, ...]
        # bad_phys_par = infile['physPred2D'][bad_i, ...]
        # bad_v_pred = sim._data_for(*bad_phys_par)

        # good_ax = plt.subplot(gs[0, 3*model_i:3*(model_i+1)])
        # good_ax.plot(t_axis, good_v_truth, color='lime', linewidth=0.5, label='True params')
        # good_ax.plot(t_axis, good_v_pred, color='green', linewidth=0.5, label='Predicted')
        # good_
        # good_ax.set_title(pname)

        # bad_ax = plt.subplot(gs[1, 3*model_i:3*(model_i+1)])
        # bad_ax.plot(t_axis, bad_v_truth, color='cyan', linewidth=0.5, label='True params')
        # bad_ax.plot(t_axis, bad_v_pred, color='blue', linewidth=0.5, label='Predicted')
        # bad_ax.set_xlabel('Time (ms)')




    # BOTTOM ROW FROM ROYS DATA
    roysdata = np.genfromtxt('figures/forVyassa.csv')
    pred_v_lg = roysdata[0][5500:14500]
    truth_v_lg = roysdata[1][5500:14500]
    pred_v_sm = roysdata[2][5500:14500]
    truth_v_sm = roysdata[3][5500:14500]
    
    trace_gs = gridspec.GridSpecFromSubplotSpec(1, NTRACES, subplot_spec=top_gs[2*NMODELS-1, :], wspace=0)
    ax = plt.subplot(trace_gs[0], sharex=prev_ax)#, sharey=prev_ax)
    ax.axis('off')
    ax.plot(t_axis, truth_v_sm, color='k', linewidth=0.5, label='True')
    ax.plot(t_axis, pred_v_sm, color='red', linewidth=0.5, linestyle='--', label='Predicted')
    ax.text(0.5, -0.3, "Similar",
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax = plt.subplot(trace_gs[1], sharex=prev_ax)#, sharey=prev_ax)
    ax.axis('off')
    ax.plot(t_axis, truth_v_lg, color='k', linewidth=0.5)
    ax.plot(t_axis, pred_v_lg, color='red', linewidth=0.5, linestyle='--')
    ax.text(0.5, -0.3, "Dissimilar",
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    title_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top_gs[2*NMODELS-2, :])
    title_ax = plt.subplot(title_gs[:])
    title_ax.axis('off')
    title_ax.text(0.5, 0, "Mainen 10 param", horizontalalignment='center',
                        verticalalignment='center', transform=title_ax.transAxes)

    # plt.savefig('trace_compare.png'.format(FILE_i)) 
    plt.show()
            
        # for phys_pred, unit_pred, unit_truth, v_truth in sim.iter_sim_predictions([simpredfile]):
        #     v_pred = self._data_for(*phys_pred)
        #     isi = self._similarity(v_truth, v_pred)

        

    
