"""
Executes one simulation with specified stimulus.
Displays response and reports spike frequency
"""

from neuron import h, gui
import math
import matplotlib.pyplot as plt
# simulation parameter
h.tstop = 500  # ms, more than long enough for 15 spikes at ISI = 25 ms


# model specification


class Cell:
    def __init__(self):
        soma = h.Section(name="soma", cell=self)
        dend = h.Section(name="dend", cell=self)
        dend.connect(soma)

        soma.nseg = 1
        soma.L = 10
        soma.diam = 100.0 / soma.L / math.pi
        soma.insert("hh")

        dend.nseg = 25
        dend.L = 1000
        dend.diam = 2
        dend.insert("hh")
        for seg in dend:
            seg.hh.gnabar /= 2
            seg.hh.gkbar /= 2
            seg.hh.gl /= 2

        self.dend = dend
        self.soma = soma
        for sec in [soma, dend]:
            sec.Ra = 100
            sec.cm = 1



cell = Cell()



# instrumentation

# experimental manipulations
stim = h.IClamp(cell.soma(0.5))
stim.delay = 1  # ms
stim.dur = 1e9
stim.amp = 0.1  # nA

# data recording and analysis
# count only those spikes that get to distal end of dend
nc = h.NetCon(cell.dend(1)._ref_v, None, sec=cell.dend)
nc.threshold = -10  # mV
spvec = h.Vector()
nc.record(spvec)

NSETTLE = 5  # ignore the first NSETTLE ISI (allow freq to stablize)
NINVL = 10  # num ISI from which frequency will be calculated
NMIN = NSETTLE + NINVL  # ignore recordings with fewer than this num of ISIs


def get_frequency():
    nspikes = spvec.size()
    if nspikes > NMIN:
        t2 = spvec[-1]  # last spike
        t1 = spvec[-(1 + NINVL)]  # NINVL prior to last spike
        return NINVL * 1.0e3 / (t2 - t1)
    else:
        return 0


# Simulation control and reporting of results


def onerun(amp):
    # amp = amplitude of stimulus current
    g = h.Graph(0)
    g.size(0, 500, -80, 40)
    g.view(0, -80, 500, 120, 2, 105, 300.48, 200.32)
    # update graph throughout the simulation
    h.graphList[0].append(g)
    # plot v at distal end of dend
    g.addvar("dend(1).v", cell.dend(1)._ref_v)
    stim.amp = amp
    h.run()
    print("TEST",spvec.size())
    freq = get_frequency()
    print("stimulus %g frequency %g" % (amp, freq))
    plt.plot(spvec)
    spvec2 = h.Vector()
    nc.record(spvec2)
    h.run()
    print("TEST",spvec2.size())
    freq = get_frequency()
    print("stimulus %g frequency %g" % (amp, freq))


onerun(0.15)
plt.savefig("K123.png")