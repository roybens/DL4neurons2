"""
This file defines the stimuli we use
"""
import numpy as np
# from pynwb import TimeSeries

# Defaults:
DT = 0.02 # simulation timestep
PREPULSE = 2
PULSELEN = 144
POSTPULSE = 6
TAU = 3

class StimulusGenerator(object):

    def __init__(self, *args, **kwargs):
        self.dt = kwargs.pop('dt', DT)
        self.prepulse = kwargs.pop('prepulse', PREPULSE)
        self.pulselen = kwargs.pop('pulselen', PULSELEN)
        self.postpulse = kwargs.pop('postpulse', POSTPULSE)

    def _pulse(self, n_pulse, **stim_args):
        raise NotImplementedError()

    def generate(self, **stim_args):
        n_pre = int(self.prepulse / self.dt)
        n_pulse = int(self.pulselen / self.dt)
        n_post = int(self.postpulse / self.dt)
        n_tot = n_pre + n_pulse + n_post

        stim = np.zeros(n_tot, dtype=np.float64)
        stim[n_pre:n_pre+n_pulse] = self._pulse(n_pulse, **stim_args)

        return stim

    def write_csv(self, filename, delimiter='\n', append=False, **stim_args):
        # NOT TESTED
        print("WARNING: StimulusGenerator.write_csv() has not been tested")
        stim = self.generate(**stim_args)
        mode = 'ab' if append else 'wb' # 'b' for binary mode
        with open(filename, mode) as f:
            np.savetxt(f, stim, delimiter=delimiter)


class StimGeneratorFromFile(StimulusGenerator):
    # TODO: Not tested
    def __init__(self, filename, *args, **kwargs):
        self.pulse = np.genfromtxt(dtype=np.float64)
        if 'pulselen' in kwargs:
            assert pulselen == len(self.stim), "If specifying pulselen, it must agree with the length of the file"
        super(StimGeneratorFromFile, self).__init__(*args, **kwargs)

    def _pulse(self, n_pulse, **stim_args):
        return self.pulse
    

class RampGenerator(StimulusGenerator):
    def _pulse(self, n_pulse, rampval=0.):
        return np.linspace(start=0, stop=rampval, num=n_pulse, dtype=np.float64)

    
class NegRampGenerator(RampGenerator):
    def _pulse(self, n_pulse, rampval=0.):
        return np.linspace(start=rampval, stop=0, num=n_pulse, dtype=np.float64)

    
class StepGenerator(StimulusGenerator):
    def _pulse(self, n_pulse, stepval):
        return np.ones(n_pulse, dtype=np.float64) * stepval

class NoiseGenerator(StimulusGenerator):
    def _pulse(self, n_pulse, mean=0., sd=1., tau=TAU):
        pulse = np.zeros(n_pulse, dtype=np.float64)
        for i in range(1, n_pulse):
            diff = ((mean - pulse[i-1])*self.dt/tau
                    + sd*np.sqrt(2*self.dt/tau)*np.random.normal(0, 1))
            pulse[i] = pulse[i-1] + diff

        return pulse


class SinGenerator(StimulusGenerator):
    pass # TODO


class ChirpGenerator(StimulusGenerator):
    pass # TODO



n_stims = 8

rampvals = [4.0 + 1.0*i for i in range(n_stims)]
stepvals = [4.0 + 0.5*i for i in range(n_stims)]
noise_means = [4.0 + 0.5*i for i in range(n_stims)]
sds = [0.2*(i + 1) for i in range(n_stims)]

stims = {
    'ramp': [RampGenerator().generate(rampval=val) for val in rampvals],
    # 'neg_ramp': [NegRampGenerator().generate(rampval=-val) for val in rampvals],
    'step': [StepGenerator().generate(stepval=val) for val in stepvals],
    'noise': [NoiseGenerator().generate(mean=mean, sd=sd)
              for mean, sd in zip(noise_means, sds)],
    'sin': [],
    'chirp': [],
}

def add_stims(nwb):
    for stim_type, stim_list in iter(stims.items()):
        for i, stim in enumerate(stim_list):
            stim_name = '{}_{:02d}'.format(stim_type, i)
            stim_timeseries = TimeSeries(stim_name, stim, 'nA', rate=1.0/DT)
            nwb.add_stimulus(stim_timeseries)
