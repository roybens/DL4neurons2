from __future__ import print_function

import os
import sys
import json
import logging as log
from datetime import datetime
from argparse import ArgumentParser
from collections import OrderedDict
from neuron.units import ms, mV
import matplotlib.pyplot as plt
import numpy as np

from neuron import h, gui

from get_rec_points import get_rec_points,get_rec_pts_from_distances,get_rec_pts_for_M1

class BaseModel(object):
    def __init__(self, *args, **kwargs):
        h.celsius = kwargs.pop('celsius', 34)
        # self.log = kwargs.pop('log', print)
        self._set_self_params(*args)

    def _set_self_params(self, *args):
        if len(args) == 0 and hasattr(self, 'DEFAULT_PARAMS'):
            args = self.DEFAULT_PARAMS
        params = {name: arg for name, arg in zip(self.PARAM_NAMES, args)}
        # Model params
        for (var, val) in params.items():
            setattr(self, var, val)

    @property
    def stim_variable_str(self):
        return "clamp.amp"

    def param_dict(self):
        return {name: getattr(self, name) for name in self.PARAM_NAMES}

    def init_hoc(self, dt, tstop):
        h.tstop = tstop
        h.steps_per_ms = 1./dt
        h.stdinit()

    def attach_clamp(self):
        h('objref clamp')
        # print(h.cell)
        clamp = h.IClamp(h.cell(0.5))
        clamp.delay = 0
        clamp.dur = h.tstop
        h.clamp = clamp

    def attach_stim(self, stim):
        # assign to self to persist it
        self.stimvals = h.Vector().from_python(stim)
        self.stimvals.play("{} = $1".format(self.stim_variable_str), h.dt)

    def attach_recordings(self, ntimepts):
        hoc_vectors = {
            'v': h.Vector(ntimepts),
            'ina': h.Vector(ntimepts),
            'ik': h.Vector(ntimepts),
            'ica': h.Vector(ntimepts),
            'i_leak': h.Vector(ntimepts),
            'i_cap': h.Vector(ntimepts),
        }
        
        hoc_vectors['v'].record(h.cell(0.5)._ref_v)
        hoc_vectors['ina'].record(h.cell(0.5)._ref_ina)
        hoc_vectors['ica'].record(h.cell(0.5)._ref_ica)
        hoc_vectors['ik'].record(h.cell(0.5)._ref_ik)
        hoc_vectors['i_leak'].record(h.cell(0.5).pas._ref_i)
        hoc_vectors['i_cap'].record(h.cell(0.5)._ref_i_cap)


        return hoc_vectors

    def set_attachments(self,stim,stim_len,dt):
        ntimepts = stim_len
        tstop = ntimepts * dt
        
        h('objref cell')
        h.cell = self.create_cell()
        self.attach_clamp()
        self.attach_stim(stim)
        self.hoc_vectors = self.attach_recordings(ntimepts)
        self.init_hoc(dt, tstop)

    def simulate(self, stim, dt=0.025,v_init=-68):
        _start = datetime.now()
        # SYNAPSES, NO_SYNAPSES = 1, 0
        # template_name = self.cell_kwargs['model_template'].split(':', 1)[-1]
        # hobj = getattr(h, template_name)(NO_SYNAPSES)
        # self.entire_cell = hobj # do not garbage collect
        
        ntimepts = len(stim)
        tstop = ntimepts * dt
        h.cell=self.entire_cell.soma[0]
        # self.init_hoc(dt, tstop)
        #h.finitialize()
        # h('objref cell')
        # h.cell = self.create_cell()
        
        # print(1)
        # self.attach_clamp()
        # self.attach_stim(stim)
        # print(2)
        # hoc_vectors = self.attach_recordings(ntimepts)
        # self.init_hoc(dt, tstop)
        #hoc_vectors=self.hoc_vectors
        #hoc_vectors = self.attach_recordings(ntimepts)
        #h.frecord_init()
        #hoc_vector ={}
        #self.hoc_vectors = self.attach_recordings(ntimepts)

        h.dt=dt
        self.stimvals = h.Vector().from_python(stim)
        self.stimvals.play("{} = $1".format(self.stim_variable_str), h.dt)
        h.finitialize()
        # print(self.hoc_vectors)
        

        # print(4)
        
        # print("TOPOLOGY",h.topology())
        # self.log.debug("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
        # self.log.debug("({} total timesteps)".format(ntimepts))
        # print("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
        now = datetime.now()        
        h.continuerun(tstop*ms)
        now2 = datetime.now()
        diff = now2-now
        # print("TIME to sim",diff.total_seconds())
        #print(f"hoc vectors is {hoc_vectors['v']}")

        # self.log.debug("Time to simulate: {}".format(datetime.now() - _start))

        # print("After Simulation",self.hoc_vectors)
        return OrderedDict([(k, np.array(v)) for (k, v) in self.hoc_vectors.items()])


class BBP(BaseModel):
    def __init__(self, m_type, e_type, cell_i, *args, **kwargs):
        with open('cells.json') as infile:
            cells = json.load(infile)
        self.args = args
        self.e_type = e_type
        self.m_type = m_type
        self.cell_i = cell_i
        self.cell_kwargs = cells[m_type][e_type][cell_i]

        # If args are not passed in, we use default arguments. The value of self.use_defaults is checked in BBP.create_cell()
        self.use_defaults = (len(args) == 0)

        super(BBP, self).__init__(*args, **kwargs)

    STIM_MULTIPLIER = 1.0

    def _get_rec_pts(self):
        if not hasattr(self, 'probes'):
            self.probes = list(OrderedDict.fromkeys(get_rec_points(self.entire_cell)))
        return self.probes
        
    def _n_rec_pts(self):
        return len(self._get_rec_pts())

    def attach_recordings(self, ntimepts):
        hoc_vectors = OrderedDict()
        for sec in self._get_rec_pts():
            hoc_vectors[sec.hname()] = h.Vector(ntimepts)
            hoc_vectors[sec.hname()].record(sec(0.5)._ref_v)

        return hoc_vectors
    def quit(self):
        h.quit()
    def create_cell(self):
        h.load_file('stdrun.hoc')
        h.load_file('import3d.hoc')
        cell_dir = self.cell_kwargs['model_directory']
        log.debug("cell_dir = {}".format(cell_dir))
        template_name = self.cell_kwargs['model_template'].split(':', 1)[-1]
        templates_dir = '/global/cfs/cdirs/m2043/hoc_templates/hoc_templates'
        
        constants = '/'.join([templates_dir, cell_dir, 'constants.hoc'])
        log.debug(constants)
        h.load_file(constants)

        # morpho_template = '/'.join([templates_dir, cell_dir, 'morphology.hoc'])
        # log.debug(morpho_template)
        # h.load_file(morpho_template)
        
        # biophys_template = '/'.join([templates_dir, cell_dir, 'biophysics.hoc'])
        # log.debug(biophys_template)
        # h.load_file(biophys_template)
        
        # synapse_template = '/'.join([templates_dir, cell_dir, 'synapses/synapses.hoc'])
        # log.debug(synapse_template)
        # h.load_file(synapse_template)
        
        cell_template = '/'.join([templates_dir, cell_dir, 'template.hoc'])
        log.debug(cell_template)
        h.load_file(cell_template)
        
        # For some reason, need to instantiate cell from within the templates directory?
        cwd = os.getcwd()
        os.chdir(os.path.join(templates_dir, cell_dir))
        
        SYNAPSES, NO_SYNAPSES = 1, 0
        hobj = getattr(h, template_name)(NO_SYNAPSES)
        self.entire_cell = hobj # do not garbage collect
        # print(self.entire_cell,"Created CELL")

        os.chdir(cwd)

        # assign self.PARAM_RANGES and self.DEFAULT_PARAMS
        self.PARAM_RANGES, self.DEFAULT_PARAMS = [], []
        for name, sec, param_name, seclist in self.iter_name_sec_param_name_seclist():
#             default = getattr(seclist[0], name, -1)
            default = -1
            if len(seclist) != 0:
                default = getattr(seclist[0], name, -1)
            self.DEFAULT_PARAMS.append(default)
            self.PARAM_RANGES.append((default/10.0, default*10.0) if default != -1 else (0, 0))
        self.DEFAULT_PARAMS = tuple(self.DEFAULT_PARAMS)
        self.PARAM_RANGES = tuple(self.PARAM_RANGES)

        # change biophysics parameters
        if not self.use_defaults:
            for name, sec, param_name, seclist in self.iter_name_sec_param_name_seclist():
                for sec in seclist:
                    if hasattr(sec, name):
                        #print(f'sec is {sec} name is {name} param_name is {param_name} get_attr is {getattr(self, param_name)}')
                        #print(self.DEFAULT_PARAMS)
                        setattr(sec, name, getattr(self, param_name))
                    else:
                        log.debug("Not setting {} (absent from this cell)".format(param_name))
                        continue

        return hobj.soma[0]

    def iter_name_sec_param_name(self):
        """
        The param_names for the BBP model are <parameter>_<section>
        This yields (<parameter>, <section>, <parameter>_<section>) for each
        """
        name_sec = [p.rsplit('_', 1) for p in self.PARAM_NAMES]
        for (name, sec), param_name in zip(name_sec, self.PARAM_NAMES):
            yield name, sec, param_name

    def iter_name_sec_param_name_seclist(self):
        """
        The param_names for the BBP model are <parameter>_<section>
        This yields (<parameter>, <section>, <parameter>_<section>, seclist) for each
        where seclist is a Python list of the Neuron segments in that section
        """
        for name, sec, param_name in self.iter_name_sec_param_name():
            if sec == 'apical':
                seclist = list(self.entire_cell.apical)
            elif sec == 'basal':
                seclist = list(self.entire_cell.basal)
            elif sec == 'dend':
                seclist = list(self.entire_cell.basal) + list(self.entire_cell.apical)
            elif sec == 'somatic':
                seclist = list(self.entire_cell.somatic)
            elif sec == 'axonal':
                seclist = list(self.entire_cell.axonal)
            elif sec == 'all':
                seclist = list(self.entire_cell.all)
            else:
                raise NotImplementedError("Unrecognized section identifier: {}".format(sec))
            yield name, sec, param_name, seclist
            
    def get_varied_params(self):
        """
        Get a list of booleans denoting whether each parameter is varied in this cell or not
        A parameter is varied if 1.) it is present in the section, and 2.) its value is nonzero
        """
        boolarray = []
        for name, sec, param_name, seclist in self.iter_name_sec_param_name_seclist():
#             boolarray.append(getattr(seclist[0], name, 0) != 0)
            currentvalue = 0
            if len(seclist) != 0:
                currentvalue = getattr(seclist[0], name, 0)
            boolarray.append(currentvalue != 0)
        return boolarray

    def get_probe_names(self):
        return ['soma'] + \
            [
                sec.hname().rsplit('.')[-1].replace('[', '_').replace(']', '')
                for sec in self._get_rec_pts()[1:]
            ]
        

class BBPInh(BBP):
    PARAM_NAMES = (
        'gNaTa_tbar_NaTa_t_axonal',
        'gK_Tstbar_K_Tst_axonal',
        'gNap_Et2bar_Nap_Et2_axonal',
        'gCa_LVAstbar_Ca_LVAst_axonal',
        'gSK_E2bar_SK_E2_axonal',
        'gK_Pstbar_K_Pst_axonal',
        'gSKv3_1bar_SKv3_1_axonal',
        'g_pas_axonal',
        'gImbar_Im_axonal',
        'gCabar_Ca_axonal',
        'gK_Tstbar_K_Tst_somatic',
        'gNap_Et2bar_Nap_Et2_somatic',
        'gCa_LVAstbar_Ca_LVAst_somatic',
        'gSK_E2bar_SK_E2_somatic',
        'gK_Pstbar_K_Pst_somatic',
        'gSKv3_1bar_SKv3_1_somatic',
        'g_pas_somatic',
        'gImbar_Im_somatic',
        'gNaTs2_tbar_NaTs2_t_somatic',
        'gCabar_Ca_somatic',
        'gK_Tstbar_K_Tst_dend',
        'gSKv3_1bar_SKv3_1_dend',
        'gNap_Et2bar_Nap_Et2_dend',
        'gNaTs2_tbar_NaTs2_t_dend',
        'gIhbar_Ih_dend',
        'g_pas_dend',
        'gImbar_Im_dend',
        'gkbar_StochKv_somatic',
        'gkbar_KdShu2007_somatic',
        'gkbar_StochKv_dend',
        'gkbar_KdShu2007_dend',
    )

class BBPExc(BBP):
    PARAM_NAMES = (
        'gNaTs2_tbar_NaTs2_t_apical',
        'gSKv3_1bar_SKv3_1_apical',
        'gImbar_Im_apical',
        'gNaTa_tbar_NaTa_t_axonal',
        'gK_Tstbar_K_Tst_axonal',
        'gNap_Et2bar_Nap_Et2_axonal',
        'gSK_E2bar_SK_E2_axonal',
        'gCa_HVAbar_Ca_HVA_axonal',
        'gK_Pstbar_K_Pst_axonal',
        'gSKv3_1bar_SKv3_1_axonal',
        'gCa_LVAstbar_Ca_LVAst_axonal',
        'gSKv3_1bar_SKv3_1_somatic',
        'gSK_E2bar_SK_E2_somatic',
        'gCa_HVAbar_Ca_HVA_somatic',
        'gNaTs2_tbar_NaTs2_t_somatic',
        'gIhbar_Ih_somatic',
        'gCa_LVAstbar_Ca_LVAst_somatic',
        'gIhbar_Ih_dend',
    )
class BBPExcV2(BBP):
    PARAM_NAMES = (
        'gNaTs2_tbar_NaTs2_t_apical',
        'gSKv3_1bar_SKv3_1_apical',
        'gImbar_Im_apical',
        'gIhbar_Ih_dend',
        'gNaTa_tbar_NaTa_t_axonal',
        'gK_Tstbar_K_Tst_axonal',
        'gNap_Et2bar_Nap_Et2_axonal',
        'gSK_E2bar_SK_E2_axonal',
        'gCa_HVAbar_Ca_HVA_axonal',
        'gK_Pstbar_K_Pst_axonal',
        'gCa_LVAstbar_Ca_LVAst_axonal',
        'g_pas_axonal',
        'cm_axonal',
        'gSKv3_1bar_SKv3_1_somatic',
        'gNaTs2_tbar_NaTs2_t_somatic',
        'gCa_LVAstbar_Ca_LVAst_somatic',
        'g_pas_somatic',
        'cm_somatic',
        'e_pas_all'
    )

    UNIT_RANGES = [
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1]]
    
    UNIT_PARAMS =[
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1]
    ]
    
    #these params would be assigned from values of other free parameters (parname:cloned_value)
    CLONED_PARAMS = {'g_pas_dend': 'g_pas_somatic', 'cm_dend': 'cm_somatic', 'gIhbar_Ih_somatic': 'gIhbar_Ih_dend'}
    
    def init_parameters(self):
            # print("INITIALIZING PARAMETERS")
            for name, sec, param_name, seclist in self.iter_name_sec_param_name_seclist():
                for sec in seclist:
                    if hasattr(sec, name):
                        #print(f'sec is {sec} name is {name} param_name is {param_name} get_attr is {getattr(self, param_name)}')
                        #print(self.DEFAULT_PARAMS)
                        setattr(sec, name, getattr(self, param_name))
                    else:
                        log.debug("Not setting {} (absent from this cell)".format(param_name))
                        continue
        

    def _set_self_params(self, *args):
        # print("Setting PARAM Names")
        #print(f'args is {args} param_names are {self.PARAM_NAMES}')
        if len(args) == 0 and hasattr(self, 'DEFAULT_PARAMS'):
            args = self.DEFAULT_PARAMS
        params = {name: arg for name, arg in zip(self.PARAM_NAMES, args)}
        # Model params
        for (var, val) in params.items():
            setattr(self, var, val)
        if(len(params.items())>0):
            for (var,cloned_var) in self.CLONED_PARAMS.items():
                #print(params)
                #print(f'{var} {cloned_var} ,{params[cloned_var]}')
                setattr(self,var,params[cloned_var])
        
            
    def create_cell(self):
        #print('creating BBPExcV2')
        cell = BBP.create_cell(self)
        self.PARAM_RANGES = (
        (1.2E-03,2.6E+00),
        (5.1E-05,4.0E-01),
        (1.4E-05,1.0E-02),
        (8.0E-07,8.0E-01),
        (3.1E-01,4.0E+01),
        (1.0E-04,8.9E-01),
        (5.6E-06,9.8E-02),
        (7.1E-04,9.8E-01),
        (3.1E-05,9.9E-03),
        (4.3E-02,9.7E+00),
        (7.0E-07,8.8E-02),
        (3.0E-07,3.0E-03),
        (0.5,3),
        (7.3E-03,3.0E+00),
        (9.3E-02,1.0E+01),
        (3.3E-05,6.9E-02),
        (3.0E-07,3.0E-03),
        (0.5,3),
        (-100,-50))
        return cell

    def _get_rec_pts(self):
        
        #if not hasattr(self, 'probes'):
        # 
            # print("HI4")
            #This should be taken from probes
            # self.probes = list(OrderedDict.fromkeys(get_rec_pts_from_distances(self.entire_cell,axon_targets = [150],dend_targets = [50])))
            
        self.probes = list(OrderedDict.fromkeys(get_rec_pts_from_distances(self.entire_cell,axon_targets = [150],dend_targets = [50])))
        # print(self.probes)
        # print(self.entire_cell)

        return self.probes
        

class Mainen(BaseModel):
    PARAM_NAMES = (
        'gna_dend',
        'gna_node',
        'gna_soma',
        'gkv_axon',
        'gkv_soma',
        'gca_dend',
        'gkm_dend',
        'gkca_dend',
        'gca_soma',
        'gkm_soma',
        'gkca_soma',
        'depth_cad',
        'c_m',
        'rm',
    )
    DEFAULT_PARAMS = (
        20, 30000, 20, 2000, 200, 0.3, 0.1, 3, 0.3, 0.1, 3, 0.1, 0.75, 30000
    )
    PARAM_RANGES = (
        (10, 250),
        (15000, 80000),
        (10, 60000),
        (1000, 5000),
        (100, 2000),
        (0.15, 0.6),
        (0.05, 0.2),
        (1.5, 6),
        (.15, 6),
        (0.001, 10),
        (0.03, 300),
        (0.05, 0.2),
        (0.3, 1.5),
        (15000, 60000),
    )
    STIM_MULTIPLIER = 1.0

    def __init__(self, *args, **kwargs):
        h.load_file('demofig1.hoc')
        super(Mainen, self).__init__(*args, **kwargs)

    @property
    def stim_variable_str(self):
        return "st.amp"

    def create_cell(self):
        # replication of top of demofig1.hoc and fitCori.hoc
        h.gna_dend = self.gna_dend
        h.gna_node = self.gna_node
        h.gna_soma = self.gna_soma
        h.gkv_axon = self.gkv_axon
        h.gkv_soma = self.gkv_soma
        h.gca = self.gca_dend
        h.gkm = self.gkm_dend
        h.gkca = self.gkca_dend
        h.gca_soma = self.gca_soma
        h.gkm_soma = self.gkm_soma
        h.gkca_soma = self.gkca_soma
        h.depth_cad = self.depth_cad
        h.rm = self.rm
        h.c_m = self.c_m
        
        h.load_3dcell('cells/j7.hoc')
        return h.soma

    def attach_clamp(self):
        self.log.debug("Mainen, not using separate clamp")


class Izhi(BaseModel):
    PARAM_NAMES = ('a', 'b', 'c', 'd')
    DEFAULT_PARAMS = (0.01, 0.2, -65., 2.)
    PARAM_RANGES = ( (0.01, 0.1), (0.1, 0.4), (-80, -50), (0.5, 5) ) # v5 blind sample 
    # PARAM_RANGES = ( (0.01, 0.1), (0.1, 0.4), (-80, -50), (0.5, 10) ) # v6b, not used
    # PARAM_RANGES = ( (-.03, 0.06), (-1.1, 0.4), (-70, -40), (0, 10) )
    STIM_MULTIPLIER = 15.0

    @property
    def stim_variable_str(self):
        return "cell.Iin"

    def create_cell(self):
        self.dummy = h.Section()
        cell = h.Izhi2003a(0.5,sec=self.dummy)

        for var in self.PARAM_NAMES:
            setattr(cell, var, getattr(self, var))

        return cell

    def attach_clamp(self):
        self.log.debug("Izhi cell, not using IClamp")

    def attach_recordings(self, ntimepts):
        vec = h.Vector(ntimepts)
        vec.record(h.cell._ref_V) # Capital V because it's not the real membrane voltage
        return {'v': vec}


class HHPoint5Param(BaseModel):
    PARAM_NAMES = ('gnabar', 'gkbar', 'gcabar', 'gl', 'cm')
    DEFAULT_PARAMS = (500, 10, 1.5, .0005, 0.5)
    PARAM_RANGES = tuple((0.5*default, 2.*default) for default in DEFAULT_PARAMS)
    PARAM_RANGES_v4 = ( (200, 800), (8, 15), (1, 2), (0.0004, 0.00055), (0.3, 0.7) )
    STIM_MULTIPLIER = 20.0

    def create_cell(self):
        cell = h.Section()
        cell.insert('na')
        cell.insert('kv')
        cell.insert('ca')
        cell.insert('pas')

        cell(0.5).na.gbar = self.gnabar
        cell(0.5).kv.gbar = self.gkbar
        cell(0.5).ca.gbar = self.gcabar
        cell(0.5).pas.g = self.gl
        cell.cm = self.cm

        return cell

class HHBallStick7Param(BaseModel):
    PARAM_NAMES = (
        'gnabar_soma',
        'gnabar_dend',
        'gkbar_soma',
        'gkbar_dend',
        'gcabar_soma',
        'gl_soma',
        'cm'
    )
    DEFAULT_PARAMS = (500, 500, 10, 10, 1.5, .0005, 0.5)
    # PARAM_RANGES = (
    #     (200, 800),
    #     (200, 800),
    #     (8, 15),
    #     (8, 15),
    #     (1, 2),
    #     (0.0004, 0.00055),
    #     (0.3, 0.7)
    # )
    PARAM_RANGES = tuple((0.5*default, 2.*default) for default in DEFAULT_PARAMS)
    STIM_MULTIPLIER = 0.18

    DEFAULT_SOMA_DIAM = 21 # source: https://synapseweb.clm.utexas.edu/dimensions-dendrites and Fiala and Harris, 1999, table 1.1

    def __init__(self, *args, **kwargs):
        self.soma_diam = kwargs.pop('soma_diam', self.DEFAULT_SOMA_DIAM)
        self.dend_diam = kwargs.pop('dend_diam', self.DEFAULT_SOMA_DIAM / 10)
        self.dend_length = kwargs.pop('dend_length', self.DEFAULT_SOMA_DIAM * 10)

        super(HHBallStick7Param, self).__init__(*args, **kwargs)
    
    def create_cell(self):
        soma = h.Section()
        soma.L = soma.diam = self.soma_diam
        soma.insert('na')
        soma.insert('kv')
        soma.insert('ca')
        soma.insert('pas')

        dend = h.Section()
        dend.L = self.dend_length
        dend.diam = self.dend_diam
        dend.insert('na')
        dend.insert('kv')

        dend.connect(soma(1))

        # Persist them
        self.soma = soma
        self.dend = dend
        
        for sec in h.allsec():
            sec.cm = self.cm
        for seg in soma:
            seg.na.gbar = self.gnabar_soma
            seg.kv.gbar = self.gkbar_soma
            seg.ca.gbar = self.gcabar_soma
            seg.pas.g = self.gl_soma
        for seg in dend:
            seg.na.gbar = self.gnabar_dend
            seg.kv.gbar = self.gkbar_dend

        return soma

    def attach_recordings(self, ntimepts):
        hoc_vectors = super(HHBallStick7Param, self).attach_recordings(ntimepts)

        hoc_vectors['v_dend'] = h.Vector(ntimepts)
        hoc_vectors['v_dend'].record(self.dend(1)._ref_v) # record from distal end of stick
        
        return hoc_vectors


class HHBallStick9Param(HHBallStick7Param):
    PARAM_NAMES = (
        'gnabar_soma',
        'gnabar_dend',
        'gkbar_soma',
        'gkbar_dend',
        'gcabar_soma',
        'gcabar_dend',
        'gl_soma',
        'gl_dend',
        'cm'
    )
    DEFAULT_PARAMS = (500, 500, 10, 10, 1.5, 1.5, .0005, .0005, 0.5)
    # PARAM_RANGES = tuple((0.7*default, 1.8*default) for default in DEFAULT_PARAMS)
    PARAM_RANGES = tuple((0.5*default, 2.0*default) for default in DEFAULT_PARAMS)
    STIM_MULTIPLIER = 0.3

    def create_cell(self):
        super(HHBallStick9Param, self).create_cell()

        self.dend.insert('ca')
        self.dend.insert('pas')
        for seg in self.dend:
            seg.ca.gbar = self.gcabar_dend
            seg.pas.g = self.gl_dend

        return self.soma

class HHTwoDend13Param(HHBallStick9Param):
    PARAM_NAMES = (
        'gnabar_soma',
        'gnabar_apic',
        'gnabar_basal',
        'gkbar_soma',
        'gkbar_apic',
        'gkbar_basal',
        'gcabar_soma',
        'gcabar_apic',
        'gcabar_basal',
        'gl_soma',
        'gl_apic',
        'gl_basal',
        'cm'
    )
    # DEFAULT_PARAMS = (500, 500, 500, 100, 100, 100, 5, 5, 10, .0005, .0005, .0005, 0.5) # Until 10par v1
    DEFAULT_PARAMS = (500, 500, 500, 10, 10, 10, 1.5, 1.5, 1.5, .0005, .0005, .0005, 0.5) # not used yet
    PARAM_RANGES = tuple((0.5*default, 2.0*default) for default in DEFAULT_PARAMS)
    STIM_MULTIPLIER = 1.0

    def __init__(self, *args, **kwargs):
        super(HHTwoDend13Param, self).__init__(*args, **kwargs)

        # Rename *_apic to *_dend (super ctor sets them based on PARAM_NAME
        self.gnabar_dend = self.gnabar_apic
        self.gkbar_dend = self.gkbar_apic
        self.gcabar_dend = self.gcabar_apic
        self.gl_dend = self.gl_apic

    def create_cell(self):
        super(HHTwoDend13Param, self).create_cell()

        self.apic = self.dend
        
        self.basal = [h.Section(), h.Section()]

        for sec in self.basal:
            sec.L = self.dend_length / 4.
            sec.diam = self.dend_diam

            sec.connect(self.soma(0))
            
            sec.insert('na')
            sec.insert('kv')
            sec.insert('ca')
            sec.insert('pas')
            for seg in sec:
                seg.na.gbar = self.gnabar_basal
                seg.kv.gbar = self.gkbar_basal
                seg.ca.gbar = self.gcabar_basal
                seg.pas.g = self.gl_basal

            
        return self.soma

    
def _mask_in_args(defaults, mask, args):
    i = 0
    for src, default in zip(mask, defaults):
        if src:
            yield args[i]
            i += 1
        else:
            yield default
            
def mask_in_args(*args, **kwargs):
    return list(_mask_in_args(*args, **kwargs))


class HHBallStick4ParamEasy(HHBallStick9Param):

    def __init__(self, *args, **kwargs):
        mask = [1, 0, 0, 1, 1, 0, 0, 1, 0] # 1 = get from these args, 0 = get from superclass
        newargs = mask_in_args(HHBallStick9Param.DEFAULT_PARAMS, mask, args)
        super(HHBallStick4ParamEasy, self).__init__(*newargs, **kwargs)


class HHBallStick4ParamHard(HHBallStick9Param):
    
    def __init__(self, *args, **kwargs):
        mask = [1, 1, 1, 1, 0, 0, 0, 0, 0] # 1 = get from these args, 0 = get from superclass
        newargs = mask_in_args(HHBallStick9Param.DEFAULT_PARAMS, mask, args)
        super(HHBallStick4ParamHard, self).__init__(*newargs, **kwargs)

class HHBallStick7ParamLatched(HHBallStick9Param):
    PARAM_NAMES = (
        'gnabar_soma',
        'gnabar_dend',
        'gkbar_soma',
        'gkbar_dend',
        'gcabar_soma',
        'gcabar_dend',
        'gl_soma',
        'gl_dend',
        'cm'
    )
    
    """ Latch g_l soma and dend """
    def __init__(self, *args, **kwargs):
        # args = list(args)
        # args.append(args[-1]) # use gl_soma as gl_dend
        # args.append(self.DEFAULT_PARAMS[-1]) # default cm
        mask = [1, 1, 1, 1, 1, 1, 1, 0, 0] # 1 = get from these args, 0 = get from superclass
        newargs = mask_in_args(HHBallStick9Param.DEFAULT_PARAMS, mask, args)
        super(HHBallStick7ParamLatched, self).__init__(*newargs, **kwargs)
        self.gl_dend = self.gl_soma

class HHTwoDend10ParamLatched(HHTwoDend13Param):
    def __init__(self, *args, **kwargs):
        mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        newargs = mask_in_args(HHTwoDend13Param.DEFAULT_PARAMS, mask, args)
        super(HHTwoDend10ParamLatched, self).__init__(*newargs, **kwargs)
        self.gl_apic = self.gl_basal = self.gl_soma

# from NeuronModelClass import NeuronModel
# class M2_TTPC_NA_HH:
#     def __init__(self,mod_dir = './Neuron_Model_HH'):
#         self.model = NeuronModel(mod_dir = mod_dir)
        
#     def _set_self_params(self, param_set):
#         self.model.update_params(param_set)
#     def simulate(self,stim, dt,v_init):
#         data,_,_,_ = self.model.run_model_compare(stim, dt, v_init)
#         return data


class M1_TTPC_NA_HH(BaseModel):
    PARAM_NAMES = (
            'dend_na12',
            'soma_na12',
            'ais_na12',
            'dend_na16',
            'soma_na16',
            'ais_na16',
            'ais_ca',
            'ais_KCa',
            'axon_KP',
            'axon_KT',
            'axon_K',
            'axon_KCA',
            'axon_HVA',
            'axon_LVA',
            'node_na',
            'soma_K',
            'dend_k',
            'gpas_all',
            'cm_all'
        )
    DEFAULT_PARAMS = (
        0.006,
        0.0983955,
        4,
        0.006,
        0.0983955,
        4,
        0.0396,
        0.0071,
        0.973538,
        1.7 ,
        1.021945,
        1.8,
        0.00012,
        0.0014,
        2,
        0.0840,
        0.004226,
        0.0000300,
        1 
     )
    #DUMMY Ranges
    PARAM_RANGES = (
        (1.2E-03,2.6E+00),
        (5.1E-05,4.0E-01),
        (1.4E-05,1.0E-02),
        (8.0E-07,8.0E-01),
        (3.1E-01,4.0E+01),
        (1.0E-04,8.9E-01),
        (5.6E-06,9.8E-02),
        (7.1E-04,9.8E-01),
        (3.1E-05,9.9E-03),
        (4.3E-02,9.7E+00),
        (7.0E-07,8.8E-02),
        (3.0E-07,3.0E-03),
        (0.5,3),
        (7.3E-03,3.0E+00),
        (9.3E-02,1.0E+01),
        (3.3E-05,6.9E-02),
        (3.0E-07,3.0E-03),
        (0.5,3),
        (-100,-50))
    
    UNIT_RANGES = [
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1],
        [-1,+1]]
    
    UNIT_PARAMS =[
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1]
    ]
    
    default = {}
    def __init__(self,mod_dir,*args,**kwargs):
        
        self.mod_dir = mod_dir
        # S, self.DEFAULT_PARAMS = [], []
        super(M1_TTPC_NA_HH, self).__init__(*args, **kwargs)
    def update_params(self, params):
        if type(params) == dict:
            h.dend_na12 = params['dend_na12']
            h.soma_na12 = params['soma_na12']
            h.ais_na12 = params['ais_na12']
            h.dend_na16 = params['dend_na16']
            h.soma_na16 = params['soma_na16']
            h.ais_na16 = params['ais_na16']
            h.ais_ca = params['ais_ca']
            h.ais_KCa = params['ais_KCa']
            h.axon_KP = params['axon_KP']
            h.axon_KT = params['axon_KT']
            h.axon_K = params['axon_K']
            h.axon_KCA = params['axon_KCa']
            h.axon_HVA = params['axon_HVA']
            h.axon_LVA = params['axon_LVA']
            h.node_na = params['node_na']
            h.soma_K = params['soma_K']
            h.dend_k = params['dend_K']
            h.gpas_all = params['gpas_all']
            h.cm_all = params['cm_all']
        elif type(params) == list or type(params) == np.ndarray:
            h.dend_na12 = params[0]
            h.soma_na12 = params[1]
            h.ais_na12 = params[2]
            h.dend_na16 = params[3]
            h.soma_na16 = params[4]
            h.ais_na16 = params[5]
            h.ais_ca = params[6]
            h.ais_KCa = params[7]
            h.axon_KP = params[8]
            h.axon_KT = params[9]
            h.axon_K = params[10]
            h.axon_KCA = params[11]
            h.axon_HVA = params[12]
            h.axon_LVA = params[13]
            h.node_na = params[14]
            h.soma_K = params[15]
            h.dend_k = params[16]
            h.gpas_all = params[17]
            h.cm_all = params[18]
            if len(params) > 19:
                h.cell.soma[0].e_pas = params[19]
                h.cell.axon[0].e_pas = params[19]
                h.cell.axon[1].e_pas = params[19]
                for i in range(len(h.cell.dend)):
                    h.cell.dend[i].e_pas = params[19]
        h.working()



    def tes1(self, mod_dir = './neuron_files/M1_TTPC_NA_HH/',#'./Neuron_Model_HH/', 
    
    nav12=1,
                      nav16=1,
                      dend_nav12=1,
                      soma_nav12=1,
                      ais_nav12=1,
                      dend_nav16=1,
                      soma_nav16=1,
                      ais_nav16=1,
                      ais_ca = 1,
                      ais_KCa = 1,
                      axon_Kp=1,
                      axon_Kt =1,
                      axon_K=1,
                      axon_Kca =1,
                      axon_HVA = 1,
                      axon_LVA = 1,
                      node_na = 1,
                      soma_K=1,
                      dend_K=1,
                      gpas_all=1):
        # print(f"nav12={nav12}, nav16={nav16}, dend_nav12={dend_nav12}, soma_nav12={soma_nav12}, ais_nav12={ais_nav12}, "
        # f"dend_nav16={dend_nav16}, soma_nav16={soma_nav16}, ais_nav16={ais_nav16}, ais_ca={ais_ca}, ais_KCa={ais_KCa}, "
        # f"axon_Kp={axon_Kp}, axon_Kt={axon_Kt}, axon_K={axon_K}, axon_Kca={axon_Kca}, axon_HVA={axon_HVA}, "
        # f"axon_LVA={axon_LVA}, node_na={node_na}, soma_K={soma_K}, dend_K={dend_K}, gpas_all={gpas_all}")
        run_dir = os.getcwd()

        os.chdir(mod_dir)
        self.h = h  # NEURON h
        print(f'running model at {os.getcwd()} run dir is {run_dir}')
        #pdb.set_trace()
        h.load_file("runModel.hoc")
        self.soma_ref = h.root.sec
        self.soma = h.secname(sec=self.soma_ref)
        self.sl = h.SectionList()
        self.sl.wholetree(sec=self.soma_ref)
        self.nexus = h.cell.apic[66]
        self.dist_dend = h.cell.apic[91]
        self.ais = h.cell.axon[0]
        self.axon_proper = h.cell.axon[1]
        h.dend_na12 = 0.012/2
        h.dend_na16 = h.dend_na12
        h.dend_k = 0.004226 * soma_K


        h.soma_na12 = 0.983955/10
        h.soma_na16 = h.soma_na12
        h.soma_K = 8.396194779331378477e-02 * soma_K

        h.ais_na16 = 4
        h.ais_na12 = 4
        h.ais_ca = 0.00990*4*ais_ca
        h.ais_KCa = 0.007104*ais_KCa

        h.node_na = 2 * node_na

        h.axon_KP = 0.973538 * axon_Kp
        h.axon_KT = 1.7 * axon_Kt
        h.axon_K = 1.021945 * axon_K
        h.axon_LVA = 0.0014 * axon_LVA
        h.axon_HVA = 0.00012 * axon_HVA
        h.axon_KCA = 1.8 * axon_Kca


        #h.cell.axon[0].gCa_LVAstbar_Ca_LVAst = 0.001376286159287454

        #h.soma_na12 = h.soma_na12/2
        h.naked_axon_na = h.soma_na16/5
        h.navshift = -10
        h.myelin_na = h.naked_axon_na
        h.myelin_K = 0.303472
        h.myelin_scale = 10
        h.gpas_all = 3e-5 * gpas_all
        h.cm_all = 1


        h.dend_na12 = h.dend_na12 * nav12 * dend_nav12
        h.soma_na12 = h.soma_na12 * nav12 * soma_nav12
        h.ais_na12 = h.ais_na12 * nav12 * ais_nav12

        h.dend_na16 = h.dend_na16 * nav16 * dend_nav16
        h.soma_na16 = h.soma_na16 * nav16 * soma_nav16
        h.ais_na16 = h.ais_na16 * nav16 * ais_nav16
        h.working()
        os.chdir(run_dir)


    



    def create_cell(self):
        self.h = h
        run_model = os.path.join(self.mod_dir,"runModel.hoc")
        h.load_file(run_model)
        self.soma_ref = h.root.sec
        self.soma = h.secname(sec=self.soma_ref)
        self.sl = h.SectionList()
        self.sl.wholetree(sec=self.soma_ref)
        self.nexus = h.cell.apic[66]
        self.dist_dend = h.cell.apic[91]
        self.ais = h.cell.axon[0]
        self.axon_proper = h.cell.axon[1]

        #Setting Fixed parameters
        h.naked_axon_na = 0.0196791 #h.naked_axon_na = h.soma_na16/5
        h.navshift = -10
        h.myelin_na = 0.0196791 #h.myelin_na = h.naked_axon_na
        h.myelin_K = 0.303472
        h.myelin_scale = 10
        # self.tes1("/pscratch/sd/k/ktub1999/main/DL4neurons2/Neuron_Model_HH")
        hobj = h.cell
        self.entire_cell = hobj
        return h.cell.soma[0]
        
        
    def _set_self_params(self, *args):
        if len(args) == 0 and hasattr(self, 'DEFAULT_PARAMS'):
            args = self.DEFAULT_PARAMS
        params = {name: arg for name, arg in zip(self.PARAM_NAMES, args)}
        # Model params
        for (var, val) in params.items():
            setattr(self, var, val)

    def init_parameters(self):
        #iterate over default and fixed params and set the vaues.
        for param in self.PARAM_NAMES:
            # if hasattr(sec, name):
            # parameter = getattr(h, param)
            
            # if not hasattr(h, param):
            #     setattr(h, param, h.Section())
            # getattr(h, param).value = getattr(self,param)
            
            setattr(h,param,getattr(self,param))
            # h.param = getattr(self,param)
        h.cell= self.entire_cell
        h.working()
        h.cell = self.entire_cell.soma[0]
    
    def _get_rec_pts(self):
        # if not hasattr(self, 'probes'):
        self.probes = list(OrderedDict.fromkeys(get_rec_pts_for_M1(self.entire_cell)))
        return self.probes
    def _n_rec_pts(self):
        return len(self._get_rec_pts())
    
    def attach_recordings(self, ntimepts):
        hoc_vectors = OrderedDict()
        for sec in self._get_rec_pts():
            hoc_vectors[sec.hname()] = h.Vector(ntimepts)
            hoc_vectors[sec.hname()].record(sec(0.5)._ref_v)
        return hoc_vectors
    def set_attachments(self,stim,stim_len,dt):
        ntimepts = stim_len
        tstop = ntimepts * dt
        
        # h('objref cell')
        h.cell = self.create_cell()
        self.attach_clamp()
        self.attach_stim(stim)
        self.hoc_vectors = self.attach_recordings(ntimepts)
        self.init_hoc(dt, tstop)
    def get_probe_names(self):
        return ['soma'] + \
            [
                sec.hname().rsplit('.')[-1].replace('[', '_').replace(']', '')
                for sec in self._get_rec_pts()[1:]
            ]

MODELS_BY_NAME = {
    'izhi': Izhi,
    'hh_point_5param': HHPoint5Param,
    'hh_ball_stick_7param': HHBallStick7Param,
    'hh_ball_stick_7param_latched': HHBallStick7ParamLatched,
    'hh_ball_stick_4param_easy': HHBallStick4ParamEasy,
    'hh_ball_stick_4param_hard': HHBallStick4ParamHard,
    'hh_ball_stick_9param': HHBallStick9Param,
    'hh_two_dend_13param': HHTwoDend13Param,
    'hh_two_dend_10param': HHTwoDend10ParamLatched,
    'mainen': Mainen,
    'BBP': BBP,
    'M1_TTPC_NA_HH':M1_TTPC_NA_HH
}


if __name__ == '__main__':
    # When executed as a script, this will generate and display traces of the given model at the given params (or its defaults) and overlay a trace with the params shifted 1 rmse
    # parser = ArgumentParser()

    # parser.add_argument('--model', choices=MODELS_BY_NAME.keys(), default='izhi')
    # parser.add_argument('--params', nargs='+', type=float, required=False, default=None)
    # parser.add_argument('--rmse', nargs='+', type=float, required=True)

    # args = parser.parse_args()

    # model_cls = MODELS_BY_NAME[args.model]

    all_rmse = {
        'izhi': [0.0045, 0.011, 0.068, 0.27],
        'hh_point_5param': [rmse * (_max - _min)/2.0 for rmse, (_min, _max)
                            in zip([0.09, 0.39, 0.38, 0.04, 0.05],
                                   HHPoint5Param.PARAM_RANGES)], # could not find rmse valuse in physical units
        # 'hh_point_5param': [rmse * (_max - _min)/2.0 for rmse, (_min, _max)
        #                     in zip([.07, .11, .1, .05, .05],
        #                            HHPoint5Param.PARAM_RANGES_v4)],
        'hh_ball_stick_7param': [49, 55, 1.3, 1.4, 0.16, 1e-5, 0.012],
        'hh_ball_stick_9param': [12, 16, .32, .44, .061, .068, 5.2e-6, 6.8e-6, .0042],
        'hh_two_dend_13param': [51, 34, 110, 12, 9.7, 29, .7, .61, 1.7, 3.9e-5, 2e-5, 7.5e-5, .011],
    }

    print(all_rmse['hh_point_5param'])
    exit()

    
    stim = np.genfromtxt('stims/chirp16a.csv')


    for i, (model_name, model_cls) in enumerate(MODELS_BY_NAME.items()):
        nparam = len(model_cls.PARAM_NAMES)

        plt.figure(figsize=(8, 2*(nparam+2)))
        
        plt.subplot(nparam+2, 1, 1)
        plt.plot(stim, color='red', label='stimulus')
        plt.title("Stimulus")
        plt.legend()

        x_axis = np.linspace(0, len(stim)*0.02, len(stim))
        
        thisstim = stim * MODELS_BY_NAME[model_name].STIM_MULTIPLIER

        model = model_cls(*model_cls.DEFAULT_PARAMS, log=log)
        default_trace = model.simulate(thisstim, 0.02)['v'][:len(stim)]
        
        for i, (param_name, rmse) in enumerate(zip(model_cls.PARAM_NAMES, all_rmse[model_name])):
            plt.subplot(nparam+2, 1, i+2)
            plt.title(param_name)

            plt.plot(x_axis, default_trace, label='Default params', color='k')

            params = list(model_cls.DEFAULT_PARAMS)
            params[i] += rmse
            model = model_cls(*params, log=log)
            trace = model.simulate(thisstim, 0.02)
            plt.plot(x_axis, trace['v'][:len(stim)], label='Default + 1 rmse', color='blue')
            
            params = list(model_cls.DEFAULT_PARAMS)
            params[i] -= rmse
            model = model_cls(*params, log=log)
            trace = model.simulate(thisstim, 0.02)
            plt.plot(x_axis, trace['v'][:len(stim)], label='Default - 1 rmse', color='orange')

            plt.gca().get_xaxis().set_visible(False)


        # extreme smears
        plt.subplot(nparam+2, 1, nparam+2)
        plt.title("All param smear")


        params_add = [param + rmse for param, rmse in zip(model_cls.DEFAULT_PARAMS, all_rmse[model_name])]
        params_sub = [param - rmse for param, rmse in zip(model_cls.DEFAULT_PARAMS, all_rmse[model_name])]
        
        plt.plot(x_axis, default_trace, label='Default params', color='k')
        
        model_add = model_cls(*params_add, log=log)
        trace = model_add.simulate(thisstim, 0.02)
        plt.plot(x_axis, trace['v'][:len(stim)], label='Default + 1 rmse', color='blue')

        model_sub = model_cls(*params_sub, log=log)
        trace = model_sub.simulate(thisstim, 0.02)
        plt.plot(x_axis, trace['v'][:len(stim)], label='Default - 1 rmse', color='orange')

        # on the last plot only
        plt.xlabel("Time (ms)")
        plt.legend()

        plt.subplots_adjust(hspace=0.4)

        plt.savefig('pred_actual_voltages/{}.png'.format(model_name))


