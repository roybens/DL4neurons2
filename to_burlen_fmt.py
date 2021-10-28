"""
Pass --m-type, --e-type, and --outdir as args to this script

Outputs the format expected by Burlen's VTK plugin
specified in https://docs.google.com/document/d/1sYXKnYKfr_5ljKqns4aFDi67N2Hq7nPeY0B7qKanEog

The output will look like it came from my simulation, running on one
cell for a single timepoint. The data arrays (membrane variables,
spikes) are all [0]. The only real data is the morphology and section
names in <outdir>/seg_coords/0.h5

The section names are stored in 2 arrays within the file
<outdir>/seg_coords/0.h5. Each element of each array corresponds to a section.

  part_name = elements are a coded version of the part name
              0 = soma, 1 = dend, 2 = apic, 3 = basal, 4 = axon

  part_idx  = elements are the index within the named part

"""
import shutil
import logging as log
import os
from argparse import ArgumentParser
import h5py
import numpy as np

from neuron import h

from morphology import Morphology
from run import get_model

ZERO_ARR = np.array([0], dtype=np.float32)

def create_master_file(outdir):
    masterfilename = os.path.join(outdir, 'master.h5')
    with h5py.File(masterfilename, 'w') as masterfile:
        masterfile.create_dataset('neuro-data-format-version', data=0.3)
        masterfile.create_dataset('num_cells', data=1)
        masterfile.create_dataset('spikerate_bins', data=ZERO_ARR)
        masterfile.create_dataset('thal_spikerate', data=ZERO_ARR)
        masterfile.create_dataset('bkg_spikerate', data=ZERO_ARR)

def create_spikes_h5(outdir):
    with h5py.File(os.path.join(outdir, 'spikes.h5'), 'w') as outfile:
        outfile.create_dataset('spikes/timestamps', data=ZERO_ARR)
        outfile.create_dataset('spikes/gids', data=ZERO_ARR)

def create_seg_coords(outdir, m_type, e_type):
    with h5py.File(os.path.join(outdir, 'seg_coords', '0.h5'), 'w') as outfile:
        model = get_model('BBP', log, m_type=m_type, e_type=e_type)
        morph = Morphology(model.entire_cell)
        coords = morph.calc_seg_coords()
        ei = 'e' if e_type == 'cADpyr' else 'i'
        layer = int(m_type[1]) # L5_TTPC1, or L23_DBC

        # Find section name for each segment
        secmap = {'soma': 0, 'dend': 1, 'apic': 2, 'basal': 3, 'axon': 4}
        part_name, part_idx = [], []
        for sec in model.entire_cell.all:
            nseg = sec.nseg
            part_idx.extend(
                [int(sec.name().split('.')[-1].split('[')[1][:-1])] * nseg
            )
            part_name.extend(
                [secmap[sec.name().split('.')[-1].split('[')[0]]] * nseg
            )

        outfile.create_dataset('p0', data=coords['p0'])
        outfile.create_dataset('p05', data=coords['p05'])
        outfile.create_dataset('p1', data=coords['p1'])
        
        outfile.create_dataset('d0', data=coords['d0'])
        outfile.create_dataset('d1', data=coords['d1'])
        outfile.create_dataset('ei', data=ei)
        outfile.create_dataset('part_name', data=part_name)
        outfile.create_dataset('part_idx', data=part_idx)
        outfile.create_dataset('m_type', data=m_type)
        outfile.create_dataset('e_type', data=e_type)
        outfile.create_dataset('layer', data=layer)
        outfile.create_dataset('soma_pos', data=morph.get_soma_pos())

        return len(part_name)
        

def create_im_h5(outdir, nsegs):
    with h5py.File(os.path.join(outdir, 'im.h5'), 'w') as outfile:
        arr = np.atleast_2d(np.array([0]*nsegs, dtype=np.float32))
        outfile.create_dataset('/im/data', data=arr)
        outfile.create_dataset('/v/data', data=arr)
        outfile.create_dataset('/mapping/gids', data=[0])
        outfile.create_dataset('/mapping/time/', data=[0.0, 0.1, 0.1])
        outfile.create_dataset('/mapping/element_id', data=range(nsegs))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--m-type', type=str, required=True)
    parser.add_argument('--e-type', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--force', action='store_true', required=False, default=False)
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        if args.force:
            shutil.rmtree(args.outdir)
        else:
            log.error("Path already exists: {}".format(args.outdir))
    os.mkdir(args.outdir)
    os.mkdir(os.path.join(args.outdir, 'seg_coords'))

    create_master_file(args.outdir)
    create_spikes_h5(args.outdir)
    nsegs = create_seg_coords(args.outdir, args.m_type, args.e_type)
    create_im_h5(args.outdir, nsegs)
