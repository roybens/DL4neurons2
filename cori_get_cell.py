"""
Script to generate the m-type or e-type from the slurm array task id for the 10% run
"""

import os, sys
import json

with open('cells.json') as infile:
    cells = json.load(infile)

all_m_types = sorted(cells.keys())

i = int(os.environ['SLURM_ARRAY_TASK_ID']) * 2

m_type = all_m_types[i]

CELL_I = 0

e_type = sorted(cells[m_type].keys())[CELL_I]

bbp_name = cells[m_type][e_type][CELL_I]['model_directory']

if '--m-type' in sys.argv:
    print(m_type)

if '--e-type' in sys.argv:
    print(e_type)

if '--bbp-name' in sys.argv:
    print(bbp_name)
