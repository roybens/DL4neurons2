"""
Script to generate the m-type or e-type from the slurm array task id for the full production run
"""

import os, sys
import json

with open('cells.json') as infile:
    cells = json.load(infile)

CELL_I = 1 # of the 5 BBP clones

part_i = int(sys.argv[1]) # which cell in the current slurm job are we running?
total_parts = int(os.environ['CELLS_PER_JOB']) # how many cells per slurm job?

# Flatten cells.json
all_celldata = []
for m_type in sorted(cells.keys()):
    for e_type in sorted(cells[m_type].keys()):
        bbp_name = cells[m_type][e_type][0]['model_directory']
        all_celldata.append((m_type, e_type, bbp_name))

# Remove cell data for the ones done in the 10% prod run
done_cells = [
    'L5_TTPC1_cADpyr',
    'L1_DAC_bNAC',
    'L1_HAC_bNAC',
    'L1_NGC-SA_cNAC',
    'L23_BP_bAC',
    'L23_ChC_cAC',
    'L23_LBC_bAC',
    'L23_NBC_bAC',
    'L23_PC_cADpyr',
    'L4_BP_bAC',
    'L4_ChC_cAC',
    'L4_LBC_cAC',
    'L4_NBC_cAC',
    'L4_PC_cADpyr',
    'L4_SP_cADpyr',
    'L5_BP_bAC',
    'L5_ChC_cAC',
    'L5_LBC_bAC',
    'L5_NBC_bAC',
    'L5_SBC_bNAC',
    'L5_TTPC1_cADpyr',
    'L5_UTPC_cADpyr',
    'L6_BPC_cADpyr',
]
def already_done(bbp_name):
    for cellname in done_cells:
        if cellname in bbp_name:
            return True
    return False
all_celldata = [(m, e, bbp) for (m, e, bbp) in all_celldata if not already_done(bbp)]

# Select one
array_i = int(os.environ['SLURM_ARRAY_TASK_ID'])
i = array_i * total_parts + part_i
selected_m, selected_e, selected_bbp_name = all_celldata[i]

# Output
if '--m-type' in sys.argv:
    print(selected_m)

if '--e-type' in sys.argv:
    print(selected_e)

if '--bbp-name' in sys.argv:
    print(selected_bbp_name)

