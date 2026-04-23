"""
simulate_and_store.py
---------------------
For each neuron defined in CELLS_CSV, build a NEURON model, run one simulation
per parameter set ("probe"), and save every voltage trace into a per-neuron
HDF5 file.

Parameter-set sources (applied in priority order)
--------------------------------------------------
Option A  BASE_CSV    — whitespace-separated matrix, no header.
                        Rows = probes, columns = raw parameter values.
                        Set BASE_CSV = None to skip.

Option B  NAMED_CSVS  — list of (label, path) tuples.
                        Each CSV must have columns 'Parameters'/'Prameters'
                        and 'Values'.  One probe is created per entry.

Option C  AUTO_GENERATE — if neither A nor B produces any rows, generate
                          N_AUTO_PROBES probes by log-uniformly scaling
                          the model's DEFAULT_PARAMS.  The first auto-probe
                          is always the unscaled defaults (scale = 1).

HDF5 layout  (one file per neuron)
-----------------------------------
  <mType>_<eType>_<iType>.h5
  ├── param_names          str array  (N_params,)
  ├── attrs['meta']        JSON string with neuron info
  └── probe_<i>/           group per probe
        ├── attrs['label']        human-readable name
        ├── attrs['probe_index']  integer index
        ├── params               float64 array  (N_params,)
        └── volts                float32 array  (T - STIM_SKIP,)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import json
import logging as log
import numpy as np
import pandas as pd
import h5py
from datetime import datetime

# ── make DL4neurons2 importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run import get_model

# =============================================================================
# Configuration — edit these paths / constants
# =============================================================================
CELLS_CSV = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/testcell.csv'
CELLS_CSV = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhFirst.csv'
CELLS_CSV = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhNew.csv'

# ── Option A: matrix CSV (whitespace-separated, no header) ───────────────────
BASE_CSV = None   # e.g. '/path/to/BaseTest.csv'  or  None to skip

# ── Option B: named single-param CSVs (Parameters + Values columns) ──────────
NAMED_CSVS = [
    ('MeanParams',
     '/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/InhibitoryMeanParams0.csv'),
    ('MeanParams_x0.1',
     '/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/InhibitoryMeanParams0x0.1.csv'),
    ('MeanParams_x10',
     '/global/homes/k/ktub1999/mainDL4/DL4neurons2/sensitivity_analysis/NewBase2/InhibitoryMeanParams0x10.csv'),
]
NAMED_CSVS = []

# ── Option C: auto-generate from DEFAULT_PARAMS ───────────────────────────────
AUTO_GENERATE  = True    # used only when A and B yield no rows
N_AUTO_PROBES  = 10      # total probes to generate (includes the unscaled default)
AUTO_LOG_SCALE = 1.0     # spread: param * 10^(uniform(-s, +s))
AUTO_SEED      = 42      # RNG seed for reproducibility

# ── Simulation settings ───────────────────────────────────────────────────────
STIM_FILE  = '/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/5k50kInterChaoticB.csv'
OUTPUT_DIR = '/global/cfs/cdirs/m2043/roybens/sens_ana/model_sensitivity_analysis/SimOutputsInh'
ITYPE      = 1       # fallback iType when CSV has no iType column
DT         = 0.025   # simulation timestep (ms) — 0.1 causes skv.mod negative-state warnings for some inh cells
V_INIT     = -75.0   # initial membrane voltage (mV)
STIM_SKIP  = 1000    # leading samples to drop before storing

# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


@contextlib.contextmanager
def _suppress_stderr_fd():
    """Redirect file-descriptor 2 to /dev/null for the duration of the block.
    Silences C-level fprintf(stderr, ...) calls such as the StochKv.mod
    'skv.mod:strap: negative state' spam that bypasses Python's sys.stderr.
    """
    saved_fd   = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


# =============================================================================
# Command-line arguments
# =============================================================================
_parser = argparse.ArgumentParser(
    description='Simulate neuron models and store voltage traces to HDF5.',
)
_grp = _parser.add_mutually_exclusive_group()
_grp.add_argument(
    '--index', type=int, default=None,
    help='Simulate a single neuron by 0-based CSV row index.',
)
_grp.add_argument(
    '--start', type=int, default=None,
    help='Start of neuron range in CSV (0-based, inclusive).',
)
_parser.add_argument(
    '--end', type=int, default=None,
    help='End of neuron range (exclusive). Used with --start.',
)
_parser.add_argument(
    '--o_cell', type=str, default='csv',
    help=(
        'iType selection: '
        '"csv" = read iType column from CSV (fallback: ITYPE constant); '
        '"range" = iterate iType 0..5 for each neuron; '
        'integer = use that fixed iType value for every neuron.'
    ),
)
_args = _parser.parse_args()
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_named_csv(path: str) -> np.ndarray:
    """Read a CSV with columns 'Parameters'/'Prameters' + 'Values'.
    Returns a 1-D float64 array of values in file order."""
    df = pd.read_csv(path)
    for col in ('Values', 'values'):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float64)
    raise ValueError(f"No 'Values' column found in {path}")


def load_csv_probes(
    cells_df: pd.DataFrame,
) -> tuple[list[np.ndarray], list[str], dict[str, np.ndarray]]:
    """
    Load probe rows and, when no CSV sources are configured, pre-load the
    DEFAULT_PARAMS for every neuron in *cells_df* as a per-neuron fallback.

    Returns
    -------
    rows                : list of 1-D float64 arrays — shared probes from CSVs
                          (empty when no CSVs are configured)
    labels              : list of str, one label per shared probe
    per_neuron_defaults : dict  neuron_label -> 1-D float64 DEFAULT_PARAMS
                          Populated only when *rows* is empty (no CSV probes);
                          the main loop uses this for the "defaults-only" mode.
    """
    rows:               list[np.ndarray]      = []
    labels:             list[str]             = []
    per_neuron_defaults: dict[str, np.ndarray] = {}

    # ── Option A: matrix CSV ─────────────────────────────────────────────────
    if BASE_CSV is not None and os.path.exists(BASE_CSV):
        mat = np.genfromtxt(BASE_CSV, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        for i, r in enumerate(mat):
            rows.append(r.copy())
            labels.append(f'BaseCsv_probe{i + 1}')
        print(f'[probes] Loaded {len(mat)} probe(s) from BASE_CSV ({BASE_CSV})')
    elif BASE_CSV is not None:
        print(f'[probes] BASE_CSV not found: {BASE_CSV}')

    # ── Option B: named CSVs ──────────────────────────────────────────────────
    for name, path in NAMED_CSVS:
        if not os.path.exists(path):
            print(f'[probes] ✗ Named CSV not found: {path}')
            continue
        try:
            r = load_named_csv(path)
            rows.append(r)
            labels.append(name)
            print(f'[probes] ✓ Named CSV "{name}": {len(r)} parameters')
        except Exception as exc:
            print(f'[probes] ✗ Could not load "{name}": {exc}')

    # ── Fallback: collect DEFAULT_PARAMS from every neuron as shared probes ──
    # Each neuron's unscaled defaults becomes one probe row, shared across
    # all neurons in the simulation loop (one parameter set per neuron type).
    if not rows:
        print('[probes] No CSV probes configured — loading DEFAULT_PARAMS '
              'from all neurons in CELLS_CSV as shared probes ...')
        for _, cell_row in cells_df.iterrows():
            mtype = str(cell_row['mType'])
            etype = str(cell_row['eType'])
            itype = int(cell_row['iType']) if 'iType' in cell_row.index else ITYPE
            nlabel = f'{mtype}_{etype}_{itype}'
            try:
                tmp = get_model('BBP', log, mtype, etype, itype)
                defaults = np.array(tmp.DEFAULT_PARAMS, dtype=np.float64)
                rows.append(defaults)
                labels.append(f'Default_{nlabel}')
                print(f'  ✓ {nlabel}: {len(defaults)} params')
            except Exception as exc:
                print(f'  ✗ {nlabel}: could not load model — {exc}')
        print(f'[probes] Built {len(rows)} shared probe(s) from DEFAULT_PARAMS.\n')

    return rows, labels, per_neuron_defaults


def autogenerate_probes(
    default_params: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Option C: generate N_AUTO_PROBES probes by log-uniformly scaling
    the model's DEFAULT_PARAMS.  Probe 0 is always the unscaled defaults.
    Called per-neuron only when no global CSV rows are available.
    """
    n_params = len(default_params)
    print(f'  [auto] Generating {N_AUTO_PROBES} probes from DEFAULT_PARAMS '
          f'(seed={AUTO_SEED}, log_scale=±{AUTO_LOG_SCALE})')
    rows:   list[np.ndarray] = [default_params.copy()]
    labels: list[str]        = ['AutoProbe_default']
    exponents = rng.uniform(
        -AUTO_LOG_SCALE, AUTO_LOG_SCALE,
        size=(N_AUTO_PROBES - 1, n_params),
    )
    for k in range(N_AUTO_PROBES - 1):
        rows.append(default_params * np.power(10.0, exponents[k]))
        labels.append(f'AutoProbe_{k + 2}')
    return rows, labels


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

cells_df = pd.read_csv(CELLS_CSV)
print(f'Loaded {len(cells_df)} neurons from {CELLS_CSV}')

# ── Apply neuron-range CLI filter ────────────────────────────────────────────
if _args.index is not None:
    _row_start, _row_end = _args.index, _args.index + 1
elif _args.start is not None:
    _row_start = _args.start
    _row_end   = _args.end if _args.end is not None else len(cells_df)
else:
    _row_start, _row_end = 0, len(cells_df)

cells_df = cells_df.iloc[_row_start:_row_end].reset_index(drop=True)
print(f'Simulating neurons [{_row_start}:{_row_end}] → {len(cells_df)} row(s)\n')

# ── Resolve --o_cell into an iType list (None = per-neuron from CSV) ─────────
if _args.o_cell == 'csv':
    _itype_list: list[int] | None = None        # determined per-neuron in the loop
elif _args.o_cell == 'range':
    _itype_list = list(range(6))                 # iType 0, 1, 2, 3, 4, 5
else:
    _itype_list = [int(_args.o_cell)]            # single fixed iType

stim_arr = np.genfromtxt(STIM_FILE, dtype=np.float32)
rng      = np.random.default_rng(AUTO_SEED)

# ── Load CSV probes once (shared across all neurons) ─────────────────────────
GLOBAL_PROBE_ROWS, GLOBAL_PROBE_LABELS, NEURON_DEFAULT_PARAMS = \
    load_csv_probes(cells_df)

if GLOBAL_PROBE_ROWS:
    print(f'[probes] {len(GLOBAL_PROBE_ROWS)} probe(s) loaded globally '
          f'— will be reused for every neuron.\n')
elif AUTO_GENERATE:
    print(f'[probes] No CSV or default probes — will auto-generate '
          f'{N_AUTO_PROBES} randomised probe(s) per neuron.\n')
else:
    raise RuntimeError(
        'No probe rows loaded and AUTO_GENERATE is False. '
        'Set BASE_CSV, add entries to NAMED_CSVS, or enable AUTO_GENERATE.'
    )

for idx, cell_row in cells_df.iterrows():
    mtype = str(cell_row['mType'])
    etype = str(cell_row['eType'])

    # Determine which iType values to simulate for this neuron
    if _itype_list is not None:
        itypes_to_run = _itype_list
    else:
        itypes_to_run = [int(cell_row['iType']) if 'iType' in cell_row.index else ITYPE]

    for itype in itypes_to_run:
        neuron_label = f'{mtype}_{etype}_{itype}'
        h5_path      = os.path.join(OUTPUT_DIR, f'{neuron_label}.h5')

        print(f'\n[{idx + 1}/{len(cells_df)}]  {mtype}  {etype}  iType={itype}')
        print(f'  → {h5_path}')

        # ── build NEURON model ────────────────────────────────────────────────
        try:
            model = get_model('BBP', log, mtype, etype, itype)
            model.set_attachments(stim_arr, len(stim_arr), DT)
        except Exception as exc:
            print(f'  ✗ Could not build model: {exc}')
            continue

        # ── select probe rows ────────────────────────────────────────────────
        if GLOBAL_PROBE_ROWS:
            # shared probes (from CSVs or all-neurons' defaults) — same for every neuron
            probe_rows   = GLOBAL_PROBE_ROWS
            probe_labels = GLOBAL_PROBE_LABELS
        elif AUTO_GENERATE:
            # randomised probes around this neuron's DEFAULT_PARAMS
            probe_rows, probe_labels = autogenerate_probes(
                np.array(model.DEFAULT_PARAMS, dtype=np.float64), rng
            )
        else:
            print('  ✗ No probe rows available — skipping neuron')
            continue

        n_probes = len(probe_rows)
        print(f'  Running {n_probes} probe(s) ...')

        # ── write HDF5 ────────────────────────────────────────────────────────
        with h5py.File(h5_path, 'w') as hf:

            # parameter names as fixed-length byte strings
            hf.create_dataset(
                'param_names',
                data=np.array(model.PARAM_NAMES, dtype='S'),
                dtype=h5py.string_dtype(),
            )

            # file-level metadata
            meta = {
                'mType':     mtype,
                'eType':     etype,
                'iType':     itype,
                'stim_file': STIM_FILE,
                'dt':        DT,
                'v_init':    V_INIT,
                'stim_skip': STIM_SKIP,
                'n_probes':  n_probes,
                'created':   datetime.now().isoformat(),
            }
            hf.attrs['meta'] = json.dumps(meta)

            for p_idx, (params, probe_label) in enumerate(
                    zip(probe_rows, probe_labels)):
                now = datetime.now().strftime('%H:%M:%S')
                print(f'    probe {p_idx + 1}/{n_probes}  [{probe_label}]  [{now}]')

                try:
                    model._set_self_params(*params)
                    model.init_parameters()
                    with _suppress_stderr_fd():
                        volts_dict = model.simulate(stim_arr, DT, V_INIT)
                    # use the first recorded section (soma)
                    first_key   = list(volts_dict.keys())[0]
                    volts_full  = np.array(volts_dict[first_key], dtype=np.float32)
                    volts_store = volts_full[STIM_SKIP:]
                except Exception as exc:
                    print(f'    ✗ Simulation failed: {exc}')
                    continue

                grp = hf.create_group(f'probe_{p_idx}')
                grp.attrs['label']       = probe_label
                grp.attrs['probe_index'] = p_idx
                grp.create_dataset('params', data=np.array(params, dtype=np.float64))
                grp.create_dataset('volts',  data=volts_store,
                                   compression='gzip', compression_opts=4)

        print(f'  ✓ Saved {h5_path}')

print('\nDone.')
