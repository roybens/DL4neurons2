"""
compute_sensitivity_scores.py
------------------------------
Load per-neuron HDF5 files produced by simulate_and_store.py and compare
voltage traces for the **same probe index across different neurons**.

Comparison logic
----------------
  For each probe index present in the data:
    - Take the voltage trace of that probe from every neuron that has it.
    - Compare all neuron pairs (neuronA vs neuronB) using efel features.

Output
------
  sensitivity_scores.csv   — detailed table, one row per (probe, neuronA, neuronB)
  sensitivity_scores_pivot_weighted.csv — pivot: rows = probes, columns = neuron pairs,
                                          values = weighted_score

CSV columns (detailed)
----------------------
  probe_index     : integer index of the probe inside the HDF5
  probe_label     : human-readable probe label (from HDF5 attrs)
  neuron_A        : name of the first neuron  (<mType>_<eType>_<iType>)
  neuron_B        : name of the second neuron
  weighted_score  : sum of all per-feature RMSEs
  <feature_name>  : per-feature RMSE (one column per efel feature)
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import h5py
import efel

# ── make DL4neurons2 importable (for any shared utilities if needed) ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# Configuration — edit these paths / constants
# =============================================================================
INPUT_DIR   = '/global/cfs/cdirs/m2043/roybens/sens_ana/model_sensitivity_analysis/SimOutputsExc65'
OUTPUT_DIR  = '/global/cfs/cdirs/m2043/roybens/sens_ana/model_sensitivity_analysis/SimOutputsExc65'
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, 'sensitivity_scores.csv')
PIVOT_CSV   = os.path.join(OUTPUT_DIR, 'sensitivity_scores_pivot_weighted.csv')

DT = 0.1  # ms — must match the value used during simulation

FEATURES_TO_COMPUTE = [
    'mean_frequency',
    'AP_amplitude',
    'AHP_depth_abs_slow',
    'fast_AHP_change',
    'AHP_slow_time',
    'spike_half_width',
    'time_to_first_spike',
    'inv_first_ISI',
    'ISI_CV',
    'ISI_values',
    'adaptation_index',
]

# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Feature / score helpers  (mirrors analyze_sensitivity_using_scores.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(voltage_trace: np.ndarray, dt: float = DT) -> dict:
    """Compute efel features for a 1-D voltage trace (mV)."""
    n = len(voltage_trace)
    trace = {
        'T':          [x * dt for x in range(n)],
        'V':          list(voltage_trace.astype(float)),
        'stim_start': [0.0],
        'stim_end':   [n * dt],
    }
    return efel.get_feature_values([trace], FEATURES_TO_COMPUTE)[0]


def safe_mean(arr) -> float:
    if arr is None or np.size(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def weighted_score_compute(feature1: dict, feature2: dict):
    """
    Compute per-feature RMSE and total weighted score.
    Missing features → zero; arrays of different lengths → zero-padded.
    """
    all_names = list({*feature1.keys(), *feature2.keys()})
    score_dict: dict[str, float] = {}

    for fname in all_names:
        v1 = feature1.get(fname)
        v2 = feature2.get(fname)
        if v1 is None:
            v1 = np.array([0.0])
        if v2 is None:
            v2 = np.array([0.0])
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)

        diff = len(v1) - len(v2)
        if diff > 0:
            v2 = np.concatenate([v2, np.zeros(diff)])
        elif diff < 0:
            v1 = np.concatenate([v1, np.zeros(-diff)])

        score_dict[fname] = float(np.sqrt(safe_mean((v1 - v2) ** 2)))

    return sum(score_dict.values()), score_dict


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — load all H5 files into memory
# Structure: all_data[neuron_key][probe_idx] = {'label': str, 'volts': ndarray}
# ─────────────────────────────────────────────────────────────────────────────

h5_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.h5')])
if not h5_files:
    raise FileNotFoundError(f'No .h5 files found in {INPUT_DIR}')

print(f'Found {len(h5_files)} HDF5 file(s) in {INPUT_DIR}\n')

all_data: dict[str, dict[int, dict]] = {}
dt_per_neuron:    dict[str, float] = {}
mtype_per_neuron: dict[str, str]   = {}   # neuron_key -> mType string

for fname in h5_files:
    h5_path    = os.path.join(INPUT_DIR, fname)
    neuron_key = os.path.splitext(fname)[0]

    with h5py.File(h5_path, 'r') as hf:
        meta = {}
        if 'meta' in hf.attrs:
            try:
                meta = json.loads(hf.attrs['meta'])
            except Exception:
                pass
        dt_per_neuron[neuron_key]    = float(meta.get('dt', DT))
        mtype_per_neuron[neuron_key] = str(meta.get('mType', ''))  # e.g. 'L6_TPC_L1'

        probe_keys = sorted(
            [k for k in hf.keys() if k.startswith('probe_')],
            key=lambda s: int(s.split('_')[1]),
        )
        probes: dict[int, dict] = {}
        for pk in probe_keys:
            p_idx = int(pk.split('_')[1])
            probes[p_idx] = {
                'label': str(hf[pk].attrs.get('label', pk)),
                'volts': hf[pk]['volts'][:].astype(np.float64),
            }

    all_data[neuron_key] = probes
    print(f'  Loaded {neuron_key}: {len(probes)} probe(s)')

neuron_keys = sorted(all_data.keys())
n_neurons   = len(neuron_keys)
print(f'\nTotal neurons loaded: {n_neurons}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — find all probe indices present in at least 2 neurons
# ─────────────────────────────────────────────────────────────────────────────

from collections import defaultdict
probe_to_neurons: dict[int, list[str]] = defaultdict(list)
probe_labels_map: dict[int, str] = {}

for neuron_key, probes in all_data.items():
    for p_idx, info in probes.items():
        probe_to_neurons[p_idx].append(neuron_key)
        probe_labels_map[p_idx] = info['label']   # last-writer wins; labels should be consistent

valid_probes = sorted(
    p for p, neurons in probe_to_neurons.items() if len(neurons) >= 2
)
print(f'Probe indices with ≥2 neurons: {valid_probes}\n')

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — for each probe, compare every neuron pair
# ─────────────────────────────────────────────────────────────────────────────

records: list[dict] = []

for p_idx in valid_probes:
    probe_label   = probe_labels_map[p_idx]
    neurons_here  = probe_to_neurons[p_idx]   # neurons that have this probe
    n_here        = len(neurons_here)
    # cross-mType pairs only
    n_pairs_cross = sum(
        1 for ai in range(n_here) for bi in range(ai + 1, n_here)
        if mtype_per_neuron.get(neurons_here[ai]) != mtype_per_neuron.get(neurons_here[bi])
    )
    n_pairs_skip  = n_here * (n_here - 1) // 2 - n_pairs_cross
    print(f'Probe {p_idx} [{probe_label}]  —  {n_here} neurons, '
          f'{n_pairs_cross} cross-mType pair(s)  ({n_pairs_skip} same-mType skipped)')

    for ai in range(n_here):
        for bi in range(ai + 1, n_here):
            nA = neurons_here[ai]
            nB = neurons_here[bi]

            # Skip pairs from the same mType (e.g. two variants of L6_TPC_L1)
            if mtype_per_neuron.get(nA) and mtype_per_neuron[nA] == mtype_per_neuron[nB]:
                print(f'  [skip same mType={mtype_per_neuron[nA]}]  {nA}  vs  {nB}')
                continue

            volts_A = all_data[nA][p_idx]['volts']
            volts_B = all_data[nB][p_idx]['volts']
            dt_A    = dt_per_neuron[nA]
            dt_B    = dt_per_neuron[nB]
            # use the smaller dt for both (they should match, but be safe)
            dt_use  = min(dt_A, dt_B)

            try:
                feat_A = compute_features(volts_A, dt=dt_use)
                feat_B = compute_features(volts_B, dt=dt_use)
                w_score, score_dict = weighted_score_compute(feat_A, feat_B)
            except Exception as exc:
                print(f'  ✗ Feature error {nA} vs {nB}: {exc}')
                score_dict = {f: np.nan for f in FEATURES_TO_COMPUTE}
                w_score    = np.nan

            rec = {
                'probe_index':    p_idx,
                'probe_label':    probe_label,
                'neuron_A':       nA,
                'neuron_B':       nB,
                'weighted_score': w_score,
            }
            rec.update(score_dict)
            records.append(rec)

            print(f'  [{nA}] vs [{nB}]  weighted_score = {w_score:.4f}')

    print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — write outputs
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not records:
    print('✗ No records to write.')
else:
    out_df = pd.DataFrame(records)

    # fixed columns first, feature columns sorted after
    fixed_cols = ['probe_index', 'probe_label', 'neuron_A', 'neuron_B', 'weighted_score']
    feat_cols  = sorted([c for c in out_df.columns if c not in fixed_cols])
    out_df     = out_df[fixed_cols + feat_cols]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f'✓ Detailed scores saved to  {OUTPUT_CSV}')
    print(f'  {len(out_df)} rows  ×  {len(out_df.columns)} columns')

    # ── pivot: rows = probes, columns = "neuronA_vs_neuronB" ─────────────────
    out_df['neuron_pair'] = out_df['neuron_A'] + '_vs_' + out_df['neuron_B']
    pivot = out_df.pivot_table(
        index=['probe_index', 'probe_label'],
        columns='neuron_pair',
        values='weighted_score',
    )
    pivot.to_csv(PIVOT_CSV)
    print(f'✓ Pivot (weighted_score) saved to  {PIVOT_CSV}')
    print(f'  {pivot.shape[0]} probe row(s)  ×  {pivot.shape[1]} neuron-pair column(s)')

print('\nDone.')
