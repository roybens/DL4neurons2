#!/bin/bash -l
# =============================================================================
# sensitivity_sbatch.sh
# ---------------------
# Submit with:
#   sbatch sensitivity_sbatch.sh [CELLS_CSV] [CHUNK_SIZE] [O_CELL]
#
# Arguments (positional, all optional):
#   CELLS_CSV   path to the neuron CSV                 (default: AllInhFirst.csv)
#   CHUNK_SIZE  neurons per srun call                  (default: 1)
#   O_CELL      iType mode: "csv" | "range" | integer  (default: "csv")
#
# Examples:
#   sbatch sensitivity_sbatch.sh
#   sbatch sensitivity_sbatch.sh AllInhFirst.csv 4 range
#   sbatch sensitivity_sbatch.sh AllInhFirst.csv 2 1
# =============================================================================

#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -q regular
#SBATCH -J sens_sim
#SBATCH -C cpu
#SBATCH --output logs/sens_%j.out
#SBATCH --error  logs/sens_%j.err

# ── Resolve paths and parameters ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CELLS_CSV="${1:-/global/homes/k/ktub1999/mainDL4/DL4neurons2/AllInhFirst.csv}"
CHUNK_SIZE="${2:-1}"
O_CELL="${3:-csv}"

# optional: export HDF5_USE_FILE_LOCKING=FALSE  # for Lustre filesystems
# optional: module load python / source venv/bin/activate

# ── Compute number of chunks ──────────────────────────────────────────────────
if [[ ! -f "$CELLS_CSV" ]]; then
    echo "ERROR: CELLS_CSV not found: $CELLS_CSV"; exit 1
fi

N_NEURONS=$(( $(wc -l < "$CELLS_CSV") - 1 ))   # subtract header row
N_CHUNKS=$(( (N_NEURONS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "================================================"
echo "  SLURM_JOB_ID : ${SLURM_JOB_ID}"
echo "  NODE         : $(hostname)"
echo "  CELLS_CSV    : ${CELLS_CSV}"
echo "  N_NEURONS    : ${N_NEURONS}"
echo "  CHUNK_SIZE   : ${CHUNK_SIZE}  →  ${N_CHUNKS} srun call(s)"
echo "  O_CELL       : ${O_CELL}"
echo "  DATE         : $(date)"
echo "================================================"

# ── Run each chunk sequentially via srun ──────────────────────────────────────
for (( i=0; i<N_CHUNKS; i++ )); do
    START=$(( i * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE ))

    echo ""
    echo "--- chunk ${i}/${N_CHUNKS}  neurons [${START}:${END}) ---"

    
        python3 "${SCRIPT_DIR}/simulate_and_store.py" \
            --start  "${START}" \
            --end    "${END}"   \
            --o_cell "${O_CELL}"&
done
wait
echo ""
echo "================================================"
echo "  All chunks done at $(date)"
echo "================================================"

# ./model_sensitivity_analysis/sensitivity_sbatch.sh AllInhFirst.csv 10 range