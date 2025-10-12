#!/bin/bash -l

#SBATCH --account=y95-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --time=4:00:00
#SBATCH --job-name=esmfold

set -euo pipefail

# Load required module
module load singularity/3.11.4-nompi

# ---- CONFIGURATION ----

# Input file
INPUT="/scratch/pawsey1142/kgagalova/test-esmfold/test3/rd21a.fasta"

# Result directory
RESULTS_DIR="${MYSCRATCH}/test-esmfold/test3/results/${SLURM_JOB_ID}/"
mkdir -p "${RESULTS_DIR}"

# Reference directory (unused by esm-fold here, but kept for completeness)
REF_DIR="/scratch/references/alphafold_feb2024/databases"

# Container image (ROCm build)
CONTAINER="docker://quay.io/pawsey/esmfold_openfold:rocm6.3.3"

# Transformer cache (Hugging Face / Torch)
CACHE_DIR="${MYSCRATCH}/test-esmfold/esmfold_cache"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}/torch"
mkdir -p "${TORCH_HOME}"

# ---- SINGULARITY CACHE ----
export SINGULARITY_CACHEDIR="/scratch/pawsey1142/kgagalova/test-esmfold"
export SINGULARITY_TMPDIR="/scratch/pawsey1142/kgagalova/test-esmfold/tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

echo "Running ESMFold..."
echo "Input: ${INPUT}"
echo "Results: ${RESULTS_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Singularity cache: ${SINGULARITY_CACHEDIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

# ---- ROCm SMI logging (every 5s) ----
ROCM_LOG="${RESULTS_DIR}/rocm_smi_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "${ROCM_LOG}")"

# Background sampler. Clean up on exit.
(
  while true; do
    echo "===== $(date -Is) ====="
    /opt/rocm/bin/rocm-smi --showuse --showmeminfo vram --showtemp || true
    sleep 5
  done
) > "${ROCM_LOG}" 2>&1 &
SMI_PID=$!
cleanup() { kill "${SMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ---- RUN ----
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec \
  -B "${CACHE_DIR}" \
  -B "${SINGULARITY_CACHEDIR}" \
  "${CONTAINER}" \
  esm-fold -i "${INPUT}" -o "${RESULTS_DIR}" --chunk-size 16
