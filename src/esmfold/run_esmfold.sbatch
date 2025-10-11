#!/bin/bash -l

#SBATCH --account=y95-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=esmfold

# Load required module
module load singularity/3.11.4-nompi

# ---- CONFIGURATION ----

# Input directory
INPUT="/scratch/pawsey1142/kgagalova/test-esmfold/test1"

# Result directory
RESULTS_DIR="${MYSCRATCH}/test-esmfold/test1/results/${SLURM_JOB_ID}/"
mkdir -p "${RESULTS_DIR}"

# Reference directory
REF_DIR="/scratch/references/alphafold_feb2024/databases"

# Container image (from Pawsey registry)
CONTAINER="docker://quay.io/pawsey/esmfold_openfold:rocm6.3.3"

# Transformer cache (for Hugging Face, PyTorch, etc.)
CACHE_DIR="${MYSCRATCH}/test-esmfold/esmfold_cache"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}/torch"
mkdir -p "${TORCH_HOME}"

# ---- SINGULARITY CACHE ----
# Set Singularity cache to /scratch/pawsey1142/kgagalova/test-esmfold
export SINGULARITY_CACHEDIR="/scratch/pawsey1142/kgagalova/test-esmfold"
export SINGULARITY_TMPDIR="/scratch/pawsey1142/kgagalova/test-esmfold/tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# ---- RUN ----
echo "Running ESMFold..."
echo "Input: ${INPUT}"
echo "Results: ${RESULTS_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Singularity cache: ${SINGULARITY_CACHEDIR}"

srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec \
  -B "${CACHE_DIR}" \
  -B "${SINGULARITY_CACHEDIR}" \
  "${CONTAINER}" \
  esm-fold -i "${INPUT}" -o "${RESULTS_DIR}" --chunk-size 16
