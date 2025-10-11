#!/bin/bash -l

#SBATCH --account=y95-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=boltz_prediction

# Load required module
module load singularity/3.11.4-nompi

# ---- SINGULARITY CACHE ----
# Set Singularity cache to /scratch/pawsey1142/kgagalova/test-boltz
export SINGULARITY_CACHEDIR="/scratch/pawsey1142/kgagalova/test-boltz"
export SINGULARITY_TMPDIR="/scratch/pawsey1142/kgagalova/test-boltz/tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# Set container
containerImage="docker://quay.io/pawsey/boltz2:v2.0.3"
# Set input directory
INPUTDIR="/scratch/pawsey1142/kgagalova/test-boltz/test1/"
# Set output directory
OUTDIR="/scratch/pawsey1142/kgagalova/test-boltz/test1/results/${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"
# Set cache directory
CACHEDIR="/scratch/pawsey1142/kgagalova/test-boltz/test1/cache"
mkdir -p "${CACHEDIR}"
# Set numba cache dir
export numba_cache_dir="/scratch/pawsey1142/kgagalova/test-boltz/test1/numba_cache_dir/${SLURM_JOB_ID}"
mkdir -p "${numba_cache_dir}"
# Run Boltz prediction with bind mounting for the directories we need
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
    singularity exec \
    -B "${numba_cache_dir}" \
    -B "${INPUTDIR}" \
    -B "${OUTDIR}" \
    -B "${CACHEDIR}" \
    -B "${CACHEDIR}:/usr/local/lib/python3.12/dist-packages/trifast/configs/" \
    "${containerImage}" boltz predict \
    "${INPUTDIR}/iaa_aux.fasta" \
    --cache "${CACHEDIR}" \
    --use_msa_server \
    --out_dir "${OUTDIR}"
