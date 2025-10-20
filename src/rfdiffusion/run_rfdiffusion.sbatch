#!/bin/bash -l

#SBATCH --job-name=rfdiffusion
#SBATCH -A y95-gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

# Load required modules
module load singularity/3.11.4-nompi

#Set up container variable
#ContainerImage=docker://quay.io/pawsey/rfdiffusion:rocm6.4
# local container
ContainerImage=/scratch/pawsey1142/kgagalova/test-rfdiffusion/test1/rfdiffusion_rocm6.4.sif

# Set up working directory and output directory.
MYSCRATCH=/scratch/pawsey1142/kgagalova
WORKDIR=${MYSCRATCH}/test-rfdiffusion/test1/rfdiffusion_runs/${SLURM_JOB_ID}
mkdir -p ${WORKDIR}
cd ${WORKDIR}
OUTDIR=${WORKDIR}/outputs

#Create schedules dir for RFDiffusion to use
mkdir ${WORKDIR}/schedules

# before srun, set cache dirs to scratch
export SINGULARITY_CACHEDIR=${MYSCRATCH}/.singularity/cache
export SINGULARITY_TMPDIR=${MYSCRATCH}/.singularity/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$WORKDIR/home" "$WORKDIR/schedules"

srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec \
  --cleanenv \
  --containall \
  -B schedules:/app/RFdiffusion/rfdiffusion/inference/../../schedules \
  ${ContainerImage} \
  run_inference.py \
    inference.model_directory_path=/app/RFdiffusion/models \
    inference.output_prefix=${OUTDIR} \
    'contigmap.contigs=[100-100]' \
    inference.num_designs=2
