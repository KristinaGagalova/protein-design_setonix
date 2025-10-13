#!/bin/bash -l

#SBATCH --job-name=alphafold-multimer
#SBATCH -A ${PAWSEY_PROJECT}-gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2

# Load required module
module load singularity/3.11.4-nompi
REF_DIR='/scratch/references/alphafold_feb2024/databases'

ROCM_LOG="${MYSCRATCH}/alphafold2/logs/${SLURM_JOB_ID}_rocm_smi.log"
# Background sampler. Clean up on exit.
(
  while true; do
    echo "===== $(date -Is) ====="
    /opt/rocm/bin/rocm-smi --showuse --showmeminfo vram --showtemp || true
    sleep 1
  done
) > "${ROCM_LOG}" 2>&1 &
SMI_PID=$!
cleanup() { kill "${SMI_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# Run AlphaFold2
srun -N 1 -n 1 -c 16 --gres=gpu:2 \
  singularity exec docker://quay.io/pawsey/alphafold2-amd-gpu:v2.3.2_rocm6.2.4 \
  python /app/alphafold/run_alphafold.py \
  --fasta_paths=/path/to/fasta/protein.fasta \
  --model_preset=multimer \
  --use_gpu_relax=True \
  --benchmark=False \
  --mgnify_database_path=${REF_DIR}/mgnify/mgy_clusters_2022_05.fa \
  --pdb_seqres_database_path=${REF_DIR}/pdb_seqres/pdb_seqres.txt \
  --uniprot_database_path=${REF_DIR}/uniprot \
  --uniref90_database_path=${REF_DIR}/uniref90/uniref90.fasta \
  --data_dir=${REF_DIR} \
  --template_mmcif_dir=${REF_DIR}/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=${REF_DIR}/pdb_mmcif/obsolete.dat \
  --small_bfd_database_path=${REF_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --output_dir=${MYSCRATCH}/alphafold2/output/${SLURM_JOB_ID} \
  --max_template_date=2025-10-05 \
  --db_preset=reduced_dbs \
  --logtostderr \
  --hhsearch_binary_path=/opt/miniforge3/bin/hhsearch \
  --hhblits_binary_path=/opt/miniforge3/bin/hhblits
