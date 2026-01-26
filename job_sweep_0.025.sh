#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N sweep_0.025
#$ -l h_rt=00:05:00
#$ -l gpu_1=1
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err

set -euo pipefail

mkdir -p logs

echo "===== JOB START ====="
date
hostname
echo "JOB_ID=$JOB_ID"
echo "NSLOTS=${NSLOTS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# load CUDA environment explicitly as it is required
source /etc/profile.d/modules.sh || true
module purge || true
module load cuda || true

# run experiments
./build.sh
./sweep_warmup_interval.sh --p 0.025 --seed 3 --max-warmup 8 --trials 50000 --out data/results_warmup_interval_025.tsv

echo "===== JOB END ====="
date
