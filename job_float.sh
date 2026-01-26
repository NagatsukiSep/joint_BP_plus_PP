#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N float_vs_double
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
# double
nvcc -O2 -std=c++17 -DUSE_CUDA -o jointbp_ets jointbp_ets.cpp jointbp_cuda.cu
./run.sh --p 0.04 --trials 50000 --report-every 50000 --check-warmup 0 --check-interval 1
# float
./build.sh
./run.sh --p 0.04 --trials 50000 --report-every 50000 --check-warmup 0 --check-interval 1

echo "===== JOB END ====="
date
