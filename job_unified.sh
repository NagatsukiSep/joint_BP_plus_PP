#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N unified_job
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

echo "Running single precision benchmark"
./build.sh
./jointbp_ets \
    --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt --simulate --p 0.04 --max-iter 200 --seed 106 --no-pp --cuda --cuda-device 0 --cuda-graph \
    --cuda-check-warmup 0 \
    --cuda-check-interval 1 \
    --trials 50000 \
    --report-every 50000

echo "===== JOB END ====="
date
