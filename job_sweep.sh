#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N sweep_warmup_interval
#$ -g {your_group_here}
#$ -l h_rt=02:00:00
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

# CUDA 環境が必要なので明示的にロード（必要に応じて modulepath を調整）
source /etc/profile.d/modules.sh || true
module purge || true
module load cuda || true

# 実行（例：あなたの実行コマンドに置き換え）
./sweep_warmup_interval.sh --max-warmup 14 --trials 50000 --out data/results_warmup_interval.tsv

echo "===== JOB END ====="
date
