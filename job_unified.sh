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

# --- GPU/CPU monitor (start) ---
MON_PIDS=()

cleanup() {
  echo "===== CLEANUP ====="
  for pid in "${MON_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "===== GPU INFO ====="
nvidia-smi -L || true
nvidia-smi || true

# 1) GPU utilization log (1Hz)
#   - device index は "見えている世界" の 0,1,...（CUDA_VISIBLE_DEVICES により絞られていれば通常 0 のみ）
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem \
  --format=csv -l 1 > "logs/gpu.${JOB_ID}.csv" &
MON_PIDS+=("$!")

# 2) (任意) プロセス別の GPU 使用状況（軽量）
#    pmon は環境により使えない/項目が違う場合あり。動かなければ外してください。
nvidia-smi pmon -s um -c 0 > "logs/gpu_pmon.${JOB_ID}.log" 2>/dev/null &
MON_PIDS+=("$!")

# 3) (任意) CPU 側も 1Hz で見る（GPU が低稼働のときの原因切り分け用）
mpstat -P ALL 1 > "logs/cpu.${JOB_ID}.log" &
MON_PIDS+=("$!")
# --- GPU/CPU monitor (end) ---

# build
./build.sh

# run experiments
/usr/bin/time -v ./jointbp_ets \
  --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt --simulate --p 0.04 --max-iter 200 --seed 106 --no-pp --cuda --cuda-device 0 --cuda-graph \
  --cuda-check-warmup 0 \
  --cuda-check-interval 1 \
  --trials 10000 \
  --report-every 10000 \
  2> "logs/time.${JOB_ID}.txt"

echo "===== JOB END ====="
date