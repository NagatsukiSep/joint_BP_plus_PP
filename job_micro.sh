#!/usr/bin/env bash
set -euo pipefail

PARAMS="H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt"
P="0.04"
SEED="106"
CUDA_DEVICE="0"
MB_ITERS="100000"
MODE="both"

usage() {
  echo "Usage: $0 [--p value] [--seed value] [--device N] [--iters N] [--mode iter|check|both]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --p)
      P="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --device)
      CUDA_DEVICE="${2:-}"
      shift 2
      ;;
    --iters)
      MB_ITERS="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

common_args=(
  --params "$PARAMS"
  --simulate
  --p "$P"
  --seed "$SEED"
  --no-pp
  --cuda
  --cuda-device "$CUDA_DEVICE"
  --cuda-check-warmup 0
  --cuda-check-interval 1
  --cuda-microbench-mode "$MODE"
  --cuda-microbench "$MB_ITERS"
)

echo "[microbench] graph=1"
./jointbp_ets "${common_args[@]}" --cuda-graph
echo
echo "[microbench] graph=0"
./jointbp_ets "${common_args[@]}"
