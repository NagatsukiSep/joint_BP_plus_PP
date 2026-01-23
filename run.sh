P=0.04
TRIALS=1000
CUDA_CHECK_WARMUP=0
CUDA_CHECK_INTERVAL=1
COSTS_OUT=0

usage() {
  echo "Usage: $0 [--p value] [--warmup value] [--interval value] [--costs-out] [--costs-file path]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --p)
      P="${2:-}"
      shift 2
      ;;
    --warmup)
      CUDA_CHECK_WARMUP="${2:-}"
      shift 2
      ;;
    --interval)
      CUDA_CHECK_INTERVAL="${2:-}"
      shift 2
      ;;
    --costs-out)
      COSTS_OUT=1
      shift
      ;;
    --costs-file)
      COSTS_FILE="${2:-}"
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

if [ "$COSTS_OUT" -eq 1 ]; then
  CUDA_CHECK_WARMUP=0
  CUDA_CHECK_INTERVAL=1
fi

if [ "$COSTS_OUT" -eq 1 ]; then
  costs_path="${COSTS_FILE:-data/p${P}_costs.txt}"
  case "$costs_path" in
    data/*) mkdir -p data ;;
  esac
fi

CMD=(./jointbp_ets --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt --simulate --p "$P" --trials "$TRIALS" --max-iter 200 --no-pp --cuda --cuda-device 0 --cuda-graph --cuda-check-interval "$CUDA_CHECK_INTERVAL" --seed 1234 --report-every 1000 --cuda-check-warmup "$CUDA_CHECK_WARMUP")
if [ "$COSTS_OUT" -eq 1 ]; then
  if [ -n "${COSTS_FILE:-}" ]; then
    CMD+=(--costs-out "$COSTS_FILE")
  else
    CMD+=(--costs-out "data/p${P}_costs.txt")
  fi
fi

"${CMD[@]}"
