P=0.04
TRIALS=100000
CUDA_CHECK_WARMUP=0
CUDA_CHECK_INTERVAL=1
HIST_OUT=0
HIST_FILE=""

usage() {
  echo "Usage: $0 [--p value] [--warmup value] [--interval value] [--iter-hist-out] [--hist-file path]" >&2
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
    --iter-hist-out)
      HIST_OUT=1
      shift
      ;;
    --hist-file)
      HIST_FILE="${2:-}"
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

if [ "$HIST_OUT" -eq 1 ]; then
  CUDA_CHECK_WARMUP=0
  CUDA_CHECK_INTERVAL=1
  hist_path="${HIST_FILE:-data/p${P}_hist.txt}"
  case "$hist_path" in
    data/*) mkdir -p data ;;
  esac
fi

CMD=(./jointbp_ets --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt --simulate --p "$P" --trials "$TRIALS" --max-iter 200 --no-pp --cuda --cuda-device 0 --cuda-graph --cuda-check-interval "$CUDA_CHECK_INTERVAL" --seed 1234 --report-every 1000 --cuda-check-warmup "$CUDA_CHECK_WARMUP")
if [ "$HIST_OUT" -eq 1 ]; then
  CMD+=(--iter-hist-out "$hist_path")
fi

"${CMD[@]}"
