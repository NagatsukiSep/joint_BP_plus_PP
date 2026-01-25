P=0.04
TRIALS=50000
CUDA_CHECK_WARMUP=0
CUDA_CHECK_INTERVAL=1
HIST_OUT=0
HIST_FILE=""
PRE_RUNS=0
REPORT_EVERY=1000
REPORT_EVERY_SPECIFIED=0

usage() {
  echo "Usage: $0 [--p value] [--check-warmup value] [--interval value] [--trials value] [--pre-runs value] [--report-every value] [--iter-hist-out] [--hist-file path]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --p)
      P="${2:-}"
      shift 2
      ;;
    --check-warmup)
      CUDA_CHECK_WARMUP="${2:-}"
      shift 2
      ;;
    --interval)
      CUDA_CHECK_INTERVAL="${2:-}"
      shift 2
      ;;
    --trials)
      TRIALS="${2:-}"
      shift 2
      ;;
    --pre-runs)
      PRE_RUNS="${2:-}"
      shift 2
      ;;
    --report-every)
      REPORT_EVERY="${2:-}"
      REPORT_EVERY_SPECIFIED=1
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

BASE_CMD=(./jointbp_ets
  --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt
  --simulate
  --p "$P"
  --max-iter 200
  --no-pp
  --cuda
  --cuda-device 0
  --cuda-graph
  --cuda-check-interval "$CUDA_CHECK_INTERVAL"
  --cuda-check-warmup "$CUDA_CHECK_WARMUP"
)

if [ "$PRE_RUNS" -gt 0 ]; then
  echo "Pre-run: $PRE_RUNS trials (no-output)"
  WARMUP_CMD=("${BASE_CMD[@]}"
    --trials "$PRE_RUNS"
    --seed 106
    --report-every 0
  )
  if ! "${WARMUP_CMD[@]}" >/dev/null 2>&1; then
    echo "Pre-run failed" >&2
    exit 1
  fi
fi

CMD=("${BASE_CMD[@]}"
  --trials "$TRIALS"
  --seed 106
  --report-every "$REPORT_EVERY"
)

if [ "$HIST_OUT" -eq 1 ]; then
  CMD+=(--iter-hist-out "$hist_path")
fi

"${CMD[@]}"
