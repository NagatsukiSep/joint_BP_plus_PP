#!/usr/bin/env bash
set -euo pipefail

BASE_CMD=(./jointbp_ets \
  --params H_P768_J3_L12_dmax3_nc0-3_1-2_seed11579811919164041.txt \
  --simulate --p 0.04 --trials 10000 --max-iter 200 --no-pp \
  --cuda --cuda-device 0 --seed 1234 --report-every 1000)

# Candidates from objective_search (I=interval, W=warmup)
CANDIDATE_INTERVALS=(1 1 1 1 2)
CANDIDATE_WARMUPS=(7 6 8 5 6)

OUT_TSV=data/results_cuda_sweep.tsv
CANDIDATES_FILE=""
TOP_N=10

usage() {
  echo "Usage: $0 [--out output_tsv] [--candidates objective_tsv] [--top N]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      OUT_TSV="${2:-}"
      shift 2
      ;;
    --candidates)
      CANDIDATES_FILE="${2:-}"
      shift 2
      ;;
    --top)
      TOP_N="${2:-}"
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

if [ -z "$OUT_TSV" ]; then
  echo "Missing value for --out" >&2
  usage
  exit 1
fi

case "$OUT_TSV" in
  data/*) mkdir -p data ;;
esac

if [ -n "$CANDIDATES_FILE" ]; then
  if [ ! -f "$CANDIDATES_FILE" ]; then
    echo "Candidates file not found: $CANDIDATES_FILE" >&2
    exit 1
  fi
  CANDIDATE_INTERVALS=()
  CANDIDATE_WARMUPS=()
  while read -r interval warmup; do
    CANDIDATE_INTERVALS+=("$interval")
    CANDIDATE_WARMUPS+=("$warmup")
  done < <(awk -v n="$TOP_N" 'NR==1{next} NR<=n+1{print $1, $2}' "$CANDIDATES_FILE")
fi
echo -e "interval\twarmup\tlatency_ms" > "$OUT_TSV"

for i in "${!CANDIDATE_INTERVALS[@]}"; do
  interval="${CANDIDATE_INTERVALS[$i]}"
  warmup="${CANDIDATE_WARMUPS[$i]}"
  echo "Running interval=$interval warmup=$warmup" >&2
  tmp=$(mktemp)
  "${BASE_CMD[@]}" --cuda-check-interval "$interval" --cuda-check-warmup "$warmup" \
    | tee "$tmp" >/dev/null

  latency_ms=$(awk '
    match($0, /(avg_)?latency[[:space:]]*=[[:space:]]*([0-9.]+)(us|ms)/, m) {
      val = m[2];
      unit = m[3];
      if (unit == "us") {
        lat = val / 1000.0;
      } else {
        lat = val;
      }
    }
    END {if (lat=="") {print "nan"} else {print lat}}
  ' "$tmp")

  echo -e "${interval}\t${warmup}\t${latency_ms}" >> "$OUT_TSV"
  rm -f "$tmp"
done

echo "Done. Results in $OUT_TSV" >&2
