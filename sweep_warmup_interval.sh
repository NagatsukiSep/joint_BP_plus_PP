#!/usr/bin/env bash
set -euo pipefail

# Hardcoded sweep limit
N=30

P=0.04
OUT_TSV=data/results_warmup_interval.tsv

usage() {
  echo "Usage: $0 [--p value] [--out output_tsv]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --p)
      P="${2:-}"
      shift 2
      ;;
    --out)
      OUT_TSV="${2:-}"
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

echo -e "interval\twarmup\tlatency_ms" > "$OUT_TSV"

for ((warmup=0; warmup<=N; warmup++)); do
  max_interval=$((N - warmup + 1))
  for ((interval=1; interval<=max_interval; interval++)); do
    echo "Running interval=$interval warmup=$warmup" >&2
    tmp=$(mktemp)
    ./run.sh --p "$P" --warmup "$warmup" --interval "$interval" \
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
done

echo "Done. Results in $OUT_TSV" >&2
