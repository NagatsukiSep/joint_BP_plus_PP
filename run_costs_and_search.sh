#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --p value" >&2
}

P=""
while [ $# -gt 0 ]; do
  case "$1" in
    --p)
      P="${2:-}"
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

if [ -z "$P" ]; then
  echo "Missing required --p value" >&2
  usage
  exit 1
fi

mkdir -p data

COSTS_FILE="data/p${P}_costs.txt"
OBJECTIVE_TSV="data/p${P}_objective.tsv"
SWEEP_TSV="data/p${P}_sweep.tsv"

./run.sh --p "$P" --costs-out --costs-file "$COSTS_FILE"
./run_objective_search.sh "$COSTS_FILE" "$OBJECTIVE_TSV"
./sweep_cuda_params.sh --out "$SWEEP_TSV" --candidates "$OBJECTIVE_TSV" --top 10
