#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 input_file [output_tsv]" >&2
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-}"

g++ -O2 -std=c++17 -o objective_search objective_search.cpp

if [ -n "$OUTPUT_FILE" ]; then
  ./objective_search "$INPUT_FILE" "$OUTPUT_FILE"
else
  ./objective_search "$INPUT_FILE"
fi
