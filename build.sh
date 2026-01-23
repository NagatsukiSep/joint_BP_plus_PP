#!/usr/bin/env bash
set -euo pipefail

nvcc -O2 -std=c++17 -DUSE_CUDA -DUSE_CUDA_FP32 -o jointbp_ets jointbp_ets.cpp jointbp_cuda.cu
