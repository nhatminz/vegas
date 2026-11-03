#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$REPO_DIR/.cache/huggingface"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export VEGAS_BACKEND="${VEGAS_BACKEND:-portable}"
export VLLM_CACHE_ROOT="$REPO_DIR/.cache/vllm"
export TRITON_CACHE_DIR="$REPO_DIR/.cache/triton"
export TMPDIR="$REPO_DIR/.cache/tmp"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
cd "$REPO_DIR"
exec "$REPO_DIR/.venv/bin/python" benchmarks/benchmark_compare.py "$@"
