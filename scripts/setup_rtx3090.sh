#!/usr/bin/env bash
# Tested deployment route: Python changes from Vegas, native ops from its
# exact upstream parent wheel, portable Vegas kernels JIT-compiled by Triton.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export UV_CACHE_DIR="$REPO_DIR/.cache/uv"
export TMPDIR="$REPO_DIR/.cache/tmp"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" .cache/wheels
if [[ ! -x .venv/bin/python ]]; then
    uv python install 3.12
    uv venv --python 3.12 .venv
fi
WHEEL_NAME=vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
WHEEL_FILE="$REPO_DIR/.cache/wheels/$WHEEL_NAME"
if [[ ! -f "$WHEEL_FILE" ]]; then
    curl -fL --retry 3 "https://wheels.vllm.ai/89a77b10846fd96273cce78d86d2556ea582d26e/$WHEEL_NAME" -o "$WHEEL_FILE"
fi
printf '%s  %s\n' f066b2a2f8597a4a3ada8fbbfd122b59086864b2260ca42dc81bf9fb57af0c42 "$WHEEL_FILE" | sha256sum -c -
uv pip install --python .venv/bin/python -r requirements/rtx3090.lock.txt
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_FILE" \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0 \
    uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/prepare_offline_data.py
