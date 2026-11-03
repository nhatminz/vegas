#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-aime25}"
if (($#)); then shift; fi
exec "$SCRIPT_DIR/run_compare.sh" --method vllm --dataset "$DATASET" "$@"
