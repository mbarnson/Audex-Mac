#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.audex/vendor/vllm-metal/.venv-vllm-metal/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "The pinned MLX environment is missing. Run ./start.sh --refresh-deps first." >&2
  exit 1
fi

exec env PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m audex_mac.nvfp4_conversion "$@"
