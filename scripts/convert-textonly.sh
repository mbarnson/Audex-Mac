#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_METAL_PYTHON="${ROOT_DIR}/.audex/vendor/vllm-metal/.venv-vllm-metal/bin/python"
PROJECT_PYTHON="${ROOT_DIR}/.venv/bin/python"

python_has_conversion_deps() {
  local candidate="$1"
  "${candidate}" -c 'import safetensors.torch' >/dev/null 2>&1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${VLLM_METAL_PYTHON}" ]] && python_has_conversion_deps "${VLLM_METAL_PYTHON}"; then
  PYTHON="${VLLM_METAL_PYTHON}"
elif [[ -x "${PROJECT_PYTHON}" ]] && python_has_conversion_deps "${PROJECT_PYTHON}"; then
  PYTHON="${PROJECT_PYTHON}"
else
  echo "Audex text-only conversion needs Python with safetensors.torch available." >&2
  echo "Install or refresh the vLLM Metal runtime, or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" -m audex_mac.textonly_conversion "$@"
