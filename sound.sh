#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sound Lab assets use NVIDIA's CFG3 text-to-audio recipe. This environment
# change is scoped to this process and does not alter start.sh defaults.
export AUDEX_VLLM_TTS_CFG="1"

exec "${ROOT_DIR}/start.sh" sound-lab "$@"
