#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sound Lab assets use NVIDIA's CFG3 text-to-audio recipe. This environment
# change is scoped to this process and does not alter start.sh defaults.
export AUDEX_VLLM_TTS_CFG="1"
export AUDEX_VLLM_ENABLE_CFG_WIRING="1"

# NVIDIA's reference TTA recipe admits two CFG pairs per wave: four sequences.
# Keep the 8K context scoped to Sound Lab so those slots remain inexpensive.
export AUDEX_VLLM_CFG_MAX_MODEL_LEN="8192"
export AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="4"

exec "${ROOT_DIR}/start.sh" sound-lab --profile bf16 "$@"
