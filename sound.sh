#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sound Lab assets use NVIDIA's CFG3 text-to-audio recipe. This environment
# change is scoped to this process and does not alter start.sh defaults.
export AUDEX_VLLM_TTS_CFG="1"

# Unlike the conversational path, Sound Lab emits a bounded ~2K-token sound
# from several CFG pairs at once. Reserve ten non-paged slots for five
# auditions without paying the 256K-context KV cost per slot.
export AUDEX_VLLM_CFG_MAX_MODEL_LEN="8192"
export AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="10"

exec "${ROOT_DIR}/start.sh" sound-lab "$@"
