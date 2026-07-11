#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
CORPUS="${1:-${ROOT_DIR}/scripts/tta_quant_quality_corpus.json}"
RUN_LABEL="${2:-tta-quant}"
if [[ ! "${RUN_LABEL}" =~ ^[a-z0-9-]+$ ]]; then
  echo "Run label must contain only lowercase letters, digits, and hyphens" >&2
  exit 2
fi
RUN_ROOT="${ROOT_DIR}/.audex/runs/${RUN_LABEL}-${STAMP}"
LISTENING="${ROOT_DIR}/.audex/listening/${RUN_LABEL}-${STAMP}"
KEY="${RUN_ROOT}/private-key.json"

"${ROOT_DIR}/start.sh" tta-quant-quality render \
  --profile bf16 --corpus "${CORPUS}" --output-dir "${RUN_ROOT}/bf16"
"${ROOT_DIR}/start.sh" tta-quant-quality render \
  --profile nvfp4 --corpus "${CORPUS}" --output-dir "${RUN_ROOT}/nvfp4"
"${ROOT_DIR}/start.sh" tta-quant-quality package \
  "${RUN_ROOT}/bf16/tta-quant-bf16.manifest.json" \
  "${RUN_ROOT}/nvfp4/tta-quant-nvfp4.manifest.json" \
  --output-dir "${LISTENING}" --key-out "${KEY}"

echo "Blind comparison ready: ${LISTENING}/LISTENING.md"
