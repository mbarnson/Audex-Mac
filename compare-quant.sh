#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
CORPUS="${ROOT_DIR}/scripts/tta_quant_quality_corpus.json"
RUN_ROOT="${ROOT_DIR}/.audex/runs/tta-quant-${STAMP}"
LISTENING="${ROOT_DIR}/.audex/listening/tta-quant-${STAMP}"
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
