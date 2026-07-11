#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/compare-quant.sh" \
  "${ROOT_DIR}/scripts/tta_quant_voice_corpus.json" \
  "tta-voice-quant"
