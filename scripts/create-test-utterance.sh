#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/.audex/fixtures"
BASENAME="audex-test-utterance"
TEXT="Audex Mac test utterance. Please transcribe this short sentence."
PLAY=0
VOICE="${AUDEX_TEST_VOICE:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/create-test-utterance.sh [options]

Create a deterministic local speech fixture with macOS native tools.

Options:
  --text TEXT       Spoken text for macOS say.
  --out-dir DIR     Output directory. Defaults to .audex/fixtures.
  --basename NAME   Output basename. Defaults to audex-test-utterance.
  --voice NAME      Optional macOS say voice.
  --play            Play the generated 16 kHz mono WAV with afplay.
  -h, --help        Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text)
      TEXT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --basename)
      BASENAME="$2"
      shift 2
      ;;
    --voice)
      VOICE="$2"
      shift 2
      ;;
    --play)
      PLAY=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v say >/dev/null 2>&1; then
  echo "Missing macOS say utility." >&2
  exit 1
fi
if ! command -v afconvert >/dev/null 2>&1; then
  echo "Missing macOS afconvert utility." >&2
  exit 1
fi
if ! command -v afinfo >/dev/null 2>&1; then
  echo "Missing macOS afinfo utility." >&2
  exit 1
fi
if [[ "${PLAY}" == "1" ]] && ! command -v afplay >/dev/null 2>&1; then
  echo "Missing macOS afplay utility." >&2
  exit 1
fi

audio_bytes() {
  afinfo "$1" | awk -F: '/audio bytes/ { gsub(/ /, "", $2); print $2; exit }'
}

require_nonempty_audio() {
  local path="$1"
  local bytes
  bytes="$(audio_bytes "${path}")"
  if [[ -z "${bytes}" || "${bytes}" == "0" ]]; then
    echo "Generated audio has no samples: ${path}" >&2
    echo "Try running this script from a normal macOS Terminal session or with --voice NAME." >&2
    exit 1
  fi
}

mkdir -p "${OUT_DIR}"
AIFF_PATH="${OUT_DIR}/${BASENAME}.aiff"
WAV_PATH="${OUT_DIR}/${BASENAME}-16k-mono.wav"

SAY_ARGS=(-o "${AIFF_PATH}")
if [[ -n "${VOICE}" ]]; then
  SAY_ARGS+=(-v "${VOICE}")
fi
SAY_ARGS+=("${TEXT}")

say "${SAY_ARGS[@]}"
require_nonempty_audio "${AIFF_PATH}"
afconvert -f WAVE -d LEI16@16000 -c 1 "${AIFF_PATH}" "${WAV_PATH}"
require_nonempty_audio "${WAV_PATH}"

echo "Created AIFF: ${AIFF_PATH}"
echo "Created Audex-ready WAV: ${WAV_PATH}"
echo "Sample rate: 16000 Hz"
echo "Channels: 1"

if [[ "${PLAY}" == "1" ]]; then
  afplay "${WAV_PATH}"
fi
