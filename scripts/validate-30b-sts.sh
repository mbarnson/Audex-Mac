#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_BASENAME="audex-30b-proof"
FIXTURE_WAV="${ROOT_DIR}/.audex/fixtures/${FIXTURE_BASENAME}-16k-mono.wav"
OUTPUT_LOG="$(mktemp -t audex-30b-sts.XXXXXX)"

cleanup() {
  rm -f "${OUTPUT_LOG}"
}
trap cleanup EXIT

"${ROOT_DIR}/scripts/create-test-utterance.sh" \
  --text "Audex Mac test utterance. Please answer with one short sentence about local speech on Apple silicon." \
  --basename "${FIXTURE_BASENAME}"

(
  cd "${ROOT_DIR}"
  ./start.sh --input-wav "${FIXTURE_WAV}" --no-play
) | tee "${OUTPUT_LOG}"

python3 - "${OUTPUT_LOG}" <<'PY'
from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

output_log = Path(sys.argv[1])
text = output_log.read_text(encoding="utf-8")
match = re.search(r"Speech-to-speech run log: (.+)", text)
if match is None:
    raise SystemExit("Missing Speech-to-speech run log path in CLI output.")

sts_log_path = Path(match.group(1).strip())
sts = json.loads(sts_log_path.read_text(encoding="utf-8"))
speech_log_path = Path(sts["speech_output_run_log_path"])
speech = json.loads(speech_log_path.read_text(encoding="utf-8"))
wav_path = Path(sts["output_wav_path"])

if sts["selected_model"] != "nvidia/Nemotron-Labs-Audex-30B-A3B":
    raise SystemExit(f"Expected Audex 30B, got {sts['selected_model']}")
if "apple silicon" not in sts["transcript"].lower():
    raise SystemExit(f"Transcript does not preserve the proof phrase: {sts['transcript']}")
if not sts["response_text"].strip():
    raise SystemExit("Response text is empty.")
if len(sts["response_text"].split()) > 16:
    raise SystemExit(f"Response is too long for the default spoken demo: {sts['response_text']}")
if speech["backend"] != "mlx" or speech["device"] != "Device(gpu, 0)":
    raise SystemExit(f"Expected MLX GPU backend, got {speech['backend']} {speech['device']}")
if not speech["finite"] or speech["peak_abs"] <= 0:
    raise SystemExit("Speech waveform is not finite/non-silent.")
if not speech["reached_end_token"] or speech["hit_max_tokens"]:
    raise SystemExit("Speech generation did not finish cleanly before the token cap.")
if speech["sample_rate"] != 16000:
    raise SystemExit(f"Expected 16 kHz speech output, got {speech['sample_rate']}")

codec_frames = len(speech["generated_codec_frames"])
expected_samples = codec_frames * int(speech["hop_length"])
if speech["waveform_shape"] != [expected_samples]:
    raise SystemExit(
        f"Waveform shape mismatch: {speech['waveform_shape']} vs {expected_samples}"
    )

with wave.open(str(wav_path), "rb") as wav:
    if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
        raise SystemExit("Output WAV is not mono 16 kHz PCM16.")
    if wav.getnframes() != expected_samples:
        raise SystemExit(f"Output WAV frames mismatch: {wav.getnframes()} vs {expected_samples}")

print("Audex 30B STS validation passed.")
print(f"STS log: {sts_log_path}")
print(f"Speech log: {speech_log_path}")
print(f"Output WAV: {wav_path}")
PY
