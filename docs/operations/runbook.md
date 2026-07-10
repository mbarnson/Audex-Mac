# Audex-Mac Runbook

## Target UX

```sh
git clone <repo>
cd Audex-Mac
./start.sh
```

The normal path should be quiet and automatic:

1. Bootstrap a local `.venv` if needed.
2. Install bootstrap tooling, including `huggingface_hub`, if missing.
3. Verify native Apple Silicon Python.
4. Verify or install pinned dependencies.
5. Check supported Audex models in the Hugging Face cache.
6. Select a model.
7. Apply vLLM Metal patch guards and monkey patches.
8. Launch a typed-or-push-to-talk CLI with spoken responses.

## Public Repository Rules

Before pushing publicly:

- confirm `.gitignore` excludes local environments, caches, model weights, and
  generated audio
- do not commit Hugging Face snapshots or safetensors
- do not commit local run audio captures
- summarize local model/audio runs in `docs/engineering/viability.md`
- use GitHub issues and PRs to make Codex-driven development traceable

## Hardware Assumptions

Validated target:

- Apple Silicon Mac.
- Native arm64 Python 3.12 or 3.13 for vLLM Metal.
- 128 GB unified memory for the substantially tested full-precision 30B path.
- Enough disk for model snapshots and transient cache files.

Audex-30B-A3B is the release-quality target and is selected automatically when
fully cached. Audex-2B remains the smaller first-download choice, but its live
end-to-end testing is minimal.

## Model Download Prompt

If no supported model is cached, `./start.sh` should ask before downloading.
The prompt must mention:

- selected model ID
- approximate download size
- NVIDIA model license
- that Audex-Mac's MIT license does not cover NVIDIA model weights

NVIDIA's Audex repositories contain both a root `LICENSE` file and a
lowercase `license/` directory. A full unfiltered `hf download` therefore
fails on default case-insensitive APFS when Hugging Face tries to create the
directory after the file. Audex-Mac passes `ignore_patterns=["license/*"]`
in addition to narrow checkpoint allow-lists. For a manual full download, use:

```sh
hf download nvidia/Nemotron-Labs-Audex-2B --exclude "license/*"
```

## Dependency Refresh

Dependencies should not churn on every run.

Expected behavior:

- Install dependencies when `.venv` is missing.
- Clone and install pinned vLLM Metal into `.audex/vendor/vllm-metal` when the
  generated vLLM Metal runtime is missing.
- Move an incompatible generated `.venv` aside and recreate it with native
  arm64 Python 3.12 or 3.13.
- Reinstall when pinned dependency state is stale.
- Provide `./start.sh --refresh-deps` for forced refresh.
- Provide `./start.sh --check-upstream` for advisory vLLM Metal upstream checks.
- Provide `./start.sh --preflight-text-runtime --model audex-2b` for the
  text-only milestone readiness check.
- Provide `./start.sh --preflight-audio-runtime --model audex-2b` for the
  speech artifact readiness check: full checkpoint, Audex audio
  encoder/projector metadata, Audex decoder files, and speech-codec tokenizer
  markers.
- Provide `./start.sh --preflight-audio-features --model audex-2b` for the
  native audio preprocessing smoke: one prepared Audex clip through the local
  `WhisperFeatureExtractor`, yielding `(1, 128, 3000)` input features.
- Provide `./start.sh --preflight-audio-projector --model audex-2b` for the
  native MLX projector smoke: real Audex projector tensors yielding
  `(1, 750, 2048)` text embeddings on `Device(gpu, 0)`.
- Provide `./start.sh --preflight-audio-encoder --model audex-2b` for the
  native MLX encoder smoke: real Audex encoder and projector tensors yielding
  `(1, 128, 3000) -> (1, 750, 1280) -> (1, 750, 2048)` on `Device(gpu, 0)`.
- Provide `./start.sh --preflight-audio-splice --model audex-2b` for the native
  MLX prompt-splice smoke: full Audex tokenizer prompt with 750 sound
  placeholders, projected audio embeddings replacing those placeholders, and
  one text-model generation step using `input_embeddings`.
- Provide `./start.sh --run-text-benchmark --model audex-30b-a3b` for the full
  ten-turn text benchmark transcript using the default vLLM Metal backend.
- Use `--limit-text-turns 1` only as a development smoke; it is not a substitute
  for the full text gate.

## Text-Only Checkpoint Conversion

Some Audex snapshots include text-only config/tokenizer files and an index but
do not include the referenced text-only safetensors shards. Regenerate those
shards from a complete local checkpoint with:

```sh
scripts/convert-textonly.sh --model audex-2b --dry-run
scripts/convert-textonly.sh --model audex-2b --overwrite
```

The helper picks the first complete source checkpoint from:

1. `checkpoint_folder_audiogen`
2. `checkpoint_folder_full`

Override that choice when needed:

```sh
scripts/convert-textonly.sh --model audex-30b-a3b --input-folder checkpoint_folder_full --overwrite
```

The helper requires text-only sidecar files to already exist in
`checkpoint_folder_textonly`; NVIDIA's conversion script only writes
safetensors shards and `model.safetensors.index.json`.

The current vLLM Metal pin lives in `vendor_pins.json`.

For unattended model setup on a local machine where the NVIDIA license prompt
has already been reviewed:

```sh
./start.sh --yes-download
```

## Upstream vLLM Metal Check

Out-of-date upstream `main` is advisory only.

Startup should continue if:

- installed dependency matches the pinned commit
- patch guards pass
- upstream HEAD has moved

Startup should stop if:

- installed dependency does not match the pin
- expected patch target modules/symbols are missing
- patch guards fail

Current generated-venv patch:

- `start.sh` installs `mlx_lm/models/nemotron_dense.py` into the generated vLLM
  Metal venv before launching `audex_mac.cli`.
- The generated shim imports the repo-owned implementation from
  `audex_mac.patches.mlx_lm_nemotron_dense`.
- This is required because vLLM Metal uses spawned worker processes on macOS;
  parent-process `sys.modules` monkey patches do not reach the worker.

## Metal/MLX Runtime Guard

Audex-Mac must not run vLLM Metal in MLX CPU mode.

`start.sh` requires:

- `VLLM_METAL_USE_MLX=1`
- `VLLM_MLX_DEVICE=gpu`
- `VLLM_METAL_USE_PAGED_ATTENTION=0`

vLLM Metal reports `device_config=cpu` internally as a platform compatibility
facade. That alone is not evidence of CPU inference. The live check that matters
is `mlx_default_device=Device(gpu, 0)`.

The normal CLI uses the patched vLLM Metal non-paged path. Direct MLX remains an
explicit diagnostic fallback through `--sts-backend mlx`.

If any of those are explicitly set to another value, startup exits before model
selection. The text runtime preflight also prints the live MLX Metal check,
including `mlx_metal_available` and `mlx_default_device`.

Do not use vLLM's `device_config=cpu` log line alone as evidence of CPU
inference. At the pinned vLLM Metal commit, `MetalPlatform` advertises a CPU
PyTorch device name/type for vLLM compatibility while the worker sets MLX to
`Device(gpu, 0)` and PyTorch to MPS.

## CLI Speech-to-Speech Behavior

Normal interactive modes:

- typed text submitted with Enter; Shift+Enter inserts newlines in supported
  terminals, with Option+Enter as the fallback
- empty Enter starts push-to-talk input, and the next Enter stops it
- no VAD
- typed text bypasses ASR; full recorded utterances use Audex ASR
- non-thinking mode by default
- optional thinking flag
- speech output through Audex causal speech decoder
- deterministic audio plumbing only

Normal launch:

```sh
./start.sh
```

The CLI loops over multiple turns at a `You:` editor. Type a message and press
Enter to send it directly to the LLM. Shift+Enter inserts a newline in Ghostty,
Kitty, WezTerm, and iTerm; use Option+Enter when a terminal does not expose
Shift+Enter distinctly. Submit an empty editor to start capture and press Enter
again to stop. Both paths produce local Audex speech. Type `q` as the complete
message to quit.

No-CFG is the default conversational recipe because it delivers roughly twice
the useful codec throughput on the current Metal path. Opt into the preferred
CFG3 quality recipe explicitly:

```sh
AUDEX_VLLM_TTS_CFG=1 ./start.sh
```

Spoken turns answer the audio directly and defer Audex ASR transcript
persistence until speech generation has released the engine. Set
`AUDEX_VLLM_DIRECT_AUDIO_RESPONSE=0` only to compare the serial pipeline. The
direct request trims padded audio embeddings by default; deferred ASR remains
full-width. Set `AUDEX_VLLM_DIRECT_AUDIO_TRIM_PADDED_EMBEDDINGS=0` to compare
against the fixed 750-embedding direct request.

The current 30B playback checkpoint uses the real 3.21-second PTT fixture
`ptt-input-20260709-204149.wav` with submission after the complete captured
utterance. With the local NVFP4 snapshot, no-CFG, a one-frame decoder primer,
eight-frame steady decoder chunks, an 80 ms adaptive prebuffer floor, and a
192 kHz/512-frame output stream, two consecutive runs both reached first device
write at 0.905 seconds and estimated first DAC output at 0.959 seconds after
submit. Decoded PCM was ready at 0.705 and 0.706 seconds. Both runs had no
initial underflow and no application queue underruns; each reported one later
device underflow. Evidence is in
`sts-turn-vllm-20260710-010030.json` and
`sts-turn-vllm-20260710-010136.json` under `.audex/runs/`.

The same optimized path on the full-precision BF16 30B checkpoint reached decoded
PCM at 1.539 seconds, first device write at 1.785 seconds, and estimated DAC
onset at 1.966 seconds. The sub-second claim is therefore specific to the
tested local NVFP4 30B artifact, the BF16 2B checkpoint, and output device. Run
logs intentionally keep model text, PCM, device-write, and estimated-audible
endpoints separate.

BF16 Audex-2B reproduced the semantic-audio gate on the same fixture in two
cold-process runs. The first reached decoded PCM/device write/estimated DAC at
0.427/0.561/0.578 seconds; the repeat reached 0.402/0.536/0.553 seconds. Both
transcribed “testing testing one two, testing three four,” answered “Got it,
testing,” and passed the one-second gate. Evidence is in
`sts-turn-vllm-20260710-043555.json` and
`sts-turn-vllm-20260710-043702.json` under `.audex/runs/`.

Long-form playback uses a separate zero-underrun acceptance criterion because
every reported underrun has been audible. The product keeps the interleaved
application prebuffer at 80 ms but requests 50 ms of device latency, gives an
active TTS request exclusive scheduler steps over continuing response text, and
disables interleaved tail batching by default. Text remains resident and resumes
between speech chunks; future TTS chunks run sequentially instead of reducing
the currently audible chunk below real time. The 67.5-second typed essay in
`speech-output-vllm-20260710-052928.json` recorded
`initial_device_underflow=false`, `device_underflow_count=0`, and
`queue_underrun_count=0`. Set `AUDEX_VLLM_SPEECH_FIRST_SCHEDULING=0` or
`AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL=1` only for controlled A/B diagnostics.

The Mac demo configures a 262,144-token active context. Existing transcripts
are re-rendered and re-prefilled on resume; no compaction or vLLM disk-cache
reuse is claimed. The model card advertises a one-million-token maximum, but
one-million-token Mac inference is outside this release. Audex-2B declares a
131,072-token maximum, so its engine and conversation metadata are capped at
128K instead of requesting the 30B demo ceiling.

Startup greeting behavior follows conversation identity, not cache-file
availability. A new conversation uses the fixed Audex introduction. A resumed
vLLM conversation with substantive history reuses the committed text-history
prefix to generate at most two spoken sentences that mention the latest topic
and invite the next one without reintroducing Audex. The synthetic greeting
instruction and response are not persisted as conversation turns. An empty
resumed conversation uses the deterministic returning-user greeting.

Fixture launch without microphone or playback:

```sh
./start.sh --model audex-2b --input-wav .audex/fixtures/example.wav --no-play
```

The fixture path is useful for coding-agent verification. It still uses Audex
audio input, Audex text response generation, Audex speech-codec generation, and
the Audex causal speech decoder. It does not use external STT/TTS/VAD.

Current preflightable speech contract:

- audio input is normalized 16 kHz PCM
- mono/stereo PCM is normalized to float samples in `[-1, 1]`
- the final audio clip is zero-padded to a fixed 30 second input window
- native Audex preprocessing yields `(clips, 128, 3000)` NV-Whisper input
  features
- native MLX Audex encoder/projector execution yields
  `(clips, 750, 2048)` text-space audio embeddings
- generated text prefill accepts spliced Audex audio embeddings at exactly the
  `<so_embedding>` placeholder positions
- each 30 second input clip expands to 750 `<so_embedding>` placeholders
- prompt expansion is bracketed by `<so_start>` and `<so_end>`
- generated speech output is discovered from `<speechgen_start>`,
  `<speechcodec_N>`, and `<speechgen_end>` tokenizer IDs
- response WAVs and STS turn logs are written under `.audex/runs/`
- separate Whisper, Kokoro, and Silero models remain forbidden

## Blind CFG TTS Quality Matrix

Generate the controlled long-form matrix with one warm engine per recipe. All
three commands must use the same corpus, speech budget, and seed:

```sh
./start.sh --model audex-30b-a3b --speech-max-tokens 2400 \
  --diagnose-vllm-tts-quality-corpus scripts/tts_quality_corpus.json \
  --diagnose-vllm-tts-quality-recipe plain-reference \
  --diagnose-vllm-tts-quality-seed 20260709

./start.sh --model audex-30b-a3b --speech-max-tokens 2400 \
  --diagnose-vllm-tts-quality-corpus scripts/tts_quality_corpus.json \
  --diagnose-vllm-tts-quality-recipe nvidia-tts-cfg \
  --diagnose-vllm-tts-quality-seed 20260709

./start.sh --model audex-30b-a3b --speech-max-tokens 2400 \
  --diagnose-vllm-tts-quality-corpus scripts/tts_quality_corpus.json \
  --diagnose-vllm-tts-quality-recipe audex-cfg3 \
  --diagnose-vllm-tts-quality-seed 20260709
```

Each command prints its timestamped manifest under `.audex/runs/`. Evaluate
each manifest offline with the optional MLX-Audio STT oracle:

```sh
.venv/bin/python -m pip install -e '.[oracle]'
.venv/bin/python \
  scripts/evaluate_tts_quality_manifest.py MANIFEST.json \
  --json-out MANIFEST.eval.json
```

The evaluator can exit nonzero when a strict required-term gate fails; it still
writes the JSON report. Keep those oracle results hidden until the listener has
recorded preferences.

Package exactly one manifest from each canonical recipe:

```sh
.venv/bin/python scripts/create_blind_tts_listening_set.py \
  PLAIN.manifest.json NVIDIA_CFG.manifest.json AUDEX_CFG3.manifest.json \
  --output-dir .audex/listening/tts-quality-blind-YYYYMMDD \
  --key-out .audex/runs/tts-quality-blind-YYYYMMDD.private-key.json \
  --random-seed 8675309
```

The packager rejects a matrix unless it contains the exact three recipes, six
matching cases per recipe, the same integer seed and case content, one observed
segment per run log, clean termination, and required compact-window decode.
Give the listener only the generated `LISTENING.md` and `sample-NN.wav` files.
The private decoding key must remain outside the listener directory so recipe
names, raw paths, and case labels cannot leak into the listening session.

The first blind adjudication preferred `audex-cfg3` in four of six groups,
`nvidia-tts-cfg` in one, and `plain-reference` in one; the last plain win was a
near tie with CFG3. Keep the one-listener/one-seed limitation attached to those
results.

## Native Test Utterances

Use macOS native speech tools to create repeatable local audio fixtures without
manual recording:

```sh
scripts/create-test-utterance.sh
scripts/create-test-utterance.sh --voice Allison --play
```

The script writes ignored files under `.audex/fixtures/`, converts `say` output
to a 16 kHz mono WAV with `afconvert`, and validates nonzero sample data with
`afinfo`. If `say` produces a zero-frame file in a headless or sandboxed
session, rerun from a normal macOS Terminal session.
