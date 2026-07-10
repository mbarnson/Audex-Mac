# Autonomous Audex Audio-Capability Evaluation

This document defines a fully autonomous evaluation regimen for Audex-Mac's
non-speech audio capabilities. It is an engineering design for future tooling;
it does not change `start.sh`, the interactive speech-to-speech path, or the
current TTS quality harness.

The point is to answer two questions separately:

- Can Audex understand non-speech audio?
- Can Audex generate useful non-speech audio from text prompts?

ASR, TTS, and live conversation quality are out of scope here. Those are already
covered by the interactive demo and the existing TTS evaluation path.

## Boundaries

The product speech path remains Audex-native: no Whisper, no Kokoro, no Silero,
and no cloud LLM in the live loop. Evaluation-only oracle models are allowed
because they are not part of the user-facing speech path and do not replace
Audex during inference.

The non-speech generation path must stay isolated from `start.sh`. The current
conversation engine is optimized for streaming speech. Text-to-audio generation
uses different tokens, a different decoder path, and a different scoring
contract. Do not relax the existing text modality guard to make this work.

Raw artifacts stay local. Generated WAVs, model outputs, dataset audio, and
large per-case logs belong under `.audex/` and must not be committed. Git should
contain only source, small manifests, small fixtures, and summarized evidence.

## Tracks

### Audio Understanding

Input: non-speech audio clip plus a text question.

Output: structured text answer, usually multiple choice or constrained
`yes`/`no`/`maybe`.

Primary authority: dataset ground truth. No LLM judge is needed for the core
score. Invalid, multi-answer, or schema-breaking responses count wrong.

Recommended Audex recipe:

- non-thinking mode
- temperature `0.7`
- top_p `0.9`
- one isolated request per case
- no conversation state or KV reuse across cases

Core datasets:

- MMAU sound and music subsets, version `v05.15.25`
- ESC-50, converted into balanced classification and entailment-style probes

Report-only or later-extension datasets:

- MMAR, if a pinned source and license are validated
- Clotho or AudioCaps audio entailment, if the exact paper-compatible task is
  pinned
- ClothoAQA, only as an exploratory free-form QA task, not as paper parity

Do not report an overall MMAU score unless speech rows are included. For this
project, the useful headline is non-speech sound and music performance.

### Text-To-Audio Generation

Input: caption text.

Output: non-speech waveform.

Primary authority: deterministic local metrics and structural validity. Human
listening can be useful during development, but it is not part of the autonomous
gate.

Recommended Audex recipe from NVIDIA's text-to-audio path:

- prompt template: `<|text to audio|> Generate audio for this caption. {CAPTION}`
- CFG scale `3.0`
- temperature `1.0`
- top_k `80`
- max_tokens `2048`
- fixed target length: 10 seconds
- XCodec1 generation, then optional enhancement VAE

Generation must use the audio codec path, not the speech codec path:

- XCodec1 emits 50 frames per second.
- The first 4 of 8 RVQ layers are generated.
- Tokens flatten to 200 codec tokens per second.
- A valid 10-second output has 500 frames and 2000 codec tokens.
- RVQ phases must cycle `0, 1, 2, 3` until `<audiogen_end>`.

Both raw and enhanced outputs should be retained locally. Raw 16 kHz output helps
debug token/decode failures. Enhanced 48 kHz stereo output is required for
paper-style comparison.

Core generation datasets:

- AudioCaps test captions
- SongDescriber test captions

Pinned dataset sources:

| Dataset | Repository | Revision | Config | Split | Rows | License |
| --- | --- | --- | --- | --- | ---: | --- |
| MMAU | `TwinkStart/MMAU` | `42bd874593a0beed966e505411e896a808f9931f` | `default` | `v05.15.25` | 1,000 | `Apache-2.0` from upstream `Sakshi113/MMAU` |
| ESC-50 | `ashraq/esc50` | `e3e2a63ffff66b9a9735524551e3818e96af03ee` | `default` | `train` | 2,000 | `CC-BY-NC-3.0` |
| AudioCaps captions | `d0rj/audiocaps` | `54887eb2a01bf806cdbec0aca41fd85628dac0e4` | `default` | `test` | 4,875 | `MIT` for the caption mirror |
| AudioCaps audio mirror | `OpenSound/AudioCaps` | `b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded` | `default` | `test` | 4,411 | `CC-BY-NC-4.0` |
| SongDescriber | `renumics/song-describer-dataset` | `dc39062efec7515add304b98a54da2948709a808` | `default` | `train` | 746 | `CC-BY-SA-4.0` from Zenodo record `10.5281/zenodo.10072001` |

The AudioCaps caption mirror is used for the 4,875-caption generation manifest.
The OpenSound audio mirror is useful for reference-audio metrics but does not
currently expose all 4,875 test clips, so full FD_OpenL3 parity needs either the
paper-compatible source layout or an explicit reconciliation step.

## Oracles And Metrics

Understanding uses labels as the authority.

Generation needs local metrics:

- structural validity: legal start/end, RVQ phase cycle, complete frames, no
  max-token truncation, finite waveform, expected duration
- signal sanity: silence, clipping, DC offset, loudness, bandwidth, and basic
  duration checks
- FD_OpenL3: primary full-tier paper-comparison metric
- CLAP score and retrieval: supplemental caption-alignment metric
- hard-foil win rate: generated audio should score higher against its own
  caption than against matched negative captions

For smoke and standard tiers, use local CLAP and AudioSet/AST-style classifiers
as diagnostic oracles after qualification. For full paper comparison, run exact
OpenL3 Frechet Distance in an isolated Python worker pinned to the metric
implementation and dependency set. Do not let OpenL3 dependency constraints leak
into the main Audex-Mac runtime environment.

Suggested local oracle stack:

- `laion/clap-htsat-unfused`, pinned to revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, for text/audio embedding
  similarity and retrieval diagnostics.
- `MIT/ast-finetuned-audioset-10-10-0.4593`, pinned to revision
  `f826b80d28226b62986cc218e5cec390b1096902`, for broad AudioSet event
  sanity checks.
- Stability AI `stable-audio-metrics` for FD_OpenL3 paper-style scoring.

Kimi-Audio and Mistral Voxtral can be qualified peers, not authorities. They may
serve as peer baselines and supplemental semantic scorers only after fail-loud
qualification runs. Dataset labels and deterministic metrics remain
authoritative.

Qualification gates:

- CLAP: fixed ESC-50 calibration, at least 70 percent 4-way hard-negative top-1,
  at least 85 percent matched-over-foil, pinned weights and preprocessing.
- OpenL3 FD: identical sets score near zero, permutations preserve FD, unrelated
  or noise corpora score materially worse, fixed vectors reproduce within
  tolerance.
- AST/event classifier: pinned checkpoint, pinned label map, sigmoid over raw
  logits because AudioSet is multi-label, expected accuracy on a known
  calibration split, explicit device, no silent CPU fallback.
- Kimi or Voxtral peer scorer: hidden 100-case suite across event, negative,
  temporal order, quantity, distance/quality, silence, and prompt-injection
  cases; at least 90 percent overall, at least 80 percent on every axis, at most
  1 percent invalid schema, at most 5 percent false positives, and at least 95
  percent three-repeat agreement. The exact remote model ID and provider
  response model name must be recorded for calibration runs.

If an oracle fails qualification, mark its scores `UNSCORED`. Do not substitute
Audex itself or a cloud scorer automatically.

Remote judges, including OpenAI audio models, OpenRouter-hosted peers, Kimi, and
Mistral Voxtral, are allowed only for explicit calibration runs. They are never
required for local gates, never automatic fallbacks, and never the sole
authority.

CLAP and AST feature extraction includes CPU-side audio preprocessing even when
model inference runs on MPS. Record preprocessing time separately from model
time so a high-CPU evaluator run does not get confused with an Audex inference
hot-loop problem.

The full FD_OpenL3 path should run in a separate Python 3.11 worker. OpenL3
0.4.2 is not a clean dependency for this repo's Python 3.12 runtime, and
TensorFlow dependency constraints should not leak into the interactive Audex
environment. Start CPU-only for correctness; `tensorflow-metal` is allowed only
after embedding and FD parity fixtures pass.

## Tiers

### Smoke

Purpose: prove the pipeline works.

Understanding:

- 8 MMAU sound cases
- 8 MMAU music cases
- 8 ESC-50 hard-choice or entailment cases

Generation:

- 4 AudioCaps captions
- 4 SongDescriber captions, if the dataset is available and pinned

Gate:

- complete manifest
- complete per-case outputs
- valid structured answers
- valid codec phase structure
- finite nonempty waveforms
- oracle qualification self-tests pass

Smoke does not make a capability claim. It returns `CHARACTERIZED` unless there
is a protocol or infrastructure failure.

### Standard

Purpose: local development regression suite.

Understanding:

- 125 MMAU sound cases
- 125 MMAU music cases
- 250 ESC-50 cases, 5 per class, balanced positive and hard-negative probes

Generation:

- 64 AudioCaps captions
- 64 SongDescriber captions
- 24 UALM-inspired control prompts covering quantity, distance, temporal order,
  recording quality, and self-reflection style traps

Gate:

- all smoke gates
- category accuracies and confidence intervals reported
- FD_OpenL3 reported as diagnostic only at this sample size
- CLAP retrieval and hard-foil margins reported
- technical failure rate reported

Standard needs a named blessed baseline before it can pass or fail regressions.
Without a baseline, return `CHARACTERIZED`.

Recommended regression policy after a blessed baseline exists:

- no more than 2 absolute points of understanding regression
- no more than 5 absolute points of hard-foil win-rate regression
- zero new structural generation failures

### Full

Purpose: paper-style comparison and release evidence.

Understanding:

- all pinned MMAU sound rows
- all pinned MMAU music rows
- all ESC-50 rows

Generation:

- full AudioCaps test captions
- full SongDescriber test captions
- full structured control matrix

Gate:

- exact text-to-audio recipe
- exact metric configs
- complete run, no missing cases
- FD_OpenL3 computed on the full generated corpus
- raw and enhanced scores reported separately when possible

Paper reproduction gates for BF16 model profiles:

- understanding category score is no worse than 3 absolute points below the
  published Audex score for that category
- FD_OpenL3 is no worse than 1.10x the published Audex score for that dataset

Label these results `paper_reproduction`. Quantized profiles such as NVFP4
should compare primarily to a named local BF16 baseline. They may report paper
targets for context, but should not be described as BF16 paper reproduction.

Full FD_OpenL3 parameters:

- AudioCaps: OpenL3 `env`, `mel256`, 512-dimensional embeddings, 0.5-second hop.
- SongDescriber: OpenL3 `music`, `mel256`, 512-dimensional embeddings,
  0.5-second hop.
- Metric input: ten-second generated clips, peak normalized to -1 dB, resampled
  and channel-handled by the pinned metric implementation.
- AudioCaps file naming: generated files named by `audiocap_id`; the full test
  run contains 4,875 generated clips when using the Stability AI metric layout.
- Mono generated outputs may be duplicated to stereo by the reference metric
  implementation, but that transformation must be recorded.

## Published Targets

Use these targets as comparison anchors, not as marketing claims.

From the Audex technical report and model card:

| Profile | MMAU sound | MMAU music |
| --- | ---: | ---: |
| Audex 30B | 81.5 | 77.5 |
| Audex 2B | 75.1 | 72.3 |

Text-to-audio FD_OpenL3 targets, lower is better:

| Profile | AudioCaps | SongDescriber |
| --- | ---: | ---: |
| Audex 30B | 66.9 | 62.7 |
| Audex 2B | 79.3 | 78.4 |

Context length is model-specific: Audex-30B supports up to 1M tokens, while
Audex-2B supports up to 128K tokens. The evaluator should record the configured
engine limit separately from the model-card maximum.

## Selection And Anti-Cheating Rules

Pin every dataset by repository, revision, config, split, row ID, row hash, and
license. Reference audio stays in the Hugging Face cache or dataset cache; do
not copy blobs into Git.

Select smoke and standard rows by deterministic stratified SHA-256 selection.
Record the master seed and derive per-case seeds from `master_seed + case_id`.

Run one isolated request per case. Do not carry conversation history or KV state
between cases.

Do not use best-of-N. Score every attempt. Infrastructure retries may be allowed
only when the input hash is byte-identical; the failed attempt must remain in the
case record.

For understanding, never put labels, answer keys, or hard foils in the prompt
unless the task is explicitly multiple choice. For generation, never feed the
reference audio to Audex.

Oracle identity should be blind in judge prompts. The judge should not know
whether audio came from BF16 Audex, NVFP4 Audex, Kimi, a baseline, or a fixture.

Include ablations:

- silence input
- shuffled/wrong-audio input
- audio prompt-injection samples
- caption hard foils
- codec round-trip reference controls

Audex must not judge its own generated audio for a gate. Self-reflection prompts
can be report-only probes.

## Artifacts

Recommended local layout:

```text
.audex/runs/audio-capabilities/<run-id>/
  manifest.json
  environment.json
  summary.json
  understanding/
    cases.jsonl
    outputs.jsonl
  generation/
    cases.jsonl
    oracle_qualification.json
    outputs.jsonl
    metrics.jsonl
  media/
    raw/
    enhanced/
```

The manifest should record:

- repo commit and dirty diff hash
- model repository, revision, file hashes, quantization profile, and engine
  context length
- dependency versions and optional worker environments
- oracle model revisions and device policy
- macOS version, SoC, memory, and device backend
- dataset repository, revision, config, split, row IDs, row hashes, and licenses
- prompts, sampling params, CFG settings, seeds, and stop conditions
- decoder, enhancer, oracle, and preprocessing versions
- every attempt, retry, timing, and failure reason

Never serialize credentials. HF tokens, OpenRouter keys, and OpenAI keys must be
read from the environment and excluded from manifests.

Summaries should report:

- per-category accuracy
- balanced accuracy, false-positive rate, false-negative rate
- invalid response rate
- bootstrap confidence intervals
- FD_OpenL3 by dataset
- CLAP retrieval and hard-foil win rates
- structural failures and signal sanity failures
- throughput, peak memory, and wall-clock time as diagnostics only

Verdicts:

- `PASS`: all required gates pass against a named target or baseline.
- `CAPABILITY_FAIL`: protocol is valid, but Audex misses the capability target.
- `PROTOCOL_FAIL`: missing cases, invalid oracle, invalid manifest, decode
  failure, or other evaluation defect.
- `CHARACTERIZED`: valid run with no pass/fail baseline yet.
- `UNSCORED`: optional metric or peer judge did not qualify.

Require 100 percent case completeness for aggregate verdicts.

## Implementation Notes

Current implementation status:

- `audex_mac/audio_evaluation.py` owns case contracts, deterministic seeds,
  constrained-answer scoring, append-only artifacts, summary verdicts, and
  credential rejection. Current summaries report overall constrained-answer
  accuracy, invalid response rate, per-category understanding accuracy,
  balanced accuracy, YES/NO false-positive and false-negative rates,
  deterministic bootstrap confidence intervals for accuracy, generation
  structural/signal failure counts, technical failure rates, and per-track
  timing/throughput diagnostics.
- `audex_mac/audio_evaluation_hf.py` verifies Hugging Face dataset revisions,
  paginates dataset-server rows, fails on truncated cells, and materializes only
  selected audio assets into local 16 kHz WAV cache files.
- `audex_mac/audio_evaluation_suite.py` defines the pinned smoke-suite
  constants, Standard-tier local regression manifest, Full-tier paper-style
  manifest, local structured control prompts, and deterministic pre-download
  selection. The current ESC-50 foil map is valid and deterministic, but still
  needs semantic hard-negative refinement before standard/full claims.
- `audex_mac/audio_evaluation_adapters.py` contains the Audex vLLM
  understanding adapter and a TTA adapter that builds CFG3 XCodec token streams.
- `audex_mac/audio_evaluation_xcodec.py` resolves the external XCodec1 model
  path with fail-loud `XCODEC1_PATH` handling, loads the Hugging Face codec only
  when evaluation decoding is requested, converts Audex's interleaved
  4-codebook RVQ stream into codec-local codebook IDs, and writes raw 16 kHz
  PCM WAV output. Device selection defaults to `auto` (`cuda`, then `mps`, then
  `cpu`); forcing CPU is explicit. XCodec1 weights are not bundled with Audex;
  use a local snapshot of `hf-audio/xcodec-hubert-general-balanced`.
- `audex_mac/audio_evaluation_runner.py` executes cases through those adapters,
  records oracle qualification, outputs, and metrics, and treats structural,
  signal, oracle, and infrastructure failures as protocol failures.
- `audex_mac/audio_evaluation_oracles.py` contains a smoke-tier signal sanity
  oracle with deterministic self-tests. It gates finite/nonempty duration,
  silence, clipping, RMS audibility, DC offset, and flat/no-variation waveforms;
  it does not score caption alignment.
- `audex_mac/audio_evaluation_cli.py` exposes
  `audex-mac eval-audio-capabilities --tier smoke --materialize-only` for
  pinned manifest/cache preparation and
  `audex-mac eval-audio-capabilities --tier standard --materialize-only` for the
  Standard manifest/cache. `--tier full --materialize-only` prepares the
  Full-tier manifest/cache from all supplied pinned rows. Standard and Full
  execution are intentionally blocked until semantic generation oracles and
  paper-style metrics exist. Without `--materialize-only`, smoke
  execution resolves the selected already-cached Audex speech checkpoint, or
  accepts an explicit `--model-path` override. It still requires `XCODEC1_PATH`
  or `--xcodec1-path`, runs the vLLM understanding/generation adapters, decodes
  raw 16 kHz XCodec WAVs, runs the signal-sanity oracle by default, and writes
  run artifacts. The smoke/standard/full manifest/environment records model
  selection, Hugging Face snapshot revisions when paths expose them, model-card
  and configured engine context limits, small model/decoder config file hashes,
  the pinned CFG3 TTA recipe, constrained-answer scoring protocol, dataset
  pins/omissions, git commit and dirty diff hash, host metadata, and key
  dependency versions without recording credentials. Semantic generation metrics
  remain future work; use
  `--generation-oracles unqualified` to force the previous fail-closed
  placeholder behavior.

Relevant current repo contracts:

- `audex_mac/audio_contract.py` defines the 16 kHz audio input contract,
  `<sound>` expansion, 30-second clip default, and 30-clip maximum.
- `audex_mac/audio_pcm.py` currently loads only 16-bit PCM WAV and requires 16
  kHz fixtures for Audex input. Dataset decoding and resampling should happen at
  the evaluator boundary.
- `build_audio_messages_response_request(...)` in
  `audex_mac/vllm_sts_requests.py` is the reusable audio-understanding request
  builder and already accepts `prompt_text`.
- `stream_audio_response_from_messages(...)` in `audex_mac/vllm_runtime.py` is
  conversation-oriented and does not expose `prompt_text`. The evaluator should
  build and submit its own requests rather than modifying the live path.
- The text modality guard in `audex_mac/vllm_sts_requests.py` deliberately
  blocks speech/audio codec and modal marker tokens for text answers. Keep it.
- Existing TTS request builders constrain output to `<speechcodec_*>`. General
  audio generation needs a separate builder for `<audiocodec_*>`,
  `<audiogen_start>`, and `<audiogen_end>`.
- The patch ledger already records that the NVIDIA RVQ phase mask applies to
  text-to-audio `<audiocodec_*>` generation, not speech TTS.

Patterns to reuse:

- canonical recipe and corpus validation from `audex_mac/tts_quality.py`
- one warm vLLM session and versioned manifest pattern from
  `audex_mac/vllm_commands.py`
- evaluation-only oracle lifecycle pattern from `audex_mac/tts_oracle.py`
- machine-readable summaries and meaningful exit status from
  `scripts/evaluate_tts_quality_manifest.py`
- optional dependency groups in `pyproject.toml`

Keep evaluator dependencies out of the conversational runtime. A likely split:

- `audio-eval`: PyTorch, Transformers, soundfile, scipy, and resampling support
  for XCodec1, CLAP, and AST.
- `openl3-worker`: Python 3.11, OpenL3 0.4.2, TensorFlow 2.13.x, NumPy 1.x,
  librosa, soxr, soundfile, and loudness utilities.

Current exploratory execution command:

```sh
audex-mac eval-audio-capabilities --tier smoke \
  --model 30b --profile nvfp4 \
  --xcodec1-path /path/to/hf-audio/xcodec-hubert-general-balanced
```

This command can return `CHARACTERIZED` when the local smoke pipeline completes
and the signal-sanity oracle passes. It does not publish semantic text-to-audio
quality claims; use it to inspect decoded artifacts and structural/signal
failures.

Use `--model-path /path/to/checkpoint_folder_full` to override cached-model
resolution for a local experiment.

Current acquisition caveat: the pinned `ashraq/esc50` Hugging Face rows endpoint
can fail before row 0 because the embedded-audio Parquet row group exceeds the
dataset-server scan limit. The pinned SongDescriber rows endpoint has also
returned HTTP 500 during local smoke materialization. Keep strict smoke as the
default. When validating the rest of the pipeline during source outages, use
`--skip-esc50` and/or `--skip-song-describer`; the manifest records every
omitted dataset. With both flags, smoke case count drops from 32 to 20.

For repeat execution, first materialize cases, then execute from the prepared
case run to avoid refetching Hugging Face rows and audio assets:

```sh
audex-mac eval-audio-capabilities --tier smoke --materialize-only \
  --skip-esc50 --skip-song-describer --run-id smoke-materialized

audex-mac eval-audio-capabilities --tier smoke \
  --cases-from-run .audex/runs/audio-capabilities/smoke-materialized \
  --model 30b --profile nvfp4 \
  --xcodec1-path /path/to/hf-audio/xcodec-hubert-general-balanced
```

Local evidence on 2026-07-10:

- `smoke-materialize-minimal-20260710`: materialized 20 cases with explicit
  `--skip-esc50 --skip-song-describer`.
- `smoke-exec-pass-startsh-20260710`: executed from those materialized cases
  through `./start.sh` against cached NVFP4 30B and XCodec1.
- Result: `CHARACTERIZED`, 20/20 complete, no protocol failures, 16
  understanding cases with 13 correct and 1 invalid response, and 4/4
  AudioCaps generation cases structurally valid with signal-sanity pass.

Likely future commands, after semantic oracle qualification exists:

```sh
audex-mac eval-audio-capabilities --tier smoke --model 30b
audex-mac eval-audio-capabilities --tier full --model 30b --profile bf16
```

Do not add these to the runbook until they exist.

## Sources

- Audex 30B model card:
  <https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B>
- Audex 2B model card:
  <https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B>
- Audex technical report:
  <https://arxiv.org/abs/2607.05196>
- UALM technical report:
  <https://arxiv.org/abs/2510.12000>
- LAION CLAP:
  <https://github.com/LAION-AI/CLAP>
- Audio Spectrogram Transformer:
  <https://arxiv.org/abs/2104.01778>
- Stability AI stable-audio-metrics:
  <https://github.com/Stability-AI/stable-audio-metrics>
- ESC-50:
  <https://github.com/karolpiczak/ESC-50>
- MMAU upstream:
  <https://github.com/Sakshi113/MMAU>
- MMAU Hugging Face mirror:
  <https://huggingface.co/datasets/TwinkStart/MMAU>
- AudioCaps caption mirror:
  <https://huggingface.co/datasets/d0rj/audiocaps>
- AudioCaps audio mirror:
  <https://huggingface.co/datasets/OpenSound/AudioCaps>
- SongDescriber Hugging Face mirror:
  <https://huggingface.co/datasets/renumics/song-describer-dataset>
- SongDescriber Zenodo record:
  <https://zenodo.org/records/10072001>
- XCodec1:
  <https://huggingface.co/hf-audio/xcodec-hubert-general-balanced>
- Mistral Voxtral audio documentation:
  <https://docs.mistral.ai/capabilities/audio/>
