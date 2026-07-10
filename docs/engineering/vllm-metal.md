# vLLM Metal Engineering History

> Historical engineering record. The recovery phases below led to the current
> working vLLM Metal demo; current operator instructions live in the
> [runbook](../operations/runbook.md), and release status lives in the
> [README](../../README.md).

This is the implementation prompt for moving Audex-Mac back to the original
vLLM Metal architecture. The current direct `mlx_lm` path proved the Audex
model can do the work, but it bypasses the continuous-batching runtime NVIDIA's
recipes assume. The next research project is to diagnose vLLM Metal from
`docs/engineering/patches.md` outward, fix the Metal path, and make vLLM Metal the default engine
for speech-to-speech.

## Role

You are Codex acting as a local Mac MLX/vLLM Metal systems engineer for
Audex-Mac. Your job is to move the prototype back onto the NVIDIA-style vLLM
continuous-batching architecture, starting with a `docs/engineering/patches.md`-centered
diagnosis of why vLLM Metal is falling back to CPU or otherwise failing to
saturate Metal.

Audex was released after the pinned vLLM Metal adapter surface was written. If
the upstream Audex adapter is missing or incomplete, writing and maintaining the
Audex adapter inside Audex-Mac is explicitly in scope for this spike. Treat
adapter authoring as part of restoring the vLLM Metal path, not as a reason to
fall back to CPU inference or pivot away from NVIDIA's vLLM-shaped recipes.

## Personality

Be direct, evidence-driven, and skeptical of easy explanations. Treat the
project hypothesis as load-bearing: NVIDIA's Audex recipes assume vLLM
continuous batching, so do not optimize the current direct `mlx_lm` loop as the
main path unless the vLLM Metal path is proven impossible.

## Goal

Make Audex-Mac a usable local Mac speech-to-speech prototype by restoring vLLM
Metal as the primary inference engine for ASR, text reasoning, and TTS
speech-token generation, while keeping the Audex causal speech decoder and
audio plumbing MLX/Metal-native as needed.

## Success Criteria

- `./start.sh` launches a persistent vLLM-Metal-backed Audex session by default.
- The selected model is 30B if cached, otherwise 2B, otherwise download 2B,
  preserving the existing UX.
- vLLM Metal generation is proven to run on MLX/Metal, not accidental CPU
  fallback.
- A diagnostic command records enough evidence to explain vLLM Metal device
  selection, spawned worker environment, patch installation, model adapter
  selection, and timing.
- The STS path follows NVIDIA's same-checkpoint cascade: Audex ASR/audio-QA
  request -> Audex text response request -> Audex TTS speech-token request ->
  Audex causal speech decoder playback.
- CFG TTS uses vLLM-style paired requests/continuous batching where supported,
  not serialized direct `mlx_lm` CFG.
- `docs/engineering/patches.md` is updated as the source of truth for every monkey patch,
  upstream symbol, guard, and reapply note.
- Existing direct MLX code may remain as an explicit diagnostic fallback, but it
  must not be the default STS engine and must not silently mask vLLM Metal
  failures.

## Constraints

- Use Metal/MLX. Do not accept CPU inference as "good enough."
- Do not introduce Whisper, Silero, cloud APIs, or other model stages.
- Respect NVIDIA's Audex model card and paper recipes for prompts and sampler
  settings.
- Do not invent sampler parameters.
- No thinking by default for STS.
- No silent fallback paths. If vLLM Metal cannot run a requested path, say
  exactly why and fail loudly unless the user explicitly selected a diagnostic
  fallback.
- Keep changes small and verifiable.
- Commit only when asked.
- Never use `--no-verify`.

## Primary Resources

- `docs/engineering/patches.md`: patch ledger and first-class research target.
- `vendor_pins.json`: pinned vLLM Metal commit.
- `start.sh`: vLLM Metal runtime bootstrap, environment propagation, generated
  venv patch install.
- `audex_mac/patches/runtime.py`
- `audex_mac/patches/install.py`
- `audex_mac/patch_guards.py`
- `audex_mac/metal_policy.py`
- `audex_mac/text_generation.py`
- `audex_mac/sts_cli.py`
- `audex_mac/audio_contract.py`
- `audex_mac/audio_encoder.py`
- `audex_mac/audio_projector.py`
- `audex_mac/audio_splice.py`
- `audex_mac/speech_decoder.py`
- `.audex/vendor/vllm-metal/`
- NVIDIA reference:
  `inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py`
- NVIDIA reference:
  `inference_scripts_vllm/unified_s2s_scripts/run_cascaded_s2s_web.sh`

## Phase 1: Diagnose vLLM Metal CPU Fallback

Objective: produce a factual explanation for why vLLM Metal is slow or
CPU-bound before changing architecture.

Tasks:

- Read `docs/engineering/patches.md` and classify each listed patch as implemented, stale,
  incomplete, or merely aspirational.
- Inspect the pinned vLLM Metal source under `.audex/vendor/vllm-metal` at the
  exact symbols named in `docs/engineering/patches.md`.
- Add or run a focused `--diagnose-vllm-metal` style path that records:
  - parent process environment: `VLLM_METAL_USE_MLX`, `VLLM_MLX_DEVICE`,
    `VLLM_METAL_USE_PAGED_ATTENTION`
  - spawned `EngineCore` environment
  - `mlx.core.default_device()`
  - `mx.metal.is_available()`
  - vLLM platform plugin selected
  - vLLM Metal model adapter selected
  - whether paged attention is enabled
  - whether model weights are materialized on GPU-backed MLX arrays
  - whether any code path explicitly uses `mx.cpu`, torch CPU, or vLLM CPU
    fallback
  - first-token latency and tokens/sec for a tiny text prompt and a 4096-token
    benchmark prompt
- Determine whether the earlier "CPU fallback" was:
  - true CPU execution,
  - vLLM's compatibility `device_config=cpu` facade while MLX uses GPU,
  - spawned worker environment not inheriting Metal settings,
  - model loader fallback because Audex model registration failed,
  - paged-attention unsupported/fallback,
  - safetensors/load materialization on CPU followed by missing transfer/eval,
  - or serialized CFG/direct MLX work being mistaken for vLLM execution.

Verification:

- A JSON diagnostic report under `.audex/runs/`.
- A `docs/engineering/patches.md` section titled something like "vLLM Metal CPU fallback
  investigation" with evidence and conclusion.
- Targeted tests for any new diagnostic parser or guard.
- `scripts/lint.sh`.

Stopping condition:

Do not rewrite STS around vLLM Metal until the fallback/slow-path cause is known
or explicitly documented as blocked by upstream.

## Phase 2: Restore vLLM Metal Text Generation

Objective: make text-only Audex inference go through a persistent vLLM Metal
engine with Metal evidence.

Tasks:

- Replace "direct `mlx_lm` by default" for text with a vLLM-Metal-backed session
  path.
- Keep `--text-backend mlx` only as an explicit diagnostic fallback.
- Ensure Audex model registration and generated venv shims work in spawned
  `EngineCore` workers.
- Align prompt construction with NVIDIA's text-only/non-thinking mode.
- Benchmark 2B and 30B if cached with `max_tokens >= 4096`.

Verification:

- `./start.sh --run-text-benchmark --model audex-2b --text-backend vllm`
- The same benchmark for 30B if cached.
- Report first-token latency, total latency, tokens/sec, MLX device, and vLLM
  Metal worker evidence.
- Do not use Activity Monitor alone as evidence; logs must support the
  conclusion.

## Phase 3: Port NVIDIA Unified S2S Shape Into CLI

Objective: stop treating STS as ad hoc direct MLX calls and mirror NVIDIA's
vLLM structure.

Tasks:

- Build an `AudexVllmRuntime` or equivalent that owns one persistent vLLM Metal
  engine.
- Own the vLLM Metal Audex adapter patch in this repository. Audex is newer than
  the pinned vLLM Metal adapter allowlist; missing upstream support is not a
  reason to leave the runtime on direct MLX.
- Implement request builders matching NVIDIA:
  - ASR/audio-QA prompt with `<sound>` expansion.
  - text reasoning prompt with non-thinking default.
  - TTS prompt ending with `<speechgen_start>`.
- Reuse existing MLX audio preprocessing/encoder/projector only where vLLM
  Metal cannot own multimodal processing yet.
- Prefer the vLLM multimodal data path if vLLM Metal can accept Audex audio
  embeddings cleanly after patching.
- Stream generated speech codec token IDs out of vLLM as they arrive.

Verification:

- One-turn fixture STS with `say`/`afconvert` generated WAV.
- JSON run log includes ASR timing, text timing, TTS first-token timing, first
  decoded PCM timing, and playback diagnostics.
- Response text prints alongside generated speech tokens.

## Phase 4: Implement vLLM-Style CFG TTS

Objective: match NVIDIA's continuous-batching assumption for TTS.

Tasks:

- Study NVIDIA's `_stream_tts_frames_with_cfg` in
  `cascaded_s2s_web_server.py`.
- Implement paired conditional/unconditional TTS requests through vLLM Metal if
  supported.
- Ensure both requests run through the same persistent engine scheduler.
- Keep NVIDIA sampler settings and CFG scale from the model card/reference.
- If vLLM Metal lacks required logits-processor or `extra_args` support, patch
  it and document the patch in `docs/engineering/patches.md`.

Verification:

- TTS smoke generates speech codec tokens with CFG enabled.
- Diagnostic log proves paired requests are submitted to vLLM Metal, not
  serialized direct `mlx_lm`.
- Compare latency against direct MLX CFG to confirm the expected improvement.

Checkpoint:

- Audex-Mac now imports NVIDIA's `CFGLogitsProcessor` and `vllm_cfg_patch`
  directly from the selected Hugging Face snapshot instead of copying
  Apache-licensed NVIDIA code into the MIT source tree.
- `AudexVllmRuntime.from_model_path` mirrors NVIDIA's engine configuration when
  `NVIDIA_TTS_CFG_SCALE > 1.0`: prefix caching disabled, CFG logits processor
  installed, and `max_model_len`, `max_num_batched_tokens`, and `max_num_seqs`
  set from the NVIDIA recipe with a local Mac CLI sequence-budget adjustment.
  It now fails before `LLM(...)` construction if required CFG wiring is
  unavailable, so default STS cannot silently run unconfigured paired TTS
  requests.
- The default no-CFG interactive path uses a Mac-friendly
  `gpu_memory_utilization=0.60` to avoid unified-memory compression/paging on
  128 GB machines running 30B bf16 plus local audio components. Override with
  `AUDEX_VLLM_GPU_MEMORY_UTILIZATION` for diagnostics or larger KV-cache
  experiments.
- `audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller` fills the
  CUDA-specific gap in NVIDIA's recipe by installing a worker-local vLLM Metal
  sampler patch. The patch copies the conditional sampled token into the
  unconditional CFG row and batches prefill sampling so the first generated
  token can see both CFG rows together.
- `./start.sh --diagnose-vllm-metal` now includes `audex_cfg` evidence and the
  verdict fails if NVIDIA CFG, the Metal token-sync installer, or prefix-cache
  disabling are missing. The probe also applies the vLLM Metal sampler patch and
  records whether `sample_from_logits`, `sample_prefill_tokens`, and model
  runner symbol wiring are patched.
- A GPU-visible diagnostic run proved the earlier Activity Monitor concern was
  not simple vLLM CPU fallback: parent and spawned MLX probes reported
  `Device(gpu, 0)` and Audex-Mac repaired vLLM's effective platform to
  `vllm_metal.platform.MetalPlatform`. The next live blocker was vLLM engine
  construction failing in Transformers remote-code loading because local HF
  snapshot Python files were resolved into `blobs/` before relative imports
  were discovered. Audex-Mac now carries the
  `transformers_local_dynamic_modules` patch in `docs/engineering/patches.md`; the next
  GPU-visible proof must rerun the same STS smoke after that patch.

Next validation:

- Run the diagnostic in a shell with Metal available and confirm `audex_cfg` is
  ready alongside the already repaired `vllm_metal.platform.MetalPlatform`.
- Run a short TTS/STT turn and confirm the worker imports both
  `cfg_logits_processor.CFGLogitsProcessor` and
  `audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller`.
- If generation is still CPU-bound after CFG is wired, inspect the vLLM Metal
  model forward path next: adapter selection, multimodal embedding merge, and
  paged attention backend timing.

## Phase 5: Fix Playback After Runtime Architecture

Objective: address choppy audio only after vLLM token streaming is restored.

Tasks:

- Decode speech frames as vLLM emits them, using NVIDIA's causal decoder session
  pattern.
- Use NVIDIA's default decoder chunk size as the baseline, then tune only with
  evidence.
- Remove per-chunk WAV writes from the hot playback loop; write final artifacts
  after playback or on a background artifact path.
- Preserve buffer underrun/overrun diagnostics.
- Keep startup greeting and answer playback on the same audio transport.

Verification:

- Playback diagnostics show near-zero queue underruns/device underflows for a
  normal answer.
- First playable PCM starts before full TTS completion.
- Final WAV and per-turn JSON log still exist.

## Phase 6: Behavioral And Regression Coverage

Objective: make the vLLM Metal architecture hard to regress.

Tasks:

- Add Gherkin/Cucumber scenarios for:
  - startup selects cached 30B over 2B
  - startup downloads 2B when neither is cached
  - vLLM Metal pin is honored
  - upstream vLLM Metal `HEAD` drift warns but does not block
  - missing patch guard fails loudly
  - CPU/MLX policy violation fails loudly
  - STS uses vLLM Metal by default
  - direct MLX requires an explicit diagnostic flag
  - TTS CFG uses paired vLLM requests when available
  - run logs contain device, timing, and playback evidence
- Keep unit tests close to changed modules.

Validation:

- `scripts/lint.sh`
- Focused pytest for patches/startup/runtime.
- Full pytest before final handoff.
- One manual fixture STS smoke if local model cache is available.

## Current Execution Notes

As of the latest local pass, Phase 1 has an executable diagnostic path:

- `./start.sh --diagnose-vllm-metal`
- `./start.sh --diagnose-vllm-generation --diagnose-generation-max-tokens 4096`

The diagnostic isolates vLLM/MLX probes into subprocesses, records raw vs
Audex-repaired `vllm.platforms.current_platform`, and writes JSON reports under
`.audex/runs/`. In the Codex execution shell, MLX reports no accessible Metal
device, so these commands are expected to exit nonzero there. A normal
GPU-visible Mac terminal must be used to prove Phase 2 readiness.

Latest evidence from the Codex shell:

- raw vLLM platform: `vllm.platforms.cpu.CpuPlatform`
- after Audex patches: `vllm_metal.platform.MetalPlatform`
- Audex runtime patch report: all fields true in a fresh generated-venv process
- model adapter: Audex patch installed, Audex adapter selected, projected
  embedding path forward-ready
- text benchmark backend policy: default is vLLM Metal; `--text-backend mlx`
  exists only as an explicit diagnostic fallback
- vLLM text benchmark run logs now include `engine`, `model_load_seconds`,
  `audex_patches`, per-turn generated token counts, per-turn throughput, and
  finish/stop reasons
- generation probe: subprocess-isolated, `max_tokens=4096`, currently failing
  before model load only because this shell cannot access a Metal device
- Phase 3 request-building checkpoint: `audex_mac/vllm_sts_requests.py` now
  mirrors NVIDIA's cascaded vLLM ASR/text/TTS request shapes, including paired
  CFG TTS request metadata.
- Phase 3 persistent-runtime checkpoint: `audex_mac/vllm_runtime.py` now owns
  one `vllm.LLM` engine, runs ASR/text/TTS through NVIDIA-shaped requests, and
  submits the TTS CFG conditional/unconditional pair in a single engine call.
  Fast unit and BDD coverage prove the runtime reuses the same fake engine for
  all three stages. It also maps vLLM TTS token IDs back to Audex speech codec
  frames for the decoder handoff. The live CLI now defaults to this vLLM
  runtime for fixture and interactive STS; the older direct MLX session requires
  explicit `--sts-backend mlx`.
- Phase 3 conversation-context checkpoint: the vLLM STS text stage now builds
  the response prompt from the persisted chat messages plus the latest
  transcript, instead of sending only the latest transcript. The session then
  persists exactly that user/assistant turn, so follow-up turns can reference
  earlier user and assistant messages through the normal chat template until the
  configured context limit is reached.
- Phase 3 projected-audio checkpoint: the default vLLM CLI path now projects
  input WAV audio through Audex-Mac's MLX audio encoder/projector and submits
  `audex_projected_embeddings` to vLLM `multi_modal_data`, instead of sending
  raw PCM into an unproven generic vLLM audio processor. The Audex vllm-metal
  adapter already accepts that payload and validates projected embedding counts.
- Phase 3 feature-spec checkpoint:
  `audex_mac.patches.vllm_metal_audex_adapter.build_projected_audio_feature_spec`
  builds the exact vLLM `MultiModalFeatureSpec` shape needed for Audex
  projected audio, including a `PlaceholderRange` over a contiguous
  `<so_embedding>` token run whose length matches the projected embedding
  count.
- Phase 3 processor-registration checkpoint:
  `audex_mac.patches.vllm_metal_audex_adapter.patch_vllm_audex_processor`
  marks the Nemotron proxy class as multimodal and registers
  `AudexProjectedAudioProcessor` with vLLM's `MULTIMODAL_REGISTRY`. The
  processor parses `audex_projected_embeddings`, emits vLLM `mm_input(...)`
  with `mm_kwargs`, `mm_hashes`, and `mm_placeholders`, and therefore lets
  vLLM's engine input processor build the `MultiModalFeatureSpec` list that
  vllm-metal's paged runner consumes.
- Remaining Phase 3 live blocker: a GPU-visible terminal must prove pinned
  vLLM Metal invokes this registered Audex processor and reaches the vllm-metal
  paged multimodal splice path with real projected audio embeddings.
- Remaining Phase 5 blocker: vLLM TTS playback is synchronous in the current
  CLI handoff. It now maps generated vLLM token IDs to codec frames and decodes
  those frames through `AudexSpeechDecoderSession` into chunk artifacts, but
  vLLM token generation still completes before the first decoder chunk can be
  emitted.
- Latest diagnostic after the CLI backend flip:
  `.audex/runs/vllm-metal-diagnostic-20260707-221840.json`. In this Codex
  shell it exits `2` because MLX cannot access a Metal device, but Audex patch
  installation, adapter selection, adapter forward-readiness, and the repaired
  `vllm_metal.platform.MetalPlatform` are still intact.
- Latest diagnostic after routing default STS through projected embeddings:
  `.audex/runs/vllm-metal-diagnostic-20260707-222446.json`. It has the same
  expected Codex-shell Metal visibility failure, while preserving all Audex
  patch/adapter/platform-repair evidence.
- Latest diagnostic after adding the projected-audio feature-spec helper:
  `.audex/runs/vllm-metal-diagnostic-20260707-225756.json`. It has the same
  expected Codex-shell Metal visibility failure, while preserving all Audex
  patch/adapter/platform-repair evidence.
- Latest code checkpoint after registering the Audex projected-audio vLLM
  processor: focused tests
  `.venv/bin/python -m pytest tests/test_audex_patches.py tests/test_vllm_sts_requests.py tests/test_vllm_runtime.py -q`
  pass with `29 passed`, and `./scripts/lint.sh` passes. Diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260707-231623.json` shows
  `audex_patches` all true, Audex adapter selected/forward-ready, repaired
  platform `vllm_metal.platform.MetalPlatform`, and `audex_processor.ready=True`
  with multimodal output carrying `audex_projected_embeddings` plus a
  placeholder range over the expected `<so_embedding>` run. It still exits `2`
  in the Codex shell because MLX cannot access a Metal device there. A
  diagnostic from a GPU-visible terminal is still required.
- Latest code checkpoint after adding vLLM CFG wiring:
  `./scripts/lint.sh` passes, full pytest passes with
  `209 passed, 3 skipped`, and diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260707-232808.json` shows
  `audex_cfg.ready=True` with `cfg_logits_processor.CFGLogitsProcessor`,
  `audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller`,
  `enable_prefix_caching=False`, `max_model_len=5120`,
  `max_num_batched_tokens=8192`, and `max_num_seqs=8` in current local
  interactive builds. Older diagnostics used NVIDIA's more server-oriented
  `max_num_batched_tokens=10240` and `max_num_seqs=128`, which caused
  excessive unified-memory pressure on 30B human STS turns. The only diagnostic
  verdict failure remains the Codex shell's missing Metal device.
- Latest code checkpoint after adding fail-loud CFG enforcement and sampler
  patch diagnostics: `./scripts/lint.sh` passes, full pytest passes with
  `213 passed, 3 skipped`, and diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260707-233901.json` shows
  `audex_cfg.ready=True`, both CFG logits processors in engine kwargs,
  NVIDIA's CFG engine sizing, and `vllm_metal_patch.ready=False` only because
  importing the deep vLLM Metal model-runner path hits the same Codex-shell
  `[metal::load_device] No Metal device available` failure as the root MLX
  probe. The diagnostic verdict now reports only that root Metal visibility
  failure; a GPU-visible terminal still needs to prove
  `audex_cfg.vllm_metal_patch.ready=True`.
- Latest code checkpoint after wiring vLLM STS conversation context:
  `./scripts/lint.sh` passes and full pytest passes with
  `215 passed, 3 skipped`. The covered behavior proves
  `build_text_messages_response_request`, `generate_text_response_from_messages`,
  and `VllmSpeechToSpeechSession.run_turn_from_wav` preserve prior messages in
  the vLLM text prompt.
- Latest code checkpoint after adding vLLM decoder chunking:
  `./scripts/lint.sh` passes and full pytest passes with
  `215 passed, 3 skipped`. The speech-output log now records
  `decoder_streaming=True`, `vllm_token_streaming=False`,
  `decoder_chunk_frames`, and per-chunk WAV paths, so the playback side is
  prepared for a future vLLM token stream without falsely claiming
  first-audio-before-generation-complete.
- Latest code checkpoint after inspecting the pinned streaming API:
  diagnostic report `.audex/runs/vllm-metal-diagnostic-20260708-002915.json`
  records `vllm_streaming_api.sync_llm_generate_streams=False`,
  `async_engine_available=True`, and `async_generate_is_asyncgen=True`.
  Therefore the next Phase 5 implementation seam is an AsyncLLMEngine-backed
  runtime path, not trying to make the existing sync `LLM.generate` call stream.
- Latest code checkpoint after adding the async runtime skeleton:
  `AudexAsyncVllmRuntime` can construct an `AsyncLLMEngine` from shared engine
  kwargs/CFG config, stream cumulative `RequestOutput` values as token deltas,
  and submit TTS CFG cond/uncond requests concurrently while yielding speech
  codec frames from the conditional stream. Focused fake-engine coverage passes
  with `.venv/bin/python -m pytest tests/test_vllm_runtime.py -q`; full pytest
  passes with `219 passed, 3 skipped`. Diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-003450.json` still reports only
  the expected Codex-shell Metal-device failure. The default CLI has not been
  switched to this async runtime yet; the next seam is wiring
  `stream_tts_cfg_codec_frames` into `VllmSpeechToSpeechSession` playback and
  proving it in a GPU-visible terminal.
- Latest code checkpoint after wiring the optional async TTS stream into the
  STS decoder path: `VllmSpeechToSpeechSession` accepts an optional
  `async_runtime`; when present, `generate_speech_output` consumes
  `stream_tts_cfg_codec_frames`, decodes new codec-frame chunks through
  `AudexSpeechDecoderSession`, and logs `streaming=True`,
  `vllm_token_streaming=True`, `decoder_streaming=True`, and per-chunk WAV
  paths. Without an async runtime, the sync vLLM path remains explicit and logs
  `vllm_token_streaming=False`. Full pytest passes with
  `220 passed, 3 skipped`; diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-003830.json` still reports only
  the expected Codex-shell Metal-device failure while preserving async API
  evidence. The next seam is making async vLLM the default session runtime
  without loading two engines, then proving the path in a GPU-visible terminal.
- Latest code checkpoint after making async vLLM the default session runtime:
  `VllmSpeechToSpeechSession` now constructs `AudexAsyncVllmRuntime` when no
  explicit runtime is injected, uses async final-result helpers for ASR and text
  response generation, and uses the async TTS stream for speech output. The
  synchronous `AudexVllmRuntime` path remains available only when deliberately
  injected for tests/diagnostics. Focused vLLM tests pass with
  `36 passed`; `./scripts/lint.sh` passes; full pytest passes with
  `225 passed, 3 skipped`. Diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-004534.json` shows
  `vllm_metal_audex_adapter=True`, `audex_processor.ready=True`,
  `audex_cfg.ready=True`, `vllm_streaming_api.async_engine_available=True`, and
  `vllm_streaming_api.async_generate_is_asyncgen=True`; the only verdict
  failure remains the Codex-shell `[metal::load_device] No Metal device
  available` root probe. The remaining proof is a GPU-visible terminal run that
  demonstrates vLLM Metal `EngineCore` actually executes the default async path
  on Metal with acceptable first-audio latency.
- Latest diagnostic checkpoint after adding the default STS smoke probe:
  `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke` now runs an
  opt-in subprocess probe of the same default async speech-to-speech fixture
  path used by the CLI, generating a one-second silence WAV if no fixture is
  supplied. The JSON report records `sts_probe` with run-log paths, engine
  class, turn timings, first-audio timing, generated codec-frame counts, and
  whether vLLM token streaming plus decoder streaming were observed. In the
  Codex shell, diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-005139.json` exits `2` with
  `sts_probe.enabled=True` and `sts_probe.ready=False` for the same
  `[metal::load_device] No Metal device available` root failure as the parent
  MLX probe. Validation passes: `./scripts/lint.sh` and full pytest
  `227 passed, 3 skipped`. The remaining proof is to run that exact command in
  a GPU-visible terminal and require `sts_probe.ready=True`,
  `speech_streaming.vllm_token_streaming=True`, and
  `speech_streaming.first_audio_ready_seconds` within the usable-prototype
  target.
- Latest correction to the STS smoke diagnostic: `--diagnose-vllm-sts-smoke`
  now implies `--diagnose-vllm-metal` but forces speech readiness, so automatic
  model selection and explicit cache checks use `checkpoint_folder_full`
  artifacts rather than text-only artifacts. Fresh diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-024529.json` selects
  `nvidia/Nemotron-Labs-Audex-30B-A3B`, records primary `model_path` as the
  30B `checkpoint_folder_full`, records `speech_runtime.ready=True`, probes the
  full-checkpoint Audex adapter shape
  `NemotronHAudexForConditionalGeneration`/`nemotron_h_audex`, and locates the
  30B `inference_scripts_vllm/audiogen_scripts` CFG assets. It still exits `2`
  here only because the Codex shell cannot open a Metal device. Validation
  passes: focused diagnostic/startup tests `22 passed`, `./scripts/lint.sh`,
  and full pytest `229 passed, 3 skipped`.
- Latest correction to the STS smoke acceptance gate: a ready
  `--diagnose-vllm-sts-smoke` report must now prove more than subprocess
  success. The diagnostic verdict rejects a ready STS smoke unless
  `engine_class` identifies an async vLLM engine,
  `speech_streaming.vllm_token_streaming=True`,
  `speech_streaming.decoder_streaming=True`,
  `speech_streaming.first_audio_ready_seconds` is present, and generated speech
  token count, codec-frame count, and decoder chunk count are all nonzero. This
  prevents the GPU-visible proof command from passing with a sync fallback,
  empty TTS output, or a no-audio decoder path. Validation passes:
  `tests/test_vllm_diagnostics.py` `17 passed`, `./scripts/lint.sh`, and full
  pytest `231 passed, 3 skipped`.
- Latest runtime correction for audible streaming: the default async vLLM TTS
  path now enqueues each decoded waveform chunk into the existing
  `_ContinuousPcmPlayer` when `play=True` instead of waiting for the final WAV
  and then calling `afplay`. The vLLM speech-output run log now records
  `playback_transport`, `playback_prebuffer_seconds`,
  `first_playback_started_seconds`, and `playback_diagnostics`; the STS smoke
  report forwards those fields when playback is enabled. This closes the gap
  where `vllm_token_streaming=True` and `decoder_streaming=True` could still
  mean final-WAV-only audible playback. Validation passes:
  `tests/test_vllm_sts_cli.py tests/test_vllm_diagnostics.py` `21 passed`,
  `./scripts/lint.sh`, and full pytest `232 passed, 3 skipped`. Fresh
  no-Metal diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-025141.json` still selects the
  cached 30B full checkpoint and records `speech_runtime.ready=True` before
  failing at the Codex-shell Metal visibility boundary.
- Latest playback-proof diagnostic addition: `--diagnose-vllm-sts-play` now
  implies `--diagnose-vllm-sts-smoke`, passes `play=True` into the default async
  vLLM fixture turn, and requires continuous playback evidence when the STS
  probe succeeds. A GPU-visible audible proof should run
  `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
  and require `speech_streaming.playback_transport` to equal
  `sounddevice_raw_output_stream` plus
  `speech_streaming.first_playback_started_seconds` and
  `speech_streaming.playback_diagnostics` to be present. Playback diagnostics
  must include device-underflow, queue-underrun, queue-overrun, and
  chunks-written counters. The silent proof command remains useful for
  token/decoder timing and records
  `sts_probe.play_audio=false`. Validation passes:
  `tests/test_vllm_diagnostics.py tests/test_start_sh.py` `27 passed`,
  `./scripts/lint.sh`, and full pytest `235 passed, 3 skipped`. Fresh silent
  no-Metal diagnostic report
  `.audex/runs/vllm-metal-diagnostic-20260708-025504.json` records
  `play_audio=false`, selected cached 30B full speech assets, and fails only at
  the Codex-shell Metal visibility boundary.
- Latest vLLM Metal adapter recovery: Audex-Mac now supplies the missing
  Audex adapter surface required by the pinned vLLM Metal runtime. The patch
  recognizes Audex model types/architectures, registers Audex projected-audio
  processing for dense 2B and Nemotron-H 30B, attaches the multimodal adapter
  after text-backbone loading, emits real `MultiModalKwargsItems`, converts
  MLX projected audio arrays to torch CPU tensors only for vLLM IPC transport,
  and converts them back to MLX inside EngineCore. The Nemotron-H path also
  patches vLLM Metal's attention graph walker so full-attention `mixer` layers
  are wrapped while Mamba/MLP/MoE mixers are not, normalizes
  `NemotronHAttention` to the SDPA wrapper contract, and bypasses RoPE for that
  no-RoPE attention class.
- Latest STS runtime correction: async vLLM sessions now run the whole
  ASR -> text -> TTS cascade in one event loop. The previous implementation
  called `asyncio.run(...)` separately for each stage, which let ASR complete
  and then left the persistent `AsyncLLMEngine` dead before the text request.
  One-off fixture sessions now call the engine's `shutdown(timeout=5.0)` so
  diagnostics exit after a successful JSON result instead of waiting for a pipe
  timeout.
- Latest GPU-visible validation:
  - `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=120 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
    passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-075120.json`.
    Evidence: `ready=True`, `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    parent and spawned MLX probes on `Device(gpu, 0)`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `first_audio_ready_seconds=0.612`, and `generated_codec_frame_count=56`.
  - `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
    selected the cached 30B model and passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-075518.json`.
    Evidence: `ready=True`, `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    parent and spawned MLX probes on `Device(gpu, 0)`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `first_audio_ready_seconds=4.53`, and `generated_codec_frame_count=206`.
- Latest audible playback validation:
  - `--diagnose-vllm-sts-play` now bounds diagnostic speech generation with
    `AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS` or a default of `256`, while silent STS
    smoke keeps the normal speech budget. This prevents playback proof from
    becoming an accidentally unconstrained TTS demo.
  - The async vLLM streaming TTS path no longer writes per-chunk WAV files in
    the hot loop. It records `decoded_chunk_count` and writes the final WAV at
    the end.
  - vLLM Metal's upstream sampler only avoids the Torch bridge for greedy
    sampling. Audex TTS uses NVIDIA's temperature+CFG path, so Audex-Mac now
    carries a narrow native MLX CFG sampling fast path for complete CFG pairs
    with no penalties, no top-p/top-k filtering, no logprobs, no token
    constraints, and only inert builtin logits processors.
  - The native CFG path samples each complete conditional/unconditional pair
    once from the blended conditional row, then expands that token ID into both
    vLLM request slots. This preserves the existing CFG token-sync semantics
    while avoiding a duplicate categorical draw over the full Audex vocabulary
    for the unconditional row.
  - The bounded EngineCore timing line now includes cumulative
    `native_sampled_rows` and `native_output_rows`; for ordinary two-request CFG
    TTS, the derived `sts_probe.sts_timing_assessment.native_sampling_row_ratio`
    should be about `0.5`, proving the sampler is drawing once per CFG pair.
    If a ready STS smoke report includes a row ratio above `0.75`, the
    diagnostic evidence gate now fails because duplicate CFG row sampling has
    regressed.
  - The STS evidence gate now requires measured speech-token throughput. It
    fails if `sts_probe.sts_timing_assessment.codec_frames_per_second` is
    missing, or if `sts_probe.sts_timing_assessment.below_realtime=True`;
    producing streamed audio is no longer enough to mark the vLLM Metal path
    ready.
  - `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=180 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    passed before the native sampler patch with report
    `.audex/runs/vllm-metal-diagnostic-20260708-082713.json`; it generated
    56 codec frames with `last_codec_frame_seconds=14.152`, which explains the
    late 0.8 second prebuffer fill and underruns.
  - The same audible diagnostic passed after the native sampler patch with
    report `.audex/runs/vllm-metal-diagnostic-20260708-083913.json`.
    Evidence: `ready=True`, parent and spawned MLX probes on
    `Device(gpu, 0)`, EngineCore stderr
    `Audex vLLM Metal: native MLX sampling fast path used 1 time(s)`,
    `playback_transport=sounddevice_raw_output_stream`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `first_audio_ready_seconds=1.505`, `first_playback_started_seconds=4.818`,
    `last_codec_frame_seconds=4.702`, `generated_codec_frame_count=38`,
    `chunk_count=8`, `device_underflow_count=0`, and
    `queue_underrun_count=0`.
  - This proves the vLLM Metal audible streaming transport, the non-disk decoder
    hot path, and native MLX CFG sampling activation. It still does not prove
    full realtime speech generation for longer utterances; 38 codec frames in
    4.7 seconds remains well below the 50 codec frames/sec audio rate.
  - Latest instrumentation checkpoint: the debug-only vLLM Metal patch now times
    `mlx.core.eval` calls while `MetalModelRunner._sample_paged_batch` is active
    and reports `mx_eval_ms=logits:milliseconds/count,sample_tokens:...` in the
    bounded EngineCore timing lines. This should make the next normal-answer
    fixture run distinguish full-logit/model evaluation cost from native
    categorical sampling cost. Focused coverage:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q` passes with
    `11 passed`.
  - Latest diagnostic ergonomics checkpoint: `AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS`
    now bounds silent STS smoke too, so the next GPU-visible timing run can avoid
    playback/audio-device variability:
    `scripts/create-test-utterance.sh --basename normal-answer-question --text "Please explain Python context managers in two concise sentences."`
    followed by
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`.
    The report now parses EngineCore timing into
    `sts_probe.vllm_metal_timing.latest_paged_sample`, and the CLI prints a
    `vLLM Metal TTS timing:` summary with native sampled/output row counts when
    those fields exist.
    The report also derives `sts_probe.sts_timing_assessment` with
    `codec_frames_per_second`, `audio_realtime_ratio`, playback glitch count,
    paged/native/eval milliseconds per step, native sampled/output row counts,
    dominant `mx_eval` category, nonpaged TTS-window decode counters/native
    detail timings, and `likely_bottleneck`; use that structured assessment as
    the next optimization gate.
    A fresh Codex-shell attempt wrote
    `.audex/runs/vllm-metal-diagnostic-20260708-090734.json` and failed before
    EngineCore timing only because this execution context still cannot open a
    Metal device.
  - NVIDIA reference check on 2026-07-08: the tempting RVQ logit-mask path is
    for `audiogen_scripts/run_audio_gen_vllm_rvq_logit_mask.py` and
    `<audiocodec_*>` TTA generation, not unified S2S TTS. The unified S2S
    `_tts_sampling_args` uses `max_tokens`, `temperature`, `top_p`, optional
    `top_k`, and stop IDs containing `<speechgen_end>`/EOS; its CFG path adds
    `cfg_scale`, `cfg_role`, and `cfg_pair_id` in `extra_args`. Do not add vLLM
    `allowed_token_ids` or an RVQ phase mask to S2S TTS unless NVIDIA changes
    the unified recipe. Audex-Mac does pass Audex-specific codec-window metadata
    in `extra_args` so the native MLX sampler can compact sampling to the legal
    speech-token domain without changing public vLLM sampling parameters.
  - Latest completion-gate run:
    `.audex/runs/vllm-metal-diagnostic-20260708-094230.json` proved the current
    bounded normal-answer gate reaches Metal and async vLLM, but fails the
    realtime requirement: parent/spawn MLX both `Device(gpu, 0)`,
    `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `generated_codec_frame_count=240`, and
    `codec_frames_per_second=3.947`/`audio_realtime_ratio=0.079`.
  - Latest CFG sampler hot-path patch avoids full-batch CFG logits
    materialization before native categorical sampling. Instead it builds only
    the row or rows actually sampled, blending complete CFG pairs directly into
    the conditional sample row. The same completion gate then produced
    `.audex/runs/vllm-metal-diagnostic-20260708-094623.json` with
    `codec_frames_per_second=4.614`, `audio_realtime_ratio=0.092`,
    `paged_sample_avg_ms=166.8`, `native_sampling_row_ratio=0.5`,
    `native_sample_ms_per_sampled_row=84.65`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. This is a measurable
    improvement over the previous `3.947 fps`/`216.3 ms` paged average, but the
    gate still fails because realtime requires at least `50 codec frames/sec`.
  - Latest continuous-batching diagnostic: `--diagnose-vllm-tts-batch-size`
    submits multiple independent TTS CFG pairs to one async vLLM Metal engine
    without changing the default CLI conversation path. Batch size 4 produced
    `.audex/runs/vllm-metal-diagnostic-20260708-095413.json` with
    `388` codec frames in `16.07s` (`24.144 aggregate codec fps`). Batch size 8
    produced `.audex/runs/vllm-metal-diagnostic-20260708-095545.json` with
    `900` codec frames in `12.54s` (`71.77 aggregate codec fps`), proving vLLM
    Metal can generate Audex speech tokens faster than realtime when continuous
    batching is sufficiently occupied. Per single conversation, this does not
    satisfy the completion gate: the batch-8 run is about `8.97 codec fps` per
    conditional stream if divided evenly across eight prompts. The next
    optimization target is therefore single-stream/small-batch utilization, not
    Metal enablement.
  - Latest no-CFG diagnostic:
    `--diagnose-vllm-tts-batch-no-cfg` is available only to isolate overhead; it
    is not the default and is not NVIDIA's CFG recipe. A single no-CFG TTS
    request produced
    `.audex/runs/vllm-metal-diagnostic-20260708-100224.json` with
    `request_count=1`, `decode_reqs=1`, `decode_tokens=1`,
    `total_codec_frame_count=189`, and `codec_frames_per_second=2.169`. That is
    slower than the one-pair CFG completion gate, so the next target is not
    "remove CFG"; it is the small-batch vLLM Metal decode/logits path.
  - MLX categorical microbench in the generated vLLM Metal venv on
    `Device(gpu, 0)` measured raw `mx.random.categorical` over a
    `(1, 205312)` logits row at about `0.47 ms`, and `(8, 205312)` at about
    `0.87 ms` total. Therefore the tens-of-milliseconds `sample_tokens`
    timings in the vLLM reports are not the categorical kernel alone; they are
    mostly the MLX lazy graph being forced at the sampling boundary. The
    diagnostic now labels that case
    `pending_graph_eval_during_sampling`.
    A small safe sampler cleanup replaced temperature division with reciprocal
    multiplication; the same microbench measured roughly `0.37 ms` for the
    single-row categorical shape, so any remaining hundreds-of-milliseconds
    timing is still graph synchronization rather than scalar temperature math.
  - The shortcut that skips vLLM Metal's explicit full-logits `mx.eval(...)`
    during Audex CFG decode is now enabled by default when the patch can prove
    the batch is a safe CFG decode shape. Set
    `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=0` to disable it for diagnostics. The
    earlier opt-in run
    `.audex/runs/vllm-metal-diagnostic-20260708-101226.json` skipped `137`
    full-logits evals but still failed the gate at
    `codec_frames_per_second=2.855`; after the memory-policy and compact
    speech-token sampler fixes, the same safe skip is part of the passing path.
  - Restored default completion-gate run:
    `.audex/runs/vllm-metal-diagnostic-20260708-101955.json` confirms the
    default path does not enable the skip experiment
    (`skipped_logits_eval=0`) and has Metal/MLX evidence (`Device(gpu, 0)` in
    parent and spawned probes), async vLLM
    `vllm.v1.engine.async_llm.AsyncLLM`, vLLM token streaming, decoder
    streaming, and nonzero speech output. It still fails the readiness gate on
    throughput: 240 codec frames over 141.93 seconds
    (`codec_frames_per_second=1.691`, `audio_realtime_ratio=0.034`). Timing
    still points at the same small-batch boundary:
    `mx_eval_ms logits=75358.2/328`, `sample_tokens=40027.3/137`,
    `native_detail_ms sample_eval=40031.3/137`, and
    `likely_bottleneck=pending_graph_eval_during_sampling`. Treat this as the
    next runtime/adapter problem: reduce small-batch decode graph pressure and
    sampling-bound graph evaluation, not CPU fallback.
  - Earlier local validation after documenting the skip experiment and
    restoring that checkpoint's default no-skip behavior:
    focused pytest
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py tests/test_vllm_diagnostics.py -q`
    passes with `50 passed`; `./scripts/lint.sh` passes; full pytest
    `.venv/bin/python -m pytest -q` passes with
    `276 passed, 3 skipped`.
  - Post-cleanup live rerun:
    `.audex/runs/vllm-metal-diagnostic-20260708-102758.json` did not reach
    STS inference. Parent/spawn patch probes still showed the Audex patches and
    adapter selected, but `EngineCore` aborted during initialization with
    `[METAL] Command buffer execution failed: Insufficient Memory`. Do not use
    this report as throughput evidence for the reciprocal-temperature sampler
    cleanup; the last valid throughput report remains
    `.audex/runs/vllm-metal-diagnostic-20260708-101955.json`.
  - Historical memory-fraction diagnostic:
    `VLLM_METAL_MEMORY_FRACTION=0.85` matched NVIDIA's reference `GMU=0.85`,
    but this is stale for the current non-paged MLX KV cache path. Current
    pinned vLLM-Metal rejects explicit memory fractions when
    `VLLM_METAL_USE_PAGED_ATTENTION=0` and requires
    `VLLM_METAL_MEMORY_FRACTION=auto`. The old run wrote
    `.audex/runs/vllm-metal-diagnostic-20260708-102437.json` with
    `audex_cfg.max_num_seqs=128` and completed instead of OOMing, but remained
    slower than realtime: `codec_frames_per_second=1.971`,
    `paged_sample_avg_ms=477.6`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. This is a stability
    diagnostic, not the realtime fix.
  - Diagnostic-only capacity override:
    `AUDEX_VLLM_CFG_MAX_NUM_SEQS=2` is available to test whether NVIDIA's
    128-sequence server capacity setting is the Mac small-batch bottleneck.
    It is not enabled by default. The probe wrote
    `.audex/runs/vllm-metal-diagnostic-20260708-103056.json`, recorded
    `audex_cfg.max_num_seqs=2`, and still failed at
    `codec_frames_per_second=1.827`, `paged_sample_avg_ms=490.2`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. That rules out the
    128-sequence setting as the primary single-stream throughput issue in this
    path.
  - NVIDIA sampler-default correction:
    Audex-Mac had stale TTS defaults in `audex_mac/audio_contract.py`
    (`temperature=0.1`, `top_k=80`, `cfg_scale=1.5`). The cached NVIDIA
    reference scripts for both unified S2S and TTS specify
    `temperature=0.8`, `top_p=1.0`, `top_k=0`, and `cfg_scale=2.0`, so the
    code and tests now use those values. This matters for correctness and also
    removes the native sampler's positive-`top_k` branch from the default TTS
    path.
  - 2026-07-09 correction:
    The no-CFG audible fallback still preserves `temperature=0.8`, `top_p=1.0`,
    and `top_k=0`, but the renewed CFG target now follows the model-card
    audio-generation defaults: `temperature=1.0`, `top_p=1.0`, `top_k=80`,
    `cfg_scale=3.0`, and `cfg_pairs_per_batch=2`. CFG vLLM requests also carry
    the same speech-codec token-window metadata already used by the passing
    no-CFG path.
  - Corrected-sampler completion-gate run:
    `.audex/runs/vllm-metal-diagnostic-20260708-104405.json` used the corrected
    NVIDIA TTS defaults with no skip/materialization/capacity diagnostic env
    enabled. It still failed the realtime gate despite Metal evidence, async
    vLLM, vLLM token streaming, decoder streaming, and 256 generated codec
    frames: `codec_frames_per_second=1.664`, `audio_realtime_ratio=0.033`,
    `paged_sample_avg_ms=490.3`, `native_sampling_row_ratio=0.5`, and
    `likely_bottleneck=pending_graph_eval_during_sampling`. The latest timing
    still points at MLX graph work forced at the small-batch sampling boundary:
    `mx_eval_ms logits=56636.5/328`, `sample_tokens=41079.5/137`, and
    `native_detail_ms sample_eval=41084.7/137`. The wrong sampler defaults were
    a real recipe bug, but not the throughput fix.
  - Final vLLM Metal STS completion-gate run:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-111328.json` and passed
    with `verdict.ready=true`. Evidence: parent and spawned MLX probes both
    report `Device(gpu, 0)`, vllm-metal config reports
    `memory_fraction=0.85`, the engine class is
    `vllm.v1.engine.async_llm.AsyncLLM`, vLLM token streaming and decoder
    streaming are both true, first audio was ready at `0.641s`, and the run
    generated `436` codec frames. Throughput crossed the realtime gate at
    `codec_frames_per_second=82.937` and `audio_realtime_ratio=1.659`; native
    CFG sampling row ratio was `0.501`.
  - The historical passing run depended on three Audex-Mac monkey patches:
    explicit speech-domain sampling, sixteen ordered text segments, and the
    then-working `VLLM_METAL_MEMORY_FRACTION=0.85` setting. Current
    vLLM-Metal requires `VLLM_METAL_MEMORY_FRACTION=auto` for
    `VLLM_METAL_USE_PAGED_ATTENTION=0`, and Audex-Mac now enforces `auto` so
    config construction reaches the fast default path. The native MLX CFG
    sampler samples TTS from the legal Audex speech domain
    `[<speechgen_end>] + [<speechcodec_0>...<speechcodec_N>]` instead of the
    full text/audio vocabulary; and the STS TTS path defaults to sixteen ordered
    text segments so vLLM Metal has enough concurrent CFG pairs to exercise its
    continuous-batching path.
  - Startup TTS warmup was tested and rejected. The reports
    `.audex/runs/vllm-metal-diagnostic-20260708-110753.json` and
    `.audex/runs/vllm-metal-diagnostic-20260708-110919.json` show that both
    multi-pair and single-pair warmup could leave the async `EngineCore` dead
    before ASR. Do not reintroduce TTS warmup without first fixing that engine
    lifecycle issue.
  - Current local validation after the passing gate:
    `./scripts/lint.sh` passes, and full pytest
    `.venv/bin/python -m pytest -q` passes with
    `379 passed, 3 skipped`.
  - 2026-07-09 CFG CLI validation:
    `AUDEX_VLLM_TTS_CFG=1 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    now routes product TTS through `stream_tts_cfg_codec_frames(...)`, disables
    text-to-TTS interleaving for that CFG run, sanitizes code-ish text only for
    the spoken prompt, and writes `tts_cfg_enabled=true`. The generated WAV
    `.audex/runs/speech-output-vllm-20260709-081912.wav` passed the local
    MLX-Audio oracle intelligibility check with ratio `0.9223`. The same JSON shows
    `reached_end_token=true`, `hit_max_tokens=false`, and
    `first_audio_ready_seconds=2.872`, but 30B CFG throughput is still only
    `codec_frames_per_second=7.486` / `audio_realtime_ratio=0.152`. The next
    optimization target remains paired CFG scheduling / continuous batching and
    reducing the small-batch decode/logits boundary cost.
  - 2026-07-09 batched CFG segment validation:
    the product CLI now routes static multi-chunk CFG TTS through
    `stream_tts_cfg_segments_codec_frames(...)`, submitting explicit sanitized
    segments as concurrent CFG pairs with per-segment token budgets. The same
    30B fixture wrote `.audex/runs/speech-output-vllm-20260709-083038.wav` and
    reported `tts_concurrent_segments=true`, `reached_end_token=true`,
    `hit_max_tokens=false`, `stream_finished_seconds=59.156`,
    `codec_frames_per_second=11.985`, and `audio_realtime_ratio=0.243`.
    the MLX-Audio oracle passed with ratio `0.8893`. This improves total CFG throughput
    versus the previous sequential CLI CFG run, but first-audio latency moved
    from `2.872s` to `3.843s`, so the next target is preserving concurrency
    while getting segment-0 audio ready sooner.
  - 2026-07-09 CFG scheduling A/B:
    `stream_tts_cfg_segments_codec_frames(...)` can now prime segment 0 before
    submitting tail CFG pairs; the CLI records this as
    `tts_cfg_prime_first_segment`. On the fixed four-sentence 30B fixture,
    staged submission wrote `.audex/runs/speech-output-vllm-20260709-084828.wav`
    and passed the MLX-Audio oracle with ratio `0.9967`. It reported
    `first_audio_ready_seconds=4.654`, `stream_finished_seconds=64.654`,
    `codec_frames_per_second=11.43`, and `audio_realtime_ratio=0.232`. The
    upfront-batch A/B run `.audex/runs/speech-output-vllm-20260709-084419.json`
    had better first audio at `3.557s`, but worse total throughput:
    `stream_finished_seconds=78.798`, `codec_frames_per_second=9.569`, and
    `audio_realtime_ratio=0.194`.
  - 2026-07-09 CFG scheduling default update:
    a same-day rerun on the current code favored starting all CFG segment pairs
    immediately. With full upfront submission, the four-sentence 30B fixture
    wrote `.audex/runs/speech-output-vllm-20260709-094146.json`, passed
    the MLX-Audio oracle with ratio `0.9982425307557118`, and reported
    `first_audio_ready_seconds=5.39`, `stream_finished_seconds=50.013`,
    `codec_frames_per_second=14.316`, and `audio_realtime_ratio=0.29`. The
    current prime-first comparison wrote
    `.audex/runs/speech-output-vllm-20260709-094337.json`, passed the MLX-Audio oracle
    with ratio `0.9257950530035336`, and reported
    `first_audio_ready_seconds=3.65`, `stream_finished_seconds=65.016`,
    `codec_frames_per_second=11.259`, and `audio_realtime_ratio=0.228`.
    Audex-Mac therefore now defaults to full upfront CFG segment submission for
    better continuous-batching throughput. Set
    `AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT=1` to opt into the lower-first-audio
    staged experiment.
  - 2026-07-09 CFG compact TTS-window decode experiment:
    the existing compact speech-window decode path was extended to support CFG
    pairs behind `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1`. The path is correct and
    keeps conditional/unconditional tokens synchronized, but it is slower on
    30B and therefore disabled by default. The diagnostic run
    `.audex/runs/speech-output-vllm-20260709-085734.json` reached end tokens
    and passed the MLX-Audio oracle with ratio `0.9966`, while native debug showed
    `tts_window_decode_count=1438`, `tts_window_weight_cache_hits=396`, and
    `tts_window_batch_sample=27304.5/397`. The speech timing regressed to
    `stream_finished_seconds=109.209`, `codec_frames_per_second=6.831`, and
    `audio_realtime_ratio=0.138`. A guarded default rerun without the env flag
    wrote `.audex/runs/speech-output-vllm-20260709-090142.json`, passed
    the MLX-Audio oracle with ratio `0.9947`, and reported
    `stream_finished_seconds=71.767`, `codec_frames_per_second=9.712`, and
    `audio_realtime_ratio=0.197`. The next lower-level target is the
    compact-window top-k/categorical sampler or a custom Metal sampler kernel,
    not just avoiding full-vocab projection.
  - 2026-07-09 diagnostic update: `sts_timing_assessment` now carries nonpaged
    TTS-window detail fields such as `tts_window_decode_count`,
    `tts_window_weight_cache_hit_rate`,
    `nonpaged_native_detail_ms_per_step_by_category`, and
    `dominant_nonpaged_native_detail_category`. If `tts_window_sample` or
    `tts_window_batch_sample` dominates, the assessment reports
    `likely_bottleneck=pending_graph_eval_during_tts_window_sampling`, which is
    the current evidence-backed target for the CFG quality path.
  - 2026-07-09 CFG sampler benchmark update:
    `scripts/bench_vllm_metal_sampler.py` now has a CFG-shaped mode. The
    command
    `PYTHONPATH="$PWD/.audex/vendor/vllm-metal" VLLM_LOGGING_LEVEL=ERROR TRANSFORMERS_VERBOSITY=error .audex/vendor/vllm-metal/.venv-vllm-metal/bin/python scripts/bench_vllm_metal_sampler.py --cfg-pairs 2 --temperature 1.0 --top-k 80 --cfg-scale 3.0 --iterations 5 --warmup 2 --json .audex/runs/vllm-metal-sampler-bench-20260709-cfg-topk.json`
    ran on `Device(gpu, 0)` and reported raw CFG window sample `0.472 ms`,
    full-vocab projection plus window sample `13.929 ms`, and compact-window
    projection plus sample `4.639 ms`. This means the standalone CFG top-k
    sampler is not enough to explain the real `tts_window_batch_sample` stall;
    the likely target is delayed graph evaluation or runtime scheduling at the
    integrated TTS-window sample/eval boundary.
  - 2026-07-09 lazy-eval diagnostic split:
    `AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES=1` can be combined with
    `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1` for diagnostic runs. It forces
    `mx.eval` after TTS-window forward and projection, then records
    `tts_window_forward_eval`, `tts_window_project_eval`,
    `tts_window_batch_forward_eval`, and `tts_window_batch_project_eval`. Do not
    enable it for product latency measurements; it is specifically for deciding
    whether the apparent `tts_window_batch_sample` stall is true sampling cost
    or pending forward/project work forced at the sample boundary.
  - 2026-07-09 synced CFG-window diagnostic result:
    running
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES=1 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-092544.json` and
    `.audex/runs/sts-turn-vllm-20260709-092719.json`. It reached end tokens
    without playback and reported `stream_finished_seconds=94.85`,
    `first_audio_ready_seconds=5.556`, `audio_realtime_ratio=0.155`, and
    `codec_frames_per_second=7.633`. The synced native detail split showed
    `tts_window_batch_forward_eval=7278.4/101` (about `72.06 ms/step`),
    `tts_window_batch_project_eval=1223.0/101` (about `12.11 ms/step`), and
    `tts_window_batch_sample=1347.8/101` (about `13.34 ms/step`). Treat this as
    evidence that the next compact-window CFG target is model forward eval or
    vLLM-Metal scheduling/cache usage, not the sampler alone.
  - 2026-07-09 mlx-audio-inspired playback update:
    after reviewing `Blaizzy/mlx-audio`, especially
    `mlx_audio/tts/audio_player.py` and the streaming examples in
    `docs/models/tts/`, Audex-Mac now applies the same adaptive-buffering idea
    to its existing `_ContinuousPcmPlayer`. No mlx-audio source code was copied
    wholesale. The configured prebuffer remains a ceiling, but playback can
    start earlier when observed PCM arrival rate indicates a smaller buffer is
    sufficient. Run logs now expose `adaptive_prebuffer`,
    `prebuffer_target_seconds`, `actual_prebuffer_seconds`, and
    `arrival_rate_audio_realtime_ratio` inside `playback_diagnostics`.
  - 2026-07-09 nonpaged cache-copy timing update:
    `audex_mac.patches.vllm_metal_cfg` now wraps vLLM Metal's nonpaged
    `_merge_kv_caches` and `_extract_kv_cache` helper symbols in debug mode so
    CFG diagnostics can distinguish sampling-bound graph evaluation from cache
    copy overhead. A GPU-visible 30B CFG batch probe wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-101507.json` with
    `codec_frames_per_second=16.708`. At the latest nonpaged checkpoint,
    `sample_eval` was about `50.742 ms/step`,
    `nonpaged_kv_cache_extract` was about `19.097 ms/extract`, and
    `nonpaged_kv_cache_merge` was about `0.323 ms/merge`. Interpretation:
    cache extraction is significant, but this short default CFG batch is still
    dominated by pending MLX graph work forced at the sampling boundary.
  - 2026-07-09 nonpaged async-submit update:
    the default batched nonpaged decode patch now submits
    `next_token_logits` with `mx.async_eval(...)` before Python builds
    `SamplingBatch`. This is guarded by `AUDEX_VLLM_NONPAGED_ASYNC_EVAL=0` for
    diagnostics. On the same short 30B CFG batch shape, async enabled wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-102231.json` with
    `codec_frames_per_second=19.119`, `nonpaged_decode_avg_ms=95.2`, and
    `sample_eval` down to about `21.494 ms/step`. The kill-switch comparison
    `.audex/runs/vllm-metal-diagnostic-20260709-102355.json` reported
    `codec_frames_per_second=17.028`, `nonpaged_decode_avg_ms=98.1`, and
    `sample_eval` about `50.1 ms/step`. This is a real but insufficient speed
    improvement; the path still needs deeper nonpaged decode/cache/scheduling
    work for near-realtime 30B CFG TTS.
  - 2026-07-09 persistent nonpaged batch-cache update:
    the default full-vocab nonpaged decode path now keeps the merged batch cache
    alive while vLLM presents the same request IDs in the same order, flushing
    back to per-request caches when membership changes or finished requests are
    evicted. This is guarded by
    `AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE=0`. On the short 30B two-pair
    CFG batch probe, enabled
    `.audex/runs/vllm-metal-diagnostic-20260709-103256.json` reported
    `codec_frames_per_second=40.468`, `nonpaged_decode_avg_ms=36.9`,
    `nonpaged_persistent_cache_hit_rate=0.979`, and no steady-state
    `nonpaged_kv_cache_extract` category. The kill-switch comparison
    `.audex/runs/vllm-metal-diagnostic-20260709-103417.json` reported
    `codec_frames_per_second=17.133`, `nonpaged_decode_avg_ms=106.1`, and
    `nonpaged_kv_cache_extract` about `17.174 ms/extract`. The real CFG STS
    fixture `.audex/runs/sts-turn-vllm-20260709-103538.json` remained
    intelligible (`MLX-Audio oracle ratio=0.9164`, no excessive repetition) and
    reached `codec_frames_per_second=24.231`; still not realtime, but a large
    improvement over the pre-cache CFG path.
  - 2026-07-09 static CFG sentence-chunk update:
    the product CLI now uses one-sentence static chunks only when
    `AUDEX_VLLM_TTS_CFG=1`, leaving the default no-CFG chunk policy unchanged.
    This feeds the paired CFG batcher more consistently for ordinary
    multi-sentence answers. The 30B CFG fixture
    `.audex/runs/sts-turn-vllm-20260709-104439.json` reported
    `tts_concurrent_segments=true`, `tts_observed_segments=2`,
    `first_audio_ready_seconds=2.297`, `stream_finished_seconds=15.058`,
    `codec_frames_per_second=27.427`, and `audio_realtime_ratio=0.562`.
    The MLX-Audio oracle passed the generated WAV with ratio `0.9259` and no excessive
    repetition. This is another meaningful CFG improvement, but it is still
    below realtime and trades a little first-audio latency for more occupied
    paired decode.
  - 2026-07-09 decoder hot-loop update:
    the vLLM streaming output path now derives `peak_abs` from packed PCM16
    bytes instead of forcing per-chunk MLX scalar diagnostics with
    `mx.max(...).item()` and `mx.isfinite(...).item()`. The follow-up 30B CFG
    fixture `.audex/runs/sts-turn-vllm-20260709-105128.json` wrote
    `.audex/runs/speech-output-vllm-20260709-105115.json` and reported
    `first_audio_ready_seconds=1.971`, `stream_finished_seconds=12.717`,
    `codec_frames_per_second=32.476`, and `audio_realtime_ratio=0.665`.
    The MLX-Audio oracle passed the CFG WAV with ratio `1.0` and no excessive
    repetition. The same no-CFG fallback fixture also passed the MLX-Audio oracle with
    ratio `1.0`, so the fallback remains a valid control.
  - 2026-07-09 compact CFG speech-window retest:
    `_try_batched_tts_window_decode(...)` now reuses the same stable nonpaged
    batch-cache helper as the default full-vocab CFG path, so the opt-in
    compact-window diagnostic no longer pays deliberate per-step cache
    extraction. The retest with `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1` remained
    intelligible (`MLX-Audio oracle ratio=1.0`) but slower:
    `.audex/runs/speech-output-vllm-20260709-110815.json` reported
    `codec_frames_per_second=19.773` and `audio_realtime_ratio=0.405`.
    Therefore compact CFG speech-window decode stays diagnostic-only; the
    product CFG path should continue using the default full-vocab nonpaged
    decode path while the next speed target stays in model forward / async
    graph submission rather than speech-window projection.
  - 2026-07-09 CFG playback prebuffer retune:
    after the scalar-sync and CFG batching changes, the previous 8.0s CFG
    playback prebuffer was no longer justified for short answers. The product
    CLI now uses a 2.0s configured CFG prebuffer for one- or two-segment short
    answers, while preserving 4.0s for three-or-more CFG chunks and 5.0s for
    very long utterances. The adaptive continuous player can still expand the
    actual initial buffer based on observed audio-arrival rate. A same-fixture
    30B playback comparison showed the interim 3.0s setting at
    `.audex/runs/speech-output-vllm-20260709-112120.json` started playback
    `11.04s` after first audio with zero underruns. The final 2.0s setting at
    `.audex/runs/speech-output-vllm-20260709-112349.json` started playback
    `7.522s` after first audio, also with `queue_underrun_count=0` and
    `device_underflow_count=0`. It remained intelligible under the local
    MLX-Audio oracle (`ratio=1.0`, no excessive repetition). This improves
    audible latency but does not solve the deeper CFG throughput gap:
    `audio_realtime_ratio=0.429` in that playback run.
  - 2026-07-09 nonpaged async-eval target diagnostic:
    `AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET` was added as a diagnostic switch
    with product default `logits`. The same no-play 30B CFG fixture shows the
    default full-logits async submit is still best:
    `.audex/runs/speech-output-vllm-20260709-105115.json` reported
    `codec_frames_per_second=32.476` and `audio_realtime_ratio=0.665`.
    The new `sample_logits` target wrote
    `.audex/runs/speech-output-vllm-20260709-113053.json` and reported
    `codec_frames_per_second=30.654`, `audio_realtime_ratio=0.628`, with
    native timing shifted into `native_sample_logits_async_submit`. The `none`
    target wrote `.audex/runs/speech-output-vllm-20260709-113256.json` and
    reported `codec_frames_per_second=25.348`, `audio_realtime_ratio=0.519`.
    Interpretation: keep full-logits async submit as the product default. The
    next meaningful speed target is deeper model-forward / sampling-bound graph
    evaluation or scheduling, not simply moving the async-eval target.

## 2026-07-09 CFG Concurrency Sweep

The corrected 30B CFG concurrency sweep uses
`scripts/probe_vllm_tts_decode.py --cfg-segment ...` with explicit TTS
segments. A probe result only counts as CFG evidence when the JSON `runtime`
block shows `cfg_enabled=true` and `cfg_scale=3.0`.

One early 12-segment run omitted `AUDEX_VLLM_ENABLE_CFG_WIRING=1` and therefore
reported `cfg_enabled=false`, `cfg_scale=0.0`. That run is useful only as a
guardrail: it proves the diagnostic can accidentally build conditional/
unconditional request pairs without installing NVIDIA's CFG logits processor.

Valid 12-segment 30B CFG results:

- `max_num_seqs=8`, `max_num_batched_tokens=8192`:
  `.audex/runs/tts-probe-vllm-20260709-115109.json`, `1446` aggregate codec
  frames in `35.171s`, about `41.1 fps`.
- `max_num_seqs=16`, `max_num_batched_tokens=8192`:
  `.audex/runs/tts-probe-vllm-20260709-115229.json`, `1446` aggregate codec
  frames in `34.546s`, about `41.9 fps`.
- `max_num_seqs=24`, `max_num_batched_tokens=8192`:
  `.audex/runs/tts-probe-vllm-20260709-115359.json`, `1446` aggregate codec
  frames in `41.792s`, about `34.6 fps`.
- `max_num_seqs=24`, `max_num_batched_tokens=32768`:
  `.audex/runs/tts-probe-vllm-20260709-115547.json`, `1446` aggregate codec
  frames in `35.416s`, about `40.8 fps`.

The event order in all valid runs advanced two CFG segment pairs at a time:
segments `0/1`, then `2/3`, then `4/5`, and so on. Raising
`AUDEX_VLLM_CFG_MAX_NUM_SEQS` alone did not admit more active CFG pairs, and
raising `AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS` to `32768` did not change the
observed pair-admission pattern.

NVIDIA's bundled `CFGLogitsProcessor` can blend every complete pair already in
the batch. The next useful optimization target is therefore not simply the
engine config cap. It is the combination of:

- request/pair admission in the CFG scheduler patch,
- persistent nonpaged batched KV/SSM cache while batch membership is stable,
- compact/window speech-token projection and sampling once the copy tax is
  removed,
- async evaluation and decoder offloading after the larger per-step costs are
  reduced.

## 2026-07-09 Product-Path Bisect

The current CFG product path has two separable costs:

- vLLM Metal generation of speech codec frames.
- causal speech decoder / cache-clear / PCM emission work after those frames are
  available.

A same-fixture product run on `four-sentence-context-question-16k-mono.wav`
with CFG enabled generated 733 codec frames in 26.299 seconds, or 27.872 codec
fps. The same run with `AUDEX_VLLM_SKIP_SPEECH_DECODER=1` generated the same
733 codec frames in 19.053 seconds, or 38.471 codec fps. That diagnostic skips
decoder push/flush/reset, playback, PCM emission, and per-segment MLX cache
clears, so it is not an intelligibility run. It is evidence that decoder-side
work is a real hot-path cost in the product CLI.

The row-count instrumentation also changed how to read concurrency sweeps. A
12-segment direct CFG probe with `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16` and a later
8-segment probe with both `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16` and
`AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS=32768` still logged:

```text
decode_reqs=4 cfg_cond_reqs=2 cfg_uncond_reqs=2 cfg_complete_pairs=2
```

So the previous flat 8/16/24 configured-cap sweep did not measure 4 or 8 active
CFG pairs. It measured repeated two-pair waves. The next concurrency experiment
must first change admission so actual `cfg_complete_pairs` rises above 2; only
then does it test the MoE expert-overlap curve the user is asking about.

## 2026-07-09 CFG KV Admission Fix

The two-pair wave was caused by vLLM Metal's non-paged scheduler capacity
report, not by CFG request submission. For the current Nemotron-H/Audex 30B
path, `WorkerCachePlanner.determine_available_memory()` reports one
max-length-sequence estimate to vLLM's cache manager. With
`max_model_len=5120`, native debug showed:

```text
cfg kv capacity num_blocks=5 block_size=1072 max_model_len=5120 estimated_max_concurrency=1.0
cfg scheduler admission scheduled_reqs=4 scheduled_complete_pairs=2 waiting_reqs=12
cfg kv allocation rejected ... free_blocks=0
```

Lowering `AUDEX_VLLM_CFG_MAX_MODEL_LEN=2048` did not help. It reduced the
reported block pool to `num_blocks=2`, which could not even admit both sides of
the first CFG pair. Raising `AUDEX_VLLM_GPU_MEMORY_UTILIZATION=0.85` also did
not change `num_blocks`, because this non-paged MLX capacity path does not use
that knob for scheduler admission.

Audex-Mac now installs an explicit diagnostic/product override,
`AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS`, in the early runtime patch hook. The
override multiplies the non-paged scheduler-visible one-sequence capacity only
for `single_sequence_estimate` mode. At the time of this experiment,
`./start.sh` defaulted it to `8` while preserving user overrides. The 256K demo
release later reduced that default to `2`; see the current runbook and patch
ledger for the active policy.

With `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=4`,
`AUDEX_VLLM_CFG_MAX_NUM_SEQS=16`, and eight CFG TTS segments, native debug
showed the decisive admission change:

```text
cfg kv capacity num_blocks=20 block_size=1072 max_model_len=5120 estimated_max_concurrency=4.0
cfg scheduler admission scheduled_reqs=16 scheduled_complete_pairs=8 waiting_reqs=0
nonpaged decode timing ... decode_reqs=16 cfg_cond_reqs=8 cfg_uncond_reqs=8 cfg_complete_pairs=8
```

The comparable 64-token direct CFG probe wrote
`.audex/runs/tts-probe-vllm-20260709-123825.json` and generated 512 aggregate
codec frames in 9.104 seconds, or 56.239 codec fps. That is the first valid
8-pair 30B datapoint; earlier flat configured-cap sweeps should not be used as
bandwidth-ceiling evidence.

The product-path default-capacity fixture later logged the explicit admission
arithmetic:

```text
cfg kv capacity num_blocks=40 block_size=1072 max_model_len=5120 max_length_blocks_per_request=5 inferred_request_capacity=8.0
```

That run still scheduled only four complete pairs because the response had four
sentence chunks, not because admission blocked wider decode. It generated 720
codec frames at 36.362 codec fps and passed the MLX-Audio
intelligibility oracle with ratio 0.884 and no excessive repetition.

The override is still a stopgap. Audex-Mac now logs the MLX/Metal headroom at
override time and refuses the override when
`AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS * one_sequence_kv_bytes` exceeds the
current MLX working-set headroom. It also logs the stricter
`gpu_memory_utilization` headroom as a warning signal, but the non-paged
`VLLM_METAL_MEMORY_FRACTION=auto` path cannot treat that as a hard allocator
limit without breaking the current 30B CFG default. That makes the current
widening honest enough for the spike, but it is not the final cache policy.
The principled fix is per-request non-paged reservation for the actual Audex
TTS segment budget, so short TTS chunks are not held hostage by the shared
ASR/text engine's `max_model_len`.

The latest eight-sentence product fixture proves true 8-pair scheduling, but
not product real-time throughput: `.audex/runs/speech-output-vllm-20260709-130630.json`
generated 1286 codec frames at 23.276 codec fps with
`first_audio_ready_seconds=7.941`, while the MLX-Audio oracle passed
at ratio 0.9705 with no excessive repetition. A decoder-skip A/B at the same
width only improved throughput to 23.852 codec fps, so the inline speech
decoder is not the binding bottleneck for that fixture. The next scheduler
work should target uneven segment tails and capacity-aware chunking rather
than decoder overlap.

Audex-Mac now caps static CFG TTS chunking to the target pair width before it
submits the initial batch. This prevents the known `max_chars=80` failure mode,
where a ninth chunk forced a second scheduler wave behind an otherwise 8-pair
answer. The old short-tail merge and re-split follow-up were useful diagnostics,
but they were approximating a simpler problem: split long CFG sentence atoms,
then partition ordered atoms into no more than the admitted pair capacity while
minimizing the largest estimated chunk cost. The product chunker now uses
80-character CFG atoms plus an exact contiguous linear-partition dynamic program
with character count as the cost proxy.

The follow-up capacity-aware chunk work adds a diagnostic
`AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS` override and caps the effective static CFG
TTS target by `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` when the scheduler
capacity override is set. That keeps product chunking honest when capacity is
lowered for stability diagnostics. Run logs now record both
`tts_requested_target_segments` and effective `tts_target_segments`.

A `target_segments=6` sweep proved that simply making fewer chunks is the wrong
diagnosis: `.audex/runs/speech-output-vllm-20260709-140217.json` reached
33.691 codec fps with decoder skipped, but left a 344-frame final segment that
ran nearly alone. The linear partition keeps the same capacity-aware boundary,
but replaces the heuristic stack with a deterministic optimum over contiguous
sentence atoms. A fixed-text probe with the previous shaped eight chunks admitted
`scheduled_complete_pairs=8`, had `waiting_reqs=0`, and reported 1304 aggregate
codec frames at 49.792 fps in
`.audex/runs/tts-probe-vllm-20260709-140727.wav`. That is close to realtime and
confirms full-width scheduling for the shaped TTS batch, but it is still probe
evidence. The full product path must be judged on actual conversational answer
shape plus decoder/playback behavior.

Decoder overlap remains useful but is no longer the largest proven gap on this
fixture. Under the current seven-segment product shape, decoder-skipped
`.audex/runs/speech-output-vllm-20260709-135805.json` measured 32.059 fps while
decoded `.audex/runs/speech-output-vllm-20260709-140007.json` measured
28.375 fps with first audio at 4.486s. That is a real cost, not the earlier
10-fps wall.

Audex-Mac also has a deterministic product-path TTS diagnostic now:
`--diagnose-vllm-tts-text TEXT` or `--diagnose-vllm-tts-text-file PATH`. Unlike
the low-level `scripts/probe_vllm_tts_decode.py`, this path instantiates
`VllmSpeechToSpeechSession` and calls the same `generate_speech_output(...)`
method used by speech-to-speech turns. It is the right first check for CFG
chunking because it removes ASR and text-response drift while preserving
product decoder/run-log behavior. With `AUDEX_VLLM_TTS_CFG=1`, the fixed
context-manager probe wrote
`.audex/runs/tts-text-probe-vllm-20260709-141524.json`: 8 observed segments,
`scheduled_complete_pairs=8`, `waiting_reqs=0`, 1304 codec frames at
49.953 fps, first audio at 5.758s, and no max-token hit. This is strong
evidence that the current shaped CFG TTS batch is nearly realtime in the
product path, but full STS still needs end-to-end fixture and human-listening
validation.

After replacing the heuristic stack with CFG atomization plus linear partition,
three fixed-text product-path repeats wrote:

- `.audex/runs/tts-text-probe-vllm-20260709-142835.json`: 1304 frames at
  48.793 fps, first audio 5.986s.
- `.audex/runs/tts-text-probe-vllm-20260709-142956.json`: 1304 frames at
  53.264 fps, first audio 5.402s.
- `.audex/runs/tts-text-probe-vllm-20260709-143110.json`: 1304 frames at
  51.063 fps, first audio 5.714s.

All three had `speech_decoder_skipped=false`, `tts_cfg_enabled=true`, 8 observed
segments, `tts_cfg_atom_max_chars=80`, no max-token hit, and the same segment
frame counts: `{172, 124, 128, 172, 174, 207, 128, 199}`. The last WAV passed
the MLX-Audio oracle at ratio `1.0` with no excessive repetition:
`.audex/runs/tts-text-probe-vllm-20260709-143110.eval.json`.

End-to-end STS timing artifacts now record `response_word_count`,
`min_response_words`, and `valid_response_length`. The evidence gate rejects a
fixture whose generated response is too short to trust as a TTS/chunk-planner
benchmark, so a truncated answer such as "A context manager is an" cannot be
mistaken for a real throughput regression.

End-to-end STS diagnostics also surface CFG TTS segment-balance evidence from
the product speech-output JSON. The subprocess result carries
`tts_segment_codec_frame_counts`, and `sts_timing_assessment` computes segment
count, min/max/mean codec frames, max/min ratio, final-tail codec frames, and
tail/mean ratio. If a multi-segment CFG smoke run leaves the final segment below
half the mean frame count, the evidence gate reports an underfilled-tail
planner failure. This keeps the known tail-collapse bottleneck visible instead
of allowing a superficially high aggregate fps number to hide a bad final
admission wave.

The CFG static chunker now applies the matching conservative runtime rule:
when atomization already fills the requested pair capacity and the final atom is
shorter than `DEFAULT_VLLM_CFG_TTS_MIN_TAIL_CHARS=40`, merge that final atom
into its predecessor unless every chunk is short. This removes the easiest
low-width tail without turning terse answers such as "One. Two. Three." into one
oversized prompt, and the threshold is written to speech-output JSON as
`tts_cfg_min_tail_chars`.

The exact tiny-tail product fixture confirms the rule removes the failure it
targets. `.audex/runs/tts-text-probe-vllm-20260709-153044.json` admitted all 7
CFG pairs together, kept the speech decoder enabled, and produced segment frame
counts `{164, 120, 115, 108, 112, 98, 120}` at 47.518 aggregate codec fps. The
final segment was 0.97 times the segment mean rather than an underfilled second
wave. The generated WAV passed the MLX-Audio oracle at ratio `0.971`
with no excessive repetition.

### CFG Scheduler State Reset Between Text and TTS

Clean CLI measurements separated text shape from shared-engine state. Three
fresh tiny-tail product probes produced 54.129-55.261 codec fps (54.732 mean,
2.07% full spread), while two full eight-sentence STS controls produced only
49.368 and 48.772 codec fps to the last frame. Their speech-output logs were
slower again when measured through full stream completion: 46.539 and 45.758
codec fps. All WAVs passed the MLX-Audio oracle, so the gap was not a
quality/truncation trade.

Replaying the exact generated eight-sentence response in fresh TTS-only engines
produced 51.473 and 55.057 codec fps with the same 929 frames. That ruled in
shared-session state as a material part of the regression. Native timing debug
was not the cause: the debug-enabled fresh run was the faster replay.

The narrow A/B uses vLLM's existing `AsyncLLM.reset_prefix_cache()` after the
text request has finished and before static CFG pairs are submitted. Despite
its public name, this call resets scheduler KV/prefix-cache bookkeeping; it does
not reload the model, reset MLX, change sampling, or touch the Metal worker.
Two otherwise identical STS runs with the reset produced 54.977 and 52.977
codec fps to the last frame, both above the 50 fps diagnostic gate. Their full
stream-completion rates were 51.320 and 49.588 codec fps. First-audio latency
remained variable (4.949 and 5.326 seconds), so this is a throughput/state fix,
not a claimed first-audio fix.

Static CFG TTS now performs that reset by default and records
`tts_prefix_cache_reset=true` in the STS turn log. If vLLM refuses the reset,
the run fails loudly. `AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS=0` disables it
for diagnostics. The no-CFG/interleaved path keeps its prior behavior and does
not reset by default.

The final no-override product proof is
`.audex/runs/vllm-metal-diagnostic-20260709-161211.json`. Its turn log records
`tts_prefix_cache_reset=true`; the diagnostic passed at 56.574 codec fps / 1.131
realtime ratio, while the speech log reported 52.700 codec fps through full
stream completion. The WAV passed the MLX-Audio oracle at ratio
`0.9973474801061007` with no excessive repetition.

### CFG Capacity Default Does Not Leak Into No-CFG

This subsection records the superseded 5,120-token experiment. At that point,
the eight-sequence non-paged capacity default existed to admit eight CFG pairs
under CFG's 5,120-token engine budget. Applying it unconditionally to the
no-CFG engine's longer context produced a correct startup refusal:
`.audex/runs/vllm-metal-diagnostic-20260709-161400.json` reported 111.89 GB
requested worst-case KV state against 48.42 GB of MLX/Metal headroom.

For that experiment, `start.sh` preserved an explicit
`AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` override but supplied the default `8`
only when `AUDEX_VLLM_TTS_CFG` was enabled. Ordinary no-CFG startup left the
value empty and used vLLM Metal's native capacity calculation. The fixed
no-CFG run `.audex/runs/vllm-metal-diagnostic-20260709-161742.json` reached the
async Metal engine, streamed 984 codec frames, and passed its diagnostic at
134.961 last-frame codec fps / 2.699 realtime ratio with first audio at 2.486
seconds. Its full interleaved stream metric was 38.796 codec fps because that
timer includes the overlapping text/TTS interval; keep these timing boundaries
distinct. The turn log records `tts_prefix_cache_reset=false`, and the MLX-Audio oracle
passed at ratio `0.9973474801061007` with no excessive repetition.

### Context Budget Guard

This subsection also describes the superseded 5,120-token configuration. That
vLLM Metal CFG engine set `max_model_len=5120` from Audex-Mac's former
`vllm_cfg.py` arithmetic: `max(4096, asr_max_tokens + 1024,
text_max_tokens + 1024, tts_max_tokens + 1024)`. This is not an Audex model
long-context limit. It is an engine reservation knob, and on the vendored
non-paged Metal path every request reserves blocks against the full engine
length. Raising it blindly would trade away the 8-pair CFG TTS admission win.

The release policy supersedes both settings: plain and CFG3 engines configure
262,144 tokens, and CFG defaults to two worst-case max-length sequences. The
current guard and measured headroom evidence are recorded in the release
context correction in [viability.md](viability.md) and the current entry in the
[patch ledger](patches.md).

Audex-Mac therefore refuses to silently prune resumed conversation history.
Before submitting text generation, the CLI now builds the same text prompt that
vLLM will see, counts prompt tokens, and raises a clear Audex-Mac error if
`prompt_tokens + response_max_tokens` exceeds the current engine budget. The
error names the prompt token count, response token budget, engine max length,
and the fact that request-scoped KV/state reservation is the real short-term
unlock before increasing `max_model_len`.

A prior experimental pruning run shortened the local conversation file, so the
next 30B CFG fixture completed instead of overflowing:
`.audex/runs/sts-turn-vllm-20260709-144256.json`. It logged
`text_context={"fits": true, "messages_before": 108, "prompt_tokens": 4952,
"prompt_token_budget": 5024, "context_token_limit": 5120,
"response_max_tokens": 96}`. The corresponding CFG TTS log
`.audex/runs/speech-output-vllm-20260709-144242.json` had
`speech_decoder_skipped=false`, `tts_cfg_enabled=true`, 3 observed segments,
416 codec frames at 28.872 fps, first audio at 3.981s, and no max-token hit.
The MLX-Audio oracle passed with ratio `1.0` and no excessive
repetition:
`.audex/runs/speech-output-vllm-20260709-144242.eval.json`.

Longer interactive conversations still need a larger design than raising
`max_model_len`: Mamba2 state should be carried forward or prefill must become
incremental, otherwise each turn re-prefills the full transcript and first-token
latency grows with history length. A separate design/benchmark pass should
measure prefill latency at 1k, 4k, 16k, and 32k tokens before changing the
conversation architecture.

## Conversation State Carryover

The old MLX-native path had disk prompt-cache support:
`audex_mac/sts_cli.py` loads `<conversation_id>.kv.safetensors`, checks that
the current generation prompt starts with the cached history tokens, then feeds
only the suffix tokens. The vLLM Metal path does not currently have the
equivalent; it persists the text transcript and re-renders that transcript into
each new vLLM text-generation request.

The intended vLLM design is a single append-only conversation state, not generic
vLLM prefix caching. Two details make this a design constraint rather than a
preference:

- The vendored vLLM Metal prefix-caching machinery is paged-backend machinery.
  The 30B Nemotron-H Audex model runs on the non-paged contiguous path
  (`vllm_metal/v1/model_runner.py` plus `contiguous_cache.py`), where generic
  prefix-cache block reuse is absent, not merely disabled by
  `enable_prefix_caching=False`.
- Generic transformer prefix caching assumes per-token-block addressable KV
  state. Mamba2 recurrent state is a compressed function of the full prefix at a
  token boundary; there is no independent "block 7 state" to hash and reuse.
  A generic hybrid prefix cache would need complete recurrent snapshots at many
  block boundaries plus eviction policy. Audex-Mac's SLC needs exactly one
  snapshot: the state at the end of the last committed conversation turn.

At the end of a turn, Audex-Mac should hold the committed history state. At the
next turn, it should verify that the committed history prompt is an exact token
prefix of the next generation prompt, inject the saved per-layer state into the
vLLM Metal request, and prefill only the suffix tokens for the new user turn.
Cold restart can then use the same state snapshot through an oMLX-style
safetensors writer/loader; NVMe load time replaces transcript re-prefill.

The request-builder contract now exposes both renderings:
`build_text_messages_history_prompt(..., add_generation_prompt=False)` and
`build_text_messages_generation_prompt(..., add_generation_prompt=True)`.
`tests/test_vllm_sts_requests.py` asserts that a committed history prompt is a
token prefix of the next turn's generation prompt. Any future vLLM Metal state
injection must keep that test green before trusting a saved cache/state
snapshot.

Audex-Mac now also carries an inert state hint through the existing
`SamplingParams.extra_args` channel used by the CFG scheduler patches. For text
conversation requests with a persisted conversation id, the CLI sends:

- `audex_text_state_key`: the conversation id.
- `audex_text_state_mode`: currently `append`.
- `audex_text_state_prefix_token_count`: token count for the committed history
  prompt, rendered without the assistant generation marker.
- `audex_text_state_prefix_token_hash`: deterministic hash of those committed
  history tokens.

The current vLLM Metal fork does not consume these fields yet. They are the
scaffold for the next patch: capture `RequestState.cache` before non-paged
cleanup deletes it, store it under `audex_text_state_key` with the prefix
metadata, and on the next append-mode text request compare count/hash before
injecting the saved state and prefilling only suffix tokens.

The fork-side cleanup seam is now partially wired. The
`_cleanup_finished_requests` wrapper in `audex_mac/patches/vllm_metal_cfg.py`
recognizes append-mode text state hints before the non-paged runner deletes
`RequestState`, and records metadata under
`runner._audex_text_state_snapshots[state_key]`. This intentionally stores only
metadata today (`request_id`, prefix count/hash, prompt length, token count,
generated token count, and whether a cache existed). It proves the lifecycle
hook and control channel without retaining cache objects or changing generation
behavior. The next patch should promote that metadata record into a real
single-snapshot cache capture/reuse path.

One boundary constraint is now explicit in the metadata: cleanup-time
`RequestState.token_ids` is the raw generation state (`prompt + sampled tokens`),
whereas the STS conversation commits `scrub_spoken_answer(text.text)` as the
assistant message. Those boundaries can differ when prompt leakage is scrubbed,
when stop tokens are present, or when decoded text normalizes differently. For
that reason metadata-only snapshots are stamped with
`boundary=raw_generation_state`, `committed_boundary_verified=false`, and
`reuse_eligible=false`. A future cache-retention patch must either prove the raw
state exactly matches the committed history prompt or build/capture a snapshot at
the committed-history boundary before injecting it into a later request.

The metadata path now supports that second, safer shape. Requests can set
`audex_text_state_boundary=committed_history_prefill`; in that mode the runner
checks `token_ids[:prompt_len]` against the supplied committed-history
count/hash. If the prompt boundary matches, the request generated no more than
the first sampled token, and the cache object exists, metadata is stamped
`committed_boundary_verified=true` and `reuse_eligible=true`. This still does not
retain or inject the cache object yet, but it proves the future warmup request
can identify a safe committed-history boundary without relying on raw decoded
assistant text.

The invalidation rule is intentionally blunt: any history rewrite makes the
saved snapshot stale. The existing resumed-history sanitizer already invalidates
`<conversation_id>.kv.safetensors` when it rewrites a loaded transcript, and
`test_resumed_history_sanitization_invalidates_stale_state_snapshot` covers that
startup path. Future state snapshots should reuse that rule: invalid snapshot
means loud full re-prefill or explicit rebuild, never silent reuse after a
history mismatch.

## Failure Behavior

If vLLM Metal cannot support a required Audex path at the pinned commit, do not
silently fall back. Document:

- exact unsupported upstream symbol/API
- exact failed diagnostic command
- whether upstream `main` appears to contain a relevant fix
- smallest monkey patch needed
- whether the pin should be advanced

## Open Questions

- A 2026-07-09 long-form TTS quality-corpus run showed `VLLM::EngineCore` at
  roughly 85% CPU and 68% GPU in Activity Monitor, versus the greater-than-90%
  GPU utilization normally observed in the shorter diagnostic runs. Profile the
  steady decode hot loop for host-side NumPy/Python work, scalar synchronization,
  and CPU-side sampling or cache copies before treating this workload shape as
  GPU-saturated. A later snapshot also showed the parent `python3.12` process at
  roughly 63% GPU while EngineCore reported 70%; separate expected parent-side
  MLX causal speech decoding from avoidable host work when profiling, and do not
  interpret Activity Monitor's per-process GPU percentages as additive shares.
  A subsequent CFG quality arm showed a healthier EngineCore profile of roughly
  87% GPU and 43% CPU, so correlate samples with recipe and generation/decoder
  phase instead of treating the first snapshot as steady-state behavior.
- Can the current Audex-Mac CFG sampler patch remain stable over long
  interactive conversations, especially after many `EngineCore` startup/shutdown
  cycles?
- Should Audex audio encoder/projector remain repo-owned MLX code, or can it be
  pushed into vLLM Metal's multimodal adapter cleanly?
- The passing diagnostic used `--diagnose-vllm-sts-speech-max-tokens 256` and
  hit that cap before `<speechgen_end>`. For the interactive CLI, verify longer
  answers with the default speech-token budget and confirm audio quality remains
  acceptable across segment boundaries.
- Should the sixteen-segment TTS batching default be adaptive based on response
  length, or is a fixed default preferable for keeping vLLM Metal continuously
  occupied?

## Required Output From The Implementing Agent

- Updated code.
- Updated `docs/engineering/patches.md`.
- Diagnostic report path or paths.
- Validation commands and results.
- A concise summary of whether vLLM Metal is truly on Metal and what evidence
  proves it.

## Completion Gate

The current executable completion gate is the bounded normal-answer STS smoke:

```bash
scripts/create-test-utterance.sh \
  --basename normal-answer-question \
  --text "Please explain Python context managers in two concise sentences."

AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 \
  ./start.sh --model audex-2b \
  --diagnose-vllm-metal \
  --diagnose-vllm-sts-smoke \
  --diagnose-vllm-sts-speech-max-tokens 256 \
  --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav
```

It is complete only when the report verdict is `ready=true` and the report
shows all of:

- parent and spawned MLX probes on `Device(gpu, 0)`
- `engine_class` from async vLLM
- `speech_streaming.vllm_token_streaming=true`
- `speech_streaming.decoder_streaming=true`
- nonzero speech token, codec-frame, and decoder chunk counts
- `sts_timing_assessment.codec_frames_per_second >= 50`
- `sts_timing_assessment.audio_realtime_ratio >= 1`
- `sts_timing_assessment.native_sampling_row_ratio <= 0.75` when the row ratio
  is present
- `sts_timing_assessment.likely_bottleneck` and `mx_eval_ms` evidence when
  throughput is below realtime

Until that gate passes, this plan is still active even if smaller adapter,
streaming, or playback proofs pass.
