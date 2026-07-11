# Audex-Mac Engineering Evidence

## Decision Summary

Audex-Mac has passed the viability threshold and is now a working local Mac
speech-to-speech demo for NVIDIA Audex. This file preserves the evidence and
failed experiments that produced the current implementation.

Development is intended to happen publicly at
`https://github.com/mbarnson/Audex-Mac`, with Codex agents implementing through
issues and pull requests. This file is the running technical diary for evidence
that changes the viability picture.

## Model Findings

### Audex-2B

- Model ID: `nvidia/Nemotron-Labs-Audex-2B`
- Config model type: `nemotron_dense_audex`
- Architecture: `NemotronDenseAudexForConditionalGeneration`
- Dense 28-layer text backbone.
- Includes NV-Whisper-style audio encoder and Audex causal speech decoder.
- NVIDIA examples use `TP=1`.
- Smaller first-run download when no supported model is cached.
- Live end-to-end testing remains minimal compared with 30B.

### Audex-30B-A3B

- Model ID: `nvidia/Nemotron-Labs-Audex-30B-A3B`
- Config model type: `nemotron_h_audex`
- Architecture: `NemotronHAudexForConditionalGeneration`
- MoE/hybrid/Mamba-like 30B total, 3B active model.
- Primary validated demo model on a 128 GB Apple Silicon Mac.
- Use automatically when fully cached.

## vLLM Metal Findings

vLLM Metal is the preferred runtime scaffold for the spike because it already
has:

- Apple Silicon / MLX runtime infrastructure.
- vLLM-like scheduling and cache machinery.
- model adapter seams.
- STT and multimodal code paths.
- hybrid attention support.

It does not currently make Audex a drop-in model. Audex-Mac still needs
Audex-specific loader/model/runtime patches.

Pinned-source inspection on 2026-07-07:

- vLLM Metal package version at the pin is `0.3.0`.
- Runtime requirement is native arm64 Python `>=3.12,<3.14`.
- `ModelLifecycle._load_generation_model()` is the generation-model load seam.
  It selects GGUF, `mlx_vlm`, AWQ, or `mlx_lm` loading.
- `DefaultModelAdapter.build_multimodal_adapter()` is the native multimodal
  adapter seam. At the pin it recognizes Qwen3-VL and PaddleOCR-VL only.
- `MetalModelRunner` already has paged multimodal feature caching, encoder
  dispatch, and embedding splice logic. The current implementation is image
  oriented, but the runner-side shape is usable for an Audex audio adapter if
  Audex-Mac provides audio feature specs and an adapter returning MLX hidden
  states.
- This confirms the monkey-patch areas in `docs/engineering/patches.md`: Audex-Mac needs an
  Audex model/load hook plus a native audio multimodal adapter before speech
  input can work.
- vLLM Metal reports `device_config=cpu` in vLLM's engine config because
  `MetalPlatform.device_name` / `device_type` are intentionally `"cpu"` for
  PyTorch/vLLM compatibility. That string is not sufficient evidence of CPU
  inference. The authoritative worker evidence is MLX/Metal state:
  `MLX device set to: Device(gpu, 0)`, `PyTorch device set to: mps`, and native
  paged-attention Metal kernels loaded.
- Audex-Mac now enforces `VLLM_METAL_USE_MLX=1`, `VLLM_MLX_DEVICE=gpu`, and
  `VLLM_METAL_USE_PAGED_ATTENTION=0` in `start.sh` and Python preflight. A
  conflicting `VLLM_MLX_DEVICE=cpu` aborts startup before model launch.

Current pin:

- `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`
- Pin metadata lives in `vendor_pins.json`.
- `./start.sh --check-upstream` verified that upstream `main` matched the pin
  at the time of this scaffold pass.

## MLX-Audio Reference Findings

`mlx-audio` is MIT licensed and is useful reference material for MLX-native
audio model implementation, but it should not become a semantic model
dependency for Audex-Mac. The Audex SLC path still forbids separate Whisper,
Kokoro, Silero, or other helper STT/TTS/VAD models.

Inspected `Blaizzy/mlx-audio` `main` at
`c46d15312ad52b0928c0f20dc2d6a880461f32c5` on 2026-07-07.

Useful implementation patterns:

- `mlx_audio/stt/models/qwen2_audio/qwen2_audio.py` has the closest analogue
  to Audex audio input: a Whisper/Qwen2-style audio encoder, a multimodal
  projector, feature extraction to 128 mel bins at 16 kHz, and replacement of
  audio placeholder token embeddings with projected audio features before
  autoregressive generation.
- Its Qwen2-audio implementation computes a fixed 750 audio-token count from a
  30 second / 480000-sample input window, matching the Audex-2B contract found
  in NVIDIA's scripts. This supports treating 750-per-clip as a first-class
  invariant in Audex-Mac rather than a loose prompt detail.
- `mlx_audio/audio_io.py` keeps audio file boundaries simple: NumPy arrays at
  the I/O edge, normalized float samples for model input, and explicit sample
  rate/channel handling. Audex-Mac can use the same shape of boundary without
  adopting `mlx-audio`'s STT/TTS models.
- `mlx_audio/codec/models/*` contains multiple MLX codec/decoder ports
  (`Mimi`, `DAC`, `Encodec`, `SNAC`, `Vocos`, and related helpers). These are
  useful examples for transposing PyTorch convolution weights, managing causal
  decoder cache state, and structuring `encode`/`decode`/`decode_step` APIs in
  MLX.
- `mlx_audio/sts/audio_player.py` provides a practical sounddevice-backed
  playback queue with a configurable start buffer and callback diagnostics. It
  is relevant for the final CLI's local playback plumbing, not for semantic
  inference.

Audex-specific implications:

- The next M2 implementation should model Audex audio input after the
  Qwen2-audio pattern: extract NVIDIA-equivalent features, run the Audex audio
  tower/projector in MLX, embed text tokens normally, then splice projected
  audio embeddings into `<so_embedding>` positions with an exact count check.
- The next M4 implementation should first attempt a small MLX port of NVIDIA's
  `audex_causal_speech_decoder` rather than adding a generic external TTS
  dependency. The NVIDIA decoder is a causal Vocos-like transformer,
  `AudexSpeechTokenEmbedder`, lookahead buffering, and a patch head over 320
  sample hops. This is much closer to `mlx-audio` codec decoder patterns than
  to a full general-purpose speech model.
- A torch/MPS decoder smoke remains useful as a comparison, but the final demo
  should not silently settle for CPU decoder execution.

## Milestones

### M1: Text-Only Audex

Acceptance:

- load selected Audex model on Mac
- run the text benchmark with `max_tokens >= 4096`
- use NVIDIA-recommended sampler settings
- no exact token parity requirement
- produce non-empty, non-collapsed output for runtime compatibility
- record answer quality for a separate human/Codex smell test; model reasoning
  quality does not control the CLI runtime-compatibility exit status
- log timings, sampler params, selected model, and transcript

Status:

- Audex-2B text-only inference now launches locally through vLLM Metal with an
  Audex-Mac-owned MLX-LM Nemotron Dense patch.
- Fast scaffolding exists for startup policy, model selection, patch guards,
  licensing, no-extra-model invariants, and text-gate intent.
- Fast BDD command currently used:
  `./.venv/bin/python -m pytest -m fast`
- Text benchmark contract lives in `benchmarks/text_conversation.json`.
- Text benchmark metadata references NVIDIA's Audex-2B text-only vLLM example
  and records `temperature=1.0`, `top_p=0.95`, `seed=100`, and
  `max_tokens=4096`.
- `./start.sh --show-text-benchmark` prints the benchmark summary without
  touching model inference.
- `./start.sh --preflight-text-runtime --model audex-2b` verifies the selected
  model's `checkpoint_folder_textonly` files, indexed safetensors shards, Python
  version compatibility, and vLLM/vLLM Metal imports before any generation
  attempt.
- `./start.sh --run-text-benchmark --model audex-2b` runs the ten-turn text
  benchmark and writes a transcript JSON under `.audex/runs/`.
- Local evidence on 2026-07-07:
  - Audex-2B snapshot ref:
    `1e4737632af393c1b11c64af89c675cebf14f860`
  - Audex-2B full speech checkpoint has indexed full shards present.
  - Audex-2B remote `checkpoint_folder_textonly` exposes config/tokenizer/index
    files but no safetensors shard. A direct `hf download` for
    `checkpoint_folder_textonly/model-00001-of-00002.safetensors` returned
    "File not found in repository."
  - NVIDIA's `model_conversion_scripts/convert_full_HF_to_textonly_HF.py`
    successfully generated the Audex-2B text-only shards locally from
    `checkpoint_folder_full`.
  - Audex-Mac now provides `scripts/convert-textonly.sh`, a wrapper that
    discovers the local snapshot and invokes NVIDIA's cached conversion script.
    It does not vendor NVIDIA's script into this MIT-licensed repository.
  - `scripts/convert-textonly.sh --model audex-2b --dry-run` resolved the local
    2B snapshot, selected `checkpoint_folder_full` as the complete source
    checkpoint, and targeted `checkpoint_folder_textonly`.
  - `scripts/convert-textonly.sh --model audex-30b-a3b --dry-run` resolved the
    local 30B-A3B snapshot, skipped the incomplete audiogen source folder, and
    selected `checkpoint_folder_full` as the conversion source.
  - Audex-30B-A3B text-only preflight also fails because
    `checkpoint_folder_textonly` is missing all 13 indexed text-only shards.
  - NVIDIA's published Audex artifact shape has Mac-hostile rough edges that
    Audex-Mac must work around explicitly, including missing text-only shard
    files and the observed `LICENSE`/`license` case-sensitivity conflict that
    can make `hf download` fail on a case-insensitive macOS filesystem.
  - Reconfirmed on 2026-07-10: an unfiltered Audex-2B `hf download` failed with
    `FileExistsError` while creating the lowercase `license` directory after
    materializing root `LICENSE`; `--exclude "license/*"` succeeds. Audex-Mac
    now passes the equivalent Hugging Face `ignore_patterns` value explicitly.
  - That failed download advanced `refs/main` to partial revision
    `b13ccb2373764e01aa4e49311d34c9428c925138`. Snapshot verification now scans
    other materialized revisions and selected the older complete locally
    converted revision `1e4737632af393c1b11c64af89c675cebf14f860` without
    mutating the Hugging Face cache.
  - `/opt/homebrew/anaconda3/bin/python3.13` is present, reports `arm64`, and
    satisfies the pinned vLLM Metal Python range.
  - `start.sh` now moves an incompatible generated `.venv` aside and can reuse
    an installed pinned vLLM Metal runtime at
    `.audex/vendor/vllm-metal/.venv-vllm-metal`.
  - Pinned vLLM Metal installed successfully in the ignored runtime venv.
  - `./start.sh --preflight-text-runtime --model audex-2b` now reports
    `Text runtime preflight: ready` on this machine, with
    `Metal/MLX live check: mlx_metal_available=True mlx_default_device=Device(gpu, 0)`.
  - Audex-Mac patches vLLM's architecture registry for
    `NemotronDenseForCausalLM` and installs a generated shim for
    `mlx_lm.models.nemotron_dense` into the generated vLLM Metal venv.
  - Direct MLX-LM load smoke succeeds for the local Audex-2B text checkpoint:
    model type `nemotron_dense`, 28 layers, vocab size 131072, head dim 128,
    RoPE theta 100000000.0, partial rotary factor 1.0.
  - vLLM Metal source inspection showed its `device_config=cpu` report is a
    compatibility facade: the worker sets MLX to `Device(gpu, 0)` when
    `VLLM_MLX_DEVICE=gpu`, and the platform reports a PyTorch `mps` device.
  - The user's Activity Monitor concern was still valid in practice. The vLLM
    offline engine path initialized MLX on GPU and loaded native paged-attention
    Metal kernels, but produced only about 2 generated tokens/second for the
    first Audex-2B text prompt.
  - Direct `mlx_lm` generation over the same patched `nemotron_dense` model
    produced about 100-123 generated tokens/second across the 10-turn text
    benchmark. That historical result proved the model path, but
    `docs/engineering/vllm-metal.md` now supersedes the temporary default: vLLM Metal is
    the default backend again, and `--text-backend mlx` is an explicit
    diagnostic fallback only.
  - `./start.sh --run-text-benchmark --limit-text-turns 1 --model audex-2b`
    completed one direct-MLX turn with NVIDIA sampler settings and
    `max_tokens=4096`. The output was coherent Python for the palindrome prompt.
  - `./start.sh --run-text-benchmark --model audex-2b` completed the full
    direct-MLX ten-turn run in 80.416 seconds, with `backend=mlx`,
    `mlx_default_device=Device(gpu, 0)`, and transcript log
    `.audex/runs/text-benchmark-20260707-121116.json`.
  - After moving ten-turn construction to the official ChatML template, a
    direct-MLX behavioral rerun on 2026-07-10 completed all ten turns in
    89.559 seconds and passed runtime compatibility with no failures. The run
    log was `.audex/runs/text-benchmark-20260710-101630.json`; answer-specific
    misses remained visible as non-blocking model-quality observations.
  - GPT-5.5 Codex coherence judgment for that transcript: the output is
    acceptable for the SLC smell-test bar. The run completed all ten turns with
    NVIDIA sampler settings, `max_tokens=4096`, no excessive repetition, and
    mostly coherent coding/basic-factual answers. The 2B result is not strong
    enough to treat as a correctness benchmark: it carries forward buggy
    `chunked` logic and gives one required step-by-step chunking result as
    `[[3, 1, 4], [1, 5, 9], [5, 2]]` instead of
    `[[3, 1, 4], [1, 5, 9], [2]]`. For this conversational demo, that is a
    model-quality limitation rather than a Mac runtime failure.
  - `auto` model selection sees the local Audex-30B-A3B snapshot, but its
    `checkpoint_folder_textonly` is still missing all 13 indexed safetensors
    shards, so it is not yet a runnable text benchmark target until converted.
    Model selection now falls back to a complete runnable 2B checkpoint instead
    of letting an incomplete cached 30B text folder block startup.
  - The full Audex-30B-A3B checkpoint is now runnable for speech-to-speech
    fixture mode through the `nemotron_h_audex` MLX shim. This does not remove
    the text-only conversion caveat above for text benchmarks; it means the
    default STS path can use the cached full 30B model directly.

Remaining text-quality caveat:

- Convert the local 30B-A3B text checkpoint and rerun the same benchmark when
  stronger reasoning is useful. The SLC default remains 2B for clone-and-run
  usability.

### M0: Startup And Fast BDD Scaffold

Acceptance:

- package/test harness exists
- `./start.sh` creates/reuses `.venv`
- dependency reinstall is skipped unless missing, stale, or `--refresh-deps`
- model selection policy is executable with fake cache probes
- patch guard policy is executable with fake vLLM Metal modules
- fast Gherkin scenarios pass without model weights

Status:

- Implemented as of the initial scaffold pass.
- Verified locally with 14 fast BDD scenarios passing and 3 slow/local scenarios
  deselected.
- Startup smoke `./start.sh --select-model-only` reached model selection and
  selected `nvidia/Nemotron-Labs-Audex-30B-A3B` from the local Hugging Face
  cache on this machine.
- Model-cache probing now trusts Audex-Mac's direct snapshot verifier before
  consulting `huggingface_hub.snapshot_download(local_files_only=True)`. This
  avoids false missing-model prompts when a complete local snapshot exists but
  Hugging Face's filtered local download path trips over published artifact
  quirks on macOS.
- `start.sh` now installs the pinned vLLM Metal checkout from `vendor_pins.json`
  into `.audex/vendor/vllm-metal` when the generated vLLM Metal runtime is
  absent.
- Fresh-clone bootstrap evidence on 2026-07-07:
  - A local clean clone was created at
    `/private/tmp/audex-mac-fresh.CKJGOB`.
  - `PYTHON_BIN=/opt/homebrew/anaconda3/bin/python3.13 ./start.sh
    --select-model-only --model audex-2b` cloned vLLM Metal at the pinned
    commit `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`, created
    `.audex/vendor/vllm-metal/.venv-vllm-metal`, installed `vllm==0.24.0+cpu`
    and `vllm-metal==0.3.0`, installed `mlx`, `mlx-lm`, `mlx-vlm`,
    `mlx-audio`, `sounddevice`, and `huggingface-hub`, built native Metal
    artifacts (`_paged_ops`, `paged_attention_v2`, `gdn`, `paged_mla`), then
    reached Audex-Mac CLI model selection without downloading model weights.
  - A second invocation of the same command in the temp clone reused the
    generated vLLM Metal runtime and immediately reached model selection.
- The CLI now asks before downloading a missing selected Audex model and offers
  `--yes-download` for unattended local setup after the user has accepted the
  NVIDIA model-license notice.
- Local regression evidence on 2026-07-07:
  `./start.sh --select-model-only --model auto` no longer trips Bash
  `set -u` on an empty `ARGS[@]` expansion. `start.sh` now tracks
  `ARGS_COUNT` and dispatches through guarded exec helpers so `./start.sh`
  with no CLI arguments reaches the default STS path.

### M2: Audex Audio Input

Acceptance:

- accept 16 kHz speech input
- run NVIDIA-equivalent audio preprocessing
- inject audio embeddings at `<so_embedding>` placeholders
- no external STT
- fail loudly on embedding count mismatches

Status:

- Audex-Mac now has a fast audio contract module,
  `audex_mac.audio_contract`, that records NVIDIA's native audio-input shape
  for Audex-2B: 16 kHz PCM, 30 second clips, 750 `<so_embedding>` tokens per
  clip, maximum 30 clips / 900 seconds, and exactly one `<sound>` placeholder
  per prompt.
- `audex_mac.audio_pcm` now performs deterministic audio plumbing for the input
  side: mono/stereo normalization, 16-bit PCM WAV loading, fixed 30 second clip
  splitting, final-clip zero padding, and the 30-clip cap. This is plumbing, not
  a semantic helper model.
- `scripts/create-test-utterance.sh` uses macOS `say`, `afconvert`, and
  `afinfo` to generate and validate a repeatable 16 kHz mono WAV fixture for
  encoder-path testing without manual microphone input.
- `audex_mac.audio_features` validates the local `audio_preprocessor` config
  and runs Audex's `WhisperFeatureExtractor` to produce `(clips, 128, 3000)`
  NV-Whisper input features from prepared PCM clips.
- `build_audio_chat_prompt(..., thinking_enabled=False)` expands prompts into
  `<so_start>` + repeated `<so_embedding>` + `<so_end>` and prepends
  `<think></think>` for the default non-thinking STS path.
- `./start.sh --preflight-audio-runtime --model audex-2b` verifies the selected
  speech snapshot, full checkpoint shards, Audex audio encoder/projector
  metadata, Audex causal speech decoder artifacts, 16 kHz decoder config, and
  speech-token tokenizer markers.
- Local evidence on 2026-07-07: Audex-2B full checkpoint reports
  `architecture=NemotronDenseAudexForConditionalGeneration`,
  `model_type=nemotron_dense_audex`, `audio_model_type=NV-Whisper`, 32 audio
  encoder layers, hidden size 1280, 128 mel bins, max source positions 1500,
  487 `audio_encoder.*` tensors, and `audio_projector.norm/fc1/fc2` tensors in
  `model-00002-of-00002.safetensors`.
- Local feature evidence on 2026-07-07:
  `./start.sh --preflight-audio-features --model audex-2b` loaded the cached
  Audex-2B `audio_preprocessor` as `WhisperFeatureExtractor` and produced
  `float32` input features with shape `(1, 128, 3000)` from one prepared silent
  Audex clip. This exercised NVIDIA-equivalent feature extraction only; model
  inference begins after these host features are converted into an MLX array.
- Local projector evidence on 2026-07-07:
  `./start.sh --preflight-audio-projector --model audex-2b` loaded the real
  Audex-2B `audio_projector.norm/fc1/fc2` BF16 tensors with MLX and produced
  `mlx.core.bfloat16` projector output with shape `(1, 750, 2048)` on
  `Device(gpu, 0)`. This exercised the native Audex projector weights.
- Local encoder evidence on 2026-07-07:
  `./start.sh --preflight-audio-encoder --model audex-2b` loaded the real
  Audex-2B `audio_encoder.*` and `audio_projector.*` BF16 tensors with MLX, ran
  all 32 Qwen2Audio/NV-Whisper encoder layers on `Device(gpu, 0)`, and produced
  `mlx.core.bfloat16` shapes `(1, 128, 3000) -> (1, 750, 1280) -> (1, 750,
  2048)`. This exercised the native Audex audio-input embedding path up to, but
  not including, text-token embedding splice.
- Local splice evidence on 2026-07-07:
  `./start.sh --preflight-audio-splice --model audex-2b` used the full Audex
  tokenizer to produce a 770-token prompt containing 750 `<so_embedding>` IDs,
  replaced those rows with `(750, 2048)` projected audio embeddings, and ran a
  one-token MLX-LM generation step on `Device(gpu, 0)` with spliced embeddings
  shaped `(770, 2048)`. This exercises the audio-input embedding path through
  language-model prefill, but not yet speech-token generation.
- The splice uses the same mask/cumulative-index replacement strategy seen in
  current `mlx-vlm` multimodal models: build text token embeddings, identify
  modality placeholder positions, gather projected feature rows in placeholder
  order, and replace only those embedding rows.
- The full Audex tokenizer is required for audio prompts. The text-only
  tokenizer does not preserve `<so_embedding>` as token ID 29, while the full
  tokenizer does. The generated audio prompt IDs stayed below the text-only
  model's 131072-token vocabulary for this smoke.
- Encoder/projector loading intentionally uses `mx.load` instead of a NumPy or
  torch bridge. Current MLX reports `[Load::eval_gpu] Not implemented` if a
  safetensors `Load` node is evaluated directly on GPU, so Audex-Mac
  materializes the file load first and runs subsequent encoder/projector math on
  the GPU default device.
- The STS path now requests host arrays from the NVIDIA/Transformers audio
  preprocessor and converts them directly to MLX. It no longer performs a
  PyTorch `.cpu().numpy()` hop before the Audex audio encoder.
- Fast tests cover the placeholder expansion, clip math, failure on invalid
  placeholders, PCM/WAV preparation, feature-extraction shape, component
  metadata, encoder/projector metadata, and the visible Gherkin scenarios
  "Native audio input prompt uses Audex sound embeddings", "PCM audio is
  prepared as fixed Audex clips", "Audex audio preprocessing produces
  NV-Whisper features", "Audex audio projector maps encoder frames to text
  embeddings", and "Audex audio encoder maps Whisper features to encoder
  frames", and "Audex audio embeddings replace sound placeholder token
  embeddings."
- Local utility evidence on 2026-07-07: `scripts/create-test-utterance.sh`
  correctly fails when `say` returns a zero-frame audio file in the Codex shell,
  instead of silently creating an invalid encoder fixture. The same helper
  should be run from a normal macOS Terminal session if this sandbox behavior
  repeats.
- This milestone is implemented through the one-turn CLI path. The audio-input
  embedding path reaches language-model prefill in preflight mode and drives
  Audex ASR in fixture/interactive STS mode.

### M3: Speech Token Generation

Acceptance:

- generate speech/audio tokens using NVIDIA-equivalent constraints
- preserve recommended sampler settings
- no external TTS

Status:

- Audex-Mac can discover `<speechgen_start>`, `<speechgen_end>`, and
  `<speechcodec_N>` tokenizer IDs directly from `checkpoint_folder_full/tokenizer.json`.
- `iter_new_speech_frames` tracks incremental generated token IDs and emits the
  speech-codec frame indices used by NVIDIA's unified S2S server path.
- Audex-Mac now exposes `./start.sh --preflight-speech-token-generation --model
  audex-2b`, which loads `checkpoint_folder_full` through the
  `nemotron_dense_audex` MLX shim and verifies the full 205312-token LM head on
  `Device(gpu, 0)`.
- Local evidence on 2026-07-07: the full Audex tokenizer maps
  `<speechgen_start>` to 131075, `<speechgen_end>` to 131076, and
  `<speechcodec_0>` to 131077. The text-only 131072-token head cannot emit
  speech tokens, so speech output requires the full checkpoint head.
- The speech-token smoke builds NVIDIA's non-thinking TTS prompt ending in
  `<speechgen_start>` and generated eight consecutive `<speechcodec_N>` tokens
  with NVIDIA's TTS sampler values: `temperature=0.8`, `top_p=1.0`,
  `top_k=0`.
- The direct MLX speech-token path now applies NVIDIA-style
  `tts_cfg_scale=3.0` CFG. It builds the same-length `<unk>` null prompt used
  by NVIDIA's vLLM script, blends logits as
  `uncond_logits + scale * (cond_logits - uncond_logits)`, samples once, and
  feeds the same sampled token into both conditional and unconditional caches.
  Its CFG sampler now uses `temperature=1.0`, `top_p=1.0`, and `top_k=80`;
  the no-CFG vLLM Metal fallback still uses `temperature=0.8`, `top_p=1.0`,
  and `top_k=0`.
- Local evidence on 2026-07-07:
  `./start.sh --preflight-speech-token-generation --model audex-2b` completed
  on `Device(gpu, 0)` with `cfg_applied=True`, generated eight
  `<speechcodec_N>` tokens, and reported `logprobs_shape=(205312,)`.

### M4: Audex Causal Speech Decoder

Acceptance:

- run the Audex causal speech decoder natively on Mac
- stream decoder output if NVIDIA's decoder path makes that practical
- do not invent additional streaming infrastructure for the spike

Status:

- `preflight_decoder` verifies `audex_causal_speech_decoder/config.json`,
  `model.safetensors`, remote-code files, 16 kHz sample rate, lookahead steps,
  and codebook size.
- NVIDIA's decoder loader is torch/Transformers remote code and defaults to
  CUDA in the unified server, but the decoder weights and architecture are
  straightforward enough for a local MLX port.
- `audex_mac.speech_decoder` now ports the Audex causal speech decoder to MLX:
  `AudexSpeechTokenEmbedder`, project-out bias, lookahead depthwise conv,
  causal RoPE self-attention, RMSNorm, SiLU MLP, and the tanh 320-sample patch
  head.
- Local evidence on 2026-07-07:
  `./start.sh --preflight-speech-decoder --model audex-2b` loaded the real
  2.3 GB `audex_causal_speech_decoder/model.safetensors`, ran on
  `Device(gpu, 0)`, embedded 8 codec frames as `(1, 8, 2048)`, and decoded
  finite `mlx.core.float32` waveform samples shaped `(2560,)` at 16 kHz with
  320 samples per codec frame.
- `./start.sh --preflight-speech-output --model audex-2b` combines the
  full-vocab speech-token generator with the MLX decoder and writes a local
  PCM16 WAV plus JSON run log under `.audex/runs/`.
- Local evidence on 2026-07-07:
  `./start.sh --preflight-speech-output --model audex-2b` completed on
  `Device(gpu, 0)`, wrote
  `.audex/runs/speech-output-20260707-142042.wav/json`, and the JSON log
  recorded NVIDIA TTS sampler values with `cfg_applied=true`.

### M5: CLI Speech-to-Speech

Acceptance:

- `./start.sh`
- typed text or push-to-talk utterance capture
- Audex speech-in/speech-out
- no semantic helper models
- run logs written for debugging

Status:

- `./start.sh` now launches a persistent typed-or-push-to-talk CLI when no
  preflight or benchmark flags are supplied. Typed messages bypass ASR and
  retain multiline text; an empty Enter starts recording and the next Enter
  stops it. Both input paths receive spoken Audex responses.
- Microphone capture uses `sounddevice` as deterministic audio plumbing at
  16 kHz mono float32. Playback uses native macOS `afplay`.
- `./start.sh --model audex-2b --input-wav <path> --no-play` runs the same
  one-turn path from a 16 kHz PCM WAV fixture without microphone/speaker access.
- The one-turn path uses only Audex semantic components: Audex native audio
  input for transcription, Audex text generation for a short response, Audex
  speech-token generation, and the MLX Audex causal speech decoder.
- Local evidence on 2026-07-07:
  `./start.sh --model audex-2b --input-wav .audex/fixtures/cli-smoke-tone.wav
  --no-play --response-max-tokens 16 --speech-max-tokens 8` completed
  successfully on `Device(gpu, 0)`, produced transcript `s` from the tone
  fixture, generated response text
  `I see the message got cut off--what would you like to discuss?`, and wrote
  `.audex/runs/speech-output-20260707-142137.wav` plus
  `.audex/runs/sts-turn-20260707-142137.json`.
- The STS run log recorded selected model
  `nvidia/Nemotron-Labs-Audex-2B`, elapsed timing, transcript, response text,
  and the nested speech-output run log. The speech-output log recorded
  `temperature=0.8`, `top_p=1.0`, `top_k=0`, `cfg_scale=2.0`, and
  `cfg_applied=true`.
- `afinfo` verified the output WAV as mono 16 kHz Int16 with 2560 packets.
- Local playback evidence on 2026-07-07: `afplay
  .audex/runs/speech-output-20260707-142137.wav` exited successfully.
- Interactive PTT evidence on 2026-07-07:
  `./start.sh --model audex-2b --no-play --response-max-tokens 16
  --speech-max-tokens 8` reached the push-to-talk prompt, accepted start/stop
  Enter presses, captured
  `.audex/runs/ptt-input-20260707-142823.wav`, ran the Audex STS pipeline, and
  wrote `.audex/runs/sts-turn-20260707-142912.json` plus
  `.audex/runs/speech-output-20260707-142912.wav/json`. The input and output
  WAVs both verified as mono 16 kHz Int16 with `afinfo`.
- Cached 30B default evidence on 2026-07-07:
  `./start.sh --input-wav .audex/fixtures/cli-smoke-tone.wav --no-play
  --response-max-tokens 4 --speech-max-tokens 1` selected
  `nvidia/Nemotron-Labs-Audex-30B-A3B` automatically, ran the native Audex STS
  pipeline, produced transcript `Thank you.`, response `You’re welcome!`, and
  wrote `.audex/runs/speech-output-20260707-161111.wav` plus
  `.audex/runs/sts-turn-20260707-161111.json`.
- Stronger cached 30B default evidence on 2026-07-07:
  `./start.sh --input-wav .audex/fixtures/audex-30b-proof-16k-mono.wav
  --no-play` used no explicit model or token overrides, selected
  `nvidia/Nemotron-Labs-Audex-30B-A3B`, transcribed the real macOS `say`
  fixture as `OutEx Mac Test Utterance. Please answer with one short sentence
  about local speech on Apple Silicon.`, responded `Local speech works smoothly
  on Apple Silicon Macs.`, and wrote
  `.audex/runs/sts-turn-20260707-162111.json` plus
  `.audex/runs/speech-output-20260707-162111.wav/json`.
- The 30B speech-output log for that run recorded `backend=mlx`,
  `device=Device(gpu, 0)`, NVIDIA TTS sampler settings, `cfg_applied=true`,
  `reached_end_token=true`, `hit_max_tokens=false`, 144 generated codec frames,
  finite waveform shape `(46080,)`, 16 kHz sample rate, and non-zero peak
  amplitude. `afinfo` verified the output WAV as mono 16 kHz Int16 with 46080
  packets / 2.88 seconds, and `afplay` exited successfully.
- `scripts/validate-30b-sts.sh` now makes this proof repeatable: it creates the
  macOS `say` fixture, runs `./start.sh --input-wav <fixture> --no-play` with
  auto model selection, and fails if the run does not select cached 30B, use
  MLX GPU, preserve the proof phrase in the transcript, produce a short
  response, finish TTS before the cap, and write a finite mono 16 kHz PCM WAV.
- Local validator evidence on 2026-07-07:
  `scripts/validate-30b-sts.sh` completed successfully. It selected
  Audex-30B-A3B, produced transcript `OutEx Mac Test Utterance. Please answer
  with one short sentence about local speech on Apple Silicon.`, response
  `Apple Silicon runs macOS locally, enabling native speech processing.`, and
  validated `.audex/runs/sts-turn-20260707-162614.json`,
  `.audex/runs/speech-output-20260707-162614.json`, and
  `.audex/runs/speech-output-20260707-162614.wav`.
- Interactive-regression evidence on 2026-07-07:
  A no-argument `./start.sh` PTT run captured
  `.audex/runs/ptt-input-20260707-173734.wav` but appeared stuck for several
  minutes during TTS and was interrupted at
  `audex_mac.speech_generation._generate_tts_cfg_token_ids` inside
  `mx.eval(sampled, logprobs)`. That forced the full 205312-token logprob
  vector to evaluate every speech token on the 30B path. Audex-Mac now
  evaluates only the sampled token, keeps `logprobs_shape` from static shape
  metadata, and prints stage/TTS token progress during the STS turn.
- Replay evidence for that captured PTT input:
  `./start.sh --input-wav .audex/runs/ptt-input-20260707-173734.wav --no-play`
  selected Audex-30B-A3B, printed visible progress through ASR/text/TTS,
  transcribed `Hello, how's it going?`, responded
  `Hey! I'm doing great, thanks for asking.`, generated 99 speech tokens, and
  wrote `.audex/runs/sts-turn-20260707-174207.json` plus
  `.audex/runs/speech-output-20260707-174207.wav/json`. The speech-output log
  recorded `backend=mlx`, `device=Device(gpu, 0)`,
  `reached_end_token=true`, `hit_max_tokens=false`, finite waveform shape
  `(31360,)`, and `peak_abs=0.4862576723098755`; `afinfo` verified mono
  16 kHz Int16 output with 31360 packets / 1.96 seconds.
- The terminal CLI now displays incremental ASR text and incremental response
  text while the MLX generation loops decode tokens. During TTS, Audex is
  generating `<speechcodec_N>` tokens rather than display prose, so the CLI
  keeps the final spoken response visible and prints speech-token progress.
- The CLI is functionally end-to-end but not yet latency-optimized; the simple
  implementation reloads model components between ASR, text response, and TTS.

## Text Benchmark Shape

Use at least ten turns of coding/basic reasoning questions. Avoid world
knowledge as pass/fail. Judge coherence, context retention, instruction
following, repetition, and runtime stability.

## Sources

- https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B
- https://huggingface.co/nvidia/Nemotron-Labs-Audex-30B-A3B
- https://github.com/vllm-project/vllm-metal
- https://github.com/Blaizzy/mlx-audio
- https://github.com/Blaizzy/mlx-vlm
## 2026-07-09 Controlled CFG TTS Quality Gate

The long-form quality harness compares three code-enforced recipes without
changing interactive STS defaults: plain TTS at `temperature=0.8`, `top_k=0`,
and `cfg_scale=1.0`; sampler-matched NVIDIA TTS CFG at `0.8`, `0`, and `2.0`;
and the existing Audex CFG experiment at `1.0`, `80`, and `3.0`. Every arm uses
seed `20260709`, one explicit 46-to-80-word segment per passage, compact
speech-token-window decode, and the product MLX causal speech decoder. This
removes product chunking, interleaving, and decoder-window differences from the
quality comparison.

The six-case corpus covers conversational narrative, proper names, numbers and
identifiers, technical vocabulary, expressive dialogue, and long-form
stability. All 18 run logs recorded exactly one observed segment, clean
`<speechgen_end>` completion, and no max-token hit. Raw manifests and offline
Parakeet oracle reports are under `.audex/runs/`; the randomized listener set is
under `.audex/listening/`, while its recipe key is stored separately. None of
those generated WAVs or private keys are repository artifacts.

The human blind result preferred `audex-cfg3` in four of six groups and nearly
five after a close reconsideration of the long-form group. Sampler-matched
`nvidia-tts-cfg` won one group and `plain-reference` won one. The listener
repeatedly attributed robotic delivery, multi-voice starts, and harsh onsets to
samples later decoded as no-CFG. Automated transcription did not predict this
preference: mean WER was 7.0% for plain, 8.3% for CFG2, and 12.3% for CFG3.
This is evidence from one listener, one seed, and six passages—not a MOS study.

## 2026-07-09 Release Context Budget Correction

The interactive vLLM path falsely rejected a resumed 5,025-token prompt because
`_text_context_token_limit()` fell back to 5,120 and reserved 4,096 tokens for
the response. The no-CFG engine in that run was actually configured at 262,144
tokens; the 5,120 value was an Audex-Mac guard error.

The release configuration now pins both plain and CFG3 engines to a 262,144-token
Mac-demo context. CFG3 reserves capacity for two worst-case long sequences
instead of eight, leaving a pool large enough for one long text request and the
many short conditional/unconditional TTS requests. The live 30B Gilroy
regression completed under this shape: the turn log
`.audex/runs/sts-turn-vllm-20260709-191251.json` recorded 5,029 prompt tokens
against the 262,144-token limit, CFG3 enabled, and a clean TTS completion.

Conversation transcripts remain exact and are fully re-prefilled after restart.
The vLLM path does not yet serialize or inject a persisted conversation cache;
the CLI no longer prints a nonexistent `.kv.safetensors` path. No compaction or
summarization is in scope. NVIDIA advertises up to one million tokens for the
model, while this Mac demo deliberately stops at 256K.

Activity Monitor snapshots during the earlier product-shaped corpus run showed
phase-dependent utilization: EngineCore was observed around 68% GPU and 85%
CPU while the parent Python MLX decoder also used GPU, then a later CFG phase
showed roughly 87% GPU and 43% CPU. A future profile should correlate recipe and
model-decode versus waveform-decode phases before attributing the lower-GPU
snapshot to host-side NumPy, scalar synchronization, or cache-copy work.

## 2026-07-10 Sound Lab Text-to-Audio Evidence

The first owner-run `./sound.sh` auditions produced recognizable non-speech dog
barks through Audex CFG3 and the full XCodec1 decode path. The local catalog
contains eight candidates across two jobs: five ready ten-second WAVs and three
failures. All three failures were classified only as `incomplete_target`; none
recorded an RVQ phase mismatch, unexpected token, or missing end token. The old
harness discarded the early stream before recording its frame count, so those
three durations cannot be recovered retroactively.

Sound Lab now submits at most two CFG pairs per wave under an 8K
Sound-Lab-only context reservation, matching NVIDIA's documented generation
command. NVIDIA's script decodes any nonempty phase-valid stream and pads/trims
the waveform to ten seconds; Sound Lab now does the same. Phase-invalid streams
receive one bounded retry. Initial and retry
structure, timing, and actual seeds are persisted; a
retry failure cannot invalidate candidates already published from the first
pass. This revision has host-side test coverage but still needs an owner-run
timing/listening pass to measure the new failure rate and continuous-batch
throughput on Apple Silicon.

The first post-batch owner rerun failed safely during engine startup: the
headroom guard reported ten 256K non-paged slots requiring 139.86 GB against
95.08 GB of current Metal headroom. `sound.sh` had selected the CFG3 recipe but
had not enabled the separate vLLM CFG engine-wiring switch, so the 8K CFG context
override was never applied. Sound Lab now exports
`AUDEX_VLLM_ENABLE_CFG_WIRING=1` alongside its recipe, context, and capacity
settings. The memory guard was correct and remains unchanged.

The initial board implementation mislabeled a deterministic 16-to-48 kHz sample
rate/channel conversion as an enhanced WAV. NVIDIA's reported TTA benchmark
instead uses its released 2.62 GB FP32 enhancement VAE after XCodec1 decoding.
Sound Lab now loads that checkpoint lazily on MPS, resets NVIDIA's default VAE
seed `0` for each clip, preserves raw 16 kHz WAVs, and serves the learned 48 kHz
mono reconstruction. A local MPS smoke test successfully enhanced an existing
ten-second raw Sound Lab WAV.

The first controlled BF16-versus-NVFP4 TTA listening corpus completed on
2026-07-10. It rendered eight identical literal captions and seeds per profile,
in two-pair CFG3 waves, through the same XCodec1 and stochastic enhancement VAE
seed `0`. Both manifests completed all eight cases; all 16 public listening WAVs
are mono 48 kHz and exactly ten seconds. BF16 generation-result elapsed values
sum to 812.457 seconds versus 638.601 seconds for NVFP4 (per-request values
overlap within each wave and are not wall time). The blind set is local under
`.audex/listening/tta-quant-20260710-223620`; WAVs and its private decoding key
remain excluded from Git.
