# patch.md

## meta

- name: audex-mac-vllm-metal-audex-support
- version: 0.1.2
- author: Audex-Mac maintainers
- target: Audex-Mac with pinned vllm-metal cd72e7d6d5c3eec452afe2693c3a45a0564d7650
- spec: 0.1.0

## context

- LICENSE

## scope

- create: [audex_mac/interactive_input.py, audex_mac/nvfp4_conversion.py, docs/README.md, docs/engineering/nvfp4-quantization.md, docs/engineering/patches.md, docs/engineering/viability.md, docs/engineering/vllm-metal.md, docs/operations/runbook.md, docs/project/development.md, docs/project/scope.md, scripts/quantize-30b-nvfp4.sh, tests/test_interactive_input.py, tests/test_nvfp4_conversion.py]
- modify: [.github/**, PATCH.md, README.md, audex_mac/**, features/**, personas/**, pyproject.toml, start.sh, tests/**, vendor_pins.json]
- delete: [DEVELOPMENT_OUT_LOUD.md, PATCHES.md, RUNBOOK.md, SLC.md, docs/VIABILITY.md, docs/VLLM_METAL.md]

## files

### README.md (modify)

Present Audex-Mac as a working local Mac demo, distinguish the substantially
validated 30B path from minimally tested 2B support, describe the configured
256K context accurately, and keep local audio/model artifacts out of Git.

### .github/** (modify)

Keep issue and pull-request templates pointed at the maintained documentation
paths after the release documentation migration.

### start.sh (modify)

Bootstrap the pinned vllm-metal checkout and generated runtime environment
under `.audex/vendor/vllm-metal`. Export the Metal/MLX environment required for
Apple GPU execution, install Audex-Mac generated-venv shims, apply patch guards,
and fail loudly when the pinned dependency API shape is not compatible. Warn
loudly, but do not block startup, when upstream vllm-metal has moved beyond the
pinned commit. Default the widened non-paged KV capacity only for CFG product
runs; preserve explicit user overrides and leave no-CFG capacity to vLLM Metal.

### vendor_pins.json (modify)

Pin vllm-metal to an exact upstream commit. Do not advance this pin without
updating `docs/engineering/patches.md`, this patch contract, patch guards, and validation
evidence for the new upstream shape.

### audex_mac/patches/install.py (modify)

Install generated venv shims for `mlx_lm` Audex model modules and
`sitecustomize.py`. The shims must make spawned vLLM EngineCore workers apply
Audex-Mac runtime patches when `AUDEX_MAC_AUTO_PATCHES=1`.

### audex_mac/patches/runtime.py (modify)

Keep one repeatable runtime patch entrypoint that returns a structured
`AudexPatchReport`. It must apply the Transformers local dynamic-module patch,
MLX-LM Audex model aliases, vLLM Metal platform repair, current MLX device-info
API compatibility patch, vLLM model architecture registration, and the
vLLM-Metal Audex adapter patch.

### audex_mac/patches/transformers_dynamic_module.py (modify)

Repair Transformers local dynamic-module resolution for Audex snapshots whose
remote-code files are resolved through Hugging Face cache blobs before relative
imports are discovered. Keep this patch narrow and guarded so unrelated
Transformers dynamic-module behavior is not changed.

### audex_mac/patches/mlx_lm_nemotron_dense.py (modify)

Provide the MLX-LM model surface needed for Audex-2B text/speech checkpoints on
Mac when the installed MLX-LM release lacks the required Nemotron Dense Audex
module.

### audex_mac/patches/mlx_lm_nemotron_h_audex.py (modify)

Provide the MLX-LM model surface needed for Audex-30B-A3B Nemotron-H Audex
checkpoints on Mac when the installed MLX-LM release lacks the required native
Audex module.

### audex_mac/patches/vllm_metal_audex_adapter.py (modify)

Patch the pinned vllm-metal adapter surface to support Audex projected-audio
inputs. Register the Audex model architectures, multimodal processor, projected
audio feature spec, placeholder range, and model-runner adapter behavior needed
for `audex_projected_embeddings` to reach the MLX model forward path in
EngineCore.

### audex_mac/patches/vllm_metal_cfg.py (modify)

Patch the pinned vllm-metal scheduler, sampler, and nonpaged decode path for
Audex CFG TTS. Preserve NVIDIA's CFG semantics: paired conditional and
unconditional requests, `cfg_scale=3.0`, `cfg_pairs_per_batch=2`,
`temperature=1.0`, `top_k=80`, legal speech-token masks, and the conditional
sample copied into the matching unconditional row. Keep nonpaged async submit,
stable batch-cache reuse, diagnostics, and opt-in experimental decode paths
guarded by explicit environment switches.

### audex_mac/vllm_cfg.py (modify)

Build the vLLM engine kwargs for Audex CFG mode. Import NVIDIA's bundled
`CFGLogitsProcessor` from the selected model snapshot, disable prefix caching
for CFG, install the Audex-Mac Metal token-sync patch, and size
`max_model_len`, `max_num_batched_tokens`, and `max_num_seqs` with Mac unified
memory pressure in mind. Do not silently fall back to unconfigured CFG.

### audex_mac/cli.py (modify)

Bound the release demo's active conversation context to 262,144 tokens, reject
explicit CLI values outside that range, and report whether startup selected the
default CFG3 quality recipe or the explicit no-CFG speed path. Do not advertise
a conversation KV-cache artifact that the vLLM path does not create.

### audex_mac/interactive_input.py (create)

Provide one multiline terminal editor for typed-or-spoken turns. Ordinary
Enter submits text, Shift+Enter inserts a newline when the terminal reports
modified keys, Option+Enter is the portable multiline fallback, and an empty
submission selects push-to-talk recording. Enable Ghostty/Kitty keyboard
reporting only while the editor is active and restore it afterward.

### audex_mac/sts_cli.py (modify)

Keep the direct-MLX diagnostic backend aligned with the product input contract:
typed turns bypass ASR, share conversation history, generate Audex speech, and
record `input_mode=text` plus `asr_skipped=true` in their run logs. Select the
startup greeting from new-versus-resumed conversation state, never from whether
a binary cache artifact happened to load.

### audex_mac/nvfp4_conversion.py (create)

Build a deterministic Hugging Face-shaped local snapshot for
`txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx`. Quantize only the 46 fused
routed-expert projections with MLX-native NVFP4 group size 16. Preserve the
audio encoder/projector, speech decoder, router, attention, Mamba, shared
experts, embeddings, norms, and LM head at source precision. Validate the
output index before updating `refs/main`, and do not label the artifact oQe.
Generate the Hugging Face family-tree metadata, a concise linked description of
the conversion and Audex-Mac demo, NVIDIA's complete upstream card, and the
upstream diagram assets as reproducible publication contents rather than
relying on a hand-edited Hub README. Keep the detailed selective-precision
recipe and runtime evidence in the repository documentation and artifact
manifest rather than expanding the Hub introduction.

### audex_mac/models.py and audex_mac/model_select.py (modify)

Treat the NVFP4 snapshot as the preferred cached 30B speech model while
retaining the NVIDIA BF16 30B model as the source/download recommendation and
the 2B model as the uncached first-run default.

### scripts/quantize-30b-nvfp4.sh (create)

Expose the reproducible conversion through the pinned MLX/vLLM-Metal Python
environment. Keep generated weights in the Hugging Face cache, never in Git.

### audex_mac/conversations.py (modify)

Persist the 262,144-token demo context policy without compacting or silently
dropping history. On resume, prefer the most recent explicit user identity over
stale stored metadata so benchmark prompts cannot permanently pollute the
conversation greeting.

### audex_mac/vllm_sts_requests.py (modify)

Build NVIDIA-shaped Audex ASR, text response, no-CFG TTS, and CFG TTS requests.
Use the model-card sampler defaults for each task. Carry speech-token window
metadata and CFG pair metadata through `SamplingParams.extra_args` so the
vllm-metal patches can enforce the correct token contract.

### audex_mac/vllm_runtime.py (modify)

Own one persistent async vLLM Metal runtime for the STS CLI. Route ASR, text
generation, no-CFG TTS, CFG TTS, segmented TTS, and streaming token deltas
through that runtime without loading duplicate engines. Preserve ordered speech
segments while allowing vLLM continuous batching to keep paired CFG requests
active. Expose vLLM's scheduler-side prefix-cache reset so the CLI can clear
completed ASR/text KV bookkeeping before starting a differently shaped static
CFG batch.

### audex_mac/vllm_sts_cli.py (modify)

Keep `./start.sh` usable as one typed-or-push-to-talk conversation that always
answers with Audex speech. Typed messages must bypass ASR but otherwise use the
same text, TTS, conversation persistence, context, and cache-reset paths.
Record transcripts, response text, WAV artifacts, first-audio timing, playback
timing, underrun/overrun diagnostics, CFG/no-CFG routing, segment-level TTS
metadata, MLX memory snapshots, and generated speech-token metadata. Use the
no-CFG recipe as the default speed path and preserve CFG3 as an explicit
quality option/control. A resumed conversation with substantive history must
generate an ephemeral, history-grounded greeting of at most two sentences,
without reintroducing Audex or persisting the synthetic greeting prompt. Before
static CFG TTS, reset the idle scheduler's KV/prefix cache and fail loudly if
vLLM refuses the reset; do not apply that reset to the no-CFG/interleaved path.

### audex_mac/vllm_diagnostics.py (modify)

Provide structured diagnostics for vLLM Metal device selection, spawned worker
environment, patch installation, Audex adapter readiness, CFG wiring, native
sampling timing, nonpaged decode timing, persistent-cache counters, playback
evidence, and STS smoke results. Do not treat Activity Monitor alone as proof
of CPU or GPU execution.

### audex_mac/speech_output.py (modify)

Keep speech-output WAV writing and PCM packing deterministic. Avoid adding MLX
host synchronization to the hot streaming output path unless it is behind an
explicit diagnostic flag.

### tests/** (modify)

Cover every patched upstream seam, generated venv shim, CFG request shape,
sampler/token-sync behavior, nonpaged cache behavior, STS CLI behavior,
diagnostic verdict, startup behavior, CFG3 default, and no-CFG fallback
invariant. Prefer
focused tests near the changed module, then run the full suite before commit.

### features/** (modify)

Keep the executable behavior descriptions aligned with the demo terminology,
CFG3 launcher default, current documentation paths, and 256K context policy.

### personas/** (modify)

Keep the stock spoken assistant persona explicitly identified as Audex and
constrain normal answers to fewer than five sentences unless the user asks for
depth.

### pyproject.toml (modify)

Describe the package as an Apple Silicon Audex speech-to-speech demo rather
than a compatibility spike.

### docs/** (create)

Move the former root-level project notes into an indexed `docs/` hierarchy and
keep user-facing and agent-facing documentation aligned with the current patch
contract, especially vLLM Metal startup behavior, CFG/no-CFG routing,
diagnostic commands, known performance bottlenecks, and validation evidence.

### Root legacy docs (delete)

Delete the former documentation paths only after their maintained content is
relocated to the destination paths listed in `scope.create`.

### docs/engineering/patches.md (create)

Use `docs/engineering/patches.md` as the chronological research and experiment ledger. Record
attempted patches, negative results, GPU-visible fixture artifacts, oracle
results, and reapply notes there. Keep this `PATCH.md` as the stable
standards-facing patch contract.

### PATCH.md (modify)

Keep this file conformant with patch.md source-tree patch semantics: read-only
context paths must not overlap modified files, file-operation headings must use
`create`, `modify`, or `delete`, and the verify section must remain the stable
behavioral contract for Audex-Mac-owned patches.

## env

- AUDEX_MAC_AUTO_PATCHES: required for spawned vLLM workers
- VLLM_METAL_USE_MLX: required, value `1`
- VLLM_MLX_DEVICE: required, value `gpu`
- VLLM_METAL_USE_PAGED_ATTENTION: diagnostic-dependent; default may be disabled for Nemotron-H nonpaged paths
- AUDEX_VLLM_TTS_CFG: optional; defaults to `0` for the no-CFG speed path, while
  explicit `1` selects CFG3 quality mode
- AUDEX_VLLM_ENABLE_CFG_WIRING: optional; set automatically when CFG TTS is enabled
- AUDEX_VLLM_CFG_MAX_NUM_SEQS: optional CFG concurrency diagnostic override
- AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS: optional CFG scheduler/token-budget diagnostic override
- AUDEX_VLLM_CFG_MAX_MODEL_LEN: optional diagnostic override for CFG engine
  context sizing; do not treat it as a concurrency win unless native timing
  proves wider CFG admission
- AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL: optional diagnostic override for
  vLLM scheduler prompt-token reservation behavior
- AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS: optional vLLM Metal non-paged scheduler
  capacity override; `./start.sh` defaults it to `2` when CFG is enabled so the
  256K engine reserves two worst-case max-length sequences while retaining
  enough shorter blocks for product CFG pairs. No-CFG preserves vLLM Metal's
  own default unless the user explicitly overrides it
- AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS: optional diagnostic override for static
  CFG TTS chunk target width; the product path caps the effective target to
  `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` when that capacity override is set
- AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS: optional A/B override for the
  scheduler KV/prefix-cache reset before static TTS; CFG defaults it on and
  no-CFG defaults it off
- AUDEX_VLLM_SPEECH_FIRST_SCHEDULING: optional A/B override; defaults on so an
  active or waiting Audex TTS request receives exclusive scheduler steps while
  continuing response text remains resident and resumes afterward
- AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL: optional A/B override; defaults off so
  future TTS chunks cannot reduce the currently audible chunk below real time
- AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET: optional diagnostic; default `logits`, alternatives `sample_logits` or `none`
- AUDEX_VLLM_SKIP_SPEECH_DECODER: optional product-path timing diagnostic that
  discards generated codec frames instead of invoking the causal speech decoder;
  never enable for intelligibility validation or normal `./start.sh` use
- --diagnose-vllm-tts-text / --diagnose-vllm-tts-text-file: diagnostic-only
  product-path fixed-text TTS probes. These must instantiate
  `VllmSpeechToSpeechSession` and call `generate_speech_output(...)`, not the
  lower-level runtime probe, so they preserve product decoder/run-log behavior.

## run

- post: .venv/bin/python -m pytest -q
- post: .venv/bin/python -m ruff check audex_mac tests

## verify

- `./start.sh` launches the vLLM Metal STS CLI by default when a supported Audex checkpoint is cached.
- At the interactive `You:` editor, nonempty Enter submits typed text directly
  to the conversation model, Shift+Enter inserts a newline in Ghostty, and an
  empty Enter starts the existing push-to-talk capture path.
- Typed turns skip audio projection and ASR, retain exact multiline user text
  in conversation history, synthesize the answer with Audex TTS, and record
  `input_mode=text` and `asr_skipped=true`.
- When both local snapshots are complete, `./start.sh` prefers
  `txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx` over the BF16 30B source.
- The NVFP4 recipe emits exactly 46 routed-expert `.scales` tensors and no
  quantization scales for attention, Mamba, shared experts, modality
  components, embeddings, or the LM head.
- NVFP4 acceptance requires GPU-visible audio encoder/projector and causal
  speech-decoder smoke passes in addition to a backbone load/generation pass;
  text-only coherence is not sufficient evidence.
- `./start.sh --sts-backend mlx` is required for the direct MLX diagnostic fallback.
- `./start.sh --diagnose-vllm-metal` reports MLX GPU device evidence, Audex patch readiness, vLLM Metal platform repair, Audex adapter readiness, and CFG wiring readiness in a GPU-visible terminal.
- `AUDEX_VLLM_TTS_CFG=1 ./start.sh` enables the opt-in CFG3 quality recipe:
  `cfg_scale=3.0`,
  `cfg_pairs_per_batch=2`, `temperature=1.0`, `top_k=80`, legal speech-token
  masks, and paired conditional/unconditional requests.
- Plain `./start.sh` selects the intelligible no-CFG low-latency recipe.
- Interleaved playback requests 50 ms of device latency and retains the 80 ms
  application prebuffer. Every reported device or application-queue underrun is
  an audible failure; long-form acceptance requires all underrun counters to be
  zero.
- Speech-first scheduling temporarily omits non-TTS running requests from each
  scheduler step while TTS is active without aborting them or freeing their
  KV/state. Response text resumes when the speech request finishes. Interleaved
  tail TTS batching defaults off so future chunks do not compete with the chunk
  feeding playback.
- The no-CFG vLLM Metal fallback remains intelligible on the normal fixture and does not inherit CFG-only sampler or scheduler changes.
- The 256K CFG launcher defaults to two max-length non-paged sequences. The
  resulting pool must admit one long text request and the shorter product CFG
  pairs while remaining below measured Metal headroom.
- Static CFG STS resets completed ASR/text scheduler KV/prefix-cache state
  before submitting paired TTS requests. The turn log records
  `tts_prefix_cache_reset=true`; a refused reset fails loudly rather than
  silently continuing on the slower retained state.
- A fixture run with `.audex/fixtures/normal-answer-question-16k-mono.wav` writes a generated WAV and JSON run log with first-audio timing, stream timing, codec-frame throughput, decoder timing, playback diagnostics when playback is enabled, and segment-level TTS metadata.
- A generated CFG WAV passes the local ASR intelligibility oracle with no excessive repetition before a CFG speed patch is treated as product-ready.
- For CFG chunking changes, run a fixed-text product-path TTS probe before an
  ASR->LLM->TTS fixture. This avoids treating a changed text response as
  evidence for or against the TTS scheduler/chunker.
- Static CFG TTS chunking must not create more initial chunks than the target
  admitted pair width. The CLI may split long CFG sentence atoms first, then
  partitions ordered atoms into contiguous groups that minimize the largest
  character-cost group. This avoids second scheduler waves without piling up
  short-tail or re-split heuristics.
- End-to-end STS diagnostics must stamp timing evidence invalid when the LLM
  response is too short to trust as a TTS/chunking benchmark.
- The CLI must not silently prune or summarize conversation history. The Mac
  demo ceiling is 262,144 tokens, clamped to the selected checkpoint's declared
  limit (131,072 for Audex-2B), and fails loudly with prompt-token,
  response-token, and engine-budget details when that exact window is exceeded.
- vLLM conversation-state carryover must preserve an exact append-only token
  prefix: the committed text history prompt for turn N must be a token prefix
  of the generation prompt for turn N+1 before any saved cache/state is reused.
- vLLM conversation-state carryover targets the non-paged Nemotron-H path with
  one append-only snapshot, not generic paged prefix caching. Any history rewrite
  or sanitizer change invalidates the saved snapshot; reuse after mismatch must
  fail loudly or rebuild, never silently continue.
- Text conversation-state metadata is carried through `SamplingParams.extra_args`
  using Audex-owned keys so the vLLM Metal fork can capture/reuse a single
  append-only state snapshot without changing vLLM's public `generate()` call.
- Only snapshots stamped as committed-history prefill boundaries may become
  reusable. Raw generation cleanup snapshots remain diagnostic-only because
  sampled tokens may differ from the scrubbed text committed to the transcript.
- `docs/engineering/patches.md` records any experiment, negative result, upstream symbol, guard, and reapply note that materially changes this patch contract.
- Standalone CFG probe artifacts must show `runtime.cfg_enabled=true` and `runtime.cfg_scale=3.0` before they are treated as CFG performance evidence.
- CFG concurrency claims must be backed by native timing evidence for actual
  `cfg_cond_reqs`, `cfg_uncond_reqs`, and `cfg_complete_pairs`, not only by
  configured `max_num_seqs` or `max_num_batched_tokens`.
- Any change to `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` must be accompanied by
  native capacity/admission evidence showing `num_blocks`,
  `max_length_blocks_per_request`, `inferred_request_capacity`, and actual
  `cfg_complete_pairs`. The runtime patch must also log MLX/Metal headroom and
  refuse a capacity override whose max-length KV worst case exceeds the current
  MLX working-set headroom.
- STS smoke evidence must surface CFG TTS segment-balance metrics from
  `tts_segment_codec_frame_counts`; an underfilled final segment in a
  multi-segment CFG run is a chunk-planner failure, not valid throughput
  evidence.
- Full pytest passes before commit; pre-commit hooks are not bypassed.

## rollback

Restore the modified source files from git, remove generated venv shims under
`.audex/vendor/vllm-metal/.venv-vllm-metal` if they were installed, rerun
`./start.sh --diagnose-vllm-metal`, and confirm the failure mode is explicit
rather than a silent CPU or direct-MLX fallback.
