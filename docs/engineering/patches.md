# Audex-Mac Patch Ledger

Audex-Mac keeps vLLM Metal as an external pinned dependency and applies explicit
monkey patches at startup. This file is the patch ledger.

The primary patch target is vLLM Metal. The spike may also patch adjacent MLX
runtime packages installed by vLLM Metal, including `mlx_lm`, `mlx_vlm`, or
`mlx-audio`, when Audex support requires it. Those patches must be owned here
and recorded with the same level of detail as vLLM Metal patches.

## Typed-or-Spoken Interactive Turns

Added on 2026-07-09 so a user can type privately while retaining Audex's spoken
output and the existing push-to-talk path.

- `audex_mac.interactive_input` uses `prompt_toolkit` for multiline editing.
  Enter submits, empty Enter selects recording, and `q`/`quit`/`exit` as the
  complete submission exits.
- `prompt_toolkit` is a direct dependency because Python's existing `input()`
  path cannot keep modified Enter distinct from submission or provide an
  editable multiline buffer. It stays inside the local Python runtime, adds no
  service or model dependency, and avoids a repository-owned raw-terminal line
  editor; the version range is pinned to the current compatible major release.
- Ghostty, Kitty, and WezTerm use the Kitty keyboard protocol while the editor
  is active; iTerm uses xterm `modifyOtherKeys`. Shift+Enter is mapped to a
  newline without collapsing ordinary Enter, and terminal keyboard reporting
  is restored after each prompt. Option+Enter is also bound as a fallback.
- Typed turns call `run_turn_from_text(...)` on the existing persistent session.
  They bypass the audio encoder and ASR, preserve the exact multiline user
  message, then use the normal response and TTS path.
- Run logs distinguish typed turns with `input_mode=text`, `typed_text`,
  `input_wav_path=null`, and `asr_skipped=true`.
- PTY evidence under `TERM_PROGRAM=ghostty` parsed Kitty Shift+Enter as a
  newline and ordinary Enter as submission, yielding
  `First line.\nSecond line.`. The parser restored Kitty keyboard reporting on
  exit.

## Audex-30B NVFP4 Expert Conversion

Added on 2026-07-09 as the first local quality trial for reducing the 30B
model's unified-memory footprint without quantizing its modality boundaries.

- Source: `nvidia/Nemotron-Labs-Audex-30B-A3B`, revision
  `79e4bf1a5dabe09ceb938570e9617357174560e5`
- Output identity:
  `txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx`
- Recipe: `audex_mac/nvfp4_conversion.py`, invoked by
  `scripts/quantize-30b-nvfp4.sh`
- Policy: MLX NVFP4 group 16 for the 46 fused routed-expert `fc1`/`fc2`
  projections; all other backbone, audio, and decoder weights remain at source
  precision. This revision makes no oQe claim because current oMLX imatrix
  weighting is affine-only.
- Generated checkpoint metadata: 5.436 effective bits per weight, 46 scale
  tensors, 490 audio encoder/projector tensors, and 23,034,277,632 bytes in the
  combined language/audio checkpoint index.
- Local snapshot: 25 GiB logical size including the source-precision decoder
  and NV-Whisper assets. Unchanged assets are APFS hard-linked when possible.
- GPU-visible validation:
  - direct MLX backbone load used 21,727,832,396 active bytes and completed a
    one-token generation call;
  - BF16 audio encoder/projector produced shapes
    `(1, 128, 3000) -> (1, 750, 1280) -> (1, 750, 2688)`;
  - causal speech decoder produced a finite 2,560-sample 16 kHz waveform.
- Negative result: the vLLM generation diagnostic's captured subprocess could
  remain blocked on a macOS multiprocessing resource-tracker after its child
  exited, especially while another `start.sh` engine was resident. Direct MLX
  generation was used to validate the quantized loader/kernel path; normal
  interactive `start.sh` remains the user quality gate.
- Publication decision on 2026-07-10: the repository owner accepted the
  conversational quality after local use, and the completed NVFP4 artifact
  passed two consecutive semantic-audio gates with coherent model speech and
  sub-second estimated DAC onset. This authorizes an explicitly experimental
  public release; general-audio, audio-to-audio, formal BF16 equivalence, and
  long-context evaluation remain disclosed limitations rather than completed
  claims.
- Reapply notes: bump `RECIPE_ID` for any precision-policy change so a distinct
  deterministic snapshot revision is created. Do not broaden quality claims
  beyond the acceptance evidence and disclosed limitations above.

## Policy

- Do not auto-upgrade vLLM Metal.
- Pin vLLM Metal to an exact Git commit.
- Runtime uses the pinned commit even when upstream `main` has moved.
- Upstream movement is advisory only: warn loudly, log the update prompt, and
  continue startup.
- Patch guard failure against the pinned install is fatal.
- Every patch must include:
  - upstream repository and commit
  - upstream file/symbol touched
  - why the patch exists
  - expected API-shape guard
  - reapply/update notes
  - tests or BDD scenarios covering the behavior

## Current Pin

Expected dependency:

- Repository: `https://github.com/vllm-project/vllm-metal`
- Commit: `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`
- Pin metadata: `vendor_pins.json`

The pin was selected from upstream `main` during the initial Audex-Mac
scaffold. It should not be changed without updating this file, patch guards,
and the relevant fast tests.

## Pinned Source Findings

Inspected from the pinned tarball on 2026-07-07.

- Package version: `vllm-metal==0.3.0`
- Python requirement: native arm64 Python `>=3.12,<3.14`
- Load seam:
  `vllm_metal.v1.model_lifecycle.ModelLifecycle._load_generation_model`
- Model adapter seam:
  `vllm_metal.v1.model_adapter.DefaultModelAdapter`
- Native multimodal adapter seam:
  `vllm_metal.v1.model_adapter.DefaultModelAdapter.build_multimodal_adapter`
- Paged multimodal execution and embedding splice seam:
  `vllm_metal.v1.model_runner.MetalModelRunner._run_mm_paged_forward`
- Existing multimodal support at this pin is image-oriented: Qwen3-VL and
  PaddleOCR-VL. Audex audio support still needs Audex-Mac-owned patches.
- `mlx_lm==0.31.3` is installed by the pinned vLLM Metal runtime. It includes
  `nemotron_h` but does not include `nemotron_dense`, so Audex-2B text support
  needs an Audex-Mac-owned `mlx_lm` model patch or equivalent loader patch.

## vLLM Metal CPU Fallback Investigation

Added on 2026-07-07 as the first executable checkpoint for
`docs/engineering/vllm-metal.md`.

- Diagnostic command: `./start.sh --diagnose-vllm-metal`
- Heavy timing command:
  `./start.sh --diagnose-vllm-generation`
- Report artifact:
  `.audex/runs/vllm-metal-diagnostic-YYYYMMDD-HHMMSS.json`
- Audex-Mac code:
  - File: `audex_mac/vllm_diagnostics.py`
  - CLI flag: `audex_mac.cli --diagnose-vllm-metal`
  - Fast tests: `tests/test_vllm_diagnostics.py`
  - BDD scenario: `features/patch_guards.feature`
- Evidence collected:
  - parent environment for `VLLM_METAL_USE_MLX`, `VLLM_MLX_DEVICE`, and
    `VLLM_METAL_USE_PAGED_ATTENTION`
  - spawned subprocess environment, standing in for macOS `spawn` inheritance
    failures before launching a full `EngineCore`
  - parent and spawned `mlx.core.default_device()`
  - parent and spawned `mx.metal.is_available()`
  - parent and spawned probe-array device
  - vLLM Metal platform class, device facade, Ray device key, and config
  - selected Audex model path/readiness and active Audex patch report
  - `DefaultModelAdapter` source file and whether the adapter source mentions
    Audex native multimodal support
  - when `--diagnose-vllm-generation` is used, vLLM model-load time, one-token
    latency proxy, and long-prompt throughput using the benchmark's
    `max_tokens` value (4096 by default)
  - source scan hits for expected CPU facade, explicit `DeviceType.cpu`,
    paged-attention config, and MLX device config
- Verdict behavior:
  - returns ready only when parent/spawned MLX probes show GPU availability,
    vLLM's effective `current_platform` is
    `vllm_metal.platform.MetalPlatform`, all Audex runtime patches install, and
    the vLLM Metal adapter has Audex-native multimodal support
  - returns not-ready/nonzero when vLLM falls back to
    `vllm.platforms.cpu.CpuPlatform`, even though the CPU facade itself is
    expected on `MetalPlatform`
- Initial source conclusion: the pinned vLLM Metal platform intentionally
  exposes `device_type = "cpu"` and `device_name = "cpu"` to vLLM/PyTorch for
  compatibility while `MetalPlatform.set_device()` sets MLX's default device
  from `VLLM_MLX_DEVICE`. Activity Monitor CPU usage and vLLM's CPU device
  facade are therefore not sufficient proof of CPU fallback. The diagnostic
  treats MLX device evidence as authoritative.
- Actual CPU fallback indicators:
  - `VLLM_METAL_USE_MLX != 1`
  - `VLLM_MLX_DEVICE != gpu`
  - `VLLM_METAL_USE_PAGED_ATTENTION != 1` on the default path
  - `mx.default_device()` is not `Device(gpu, 0)`
  - probe/model arrays report a CPU device
  - vLLM Metal config reports `use_mlx=false` or `mlx_device=cpu`
- Remaining live investigation:
  - run the diagnostic in the generated vLLM Metal venv on the target Mac
  - add a model-loading probe that records real Audex weight devices without
    forcing a full conversational run
  - add a tiny vLLM generation timing probe once model loading is confirmed
  - prove the current Audex multimodal adapter and processor patches with real
    projected audio in a GPU-visible vLLM Metal `EngineCore`

Observed local diagnostic run from the Codex execution context:

- Command: `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-212031.json`
- Exit: `2`
- Findings:
  - parent/spawned environment preserved `VLLM_METAL_USE_MLX=1`,
    `VLLM_MLX_DEVICE=gpu`, and `VLLM_METAL_USE_PAGED_ATTENTION=1`
  - spawned MLX probe failed with `[metal::load_device] No Metal device
    available`, which is consistent with a headless/sandboxed Codex shell and
    should be rechecked from the user's normal terminal
  - fresh vLLM platform probe found the `metal -> vllm_metal:register` entry
    point and direct registration returned `vllm_metal.platform.MetalPlatform`
  - first lazy `vllm.platforms.current_platform` still resolved to
    `vllm.platforms.cpu.CpuPlatform` after the plugin's initial MLX/compat path
    failed; a later explicit resolver call returned Metal
  - Audex MLX-LM runtime patch imports failed in this shell because MLX could
    not load a Metal device
  - pinned `DefaultModelAdapter` does not yet include Audex-native multimodal
    support
- Current conclusion:
  - vLLM's CPU facade is expected on `MetalPlatform`
  - `vllm.platforms.cpu.CpuPlatform` is not expected and must be treated as
    CPU fallback
  - the next diagnostic must run outside the headless Codex shell, then add a
    real vLLM text-generation timing probe so Activity Monitor is no longer the
    primary evidence

Observed local diagnostic after installing the Audex adapter-selection patch:

- Command: `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-212706.json`
- Exit: `2`
- Findings:
  - adapter-selection failure disappeared; the patch report now includes
    `vllm_metal_audex_adapter=True`
  - remaining failures in the Codex shell are Metal device unavailability,
    first lazy vLLM `current_platform` resolving to `CpuPlatform`, and MLX-LM
    loader patch imports failing because MLX cannot initialize a Metal device
  - this confirms the Audex adapter-selection patch is a separate solved seam;
    the next seam is making the plugin's first vLLM platform initialization
    stable in a normal, GPU-visible terminal and then connecting
    `forward_ready=True` audio encoding/splicing

Observed local diagnostic after adding the projected-embedding adapter path:

- Command: `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-212930.json`
- Exit: `2`
- Findings:
  - `vllm_metal_audex_adapter=True`
  - `audex_adapter_selected=True`
  - `audex_adapter_forward_ready=True`
  - remaining failures are unchanged: Metal unavailable in the headless Codex
    shell, first lazy vLLM `current_platform` resolving to `CpuPlatform`, and
    MLX-LM loader patches failing because MLX cannot initialize

Observed local diagnostic after isolating probes into subprocesses:

- Command: `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-213526.json`
- Exit: `2`
- Findings:
  - diagnostic no longer segfaults when the Codex shell cannot access Metal
  - vLLM registry patch failure is now reported as `[metal::load_device] No
    Metal device available`
  - vLLM Metal Audex adapter patch failure is now reported as `AttributeError:
    partially initialized module 'torch' has no attribute 'library'`, consistent
    with vLLM/torch circular import during failed first plugin initialization
  - this reinforces the current hypothesis: the normal terminal must prove
    first lazy `current_platform` becomes `MetalPlatform`; otherwise we need a
    vllm-metal import-order patch before attempting live S2S

Observed local diagnostic after adding Audex-Mac's vLLM platform repair:

- Command: `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-214213.json`
- Exit: `2`
- Findings:
  - the top-level diagnostic completed without crashing
  - CLI output now prints
    `raw=vllm.platforms.cpu.CpuPlatform after_audex_patches=vllm_metal.platform.MetalPlatform`
  - `platform_resolution_probe.current_platform` still showed
    `vllm.platforms.cpu.CpuPlatform` from vLLM's first lazy import path
  - `platform_resolution_probe.current_platform_after_audex_patches` showed
    `vllm_metal.platform.MetalPlatform`, proving the repair works in isolation
  - MLX still reported no Metal device in the Codex execution shell, so this
    run remains a headless/sandbox diagnostic failure rather than a normal Mac
    terminal pass
  - diagnostic subprocesses now flush JSON and exit immediately after probe
    output so MLX/nanobind atexit cleanup noise does not masquerade as a probe
    segfault
  - the full patch-report subprocess exits cleanly but still records the real
    no-GPU/import-order failures for `vllm_metal_platform_repair`,
    `vllm_nemotron_dense`, and `vllm_metal_audex_adapter`
- Current conclusion:
  - a cached `CpuPlatform` caused by vLLM import order is now repaired by
    Audex-Mac before vLLM generation is allowed
  - a passing normal-terminal run must show MLX GPU availability, repaired
    `MetalPlatform`, and all Audex runtime patch fields true

Observed local diagnostic after isolating the vLLM generation timing probe:

- Command:
  `./start.sh --diagnose-vllm-generation --diagnose-generation-max-tokens 4096`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-214516.json`
- Exit: `2`
- Findings:
  - `generation_probe.subprocess=True`, so `vllm.LLM(...)` and both generation
    probes run outside the parent CLI process
  - `generation_probe.max_tokens=4096`, matching the text benchmark's
    false-positive/false-negative guard
  - the child generation probe exited cleanly with structured JSON instead of
    crashing the parent process
  - in the Codex execution shell, generation still failed before model load
    with `AttributeError: partially initialized module 'torch' has no attribute
    'library'`, consistent with the same no-Metal/import-order failure recorded
    by the patch and adapter probes
- Current conclusion:
  - `--diagnose-vllm-generation` is now safe to run as the Phase 2 timing gate
    from a normal Mac terminal
  - a passing run must record `model_load_seconds`, `one_token_probe`, and
    `long_probe.tokens_per_second`; until then, vLLM Metal text generation is
    not proven usable

Observed local diagnostic after fixing Audex runtime patch order:

- Commands:
  - `./start.sh --diagnose-vllm-metal`
  - `./start.sh --diagnose-vllm-generation --diagnose-generation-max-tokens 4096`
- Reports:
  - `.audex/runs/vllm-metal-diagnostic-20260707-214908.json`
  - `.audex/runs/vllm-metal-diagnostic-20260707-214931.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` now reports all patch fields true:
    `mlx_lm_nemotron_dense`, `mlx_lm_nemotron_h_audex`,
    `vllm_metal_platform_repair`, `vllm_nemotron_dense`, and
    `vllm_metal_audex_adapter`
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - `generation_probe.subprocess=True` and `generation_probe.max_tokens=4096`
  - the only remaining diagnostic failures in this shell are MLX Metal device
    visibility and the generation probe failing model load with
    `[metal::load_device] No Metal device available`
- Current conclusion:
  - Audex-Mac can install its vLLM Metal monkey patches cleanly in a fresh
    generated-venv process
  - Phase 2 is now blocked on proving live vLLM model loading/generation in a
    normal GPU-visible terminal, not on patch registration or adapter selection

Observed local diagnostic after restoring vLLM as the text benchmark default:

- Command:
  `./start.sh --diagnose-vllm-generation --diagnose-generation-max-tokens 4096`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-220215.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` all true
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - `generation_probe.subprocess=True`
  - `generation_probe.max_tokens=4096`
  - remaining failures are only MLX Metal device visibility in this shell and
    the generation model-load probe reporting the same no-Metal error
- Current conclusion:
  - text benchmark policy now matches `docs/engineering/vllm-metal.md`: vLLM Metal is the
    default backend, and direct MLX requires explicit `--text-backend mlx`
  - a normal GPU-visible terminal still needs to produce the first successful
    vLLM timing report before Phase 2 is fully proven

Observed local diagnostic after adding the persistent vLLM runtime scaffold:

- Command:
  `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-220812.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` all true
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - raw `current_platform` still reports `vllm.platforms.cpu.CpuPlatform`
    before Audex patches, while `current_platform_after_audex_patches` reports
    `vllm_metal.platform.MetalPlatform`
  - failure reason remains the Codex shell's missing Metal device:
    `[metal::load_device] No Metal device available`
- Current conclusion:
  - the newly added `AudexVllmRuntime` scaffold did not regress patch
    installation or adapter selection
  - live proof still requires the user terminal where MLX can see the Apple GPU

Observed local diagnostic after making vLLM the default STS backend:

- Command:
  `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-221840.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` all true
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - raw `current_platform` remains `vllm.platforms.cpu.CpuPlatform` before the
    Audex repair, and `current_platform_after_audex_patches` remains
    `vllm_metal.platform.MetalPlatform`
  - failure reason remains the Codex shell's missing Metal device:
    `[metal::load_device] No Metal device available`
- Current conclusion:
  - switching the CLI default from direct MLX STS to persistent vLLM STS did
    not regress patch installation, adapter selection, or platform repair
  - the next live gate is still a normal-terminal run that can access Metal and
    proves the projected-embedding vLLM multimodal feature path for Audex

Observed local diagnostic after routing default STS through projected embeddings:

- Command:
  `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-222446.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` all true
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - raw `current_platform` remains `vllm.platforms.cpu.CpuPlatform` before the
    Audex repair, and `current_platform_after_audex_patches` remains
    `vllm_metal.platform.MetalPlatform`
  - failure reason remains the Codex shell's missing Metal device:
    `[metal::load_device] No Metal device available`
- Current conclusion:
  - the projected-embedding request/runtime bridge did not regress patch
    installation, adapter selection, or platform repair
  - a GPU-visible terminal must now prove vLLM can construct
    `MultiModalFeatureSpec` for the `audex_projected_embeddings` payload

Observed local diagnostic after adding the projected-audio feature-spec helper:

- Command:
  `./start.sh --diagnose-vllm-metal`
- Report:
  `.audex/runs/vllm-metal-diagnostic-20260707-225756.json`
- Exit: `2` in the Codex execution shell
- Findings:
  - `audex_patches` all true
  - `model_adapter.audex_patch_installed=True`
  - `model_adapter.audex_adapter_selected=True`
  - `model_adapter.audex_adapter_forward_ready=True`
  - raw `current_platform` remains `vllm.platforms.cpu.CpuPlatform` before the
    Audex repair, and `current_platform_after_audex_patches` remains
    `vllm_metal.platform.MetalPlatform`
  - failure reason remains the Codex shell's missing Metal device:
    `[metal::load_device] No Metal device available`
- Current conclusion:
  - the feature-spec helper did not regress patch installation, adapter
    selection, or platform repair
  - the next implementation seam was registering or invoking that helper from
    vLLM's `MultiModalRegistry`/processor path for the Audex proxy model

Implemented Audex projected-audio vLLM processor registration:

- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_audex_adapter.AudexProjectedAudioItems`
  - `audex_mac.patches.vllm_metal_audex_adapter.AudexProjectedAudioDataParser`
  - `audex_mac.patches.vllm_metal_audex_adapter.AudexProcessingInfo`
  - `audex_mac.patches.vllm_metal_audex_adapter.AudexDummyInputsBuilder`
  - `audex_mac.patches.vllm_metal_audex_adapter.AudexProjectedAudioProcessor`
  - `audex_mac.patches.vllm_metal_audex_adapter.patch_vllm_audex_processor`
- Upstream symbols patched:
  - `vllm.model_executor.models.nemotron.NemotronForCausalLM`
    receives `supports_multimodal=True` and
    `supports_multimodal_raw_input_only=False`.
  - `vllm.multimodal.MULTIMODAL_REGISTRY.register_processor(...)` receives the
    Audex projected-audio processor factory for the Nemotron proxy class.
- Why the patch exists:
  - Audex is newer than the pinned vLLM/vllm-metal model allowlist.
  - The repository intentionally registers Audex as a Nemotron text proxy for
    vLLM loading, but the plain proxy class is text-only unless Audex-Mac also
    adds a multimodal processor.
  - The default STS CLI already sends
    `multi_modal_data={"audio": [{"audex_projected_embeddings": ...}]}` after
    the MLX audio encoder/projector. Stock vLLM's audio parser treats generic
    audio as raw samples or torch audio embeddings and does not understand this
    Audex-specific projected-embedding dictionary.
- Expected API-shape guard:
  - `NemotronForCausalLM` must remain importable from
    `vllm.model_executor.models.nemotron`.
  - `MULTIMODAL_REGISTRY.register_processor` must accept
    `(processor, info=..., dummy_inputs=...)` and return a model-class
    decorator.
  - vLLM `mm_input(...)` must accept `prompt_token_ids`, `mm_kwargs`,
    `mm_hashes`, and `mm_placeholders`.
  - vLLM/vllm-metal must keep using `MultiModalFeatureSpec` with
    `.data`, `.modality`, `.identifier`, and `.mm_position`.
- Reapply/update notes:
  - If upstream vLLM adds native Audex support, remove the proxy-class
    processor registration only after proving vLLM still emits an audio
    `MultiModalFeatureSpec` whose data contains projected Audex embeddings and
    whose placeholder range covers the expanded `<so_embedding>` run.
  - If upstream changes the multimodal processor factory API, update
    `patch_vllm_audex_processor` first; do not route projected audio around the
    vLLM multimodal path as a silent fallback.
- Tests:
  - `tests/test_audex_patches.py` verifies the proxy model processor
    registration, the `supports_multimodal` class flag, projected-audio
    processor output shape, feature-spec construction, and adapter payload
    validation.
  - Focused command:
    `.venv/bin/python -m pytest tests/test_audex_patches.py tests/test_vllm_sts_requests.py tests/test_vllm_runtime.py -q`
  - Latest focused result: `29 passed`.
  - Latest lint result: `./scripts/lint.sh` passed.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260707-231623.json`
  - Diagnostic result in the Codex shell: exit `2`, `audex_patches` all true,
    `audex_adapter_selected=True`, `audex_adapter_forward_ready=True`, repaired
    platform `vllm_metal.platform.MetalPlatform`, `audex_processor.ready=True`,
    `audex_processor.nemotron_import_hook_installed=True`, processor output
    type `multimodal`, audio payload key `audex_projected_embeddings`, and
    placeholder range `offset=1,length=3`. The only verdict failure is the
    expected headless Metal-device error:
    `[metal::load_device] No Metal device available`.
  - Live Metal proof is still pending because the Codex execution shell cannot
    access a Metal device.

Phase 2 text benchmark run-log evidence:

- `audex_mac.text_generation` now records vLLM benchmark runs with:
  - `backend="vllm"`
  - `engine="vllm.LLM"`
  - `model_load_seconds`
  - `metal_runtime` environment and MLX default-device evidence
  - `audex_patches` from `AudexPatchReport`
  - per-turn `prompt_tokens`
  - per-turn `generation_tokens`
  - per-turn `generation_tps`
  - per-turn `finish_reason` and `stop_reason`
- Expected guard: a successful
  `./start.sh --run-text-benchmark --model audex-2b` run must produce a JSON
  transcript with those fields before the text backend can be considered proven
  on vLLM Metal.

Phase 3 vLLM STS request-builder checkpoint:

- Audex-Mac code:
  - File: `audex_mac/vllm_sts_requests.py`
  - Tests: `tests/test_vllm_sts_requests.py`
  - BDD scenario: `features/speech_to_speech_cli.feature`
- NVIDIA reference files inspected from the cached Audex-2B snapshot:
  - `inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py`
  - `inference_scripts_vllm/unified_s2s_scripts/run_cascaded_s2s_web.sh`
  - `inference_scripts_vllm/audioqa_scripts/run_audioqa_vllm.py`
  - `inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/processing_audex_vllm.py`
- Implemented request shapes:
  - ASR/audio-QA request: chat-template prompt containing
    `Transcribe the input speech.\n<so_embedding>`, non-thinking mode,
    `multi_modal_data={"audio": [(samples, 16000)]}`, temperature `0.0`,
    top-p `1.0`, max tokens `2048`
  - Text response request: NVIDIA web-demo formatting prompt composed with the
    transcript, non-thinking by default, temperature `1.0`, top-p `0.95`,
    max tokens `4096`
  - TTS request: NVIDIA TTS chat template ending at `<speechgen_start>`,
    temperature `0.8`, top-p `1.0`, stop token ids including
    `<speechgen_end>`
  - TTS CFG requests: paired conditional/unconditional prompt-token requests
    with matching token lengths and vLLM `extra_args` for `cfg_scale`,
    `cfg_role`, and `cfg_pair_id`
- Current status:
  - These request builders are now used by `AudexVllmRuntime`, which backs the
    default CLI STS loop unless the user explicitly chooses the diagnostic MLX
    backend.

Phase 3 persistent vLLM runtime checkpoint:

- Audex-Mac code:
  - File: `audex_mac/vllm_runtime.py`
  - Tests: `tests/test_vllm_runtime.py`
  - BDD scenario: `features/speech_to_speech_cli.feature`
- Implemented runtime shape:
  - `AudexVllmRuntime.from_model_path(...)` applies Audex runtime patches,
    loads the selected checkpoint tokenizer/chat template, and creates one
    persistent `vllm.LLM` engine with `trust_remote_code=True`,
    `enable_prefix_caching=True`, and `limit_mm_per_prompt={"audio": 1}`.
  - `transcribe_audio(...)`, `generate_text_response(...)`, and
    `generate_tts_cfg_pair(...)` run through that same engine instead of
    creating separate model sessions per stage.
  - `generate_tts_cfg_pair(...)` submits conditional and unconditional TTS CFG
    requests as one paired `engine.generate(...)` call with matching
    `cfg_pair_id` metadata.
  - The runtime cleans NVIDIA wrapper text from ASR output, strips optional
    thinking blocks from final spoken text, and exposes model-load/engine stats
    for future run logs.
  - `extract_tts_codec_frames(...)` maps vLLM-generated token IDs to Audex
    speech codec frames and stops on `<speechgen_end>`, giving the CLI a tested
    handoff into the existing Audex causal speech decoder.
- Current limitation:
  - `./start.sh` now constructs this persistent vLLM runtime by default for
    fixture and interactive STS paths, and direct MLX requires explicit
    `--sts-backend mlx`.
  - The default vLLM STS path now projects input WAV audio through Audex-Mac's
    MLX NV-Whisper encoder and Audex projector, then submits
    `audex_projected_embeddings` through vLLM `multi_modal_data`. This removes
    raw PCM from the vLLM adapter boundary while still using vLLM for ASR/text
    generation/TTS scheduling.
  - Live proof is still pending on a GPU-visible terminal. If pinned vLLM Metal
    cannot create the `MultiModalFeatureSpec.mm_position` range for an Audex
    projected-embedding audio payload, the next patch must extend vLLM's Audex
    input processor so the placeholder span length matches the projected
    embedding count.
  - vLLM TTS playback now streams no-CFG speech codec frames through the Audex
    causal speech decoder and the continuous PCM player. Longer responses are
    split on spoken sentence/newline boundaries so each TTS request stays short
    enough for intelligible prosody.
  - CFG quality/performance is still a separate research seam. The current
    default remains no-CFG because the paired CFG path produced unintelligible
    audio before the vLLM Metal batching patch below.

Phase 4 vLLM Metal no-CFG TTS stability checkpoint:

- Audex-Mac code:
  - File: `audex_mac/vllm_sts_cli.py`
  - File: `audex_mac/vllm_runtime.py`
  - File: `audex_mac/patches/vllm_metal_cfg.py`
  - Tests: `tests/test_vllm_sts_cli.py`, `tests/test_vllm_runtime.py`,
    `tests/test_vllm_cfg.py`
- Implemented runtime changes:
  - Persisted assistant history is scrubbed before it is reused as prompt
    context. This removes old Audex/NVIDIA boilerplate and literal prompt
    controls such as `[CRITICAL] ...` from conversations that were created
    before the prompt-leak fixes.
  - Spoken answer scrubbing now catches the observed identity-leak forms:
    `Audex is created by NVIDIA ...`, `Audex is a conversational partner ...`,
    follow-on boilerplate, and `Your turn ...` lines.
  - No-CFG TTS chunking still uses sentence/newline boundaries, but each chunk
    now receives at least the normal single-utterance speech-token budget
    (`DEFAULT_S2S_TTS_MAX_TOKENS`) instead of the previous 512-token floor.
    This avoids mid-segment cutoff when the text-token-to-speech-token estimate
    is too low.
  - TTS run logs now include MLX memory snapshots when available:
    `mlx_memory_start`, `mlx_memory_after_stream`, and
    `mlx_memory_after_clear`, plus existing underrun/overrun playback
    diagnostics. This is intended to diagnose the observed Activity Monitor
    climb toward physical RAM pressure across multi-turn conversations.
  - Completed segment cleanup now calls `mx.clear_cache()` and Python
    `gc.collect()` after segment finalization and after turn-level MLX
    phase boundaries. This does not unload the model or live KV state; it only
    releases reclaimable MLX/Python buffers from completed requests.
  - Text-to-TTS interleaving is enabled by default for non-thinking vLLM STS.
    Set `AUDEX_VLLM_STREAM_TEXT_TO_TTS=0` to force the older serial
    text-then-TTS path for diagnostics.
  - Interleaved text streaming waits for a newline, three stable sentences, or
    a long stable sentence before sending non-final text to TTS. This preserves
    the same prosody-oriented chunking as the static no-CFG path instead of
    handing Audex TTS one short sentence at a time.
  - Interleaved text streaming now also releases after two complete sentences
    when the accumulated text is already a substantial spoken chunk. This avoids
    waiting for a third sentence only to split the first TTS request back down
    at the static character limit.
  - Streaming TTS logs now record `first_tts_chunk_ready_seconds`,
    `tts_segment_ready_seconds`, and `tts_stream_min_chars_per_chunk` so
    first-audio latency can be separated into text-stream wait, TTS token
    generation, decoder, and playback phases.
  - Streaming TTS logs also record wall-clock codec-frame arrival and the first
    decoder push start/finish/frame-count. This distinguishes waiting for
    enough speech-codec frames from the MLX causal decoder's first graph/eval
    cost before changing decoder chunk size.
  - The default streaming decoder path now primes the first decoder push with
    `decoder_chunk_frames + lookahead_steps` frames, then returns to normal
    `decoder_chunk_frames` pushes. Previously the first 24-frame push could not
    emit audio because the causal decoder still needed four lookahead frames,
    delaying first waveform output until the second push.
  - Interleaved playback now uses a 1s continuous-PCM prebuffer instead of the
    earlier hard 4s and later 2s buffers. After decoder priming, the larger
    buffers delayed speaker output and produced large queue high-water/overrun
    diagnostics without preventing underruns in the measured long-turn fixture.
    The 1s checkpoint on `.audex/runs/ptt-input-20260708-223521.wav` moved
    `first_playback_started_seconds` from `6.065` to `5.063` while preserving
    `device_underflow_count=0`, `queue_underrun_count=0`, and full
    `28.72s` playback.
  - Streaming vLLM TTS playback now attempts an MLX vectorized PCM16 packing
    path before falling back to Python float-list packing. This reduces host
    CPU work in the decoder/playback loop on MLX runtimes that expose
    `array.tobytes()` or the Python buffer protocol, and logs
    fast-path/fallback counts with `pcm_pack_seconds`.
  - Resumed vLLM conversations now persist prompt-leakage cleanup immediately
    and invalidate a matching `.kv.safetensors` sidecar if the text history was
    rewritten. This prevents stale assistant turns such as NVIDIA/Audex
    self-biography or leaked `[CRITICAL]` controls from re-entering future
    prompts or mismatching a binary KV cache.
  - Interleaved text-to-TTS streaming now scrubs every cumulative text delta
    before deciding which stable chunk can be spoken. The final response text,
    persisted assistant turn, and TTS prompt therefore share the same cleaned
    spoken text.
  - TTS run logs now record `tts_segment_mlx_memory_after_clear` alongside the
    existing start/after-stream/after-clear snapshots so long-conversation
    memory growth can be attributed per spoken segment.
  - vLLM engine construction now defaults `gpu_memory_utilization` to `0.60`
    instead of `0.85` to reduce unified-memory compression/paging risk on the
    30B bf16 path. Override with `AUDEX_VLLM_GPU_MEMORY_UTILIZATION` when
    intentionally trading more KV/cache room against memory pressure.
- Implemented vLLM Metal patch changes:
  - `audex_mac/patches/vllm_metal_cfg.py` patches both
    `MetalModelRunner._sequential_decode` and `_batched_decode` for Audex TTS
    window decoding.
  - The batched no-CFG window path merges per-request KV caches, runs one
    batched backbone forward pass, projects only the allowed Audex
    `<speechgen_end>` plus speech-codec token window, samples inside that
    window, then extracts each request cache back into its request state.
  - Debug timing categories under `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1`:
    `tts_window_forward`, `tts_window_project`, `tts_window_sample`,
    `tts_window_batch_forward`, `tts_window_batch_project`, and
    `tts_window_batch_sample`.
- Probe evidence:
  - Before the batched no-CFG patch:
    `.audex/runs/tts-probe-vllm-20260709-010957.json`, two parallel TTS
    requests, aggregate `26.846` tokens/sec.
  - After the batched no-CFG patch:
    `.audex/runs/tts-probe-vllm-20260709-011413.json`, two parallel TTS
    requests, aggregate `45.597` tokens/sec.
  - Three parallel TTS requests:
    `.audex/runs/tts-probe-vllm-20260709-011613.json`, aggregate `57.294`
    tokens/sec.
  - Four parallel TTS requests:
    `.audex/runs/tts-probe-vllm-20260709-011659.json`, aggregate `66.451`
    tokens/sec.
- Current conclusion:
  - Batched window decoding improves aggregate vLLM Metal throughput and keeps
    the no-CFG path on the speech-token window instead of evaluating the full
    vocabulary logits.
  - Ordered playback still depends on the first segment producing frames
    faster than the audio device consumes them. Human testing on 2026-07-09
    confirmed greeting and first-turn intelligibility, with remaining
    long-utterance choppiness/repetition and memory-pressure concerns.
- Reapply/update notes:
  - If upstream vLLM Metal changes `_sequential_decode`, `_batched_decode`,
    `SamplingBatch`, `_SamplingResult`, `_merge_kv_caches`, or
    `_extract_kv_cache`, re-check this patch before bumping the pin.
  - Do not remove the speech-token window projection unless upstream provides
    an equivalent Audex audio-token sampler. Falling back to full-vocab logits
    has already produced slow or unintelligible TTS on this Mac.
- Verification:
  - `python -m ruff check .`
  - `.venv/bin/python -m pytest -q`
  - Latest result on 2026-07-09: `333 passed, 3 skipped`.

## Required Patch Areas

These are expected areas based on the spike design. The coding agent should
replace this section with concrete patches as implementation begins.

### Transformers Local Dynamic Module Paths

- Purpose: let vLLM construct the Audex full-checkpoint engine from a local
  Hugging Face snapshot whose remote-code Python files are symlinks into the
  cache `blobs/` directory.
- Upstream package: `transformers==5.12.1` in the pinned vLLM Metal generated
  venv.
- Upstream file/symbol:
  - `transformers.dynamic_module_utils.cached_file`
  - `transformers.dynamic_module_utils._compute_local_source_files_hash`
  - `transformers.dynamic_module_utils.get_cached_module_file`
- Upstream gap: when `AutoConfig.from_pretrained(..., trust_remote_code=True)`
  loads Audex from a local HF snapshot directory, Transformers can hand
  `_compute_local_source_files_hash` a path resolved through
  `models--*/blobs/<hash>`. The hash helper then discovers relative imports
  beside the blob object, not beside the logical snapshot file. Audex's
  `configuration_nemotron_h_audio.py` imports
  `.configuration_nemotron_h`, so the vLLM engine failed before model
  construction with:
  `FileNotFoundError: .../models--nvidia--Nemotron-Labs-Audex-30B-A3B/blobs/configuration_nemotron_h.py`.
- Audex-Mac patch:
  - File: `audex_mac/patches/transformers_dynamic_module.py`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Patch report field: `transformers_local_dynamic_modules`
  - Behavior: for local model directories, `cached_file` returns the logical
    snapshot-local file path when that file exists, preserving the directory
    where relative imports live. The local-source hash helper then hashes the
    logical snapshot paths without resolving symlinks into `blobs/`.
- Guard: the patch is idempotent, stores the original upstream callables on the
  Transformers module, and only changes local directory behavior. Non-local
  Hub model IDs still go through upstream `cached_file`.
- Reapply/update notes: when Transformers changes dynamic-module loading,
  re-check whether `_compute_local_source_files_hash` still resolves
  `resolved_module_file` before discovering relative imports. Delete this
  patch once upstream preserves logical local snapshot paths or once Audex-Mac
  moves to an upstream release that does.
- Tests: `tests/test_audex_patches.py` builds a miniature HF cache layout with
  snapshot-local Python files symlinked into `blobs/` and verifies that the
  patched loader hashes the relative import from the snapshot directory instead
  of following the blob path.

### Audex Model Registration

- Purpose: register Audex model types with the vLLM Metal runtime.
- Model types:
  - `nemotron_dense_audex`
  - `nemotron_h_audex`
  - `nemotron_dense`
- Expected guard: Audex-Mac can prove both model types map to Audex-owned loader
  hooks, even if only 2B is implemented first.

### MLX Nemotron Dense Loader

- Purpose: load Audex-2B text-only `nemotron_dense` checkpoints in the pinned
  vLLM Metal runtime.
- Upstream package: `mlx_lm==0.31.3`
- Upstream gap: `mlx_lm.models` includes `nemotron_h` but not
  `nemotron_dense`.
- Audex-Mac patch:
  - File: `audex_mac/patches/mlx_lm_nemotron_dense.py`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Generated-venv shim installer: `audex_mac.patches.install`
  - Injected modules: `mlx_lm.models.nemotron_dense` and
    `mlx_lm.models.nemotron_dense_audex`
  - Model implementation mirrors NVIDIA's
    `checkpoint_folder_textonly/modeling_nemotron_dense.py`: RMSNorm,
    squared-ReLU MLP, GQA, RoPE theta/partial factor from `rope_parameters`,
    untied `lm_head`.
  - Weight sanitizer remaps `model.embeddings.*` to MLX's
    `model.embed_tokens.*` and removes speech-side `audio_encoder.*` /
    `audio_projector.*` tensors for the text-only smoke path.
  - Full-vocab speech-token smoke uses the same loader against
    `checkpoint_folder_full` where Audex-2B exposes `model.embed_tokens.weight`
    and `lm_head.weight` as `(205312, 2048)` BF16 tensors. This is required for
    speech output because `<speechgen_start>`, `<speechgen_end>`, and
    `<speechcodec_N>` IDs live above the 131072-row text-only head.
- Expected guard: `./start.sh --preflight-text-runtime` reports
  `mlx_lm_nemotron_dense=True` when the patched loader imports in a Metal-capable
  vLLM Metal runtime.
- Spawned-worker note: vLLM Metal starts engine workers with macOS `spawn`, so
  parent-process `sys.modules` injection is not enough. `start.sh` installs a
  tiny generated shims into the generated vLLM Metal venv at
  `mlx_lm/models/nemotron_dense.py`,
  `mlx_lm/models/nemotron_dense_audex.py`, and
  `mlx_lm/models/nemotron_h_audex.py`; those shims import the repo-owned
  implementations.
- Runtime mode guard: Audex-Mac requires `VLLM_METAL_USE_MLX=1`,
  `VLLM_MLX_DEVICE=gpu`, and `VLLM_METAL_USE_PAGED_ATTENTION=1` before model
  launch. This prevents accidentally running the MLX CPU backend while keeping
  vLLM Metal's expected `device_config=cpu` compatibility facade intact.
- Text benchmark backend policy: `audex_mac.text_generation` now defaults to
  vLLM Metal for the text benchmark. Direct `mlx_lm` generation remains
  available through `--text-backend mlx` as an explicit diagnostic fallback,
  because `docs/engineering/vllm-metal.md` requires vLLM Metal to be the primary path and
  forbids silent fallback to direct MLX.
- Reapply/update notes: compare against NVIDIA's
  `modeling_nemotron_dense.py` whenever Audex checkpoints rev. Confirm
  normalization, MLP projection count, RoPE parameters, and embedding key names
  before changing this patch.
- Tests: `tests/test_audex_patches.py` covers module injection; direct local
  smoke uses `mlx_lm.utils.load_model(..., lazy=True)` against the cached
  Audex-2B text checkpoint.

### MLX Nemotron-H Audex Loader

- Purpose: load the full Audex-30B-A3B `nemotron_h_audex` checkpoint in the
  pinned vLLM Metal runtime and make it usable for Audex-Mac's speech-input
  embedding splice.
- Upstream package: `mlx_lm==0.31.3`
- Upstream file/symbol: `mlx_lm.models.nemotron_h.Model` and
  `mlx_lm.models.nemotron_h.NemotronHModel`
- Upstream gap: `mlx_lm.models.nemotron_h` implements the text backbone, but
  NVIDIA publishes the Audex full checkpoint as `nemotron_h_audex` with
  `audio_encoder.*` and `audio_projector.*` tensors. The upstream text-only
  model rejects those audio tensors and does not accept MLX-LM
  `input_embeddings`, which Audex-Mac needs after replacing `<so_embedding>`
  placeholders with native Audex audio embeddings.
- Audex-Mac patch:
  - File: `audex_mac/patches/mlx_lm_nemotron_h_audex.py`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Generated-venv shim installer: `audex_mac.patches.install`
  - Injected module: `mlx_lm.models.nemotron_h_audex`
  - The shim subclasses upstream Nemotron-H, drops `audio_encoder.*` and
    `audio_projector.*` from language-model weight loading, exposes
    `model.model.embed_tokens` as an alias for the upstream backbone embedding,
    and adds an `input_embeddings` argument so MLX-LM generation can prefill
    from Audex-Mac's spliced audio/text embeddings.
- Why dropping audio tensors is correct here: Audex-Mac loads and executes the
  Audex audio encoder/projector separately in MLX before language-model
  prefill. The full-checkpoint LM load should consume only the Nemotron-H
  backbone and LM head.
- Expected guard: `./start.sh --input-wav <fixture> --no-play` with a cached
  Audex-30B-A3B snapshot selects 30B automatically, completes the one-turn STS
  path, and writes a WAV/run log without requiring text-only 30B conversion.
- Tests: `tests/test_audex_patches.py` covers `nemotron_h_audex` module
  injection, `tests/test_patch_install.py` covers generated shim installation,
  and `tests/test_start_sh.py` guards the no-argument startup path against the
  Bash empty-array regression that previously blocked `./start.sh`.

### vLLM Nemotron Dense Registration

- Purpose: pass vLLM architecture validation for Audex text and full
  speech-capable checkpoints before vLLM Metal hands model loading to MLX-LM.
- Upstream package: `vllm==0.24.0+cpu` installed by pinned vLLM Metal.
- Upstream file/symbol: `vllm.model_executor.models.registry.ModelRegistry`.
- Upstream gap: registry includes `NemotronForCausalLM` and
  `NemotronHForCausalLM`, but not Audex's newly published architecture aliases:
  `NemotronDenseForCausalLM`,
  `NemotronDenseAudexForConditionalGeneration`, or
  `NemotronHAudexForConditionalGeneration`.
- Audex-Mac patch:
  - File: `audex_mac/patches/runtime.py`
  - Generated-venv worker hook: `audex_mac/patches/install.py` writes
    `sitecustomize.py`, and `start.sh` exports `AUDEX_MAC_AUTO_PATCHES=1` so
    spawned vLLM `EngineCore` workers apply the same runtime patches before
    model registry resolution.
  - Model-info guard: vLLM caches `_ModelInfo` for lazy registered classes under
    its model-info cache. Audex-Mac wraps the registry inspection path for
    Audex architecture aliases so cached or freshly inspected metadata reports
    `supports_multimodal=True`,
    `supports_multimodal_raw_input_only=False`, and
    `requires_raw_input_tokens=False`.
  - Architectures registered:
    - `NemotronDenseForCausalLM` ->
      `vllm.model_executor.models.nemotron:NemotronForCausalLM`
    - `NemotronDenseAudexForConditionalGeneration` ->
      `vllm.model_executor.models.nemotron:NemotronForCausalLM`
    - `NemotronHAudexForConditionalGeneration` ->
      `vllm.model_executor.models.nemotron_h:NemotronHForCausalLM`
- Why proxying is acceptable in this spike: vLLM Metal loads the executable
  model through MLX-LM after vLLM's architecture inspection gate. The proxy is
  used for vLLM metadata inspection only; Audex-Mac owns the actual MLX model
  class above.
- Expected guard: `apply_audex_runtime_patches()` returns
  `vllm_nemotron_dense=True` and a GPU-visible diagnostic does not fail model
  config validation with
  `Model architectures ['NemotronHAudexForConditionalGeneration'] are not supported`.
- Tests: `tests/test_audex_patches.py` covers the registry mutation with a fake
  `ModelRegistry` and the Audex `_ModelInfo` multimodal override;
  `tests/test_patch_install.py` covers the generated `sitecustomize.py` hook;
  `tests/test_start_sh.py` covers exporting `AUDEX_MAC_AUTO_PATCHES=1`.

### vLLM Metal current_platform Repair

- Purpose: prevent vLLM's first lazy `current_platform` import from caching
  `vllm.platforms.cpu.CpuPlatform` after the Metal plugin resolver has already
  proven that `vllm_metal.platform.MetalPlatform` is available.
- Upstream package: `vllm==0.24.0+cpu` plus `vllm-metal==0.3.0`.
- Upstream file/symbol:
  `vllm.platforms._current_platform` and
  `vllm.platforms.resolve_current_platform_cls_qualname`.
- Upstream gap: during Audex-Mac diagnostics, importing vLLM through the pinned
  vLLM Metal stack can hit vLLM/torch/MLX import-order failures before the Metal
  plugin is fully initialized. vLLM then leaves `_current_platform` as
  `CpuPlatform` even though a later resolver call returns
  `vllm_metal.platform.MetalPlatform`.
- Audex-Mac patch:
  - File: `audex_mac/patches/runtime.py`
  - Function: `_repair_vllm_metal_current_platform`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Patch report field: `vllm_metal_platform_repair`
  - Guard: only mutates `_current_platform` when
    `resolve_current_platform_cls_qualname()` returns exactly
    `vllm_metal.platform.MetalPlatform`.
- Patch-order guard: `apply_audex_runtime_patches()` must repair vLLM's
  platform before installing the MLX-LM lazy module aliases. Installing the
  aliases first changes the first vLLM import graph enough to trigger the
  no-Metal/circular-import failure path in the generated venv.
- Expected guard: `./start.sh --diagnose-vllm-metal` prints
  `vLLM current_platform: raw=... after_audex_patches=...` and the
  `after_audex_patches` value is `vllm_metal.platform.MetalPlatform`.
- Tests: `tests/test_audex_patches.py` covers replacing a fake cached
  `CpuPlatform` with a fake `MetalPlatform`, and covers patch ordering;
  `tests/test_vllm_diagnostics.py` covers judging the repaired platform as the
  effective platform.

### Generation Model Loading

- Purpose: load Audex models through Audex-Mac's MLX/Metal implementation
  instead of unsupported default `mlx_lm`/`mlx_vlm` paths.
- Upstream symbols:
  - `vllm_metal.v1.model_lifecycle.ModelLifecycle._load_generation_model`
  - `vllm_metal.v1.model_adapter.DefaultModelAdapter`
- Expected guard: target symbols exist with compatible signatures before patch.

### vLLM Metal Audex Multimodal Adapter

- Purpose: make the pinned vLLM Metal `DefaultModelAdapter` recognize Audex as
  a first-class multimodal/audio model family instead of falling through the
  existing Qwen3-VL/PaddleOCR-VL-only adapter selection.
- Upstream package: `vllm-metal==0.3.0`
- Upstream file/symbol:
  `vllm_metal.v1.model_adapter.DefaultModelAdapter.build_multimodal_adapter`
- Upstream gap: the pinned adapter only builds native multimodal adapters for
  Qwen3-VL/Qwen3.5 and PaddleOCR-VL. Audex was newly released and is not
  upstream-supported. Because Audex-Mac's goal is a day-zero Mac speech-to-
  speech spike, implementing the missing Audex adapter in this repository is
  in scope. Adapter authoring is part of restoring the vLLM Metal path, not a
  fallback that permits CPU inference or a different runtime architecture. This
  patch should be treated as project-owned spike code until upstream vLLM Metal
  grows equivalent support.
- Audex-Mac patch:
  - File: `audex_mac/patches/vllm_metal_audex_adapter.py`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Patch report field: `vllm_metal_audex_adapter`
  - Recognized model types:
    `nemotron_dense_audex`, `nemotron_h_audex`
  - Recognized architectures include the Audex conditional-generation names and
    the Nemotron text backbones used by the converted/text-only smoke paths.
  - vLLM processor registration targets both proxy classes used by Audex-Mac:
    `vllm.model_executor.models.nemotron:NemotronForCausalLM` for dense/2B
    paths and `vllm.model_executor.models.nemotron_h:NemotronHForCausalLM` for
    the 30B-A3B full speech checkpoint. This is installed through an import
    hook so macOS-spawned `EngineCore` workers see the same multimodal support
    as the parent process.
  - Renderer-state guard:
    `vllm.renderers.base.BaseRenderer._process_multimodal` is wrapped to
    initialize `_mm_req_counter` and `_mm_timing_registry` when the pinned vLLM
    renderer reaches multimodal processing without those attributes. The wrapper
    also lazily creates the registered multimodal processor if vLLM recognizes
    the model as multimodal but the renderer missed processor construction.
  - Nemotron-H paged-attention guard:
    `vllm_metal.attention.patching.find_attn_attr` is wrapped so
    Nemotron-H full-attention blocks expose their `mixer` module as the
    attention attribute only when `block_type == "*"`. Mamba, MLP, and MoE
    mixers are left untouched.
  - Nemotron-H SDPA contract guard:
    `vllm_metal.attention.impls.sdpa_wrapper.SDPAPagedAttentionWrapper` is
    wrapped so `NemotronHAttention` exposes `n_heads`, `n_kv_heads`, and an
    identity RoPE callable before vLLM Metal's generic SDPA wrapper uses it.
    `vllm_metal.attention.impls.sdpa.apply_attention_rope` is also wrapped to
    bypass RoPE entirely for `NemotronHAttention`, matching the upstream
    MLX-LM Nemotron-H attention implementation.
  - Audex paged-context guard:
    `AudexMultimodalAdapter.call_lm` clears vLLM Metal's Qwen-style
    `ctx.segment_positions` before calling Audex text backbones. Audex uses
    standard text/audio positions, not M-RoPE segment-position arrays; leaving
    those arrays in the context makes standard RoPE layers fail during
    multimodal prefill.
  - Text-preprocessing guard:
    `AudexProcessingInfo.default_tok_params` implements vLLM's
    `BaseProcessingInfo` tokenization contract so text-only follow-up requests
    still preprocess correctly while the loaded model remains registered as
    multimodal for audio turns.
  - Adapter class: `AudexMultimodalAdapter`
- Current readiness: `AudexMultimodalAdapter.forward_ready=True` only for
  explicit precomputed projected Audex embedding payloads named
  `audex_projected_embeddings`, `projected_embeddings`, or `audio_embeddings`.
  Raw audio/WAV feature ingestion through vLLM's multimodal preprocessor is
  still pending and fails loudly. This keeps the next `AudexVllmRuntime` free to
  use Audex-Mac's existing MLX audio encoder/projector before handing projected
  embeddings to the vLLM scheduler, without pretending that generic vLLM audio
  preprocessing is complete.
- Expected guard: `./start.sh --diagnose-vllm-metal` reports
  `vllm_metal_audex_adapter=True`, `audex_adapter_selected=True`, and
  `audex_adapter_forward_ready=True`.
- Tests: `tests/test_audex_patches.py` covers adapter monkey-patch selection
  for an Audex-shaped config, projected audio embedding payload validation,
  Nemotron-H processor registration, paged-attention `mixer` discovery,
  Nemotron-H SDPA/no-RoPE normalization, Audex paged-context cleanup,
  default tokenization params, and the renderer multimodal-state guard.
- Latest validation:
  - `./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
    passed on the GPU-visible Mac runtime with report
    `.audex/runs/vllm-metal-diagnostic-20260708-075120.json`. Evidence:
    `ready=True`, `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    parent/spawn MLX both `Device(gpu, 0)`, `vllm_token_streaming=True`,
    `decoder_streaming=True`, `first_audio_ready_seconds=0.612`, and
    `generated_codec_frame_count=56`.
  - `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke` selected the
    cached 30B model and passed on the GPU-visible Mac runtime with report
    `.audex/runs/vllm-metal-diagnostic-20260708-075518.json`. Evidence:
    `ready=True`, `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    parent/spawn MLX both `Device(gpu, 0)`, `vllm_token_streaming=True`,
    `decoder_streaming=True`, `first_audio_ready_seconds=4.53`, and
    `generated_codec_frame_count=206`.
  - `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=180 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    passed on the GPU-visible Mac runtime with report
    `.audex/runs/vllm-metal-diagnostic-20260708-082327.json`. Evidence:
    `ready=True`, parent/spawn MLX both `Device(gpu, 0)`,
    `playback_transport=sounddevice_raw_output_stream`,
    `first_audio_ready_seconds=1.314`,
    `first_playback_started_seconds=11.707`,
    `generated_codec_frame_count=56`, `chunk_count=12`,
    `device_underflow_count=3`, and `queue_underrun_count=2`. This proves the
    vLLM Metal playback transport but does not yet prove realtime-quality
    low-underrun playback.
  - After adding the native MLX CFG sampling fast path below, the same audible
    diagnostic passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-083913.json`. Evidence:
    `ready=True`, parent/spawn MLX both `Device(gpu, 0)`,
    `Audex vLLM Metal: native MLX sampling fast path used 1 time(s)` from
    EngineCore stderr, `first_audio_ready_seconds=1.505`,
    `first_playback_started_seconds=4.818`,
    `last_codec_frame_seconds=4.702`, `generated_codec_frame_count=38`,
    `device_underflow_count=0`, and `queue_underrun_count=0`.

### vLLM STS Async Event Loop and Shutdown

- Purpose: keep a persistent `AsyncLLMEngine` alive for the whole cascaded
  speech-to-speech turn and shut it down cleanly for one-off diagnostics.
- Upstream package: `vllm==0.24.0+cpu` installed by pinned vLLM Metal.
- Upstream symbol:
  `vllm.v1.engine.async_llm.AsyncLLM.shutdown`.
- Failure evidence:
  `.audex/runs/vllm-metal-diagnostic-20260708-073447.json` reached ASR on 2B
  (`Audex STS: transcript: oh`) and then saw a dead engine when submitting the
  text request. The cause was Audex-Mac calling `asyncio.run(...)` separately
  for ASR, text, and TTS while reusing one `AsyncLLMEngine`.
- Audex-Mac patch:
  - File: `audex_mac/vllm_sts_cli.py`
  - `VllmSpeechToSpeechSession.run_turn_from_wav` delegates async sessions to
    one `_run_turn_from_wav_async` coroutine that awaits ASR, text, and
    streaming TTS on the same event loop.
  - `VllmSpeechToSpeechSession.shutdown` calls the underlying engine
    `shutdown(timeout=5.0)` when available.
  - `run_vllm_fixture_turn` shuts down only sessions it creates internally.
    Interactive sessions stay persistent for indefinite CLI chat.
- Diagnostic patch:
  - File: `audex_mac/vllm_diagnostics.py`
  - The STS smoke subprocess report now preserves progress stdout even when a
    final JSON result is parsed.
  - If a completed JSON result is emitted but a child process keeps the pipe
    open until timeout, the parser preserves `ready=True` and records
    `subprocess_timeout_after_result=True` instead of converting success into a
    false failure.
- Tests:
  - `tests/test_vllm_sts_cli.py` covers async vLLM STS behavior.
  - `tests/test_vllm_diagnostics.py` covers progress stdout preservation and
    success-json-before-timeout handling.

### vLLM Metal Native MLX CFG Sampling Fast Path

- Purpose: avoid vLLM Metal's per-token Torch sampler bridge for Audex TTS CFG
  batches when NVIDIA's requested sampler settings can be implemented directly
  in MLX.
- Upstream package: pinned vLLM Metal.
- Upstream symbols:
  - `vllm_metal.v1.sampling_batch.sample_from_logits`
  - `vllm_metal.v1.sampling_batch.sample_prefill_tokens`
  - `vllm_metal.v1.model_runner.sample_from_logits`
  - `vllm_metal.v1.model_runner.sample_prefill_tokens`
- Failure evidence:
  - Before this patch, audible vLLM TTS generated 56 codec frames over
    `last_codec_frame_seconds=14.152` in
    `.audex/runs/vllm-metal-diagnostic-20260708-082713.json`, roughly
    4 codec frames/sec for a 50 codec frames/sec audio format. Playback had to
    wait for the 0.8 second prebuffer and still recorded underruns.
  - vLLM Metal's upstream `sample_from_logits` only stays native for greedy
    sampling. Audex TTS uses NVIDIA's non-greedy temperature setting and CFG
    logits processors, so it bridged the `(batch, 205312)` logits tensor through
    Torch every speech token.
- Audex-Mac patch:
  - File: `audex_mac/patches/vllm_metal_cfg.py`
  - `_patch_sample_from_logits` first attempts `_sample_native_mlx_if_supported`
    before falling back to upstream vLLM Metal sampling.
  - The native path is deliberately narrow: it requires a complete CFG pair,
    no logprobs, no custom generators, no penalties, no top-p/top-k filtering,
    no allowed/bad token constraints, and only CFG/token-sync processors plus
    inert vLLM builtin min-p/min-tokens/logit-bias processors.
  - It blends conditional/unconditional logits in MLX, samples each complete CFG
    pair once from the blended conditional row with
    `mx.random.categorical(logits * (1 / temperature))`, and expands that sampled
    token ID into both the conditional and unconditional vLLM request slots.
    This preserves NVIDIA's paired-token sync behavior while avoiding a
    duplicate categorical draw over the full Audex vocabulary for each CFG
    uncond row.
  - The debug timing line records cumulative `native_sampled_rows` and
    `native_output_rows`, and the diagnostic assessment derives
    `native_sampling_row_ratio`. For ordinary two-request CFG pairs this should
    be about `0.5`; a value near `1.0` means duplicate uncond-row sampling has
    regressed. Ready STS smoke reports that include a row ratio above `0.75`
    fail the diagnostic evidence gate.
  - Ready STS smoke reports must include measured speech-token throughput via
    `sts_timing_assessment.codec_frames_per_second`; missing throughput or
    `sts_timing_assessment.below_realtime=True` fails the evidence gate. The
    recovery plan's target is a usable speech prototype, so streamed audio
    below the 50 codec frames/sec audio rate is diagnostic progress but not
    readiness.
  - `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1` emits bounded EngineCore stderr lines
    proving whether the native fast path was used or why it was skipped. The
    STS/TTS diagnostics enable this only when
    `--diagnose-vllm-native-sampling-debug` is passed, so default smoke probes
    measure the same fast path as `./start.sh`.
  - The same debug mode wraps `MetalModelRunner._sample_paged_batch` and emits
    bounded per-worker paged sample timing checkpoints so model/eval time can
    be separated from native sampler time.
  - The debug wrapper also records `mlx.core.eval` time while paged sampling is
    active, categorized as `logits`, `sample_tokens`, or `other`, and appends an
    `mx_eval_ms=category:milliseconds/count` summary to the same bounded timing
    lines. This is instrumentation only and is inactive unless
    `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1` is set.
  - A diagnostic experiment can skip vLLM Metal's explicit full-logits
    `mx.eval(...)` during CFG decode by setting
    `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=1`. This is deliberately disabled
    by default. The latest single-stream traces showed that skipping the logits
    eval mostly moved pending MLX graph execution into the native token
    sampling `mx.eval(tokens)` boundary and made the default completion gate
    harder to interpret.
- Expected guard: if Audex changes sampler settings to activate min-p,
  min-tokens, logit bias, top-p, top-k, penalties, logprobs, or custom
  generators, the patch must fall back to upstream vLLM Metal sampling instead
  of approximating unsupported behavior.
- Tests:
  - `tests/test_vllm_cfg.py` covers CFG token sync, sampler symbol wrapping,
    native MLX fast-path dispatch, one-sample-per-CFG-pair expansion, token
    constraint rejection, inert builtin-processor allowlisting, and the paged
    timing wrapper.
- Latest validation:
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py tests/test_vllm_sts_cli.py tests/test_vllm_diagnostics.py -q`
    passed with `35 passed`.
  - Live audible diagnostic:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=180 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-083913.json`.
  - The report records `first_token_event_seconds=0.078`,
    `last_codec_frame_seconds=4.702`, `stream_finished_seconds=4.729`,
    `first_playback_started_seconds=4.818`, and zero device/queue underruns.
  - Remaining limitation: this is still below realtime for longer utterances;
    it removes one obvious Torch bridge bottleneck but does not prove the full
    vLLM Metal model forward path can generate speech codec frames at
    50 frames/sec on this Mac.
  - Normal-answer diagnostic fixture:
    `say` must run outside the sandbox to create non-empty speech fixtures in
    this environment. A verified 7.58 second, 16 kHz mono WAV fixture was
    generated at `.audex/fixtures/normal-answer-question.wav` and used with:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question.wav`.
  - Latest normal-answer report:
    `.audex/runs/vllm-metal-diagnostic-20260708-085759.json`. Evidence:
    ASR correctly recognized the context-manager prompt, the exact NVIDIA TTS
    prompt suffix was active, and the native sampler activated through 200 TTS
    steps, but TTS hit the 256-token diagnostic cap without `<speechgen_end>`.
    It generated 256 codec frames with `last_codec_frame_seconds=78.107`,
    `device_underflow_count=43`, and `queue_underrun_count=43`.
  - Worker timing from the same report shows the remaining bottleneck is below
    the playback layer: at paged sample count 200, average
    `_sample_paged_batch` time was `182.9 ms`, last step was `19.7 ms`, the
    batch had `decode_reqs=2`, `decode_tokens=2`, and cumulative native
    sampler time was about `10048 ms`. The remaining wall time is model/eval
    and vLLM Metal paged decode overhead.
  - Latest local code checkpoint adds debug-only `mx.eval` category timing
    inside `_sample_paged_batch`; focused coverage
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q` passes with
    `9 passed`. The next GPU-visible normal-answer diagnostic should inspect
    `mx_eval_ms=logits:...` vs `sample_tokens:...` to decide whether the next
    optimization belongs in model/logit evaluation or native categorical
    sampling.
  - Follow-up checkpoint: the full-logits eval skip is now opt-in via
    `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=1` instead of default debug
    behavior. Focused tests cover both default no-skip timing and explicit
    skip-experiment timing.
  - Latest diagnostic ergonomics checkpoint lets
    `AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS` bound silent STS smoke as well as
    playback smoke. The next GPU-visible timing command can therefore avoid
    audio-device variability:
    `scripts/create-test-utterance.sh --basename normal-answer-question --text "Please explain Python context managers in two concise sentences."`
    followed by
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`.
	    The diagnostic now parses matching EngineCore stderr lines into
	    `sts_probe.vllm_metal_timing.latest_paged_sample` and prints a
	    `vLLM Metal TTS timing:` CLI summary, including native sampled/output row
	    counts, when that structured timing is present.
	    It also derives `sts_probe.sts_timing_assessment` with codec-frame
	    throughput, realtime ratio, playback glitch count, paged/native/eval
	    milliseconds per step, native sampled/output row counts and ratio,
	    dominant `mx_eval` category, nonpaged TTS-window decode counters/native
	    detail timings, and `likely_bottleneck` so the next patch can target the
	    measured bottleneck rather than raw log fragments.
  - The same cap is now available as the explicit diagnostic-only CLI flag
    `--diagnose-vllm-sts-speech-max-tokens`; the environment variable remains
    supported, but the documented completion command no longer depends on that
    hidden knob.
  - Fresh Codex-shell attempt with the same fixture and playback enabled wrote
    `.audex/runs/vllm-metal-diagnostic-20260708-090734.json` and failed before
    EngineCore timing because the sandbox has no Metal device:
    `[metal::load_device] No Metal device available`.
  - Current completion-gate evidence:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-094230.json` from a
    GPU-visible run. Evidence: parent/spawn MLX both `Device(gpu, 0)`,
    `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `generated_codec_frame_count=240`, and
    `codec_frames_per_second=3.947`. The verdict correctly failed because the
    realtime gate requires at least `50 codec frames/sec`.
  - Latest sampler hot-path patch:
    `audex_mac.patches.vllm_metal_cfg._sample_native_mlx_if_supported` no
    longer casts/blends the whole CFG logits batch before sampling. It builds
    only the sampled logits rows and blends complete CFG pairs directly into the
    conditional sample row, preserving one sampled token per CFG pair while
    avoiding redundant unconditional-row categorical work and full-batch CFG
    materialization.
  - The same completion gate after that patch wrote
    `.audex/runs/vllm-metal-diagnostic-20260708-094623.json`. Evidence:
    `codec_frames_per_second=4.614`, `audio_realtime_ratio=0.092`,
    `paged_sample_avg_ms=166.8`, `native_sampling_row_ratio=0.5`,
    `native_sample_ms_per_sampled_row=84.65`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. This is a measured
    improvement from `3.947 fps` and `216.3 ms` paged average, but it is still
    far below realtime.
  - Latest no-skip completion gate after making logits-eval skipping opt-in:
    `.audex/runs/vllm-metal-diagnostic-20260708-101955.json`. Evidence:
    parent and spawned MLX probes are `Device(gpu, 0)`, the STS smoke used
    async vLLM `vllm.v1.engine.async_llm.AsyncLLM`, vLLM token streaming and
    decoder streaming were true, and 240 codec frames were generated. The
    verdict still failed because token streaming took 141.93 seconds:
    `codec_frames_per_second=1.691`, `audio_realtime_ratio=0.034`.
    `skipped_logits_eval=0`, `mx_eval_ms logits=75358.2/328`,
    `sample_tokens=40027.3/137`, and
    `native_detail_ms sample_eval=40031.3/137`; the diagnostic labels the
    measured blocker `pending_graph_eval_during_sampling`.
  - Diagnostic assessment now reports
    `native_sample_ms_per_sampled_row`,
    `dominant_mx_eval_per_step_category`, and
    `dominant_mx_eval_ms_per_step`. The CLI prints these fields so the next
    optimization target is visible without opening JSON.
  - Continuous-batching diagnostic:
    `--diagnose-vllm-tts-batch-size N` submits `N` independent TTS CFG pairs to
    the same async vLLM Metal engine and records aggregate codec-frame
    throughput. This is diagnostic-only and does not change the default CLI
    conversation path.
  - Batch-size-4 evidence:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-tts-batch-size 4 --diagnose-vllm-tts-batch-max-tokens 128 --diagnose-vllm-tts-batch-text "Please explain Python context managers in two concise sentences."`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-095413.json` with
    `request_count=8`, `total_codec_frame_count=388`, `elapsed_seconds=16.07`,
    and `codec_frames_per_second=24.144`.
  - Batch-size-8 evidence:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-tts-batch-size 8 --diagnose-vllm-tts-batch-max-tokens 128 --diagnose-vllm-tts-batch-text "Please explain Python context managers in two concise sentences."`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-095545.json` with
    `request_count=16`, `total_codec_frame_count=900`, `elapsed_seconds=12.54`,
    and `codec_frames_per_second=71.77`. This proves the pinned vLLM Metal path
    can exceed realtime aggregate Audex speech-token throughput when continuous
    batching is occupied. It does not prove single-conversation realtime
    playback; evenly divided, batch 8 is still only about `8.97 codec fps` per
    conditional stream.
  - Current conclusion: vLLM Metal/MLX execution and continuous batching are
    working. The remaining completion-gate blocker is single-stream/small-batch
    utilization, especially native categorical sampling and decode overhead
    when the active CFG batch contains only one conditional sample row.
  - Diagnostic no-CFG comparison:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-tts-batch-size 1 --diagnose-vllm-tts-batch-max-tokens 256 --diagnose-vllm-tts-batch-no-cfg --diagnose-vllm-tts-batch-text "Please explain Python context managers in two concise sentences."`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-100224.json` with
    `cfg_enabled=False`, `request_count=1`, `decode_reqs=1`,
    `total_codec_frame_count=189`, and `codec_frames_per_second=2.169`. This is
    slower than the one-pair CFG completion gate, so dropping CFG is not a
    viable path to realtime for the demo. The small-batch decode/logits path is
    the measured bottleneck.
  - MLX microbench from the generated vLLM Metal venv on `Device(gpu, 0)`:
    raw `mx.random.categorical` over `(1, 205312)` averaged about `0.47 ms`,
    while `(8, 205312)` averaged about `0.87 ms` total. Replacing
    temperature division with reciprocal multiplication measured about
    `0.37 ms` for the same single-row shape, so the native sampler now uses the
    equivalent multiply form. This means the
    `sample_tokens` timings in vLLM reports are mostly pending MLX graph
    evaluation being forced at the sampler boundary, not categorical sampling
    itself. The diagnostic `likely_bottleneck` label now reports
    `pending_graph_eval_during_sampling` for that shape.
  - NVIDIA reference check on 2026-07-08: the repo must not import the
    audiogen RVQ phase mask into S2S TTS as a performance shortcut. The mask in
    `inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm_rvq_logit_mask.py`
    and `tta_rvq_logits_processor.py` targets `<audiocodec_*>` TTA generation.
    Unified S2S TTS in `cascaded_s2s_web_server.py` uses stop IDs containing
    `<speechgen_end>`/EOS plus CFG `extra_args`, without `allowed_token_ids` or
    an RVQ phase logits processor. Audex-Mac does pass Audex-specific
    codec-window metadata in `extra_args` so the native MLX sampler can compact
    sampling to the legal speech-token domain without changing public vLLM
    sampling parameters.
  - Full-logits-eval skip:
    the patch skips vLLM Metal's explicit full-logits `mx.eval(...)` during
    Audex CFG decode by default when it can prove the batch is a safe CFG decode
    shape. Set `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=0` to disable it for
    diagnostics. The earlier opt-in command
    `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-101226.json`. It skipped
    `137` explicit full-logits evals but failed the completion gate at
    `codec_frames_per_second=2.855`, with
    `dominant_mx_eval_per_step_category=sample_tokens` and
    `likely_bottleneck=pending_graph_eval_during_sampling`. After the memory
    policy and compact speech-token sampler fixes, the same safe skip is part
    of the passing path.
  - Restored default completion-gate run:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-101955.json`. It records
    `AUDEX_VLLM_SKIP_PAGED_LOGITS_EVAL=null`, `skipped_logits_eval=0`, parent
    and spawned MLX on `Device(gpu, 0)`, async vLLM
    `vllm.v1.engine.async_llm.AsyncLLM`, vLLM token streaming, decoder
    streaming, and 240 generated codec frames. The verdict still fails because
    throughput is far below realtime: `codec_frames_per_second=1.691`,
    `audio_realtime_ratio=0.034`. Timing points at the small-batch graph
    evaluation boundary: `paged_sample_avg_ms=587.8`,
    `native_sample_ms_per_sampled_row=292.281`,
    `mx_eval_ms logits=75358.2/328`, `sample_tokens=40027.3/137`, and
    `dominant_mx_eval_per_step_category=sample_tokens`.
  - Earlier local validation after this checkpoint:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py tests/test_vllm_diagnostics.py -q`
    passed with `50 passed`; `./scripts/lint.sh` passed; full pytest
    `.venv/bin/python -m pytest -q` passed with
    `276 passed, 3 skipped`.
  - Post-cleanup live rerun:
    `.audex/runs/vllm-metal-diagnostic-20260708-102758.json` did not reach
    STS inference. `EngineCore` aborted during initialization with
    `[METAL] Command buffer execution failed: Insufficient Memory`, so this is
    not valid throughput evidence for the reciprocal-temperature sampler
    cleanup. The latest valid throughput report remains
    `.audex/runs/vllm-metal-diagnostic-20260708-101955.json`.
  - Historical memory-fraction diagnostic:
    `VLLM_METAL_MEMORY_FRACTION=0.85 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-102437.json`. It
    completed with `audex_cfg.max_num_seqs=128`, but remained below realtime:
    `codec_frames_per_second=1.971`, `paged_sample_avg_ms=477.6`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. This is stale
    historical evidence only: the current pinned vLLM-Metal config rejects
    explicit memory fractions when `VLLM_METAL_USE_PAGED_ATTENTION=0`.
  - Diagnostic-only capacity override:
    `AUDEX_VLLM_CFG_MAX_NUM_SEQS=2 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-103056.json`. It records
    `audex_cfg.max_num_seqs=2` and still fails at
    `codec_frames_per_second=1.827`, `paged_sample_avg_ms=490.2`, and
    `dominant_mx_eval_per_step_category=sample_tokens`. This rules out
    NVIDIA's 128-sequence server capacity setting as the primary single-stream
    throughput bottleneck for this path.
  - NVIDIA sampler-default correction:
    `audex_mac/audio_contract.py` previously carried stale TTS values
    (`temperature=0.1`, `top_k=80`, `cfg_scale=1.5`). The cached NVIDIA Audex
    reference scripts use `temperature=0.8`, `top_p=1.0`, `top_k=0`, and
    `cfg_scale=2.0` for S2S/TTS, so Audex-Mac now follows those values exactly.
    This removes the native sampler's positive-`top_k` path from default TTS
    requests.
  - Corrected-sampler completion-gate evidence:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=300 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-speech-max-tokens 256 --diagnose-vllm-sts-audio-fixture .audex/fixtures/normal-answer-question-16k-mono.wav`
    wrote `.audex/runs/vllm-metal-diagnostic-20260708-104405.json` with no
    diagnostic skip/materialization/capacity env enabled. It still fails the
    gate at `codec_frames_per_second=1.664`, `audio_realtime_ratio=0.033`,
    `paged_sample_avg_ms=490.3`, `native_sampling_row_ratio=0.5`, and
    `likely_bottleneck=pending_graph_eval_during_sampling`. The current timing
    evidence is `mx_eval_ms logits=56636.5/328`,
    `sample_tokens=41079.5/137`, and
    `native_detail_ms sample_eval=41084.7/137`. Treat this as proof that the
    sampler-default bug was correctness-relevant but not sufficient to solve
    single-stream realtime throughput.
  - Final vLLM Metal STS completion-gate evidence:
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
  - The historical passing run depended on three patch behaviors: explicit
    speech-domain sampling, sixteen ordered text segments, and the then-working
    `VLLM_METAL_MEMORY_FRACTION=0.85` setting. The current pinned vLLM-Metal
    source says the non-paged MLX KV cache path requires
    `VLLM_METAL_MEMORY_FRACTION=auto`; Audex-Mac now enforces `auto` so the
    fast default can construct vLLM-Metal config instead of failing before
    inference. The native MLX CFG sampler samples TTS from the legal Audex
    speech domain
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
    `./scripts/lint.sh` passed, and full pytest
    `.venv/bin/python -m pytest -q` passed with
    `283 passed, 3 skipped`.

### Audio Embedding Injection

- Purpose: preserve NVIDIA's `<so_embedding>` placeholder behavior and replace
  placeholder embeddings with Audex audio encoder/projector outputs.
- NVIDIA Audex-2B source evidence:
  - `inference_scripts_hf/inference_hf.py` builds prompts containing one
    `<sound>` placeholder and expands it to `<so_start>`, repeated
    `<so_embedding>`, and `<so_end>`.
  - `inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/processing_audex_vllm.py`
    uses 16 kHz audio, 30 second clips, 750 sound embeddings per clip, a
    30-clip / 900-second cap, and exactly one audio item per prompt.
  - `modeling_audex_vllm.py` wires a Qwen2/NV-Whisper-style audio tower and
    Audex audio projector into `NemotronDenseAudexForConditionalGeneration`;
    this is Audex's native audio-input path, not a separate STT model.
- Upstream symbols:
  - `vllm_metal.v1.model_adapter.DefaultModelAdapter.build_multimodal_adapter`
  - `vllm_metal.v1.model_runner.MetalModelRunner._run_mm_paged_forward`
  - `vllm_metal.multimodal.embeddings.merge_multimodal_embeddings`
- Current Audex-Mac code:
  - `audex_mac.vllm_sts_requests.build_asr_projected_embeddings_request` builds
    the vLLM ASR request shape used by the default CLI: prompt plus
    `multi_modal_data={"audio": [{"audex_projected_embeddings": ...}]}`.
  - `audex_mac.vllm_runtime.AudexVllmRuntime.transcribe_projected_audio` sends
    those projected embeddings through the persistent vLLM engine.
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession.project_wav_audio`
    prepares 16 kHz WAV input, extracts Audex features, runs the MLX
    audio_encoder/audio_projector, flattens `(clips, 750, hidden)` to
    `(clips * 750, hidden)`, and hands that array to vLLM as
    `audex_projected_embeddings`.
  - `audex_mac.patches.vllm_metal_audex_adapter.build_projected_audio_feature_spec`
    builds the exact vLLM `MultiModalFeatureSpec` shape Audex needs for
    projected audio: data contains `audex_projected_embeddings`, modality is
    `audio`, and `mm_position` points at a contiguous run of `<so_embedding>`
    token IDs matching the projected embedding count.
  - `audex_mac.audio_pcm` performs deterministic PCM preparation before feature
    extraction: mono/stereo normalization to float samples in `[-1, 1]`,
    fixed 30 second clip splitting, final-clip zero padding, and a 30-clip cap.
    It also loads native 16-bit PCM WAV fixtures generated by macOS
    `say`/`afconvert` without adding another model dependency.
  - `audex_mac.audio_features` validates the local
    `audio_preprocessor/preprocessor_config.json` and runs the Audex
    `WhisperFeatureExtractor` to produce `(clips, 128, 3000)` input features for
    the native NV-Whisper audio encoder.
  - `audex_mac.audio_encoder` ports the Qwen2Audio/NV-Whisper encoder to
    functional MLX using native `mx.conv1d`,
    `mx.fast.scaled_dot_product_attention`, GELU, layer norm, and the final
    average-pool step. It loads real `audio_encoder.*` BF16 tensors with
    `mx.load` and verifies `(clips, 128, 3000) -> (clips, 750, 1280)` on the MLX
    GPU device.
  - `audex_mac.audio_projector` loads Audex's real `audio_projector.*` BF16
    tensors with MLX `mx.load`, applies NVIDIA's RMSNorm -> fc1 -> relu^2 -> fc2
    projector, and verifies `(clips, 750, 2048)` output on the MLX GPU device.
    It deliberately avoids a NumPy or torch bridge; MLX safetensors loads are
    materialized before GPU encoder/projector math because direct GPU evaluation
    of safetensors `Load` nodes currently fails with `[Load::eval_gpu] Not
    implemented`.
  - `audex_mac.audio_splice` mirrors the current `mlx-vlm` multimodal
    mask/cumulative-index splice pattern for Audex sound tokens: full-tokenizer
    prompt IDs identify `<so_embedding>` positions, projected audio embeddings
    are gathered in placeholder order, and `mx.where` replaces the corresponding
    text embedding rows before MLX-LM prefill.
  - `audex_mac.audio_contract` records the 16 kHz / 30 second / 750
    embeddings-per-clip contract and fails loudly if the prompt does not contain
    exactly one `<sound>` placeholder.
  - `audex_mac.audio_components` verifies the full checkpoint config and
    safetensors index expose Audex's native audio encoder/projector surface:
    Audex architecture, `audio_model_type=NV-Whisper`, 128 mel bins, 1500 audio
    source positions, 750 sound embeddings per clip, `audio_encoder.*` tensors,
    and `audio_projector.norm/fc1/fc2` weights.
  - `./start.sh --preflight-audio-runtime` verifies the selected speech
    snapshot, Audex causal speech decoder files, and speech-token tokenizer
    markers before live STS work begins.
  - `./start.sh --preflight-audio-projector --model audex-2b` verifies the
    local projector execution path against the cached Audex-2B full checkpoint.
  - `./start.sh --preflight-audio-encoder --model audex-2b` verifies the local
    encoder-plus-projector execution path against the cached Audex-2B full
    checkpoint.
  - `./start.sh --preflight-audio-splice --model audex-2b` verifies the local
    encoder/projector/text-embedding splice path against the cached Audex-2B
    checkpoint and text-only MLX model.
- Expected guard: prompt expansion and embedding counts fail loudly on mismatch.
- Tests: `tests/test_audio_pcm.py`, `tests/test_audio_features.py`,
  `tests/test_audio_components.py`, `tests/test_audio_encoder.py`,
  `tests/test_audio_projector.py`, `tests/test_audio_splice.py`,
  `tests/test_audio_contract.py`, `tests/test_vllm_sts_requests.py`,
  `tests/test_vllm_runtime.py`, `tests/test_vllm_sts_cli.py`, and
  `features/speech_to_speech_cli.feature` cover PCM preparation,
  feature-extraction shape, component metadata, encoder/projector metadata,
  projected-embedding vLLM request shape, projected-embedding feature-spec
  construction, persistent-runtime handoff, embedding-splice count checks, and
  prompt-token contract.

### Speech Token Generation

- Purpose: preserve NVIDIA's speech-token generation constraints, logit masks,
  and CFG behavior where applicable.
- NVIDIA unified S2S source evidence:
  - `inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py`
    maps generated `<speechcodec_N>` tokenizer IDs after `<speechgen_start>`
    into speech-token frames until `<speechgen_end>`.
  - The same script uses Audex modes for ASR/text/TTS and does not require an
    external STT/TTS model.
- Current Audex-Mac code:
  - `audex_mac.audio_contract.build_codec_token_map` discovers
    `<speechgen_start>`, `<speechgen_end>`, and `<speechcodec_N>` IDs from the
    model tokenizer.
  - `audex_mac.audio_contract.iter_new_speech_frames` implements the
    incremental generated-token scan needed by the decoder path.
  - `audex_mac.audio_contract.build_tts_prompt` mirrors NVIDIA's unified S2S
    prompt shape exactly: ChatML system/user turns, user text prefixed by
    `<|text to speech|> Generate speech for this transcription. `, and an
    assistant prefix of `<think></think><speechgen_start>`.
  - `audex_mac.audio_contract.build_tts_null_prompt` and
    `tokenize_tts_cfg_pair` mirror NVIDIA's CFG null-prompt construction:
    build a same-length unconditional `<unk>` prompt and pad the pair when
    needed.
  - `audex_mac.speech_generation` loads the full `nemotron_dense_audex`
    checkpoint through the MLX shim and runs a short Metal/GPU speech-token
    generation smoke using NVIDIA's TTS sampling values:
    `temperature=0.8`, `top_p=1.0`, `top_k=0`, and
    `tts_cfg_scale=2.0`. The direct MLX path applies CFG as
    `uncond_logits + cfg_scale * (cond_logits - uncond_logits)` and feeds the
    same sampled token into both conditional and unconditional caches.
- Full-vocab requirement: text-only Audex-2B has `vocab_size=131072`, while the
  full tokenizer maps `<speechgen_start>` to `131075`, `<speechgen_end>` to
  `131076`, and `<speechcodec_0>` to `131077`. Speech output therefore cannot
  be implemented on the text-only MLX head; the full checkpoint head is
  mandatory.
- Expected guard: speech-token generation code can name the relevant NVIDIA
  source behavior and sampler settings.
- Tests: `tests/test_audio_contract.py` covers TTS prompt construction,
  sampler constants, speech-codec token discovery, and incremental frame
  extraction. `tests/test_speech_generation.py` guards the full-vocab readiness
  predicate. Runtime smoke:
  `./start.sh --preflight-speech-token-generation --model audex-2b`.

### vLLM Metal CFG TTS Wiring

- Purpose: make the vLLM Metal STS path use NVIDIA's CFG TTS recipe instead of
  merely attaching inert `SamplingParams.extra_args` to paired requests.
- NVIDIA source evidence:
  - `inference_scripts_vllm/audiogen_scripts/cfg_logits_processor.py` defines
    `CFGLogitsProcessor`, blends conditional/unconditional logits as
    `uncond + cfg_scale * (cond - uncond)`, and patches CUDA
    `GPUModelRunner._sample` so the unconditional row receives the conditional
    sampled token.
  - `inference_scripts_vllm/audiogen_scripts/vllm_cfg_patch.py` patches the
    vLLM scheduler so CFG partners are admitted, kept adjacent, and kept at the
    same progress.
  - `inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py`
    calls `apply_cfg_patches()` before engine creation, passes
    `logits_processors=[CFGLogitsProcessor]`, disables prefix caching, and sets
    `max_model_len`, `max_num_batched_tokens`, and `max_num_seqs` from
    `DEFAULT_CFG_NUM_SEQS = 2`.
- Current Audex-Mac code:
  - `audex_mac.vllm_cfg.configure_audex_vllm_cfg` locates the selected model
    snapshot's `inference_scripts_vllm/audiogen_scripts` directory, prepends it
    to `sys.path` and `PYTHONPATH`, imports NVIDIA's Apache-licensed
    `CFGLogitsProcessor` from the local snapshot, calls NVIDIA's
    `apply_cfg_patches()`, and updates vLLM engine kwargs with NVIDIA's
    defaults.
  - `audex_mac.vllm_runtime.AudexVllmRuntime.from_model_path` applies that CFG
    configuration before constructing the persistent `LLM` engine and raises
    before engine construction if required CFG assets are missing. This keeps
    the default STS path from silently running paired TTS requests without
    NVIDIA's CFG processor.
  - `audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller` is a
    no-op vLLM logits processor that runs inside each spawned worker and
    installs Metal-specific sampler patches. This is needed because NVIDIA's
    post-sampling token-sync patch targets CUDA `GPUModelRunner`, not vLLM
    Metal.
  - `audex_mac.patches.vllm_metal_cfg.apply_vllm_metal_cfg_patches` wraps
    `vllm_metal.v1.sampling_batch.sample_from_logits` to copy each conditional
    CFG sampled token into the matching unconditional row, mirrors that symbol
    into `vllm_metal.v1.model_runner`, and replaces `sample_prefill_tokens`
    with a batched implementation so prefill rows can see CFG pairs together
    instead of sampling one row at a time.
  - `audex_mac.vllm_diagnostics` now probes CFG wiring without loading the
    model, applies the vLLM Metal CFG sampler patch in the probe process, and
    fails the vLLM Metal diagnostic verdict if NVIDIA's CFG processor, the
    Audex-Mac Metal token-sync installer, the sampler patch, or
    `enable_prefix_caching=False` are missing.
- Expected guard: CFG is only considered ready when the model snapshot includes
  NVIDIA's vLLM audiogen scripts and the engine kwargs include both NVIDIA's
  `CFGLogitsProcessor` and Audex-Mac's Metal token-sync installer. Runtime
  startup fails loudly if the required CFG script directory is not present for
  the selected model.
- Tests: `tests/test_vllm_cfg.py` covers snapshot script discovery, NVIDIA-style
  engine kwargs, CFG token synchronization, and patching vLLM Metal sampler
  symbols. `tests/test_vllm_runtime.py` covers fail-loud startup behavior when
  required CFG assets are missing. `tests/test_vllm_diagnostics.py` covers the
  CFG probe and diagnostic verdict, including rejection of a missing sampler
  patch.
- Latest validation:
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `213 passed, 3 skipped`.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260707-233901.json`
  - Diagnostic result in the Codex shell: exit `2`; `audex_cfg.ready=True`,
    NVIDIA `CFGLogitsProcessor` and
    `audex_mac.patches.vllm_metal_cfg.AudexMetalCFGTokenSyncInstaller` are in
    engine kwargs, `enable_prefix_caching=False`, `max_model_len=5120`,
    `max_num_batched_tokens=10240`, and `max_num_seqs=128`. The vLLM Metal
    sampler-patch probe cannot import the deep `vllm_metal.v1.model_runner`
    path in this shell because MLX cannot access a Metal device, so a
    GPU-visible terminal must still prove `vllm_metal_patch.ready=True`.

### vLLM STS Conversation Context

- Purpose: make the default vLLM STS session conversational across turns instead
  of generating the text response from only the latest transcript.
- Current Audex-Mac code:
  - `audex_mac.vllm_sts_requests.build_text_messages_response_request` builds a
    vLLM text response request from an explicit chat-template message list while
    preserving NVIDIA text sampler settings and non-thinking default.
  - `audex_mac.vllm_runtime.AudexVllmRuntime.generate_text_response_from_messages`
    runs that message-aware request through the persistent vLLM engine and
    still strips optional thinking blocks from the displayed response.
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession.run_turn_from_wav` now
    generates the text response from `self.messages + latest user transcript`
    and then persists the exact user/assistant turn. The older
    transcript-only runtime API remains for focused text probes and tests.
- Expected guard: the vLLM STS text stage must include prior user and assistant
  messages in the prompt whenever a conversation is already in memory.
- Tests: `tests/test_vllm_sts_requests.py`,
  `tests/test_vllm_runtime.py`, and `tests/test_vllm_sts_cli.py` cover
  message-history prompt construction, runtime handoff, and session message
  persistence.
- Latest validation:
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `215 passed, 3 skipped`.

### vLLM STS Decoder Chunking

- Purpose: move the default vLLM STS speech-output path onto the same
  lookahead-aware decoder-session pipeline needed for streamed token playback,
  while still reporting honestly that vLLM token generation is synchronous.
- Current Audex-Mac code:
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession.generate_speech_output`
    now feeds generated codec frames through `AudexSpeechDecoderSession` with
    `DEFAULT_STREAM_DECODER_CHUNK_FRAMES`, writes per-chunk WAV artifacts, then
    writes the final WAV.
  - The speech-output run log records `decoder_streaming=True`,
    `vllm_token_streaming=False`, `decoder_chunk_frames`, and
    `chunk_wav_paths`.
  - The returned `SpeechOutputSmokeResult` includes the chunk artifact paths
    while keeping `streaming=False` until vLLM token generation itself is
    actually streamed.
- Expected guard: vLLM STS must not claim end-to-end streaming until tokens are
  emitted incrementally from vLLM. Decoder chunking is a required preparatory
  step, not proof of first-audio-before-generation-complete.
- Tests: `tests/test_vllm_sts_cli.py` covers decoder-session use, chunk
  artifacts, and honest `vllm_token_streaming=False` logging.
- Latest validation:
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `215 passed, 3 skipped`.

### vLLM Streaming API Inspection

- Purpose: make the Phase 5 token-streaming gap explicit and verifiable against
  the pinned vLLM API instead of assuming the current sync `LLM.generate` path
  can emit tokens incrementally.
- Current Audex-Mac code:
  - `audex_mac.vllm_streaming.inspect_vllm_streaming_support` inspects the
    active vLLM package for `LLM.generate`, `AsyncLLMEngine.generate`, and
    `RequestOutputKind`.
  - `audex_mac.vllm_diagnostics.run_vllm_metal_diagnostics` records the result
    under `vllm_streaming_api`.
- Current conclusion:
  - The pinned `LLM.generate` API is an offline/final-return path, not an async
    token stream.
  - The pinned `AsyncLLMEngine.generate` API is an async generator and
    `RequestOutputKind.CUMULATIVE`/`FINAL_ONLY` are available, matching
    NVIDIA's reference direction for streamed text/TTS output.
- Tests: `tests/test_vllm_streaming.py` covers the API-shape inspector without
  requiring real vLLM in the fast test environment.
- Latest validation:
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `217 passed, 3 skipped`.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-002915.json`
  - Diagnostic evidence: `vllm_streaming_api.sync_llm_generate_streams=False`,
    `async_engine_available=True`, `async_generate_is_asyncgen=True`,
    `request_output_kind_available=True`,
    `cumulative_output_kind_available=True`, and
    `final_only_output_kind_available=True`.

### Async vLLM Runtime Skeleton

- Purpose: create the runtime boundary needed for NVIDIA-style streamed TTS and
  default CLI execution through `AsyncLLMEngine`.
- Current Audex-Mac code:
  - `audex_mac.vllm_runtime.AudexAsyncVllmRuntime.from_model_path` constructs
    an `AsyncLLMEngine` from `AsyncEngineArgs` using the same base engine kwargs
    and CFG configuration as the synchronous runtime.
  - `AudexAsyncVllmRuntime.stream_many` submits multiple requests concurrently
    to `AsyncLLMEngine.generate`, computes token deltas from cumulative
    `RequestOutput` payloads, and propagates worker exceptions instead of
    hanging the stream.
  - `AudexAsyncVllmRuntime.stream_tts_cfg_codec_frames` builds the same
    conditional/unconditional TTS CFG pair as the sync runtime, streams both
    requests concurrently, and yields new speech-codec frames only from the
    conditional stream.
- Expected guard: CFG TTS streaming must submit cond/uncond requests as a pair,
  not serialize them, because NVIDIA's CFG scheduler patch depends on the pair
  being in the same engine scheduling window.
- Tests: `tests/test_vllm_runtime.py` covers async token delta calculation,
  concurrent cond/uncond request submission, and conditional-only codec-frame
  streaming with a fake async engine.
- Latest validation:
  - Focused runtime tests:
    `.venv/bin/python -m pytest tests/test_vllm_runtime.py -q` passed with
    `13 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `219 passed, 3 skipped`.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-003450.json`
  - Diagnostic result in the Codex shell: exit `2`; the verdict still reports
    only the expected sandbox Metal-device failure, while
    `vllm_streaming_api` continues to prove the pinned async API shape.

### vLLM STS Async TTS Stream Handoff

- Purpose: connect the async vLLM TTS codec-frame stream to the STS decoder
  path without forcing the default CLI to load a second vLLM engine before live
  Metal validation.
- Current Audex-Mac code:
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession` accepts an optional
    `async_runtime`.
  - When configured, `generate_speech_output(...)` calls
    `generate_speech_output_streaming_from_async_runtime(...)`, consumes
    `async_runtime.stream_tts_cfg_codec_frames(...)`, feeds new codec frames
    into `AudexSpeechDecoderSession`, writes per-chunk WAV artifacts as decoder
    chunks arrive, and writes a final WAV/run log.
  - The async speech-output log records `streaming=True`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `decoder_chunk_frames`, and `chunk_wav_paths`.
  - Without an `async_runtime`, the existing synchronous vLLM speech-output path
    remains in use and logs `vllm_token_streaming=False`.
- Expected guard: async TTS streaming is only claimed when frames come from the
  async runtime stream. An explicit sync runtime must continue to log that vLLM
  token streaming is false.
- Tests: `tests/test_vllm_sts_cli.py` covers both the existing synchronous
  speech-output path and the optional async-runtime streaming path with a fake
  async codec-frame source.
- Latest validation:
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_runtime.py tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py tests/test_fast_bdd.py -q`
    passed with `67 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `220 passed, 3 skipped`.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-003830.json`
  - Diagnostic result in the Codex shell: exit `2`; the verdict still reports
    only the expected sandbox Metal-device failure, while
    `vllm_streaming_api` continues to show `async_engine_available=True` and
    `async_generate_is_asyncgen=True`.

### Default Async vLLM Session Runtime

- Purpose: make `./start.sh` exercise the vLLM continuous-batching runtime path
  by default instead of treating async generation as an optional TTS-only
  add-on.
- Audex-Mac code:
  - `audex_mac.vllm_runtime.AudexAsyncVllmRuntime.stats` exposes the same
    `AudexVllmRuntimeStats` contract as the synchronous runtime.
  - `AudexAsyncVllmRuntime.generate_many_final`, `generate_one_final`,
    `transcribe_projected_audio`, `generate_text_response_from_messages`, and
    `generate_tts_cfg_pair` collect final outputs from the async token stream
    for ASR/text call sites that still need a final response object.
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession` constructs
    `AudexAsyncVllmRuntime.from_model_path(...)` when no runtime is explicitly
    injected. ASR, text response generation, and speech output use the async
    runtime in that default case.
  - The synchronous `AudexVllmRuntime` path remains available for explicit
    injection, focused tests, and diagnostics; it is no longer the default
    constructor path for the CLI session.
- Expected guard: the default STS session must not create both sync and async
  vLLM engines. A session with no injected sync runtime must still run ASR,
  text response generation, and speech output through one async runtime surface.
- Tests:
  - `tests/test_vllm_runtime.py` covers async final-result ASR/text/TTS helpers
    and the async runtime stats contract.
  - `tests/test_vllm_sts_cli.py` covers a full fake speech-to-speech turn with
    `runtime=None` and async ASR, async text generation, and streamed async TTS.
- Latest validation:
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_runtime.py tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py tests/test_vllm_cfg.py tests/test_vllm_streaming.py -q`
    passed with `36 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `225 passed, 3 skipped`.
  - Diagnostic command: `./start.sh --diagnose-vllm-metal`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-004534.json`
  - Diagnostic evidence: `vllm_metal_audex_adapter=True`,
    `audex_processor.ready=True`, `audex_cfg.ready=True`,
    `vllm_streaming_api.async_engine_available=True`, and
    `vllm_streaming_api.async_generate_is_asyncgen=True`.
  - Diagnostic result in the Codex shell: exit `2`; the verdict still reports
    only the expected sandbox Metal-device failure,
    `RuntimeError: [metal::load_device] No Metal device available`.

### Default vLLM STS Smoke Diagnostic

- Purpose: provide the missing live proof command for the default async
  speech-to-speech runtime. The normal `--diagnose-vllm-metal` probe verifies
  patch and API shape; this opt-in probe exercises the default async STS
  fixture path and records first-audio/streaming evidence.
- Audex-Mac code:
  - CLI flags:
    - `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
    - audible playback proof:
      `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    - optional fixture:
      `--diagnose-vllm-sts-audio-fixture /path/to/16khz.wav`
  - `--diagnose-vllm-sts-smoke` implies `--diagnose-vllm-metal` and forces
    speech readiness for model selection/cache checks, so the diagnostic targets
    `checkpoint_folder_full` rather than `checkpoint_folder_textonly`.
  - `--diagnose-vllm-sts-play` implies `--diagnose-vllm-sts-smoke`, records
    `sts_probe.play_audio=True`, and passes `play=True` to the default async
    vLLM fixture turn.
  - `audex_mac.vllm_diagnostics.run_vllm_metal_diagnostics` accepts
    `run_sts_smoke` and `sts_audio_fixture`.
  - When STS smoke is enabled, the diagnostic report includes
    `speech_runtime` with full-checkpoint and decoder paths, and uses the full
    speech checkpoint for `model_adapter` and `audex_cfg` probes.
  - `audex_mac.vllm_diagnostics._probe_vllm_sts_default_runtime` runs in a
    subprocess, calls `preflight_audio_runtime`, creates a one-second 16 kHz
    silence WAV when no fixture is provided, and executes
    `run_vllm_fixture_turn(..., play=False)`.
  - The diagnostic report writes `sts_probe` with the model path, decoder path,
    input/output WAV paths, STS run-log path, speech-output run-log path,
    engine class, turn timings, transcript/response prefixes, streaming flags,
    first-audio timing, generated token/frame counts, chunk count, end-token
    status, and max-token status.
  - `_diagnostic_verdict(..., require_sts=True)` now fails if the opt-in STS
    smoke probe is not ready or if a ready STS smoke does not report
    `vllm_token_streaming=True` and `decoder_streaming=True`.
- Expected guard: GPU-visible completion of `docs/engineering/vllm-metal.md` requires this
  command to produce `sts_probe.ready=True`, an async vLLM engine class, and
  `speech_streaming.vllm_token_streaming=True`. The verdict also requires
  streaming decoder evidence, recorded first-audio timing, nonzero generated
  speech-token count, nonzero codec-frame count, and at least one decoder chunk.
  A Codex/headless shell failure with `[metal::load_device] No Metal device
  available` is evidence of sandbox Metal visibility, not proof of runtime
  success.
  When `sts_probe.play_audio=True`, the verdict additionally requires
  `speech_streaming.playback_transport=sounddevice_raw_output_stream` and
  non-null `speech_streaming.first_playback_started_seconds`. It also requires
  `speech_streaming.playback_diagnostics` with device-underflow,
  queue-underrun, queue-overrun, and chunks-written counters.
- Tests: `tests/test_vllm_diagnostics.py` covers subprocess command wiring and
  the STS-required verdict failure. `tests/test_start_sh.py` covers
  CLI/startup flag forwarding behavior and verifies the STS smoke diagnostic
  uses speech readiness.
- Latest validation:
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_diagnostics.py tests/test_start_sh.py -q`
    passed with `22 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `229 passed, 3 skipped`.
  - Diagnostic command:
    `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-024529.json`
  - Diagnostic evidence: selected model
    `nvidia/Nemotron-Labs-Audex-30B-A3B`; primary `model_path` is the 30B
    `checkpoint_folder_full`; `speech_runtime.ready=True`; `model_adapter`
    sees `NemotronHAudexForConditionalGeneration`/`nemotron_h_audex`;
    `audex_cfg.ready=True` and points at the 30B
    `inference_scripts_vllm/audiogen_scripts`.
  - Diagnostic result in the Codex shell: exit `2`; `sts_probe.enabled=True`,
    `sts_probe.ready=False`, and the STS smoke error is the same expected
    sandbox Metal-device failure:
    `RuntimeError: [metal::load_device] No Metal device available`.
- Latest guard tightening:
  - `audex_mac.vllm_diagnostics._sts_smoke_evidence_failures` rejects ready
    STS smoke reports that use a sync engine class, omit vLLM token streaming,
    omit streaming decoder evidence, omit first-audio timing, or generate zero
    speech tokens, zero codec frames, or zero decoder chunks.
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_diagnostics.py -q`
    passed with `17 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `231 passed, 3 skipped`.
- Latest playback-smoke flag:
  - `audex_mac.cli.main` makes `--diagnose-vllm-sts-play` imply
    `--diagnose-vllm-sts-smoke`.
  - `audex_mac.vllm_diagnostics._probe_vllm_sts_default_runtime` accepts
    `play_audio`, records it in `sts_probe.play_audio`, and passes it through
    to `run_vllm_fixture_turn(..., play=play_audio)`.
  - `audex_mac.vllm_diagnostics._sts_smoke_evidence_failures` requires
    playback transport, first-playback timing, and playback diagnostics only
    when `play_audio=True`.
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_diagnostics.py tests/test_start_sh.py -q`
    passed with `27 passed`.
  - `./scripts/lint.sh` passed.
  - `.venv/bin/python -m pytest -q` passed with `235 passed, 3 skipped`.
  - Silent diagnostic command:
    `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke`
  - Diagnostic report:
    `.audex/runs/vllm-metal-diagnostic-20260708-025504.json`
  - Diagnostic result in the Codex shell: exit `2`; the report records
    `sts_probe.play_audio=false`, selected cached 30B full-checkpoint speech
    assets, and fails only at the expected sandbox Metal-device boundary.
  - GPU-visible diagnostic follow-up:
    `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    reached MLX Metal (`Device(gpu, 0)`) in both parent and spawned probes,
    repaired `current_platform` to `vllm_metal.platform.MetalPlatform`,
    installed the Audex adapter and CFG sampler patches, and selected the
    cached Audex-30B-A3B full speech checkpoint. The remaining blocker was
    vLLM engine construction failing before generation because Transformers
    hashed local Audex remote-code files from the HF cache `blobs/` path and
    looked for `blobs/configuration_nemotron_h.py`. The
    `transformers_local_dynamic_modules` patch now targets that exact failure;
    rerun the same GPU-visible audible proof command after validation.

### Default Async vLLM Streaming Playback

- Purpose: make the default async vLLM speech-to-speech path audibly stream
  decoded chunks instead of only writing chunk artifacts and then playing the
  final WAV after generation completes.
- Audex-Mac code:
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession.generate_speech_output_streaming_from_async_runtime`
    now creates `_ContinuousPcmPlayer` when `play=True`, starts it before async
    TTS generation, and enqueues every decoded waveform chunk as it is emitted
    by `AudexSpeechDecoderSession`.
  - The async vLLM speech-output run log records `playback_transport`,
    `playback_prebuffer_seconds`, `first_playback_started_seconds`, and
    `playback_diagnostics` in addition to `vllm_token_streaming=True` and
    `decoder_streaming=True`.
  - The async vLLM hot path no longer writes a WAV artifact for every decoded
    chunk. It writes the final WAV at the end and records `decoded_chunk_count`
    for streaming evidence; `chunk_wav_paths` remains empty for async vLLM
    streaming turns.
  - The STS smoke diagnostic forwards playback transport/diagnostic fields when
    the underlying speech-output log contains them.
  - `--diagnose-vllm-sts-play` bounds its speech generation with
    `AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS` or the default
    `256` speech tokens. This keeps the audible diagnostic a deterministic
    playback proof instead of an unconstrained TTS demo. Non-playback STS smoke
    keeps the normal speech budget.
- Expected guard: the default interactive vLLM STS path must not call `afplay`
  on the final WAV after async TTS has completed. With `play=True`, decoded
  chunks should enter the continuous PCM queue as soon as the decoder emits
  them.
- Tests:
  - `tests/test_vllm_sts_cli.py` verifies that `play=True` on the async vLLM
    path enqueues decoded chunks into `_ContinuousPcmPlayer`, records
    `sounddevice_raw_output_stream`, and records first-playback timing plus
    playback diagnostics.
  - `tests/test_vllm_diagnostics.py` continues to verify the STS smoke evidence
    gate and report fields.
- Latest validation:
  - Focused command:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_diagnostics.py -q`
    passed with `28 passed`.
  - Live playback diagnostic:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=180 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-082327.json`.
  - The report records `vllm_token_streaming=True`,
    `decoder_streaming=True`, `playback_transport=sounddevice_raw_output_stream`,
    `first_audio_ready_seconds=1.314`, `first_playback_started_seconds=11.707`,
    `generated_codec_frame_count=56`, and `chunk_count=12`.
  - Remaining realtime-playback evidence gap: the same report records
    `device_underflow_count=3` and `queue_underrun_count=2`, so this patch
    proves the transport and removes disk writes from the hot path but does not
    yet prove low-underrun playback for a normal answer.

### Persistent Async vLLM Session and Segmented Speech Decode

- Purpose: fix the default vLLM Metal STS CLI path after startup greeting
  playback by keeping the async vLLM engine on one event loop and by decoding
  segmented TTS as separate causal speech streams.
- Failure evidence:
  - `./start.sh` could play the startup greeting, then the first recorded turn
    failed with `vllm.v1.engine.exceptions.EngineDeadError`.
  - The same log showed the vLLM engine core exiting with an MPS allocator
    assertion after startup TTS had already used the async engine.
  - Startup or turn audio could sound like distorted, voice-like garbage when
    separately generated TTS segments were fed through one continuous speech
    decoder state.
- Audex-Mac code:
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession` now creates one
    persistent `asyncio` event loop for async-only vLLM sessions and routes
    startup TTS, ASR, text generation, and turn TTS through `_run_async(...)`
    instead of separate `asyncio.run(...)` calls.
  - `run_vllm_interactive_ptt(...)` now shuts the session down in a `finally`
    block so an interrupted CLI run does not leave a stale engine process.
  - `audex_mac.vllm_runtime.VllmTtsCodecStreamEvent` carries
    `segment_index` and `segment_finished` so the streaming consumer can detect
    boundaries between independently generated TTS segments.
  - `generate_speech_output_streaming_from_async_runtime(...)` flushes and
    resets `AudexSpeechDecoderSession` at segment boundaries and inserts the
    same short silence used by the MLX segmented TTS path before the next
    segment starts when segmented TTS is explicitly used.
  - The vLLM STS path uses `tts_target_segments=1` by default so normal spoken
    answers are synthesized as coherent utterances. Higher segmentation remains
    available for experiments, but the default avoids one-word/short-fragment
    TTS that destroys prosody.
  - The vLLM STS path now defaults the MLX speech decoder stream chunk size to
    75 frames, matching NVIDIA's FastAPI server default. The earlier 5-frame
    chunks produced many tiny playback writes and underruns when single-user
    vLLM Metal TTS generation was slower than realtime.
  - `audex_mac.patches.vllm_metal_cfg` patches vLLM worker cleanup to suppress
    PyTorch's known MPS allocator assert only during intentional distributed
    shutdown. This prevents successful CLI sessions from ending with a scary
    EngineCore traceback while preserving unrelated cleanup failures.
  - `VllmSpeechToSpeechSession.shutdown()` cancels and drains pending tasks on
    its persistent async event loop before closing it, preventing parent-side
    `Event loop is closed`/`Task was destroyed but it is pending` shutdown
    noise.
- Expected guard: the async vLLM engine should survive startup greeting
  generation and then handle the first user turn in the same CLI process.
  Segmented TTS must not concatenate independent speech-token streams through
  stale causal decoder state. Audible smoke diagnostics should report zero
  device underflows and zero queue underruns. Intentional shutdown should not
  print the PyTorch MPS allocator assert or pending async task tracebacks.
- Tests:
  - `tests/test_vllm_sts_cli.py` verifies that async sessions reuse one event
    loop, that segmented vLLM TTS resets the decoder between segments, and that
    the run log records `tts_target_segments`. It also verifies shutdown
    cancellation/draining of pending async tasks.
  - `tests/test_vllm_runtime.py` continues to verify ordered segmented CFG TTS
    frame emission.
  - `tests/test_vllm_cfg.py` verifies that the worker-local CFG patch installs
    the MPS cleanup suppressor and that it suppresses only the known allocator
    assert while re-raising unrelated cleanup failures.
- Latest validation:
  - Focused command:
    `PYTHONPATH=. .venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_runtime.py -q`
    passed with `26 passed`.
  - Fast suite:
    `PYTHONPATH=. .venv/bin/python -m pytest -m fast -q`
    passed with `276 passed, 3 skipped, 9 deselected`.
  - Segmented GPU-visible audible diagnostic:
    `AUDEX_STS_SMOKE_SPEECH_MAX_TOKENS=64 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=240 ./start.sh --model audex-2b --diagnose-vllm-metal --diagnose-vllm-sts-smoke --diagnose-vllm-sts-play`
    passed with report
    `.audex/runs/vllm-metal-diagnostic-20260708-143612.json`.
  - That report records `engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    `vllm_token_streaming=True`, `decoder_streaming=True`,
    `playback_transport=sounddevice_raw_output_stream`,
    `first_audio_ready_seconds=0.743`,
    `first_playback_started_seconds=1.611`, `generated_codec_frame_count=99`,
    `codec_fps=64.833`, and `realtime_ratio=1.297`.
  - Playback diagnostics record `device_underflow_count=0` and
    `queue_underrun_count=0`. `queue_overrun_count=2` means the queue held more
    than the current 2-second high-water threshold; no samples were dropped.
  - Single-utterance GPU-visible audible diagnostic:
    after changing the default to `tts_target_segments=1`, the same 2B playback
    smoke completed the STS path with report
    `.audex/runs/vllm-metal-diagnostic-20260708-143814.json`.
  - That report records `ready=True` for the STS probe,
    `first_audio_ready_seconds=1.996`, `first_playback_started_seconds=2.04`,
    `generated_codec_frame_count=60`, `device_underflow_count=0`, and
    `queue_underrun_count=0`.
  - The overall diagnostic verdict is `not ready` for the single-utterance run
    only because the current realtime-throughput gate reports
    `codec_fps=31.898` and `realtime_ratio=0.638`. The audible result has
    materially better prosody than segmented TTS, so single-utterance TTS is the
    CLI default.
  - Startup-then-turn GPU-visible smoke:
    a temporary bounded script generated startup greeting TTS on vLLM Metal,
    then ran a fixture speech-to-speech turn on the same
    `VllmSpeechToSpeechSession` and exited `0`.
  - That smoke produced greeting artifact
    `.audex/runs/startup-then-turn-greeting-20260708-144603.wav`, turn log
    `.audex/runs/sts-turn-vllm-20260708-144612.json`,
    `turn_engine_class=vllm.v1.engine.async_llm.AsyncLLM`,
    `greeting_tts_target_segments=1`, `turn_tts_target_segments=1`, and
    `turn_generated_codec_frames=64`.
  - The final startup-then-turn run shut down cleanly: no
    `EngineDeadError`, no PyTorch MPS allocator traceback, and no pending
    async task traceback.

### 2026-07-08: vLLM Metal CFG scheduler alignment for 30B TTS

- Symptom: Audex-30B-A3B startup greeting TTS generated valid-looking speech
  codec tokens for a simple one-sentence prompt, but the audio was incoherent
  and the stream did not emit `<speechgen_end>` before the user interrupted.
  The bad artifact
  `.audex/runs/startup-greeting-vllm-20260708-141516.json` recorded 1,894
  generated tokens, 1,891 codec frames, no end token, a 37.82s WAV, and
  playback underruns/overruns.
- Rejected workaround: capping the startup greeting token budget makes the
  symptom shorter but violates the demo contract. Startup speech must not have
  a special hard cap; if the configured utterance is long, Audex should be
  allowed to speak it.
- NVIDIA reference evidence:
  - `inference_scripts_vllm/audiogen_scripts/vllm_cfg_patch.py` patches the
    vLLM scheduler so CFG partners are held until both requests are present,
    kept adjacent in the waiting queue, and equalized when one partner's
    `num_computed_tokens` runs ahead.
  - `inference_scripts_vllm/audiogen_scripts/cfg_logits_processor.py` then
    blends conditional/unconditional logits and syncs sampled tokens.
- Audex-Mac patch:
  - `audex_mac.patches.vllm_metal_cfg.apply_vllm_metal_cfg_patches` now
    installs `_patch_scheduler_for_cfg()` in addition to the Metal sampler
    wrappers.
  - The scheduler patch tracks `cfg_pair_id`/`cfg_role` on `Scheduler` requests,
    holds incomplete pairs, reorders waiting requests so pair partners are
    adjacent, finishes pair partners together, and equalizes scheduled progress
    before decode sampling.
  - The scheduler wrapper preserves arbitrary `Scheduler.schedule(*args,
    **kwargs)` arguments because the pinned vLLM Metal runtime calls
    `schedule(throttle_prefills)`, while NVIDIA's published CUDA patch snippet
    wraps a no-argument schedule method.
  - `VllmMetalCfgPatchReport.ready` now requires `scheduler=True`. Missing
    scheduler alignment is a correctness failure because the MLX sampler can
    otherwise blend desynchronized CFG rows.
- Tests:
  - `tests/test_vllm_cfg.py` verifies the patch wraps real sampler symbols and
    marks the scheduler patch installed.
  - `tests/test_vllm_cfg.py::test_vllm_metal_cfg_scheduler_keeps_cfg_pairs_aligned`
    covers queue reordering and `num_computed_tokens` equalization.
- Real-runtime probe:
  - In the vendored vLLM Metal environment,
    `apply_vllm_metal_cfg_patches()` reports
    `sample_from_logits=True`, `sample_prefill_tokens=True`, `scheduler=True`,
    `model_runner_symbols=True`, `mps_cleanup=True`, and `ready=True`.
  - An uncapped Audex-30B-A3B no-playback startup TTS probe using the normal
    `_speech_max_tokens_for_text(...)` budget (`max_tokens=2400`) generated
    396 tokens / 395 codec frames, emitted `<speechgen_end>`, did not hit the
    max-token limit, and wrote a 7.9s WAV:
    `.audex/runs/startup-greeting-vllm-scheduler-probe-20260708-152035.wav`.

### 2026-07-08: vLLM Metal MLX device-info API update

- Symptom: startup emitted
  `mx.metal.device_info is deprecated and will be removed in a future version. Use mx.device_info instead.`
- Upstream source:
  - Pinned vLLM Metal calls `mx.metal.device_info()` in
    `.audex/vendor/vllm-metal/vllm_metal/utils.py:set_wired_limit`.
  - The same checkout already uses `mx.device_info()` in
    `vllm_metal/v1/cache_policy.py`, so this is an upstream stale call site,
    not an Audex model requirement.
- Audex-Mac patch:
  - `audex_mac.patches.runtime._patch_vllm_metal_device_info_api` replaces
    `vllm_metal.utils.set_wired_limit` with an equivalent implementation that
    calls `mx.device_info()` when available and falls back to
    `mx.metal.device_info()` only for older MLX versions.
  - `AudexPatchReport` now includes `vllm_metal_device_info_api`; `ready`
    requires it so diagnostics fail loudly if this deprecation patch cannot be
    installed.
  - The generated `sitecustomize.py` path applies this patch in spawned vLLM
    workers when `AUDEX_MAC_AUTO_PATCHES=1`, so the EngineCore process gets the
    same fix as the main process.
- Tests:
  - `tests/test_audex_patches.py::test_vllm_metal_device_info_patch_uses_current_mlx_api`
    asserts the patched function uses `mx.device_info()` and does not touch the
    deprecated `mx.metal.device_info()` path when the current API is present.
  - `tests/test_text_generation.py` was updated so run logs include
    `vllm_metal_device_info_api=True`.
- Real-runtime probe:
  - In the vendored vLLM Metal environment, `apply_audex_runtime_patches()`
    reported `vllm_metal_device_info_api=True`; calling
    `vllm_metal.utils.set_wired_limit()` completed with no
    `mx.metal.device_info` deprecation warning.

### 2026-07-08: vLLM Metal text-mode modality guard and native sampler expansion

- Symptom:
  - ASR/text generations could emit Audex speech-token vocabulary such as
    `<speechcodec_20885>` even though those modes should return ordinary text.
  - ASR/text requests were also falling through to vLLM Metal's original
    sampler path because Audex-Mac's native MLX fast path only accepted CFG TTS
    pairs. In Activity Monitor this showed up as high `VLLM::EngineCore` CPU
    use with only partial GPU saturation during ASR/text turns.
- Upstream source:
  - NVIDIA's recipes rely on vLLM/CUDA behavior for the unified Audex modality
    path. The pinned vLLM Metal stack does not currently know Audex's output
    modality constraints, so Mac text-mode sampling needs an Audex-specific
    guard.
- Audex-Mac patch:
  - `audex_mac.vllm_sts_requests` now attaches
    `audex_disallow_token_ranges` / `audex_disallow_token_ids` in
    `SamplingParams.extra_args` for ASR and text-response requests. These are
    computed from the tokenizer vocabulary and cover speech/audio codec tokens
    plus modal marker tokens such as `<speechgen_start>`, `<speechgen_end>`,
    and `<so_embedding>`.
  - `audex_mac.patches.vllm_metal_cfg._apply_disallowed_token_mask_mlx`
    enforces those guards on MLX logits before sampling. It deliberately avoids
    vLLM's built-in `bad_words` / `allowed_token_ids` paths because those are
    outside Audex-Mac's native MLX sampler fast path.
  - `_sample_native_mlx_if_supported` now handles simple unpaired no-top-p
    batches, not only CFG TTS pairs. This keeps greedy ASR on the MLX path
    while deliberately leaving text top-p sampling on vLLM Metal's fallback
    sampler until a correct optimized MLX top-p implementation is available.
  - TTS CFG requests keep their existing positive codec window and do not use
    the text-mode disallow guard.
- Tests:
  - `tests/test_vllm_sts_requests.py` asserts ASR and text requests carry the
    modality guard while TTS requests do not.
  - `tests/test_vllm_cfg.py` covers guard range merge/clamp behavior, MLX logit
    masking, non-CFG native sampling, explicit top-p fallback, and the existing
    TTS codec window.
- Reapply notes:
  - If upstream vLLM Metal gains first-class Audex modality support, keep this
    guard until a real 2B/30B ASR/text probe proves `<speechcodec_*>` and
    `<audiocodec_*>` tokens cannot leak in text modes.

### 2026-07-08: vLLM Metal Audex raw-audio projection and Nemotron-H hybrid cache

- Symptom:
  - The vLLM Metal ASR path accepted Audex audio but generated coherent first
    tokens followed by pathological repetition such as `!».\n\n`.
  - Activity Monitor still showed high `VLLM::EngineCore` CPU pressure because
    the earlier raw-audio path projected audio in the API/front process, bridged
    the projected `(750, 2688)` MLX embeddings to a CPU torch tensor for vLLM
    IPC, then converted them back to MLX inside EngineCore.
- Root cause evidence:
  - Strict MLX ASR over the same WAV produced the correct transcript.
  - vLLM Metal debug showed the prompt placeholder and projected audio shape
    were correct, and an offline comparison proved
    `merge_multimodal_embeddings(...)` was byte-equivalent to the strict MLX
    splice (`merge_max_abs_diff=0.0`).
  - Strict MLX `generate_step` emitted the expected continuation:
    `Language: English. The spoken content of the audio is 'My name is Matt...`.
  - vLLM Metal's paged path created only `OffsetCache` shims. Those are
    sufficient for attention layers whose KV state lives in vLLM Metal's paged
    cache, but Audex/Nemotron-H has Mamba blocks that require MLX-LM
    `ArraysCache` conv/SSM state. Without that state, decode after the
    multimodal prefill collapsed even though the first sampled token was sane.
- Audex-Mac patch:
  - `AudexProjectedAudioProcessor` now forwards raw PCM/sample-rate through
    vLLM multimodal kwargs and computes only the required Audex placeholder
    length in the front process. The audio encoder/projector now run inside
    `VLLM::EngineCore` via `AudexMultimodalAdapter.encode_multimodal`.
  - `AudexMultimodalAdapter` receives the full checkpoint path from the
    `ModelLifecycle._install_generation_model` patch so EngineCore can load
    `audio_encoder.*`, `audio_projector.*`, and `audio_preprocessor`.
  - The adapter now marks `requires_explicit_positions=True`, routing Audex
    text-only batches through the adapter as well.
  - `AudexMultimodalAdapter.call_lm` builds a hybrid cache: real MLX-LM
    `ArraysCache` entries for Nemotron-H Mamba layers, and vLLM Metal
    `OffsetCache` entries for attention layers. This preserves Mamba conv/SSM
    continuation state while leaving attention KV state in vLLM Metal's paged
    cache.
- Tests:
  - `tests/test_audex_patches.py` asserts raw tuple payloads are forwarded as
    raw audio with 750 placeholders for the minimum Audex clip, projected
    embeddings remain accepted, and lifecycle attachment still works.
  - `tests/test_vllm_runtime.py`, `tests/test_vllm_sts_cli.py`, and
    `tests/test_vllm_sts_requests.py` cover raw ASR request construction and
    CLI use of `transcribe_audio(..., sample_rate=16000)`.
  - `tests/test_vllm_diagnostics.py` and `tests/test_start_sh.py` cover the
    updated processor probe/startup behavior.
- Live verification:
  - Raw vLLM Metal ASR probe over `.audex/runs/startsh-e2e/input.wav` now
    transcribes coherently:
    `The spoken content of the audio is 'my name is matt please answer in one
    short sentence what is a python list'.`
  - `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke
    --diagnose-vllm-sts-audio-fixture .audex/runs/startsh-e2e/input.wav
    --diagnose-vllm-sts-speech-max-tokens 128` reached STS speech streaming
    and logged a clean transcript prefix, but still failed the realtime
    threshold because TTS/logits evaluation remains slow.
- Reapply notes:
  - If upstream vLLM Metal adds first-class hybrid-model cache support, replace
    this adapter-owned cache graft with upstream's request-scoped Mamba cache.
  - Do not regress to front-process projected-audio IPC; raw audio should cross
    the vLLM boundary and projection should happen in EngineCore.
  - The current hybrid cache is intentionally conservative for the single-user
    CLI SLC. Before relying on multi-request continuous batching, add
    request-scoped `ArraysCache` ownership instead of one adapter-level cache.

### vLLM Metal TTS Intelligibility Validation

- Purpose: keep the default CLI speech output on a path that is proven
  intelligible on Mac while preserving NVIDIA-style CFG machinery as an
  explicit diagnostic/research target.
- Finding:
  - The vLLM Metal CFG pair path generated valid-looking `<speechcodec_N>`
    token IDs but produced WAVs that local speech recognition transcribed as
    empty/unintelligible. This held for both the Audex-Mac MLX decoder and
    NVIDIA's bundled PyTorch decoder over the same generated codec frames,
    ruling out playback and MLX decoder corruption for that sample.
  - A single non-CFG vLLM Metal TTS request using the same Audex TTS prompt and
    NVIDIA sampler values (`temperature=0.8`, `top_p=1.0`, `top_k=0`) emitted
    `<speechgen_end>` and produced intelligible speech. The local MLX-Audio oracle
    evaluation transcribed the generated WAV as: `A Python list is a mutable
    ordered collection of items separated by commas.`
- Audex-Mac patch:
  - `AudexAsyncVllmRuntime.stream_tts_codec_frames(...)` streams speech-codec
    frames from one normal TTS request. `AudexVllmRuntime.generate_tts(...)`
    and `AudexAsyncVllmRuntime.generate_tts(...)` provide matching final-output
    helpers.
  - `VllmSpeechToSpeechSession.generate_speech_output(...)` now uses the
    non-CFG TTS path by default for both sync and async vLLM runtimes and logs
    `tts_cfg_enabled=false` in speech-output JSON artifacts.
  - CFG builders and streams remain in the codebase for diagnostics and future
    repair, but they are not the default audible CLI path.
  - `scripts/evaluate_tts_wav.py` wraps the in-repository MLX-Audio transcription
    stack and returns a pass/fail JSON verdict against expected text.
  - `scripts/probe_vllm_tts_decode.py` can generate one vLLM Metal TTS sample,
    store exact generated token IDs/codec frames, optionally force the sampler
    codec window, and optionally disable CFG for A/B testing.
- Current no-CFG performance evidence:
  - The vLLM Metal sampler patch now lets no-CFG TTS requests carry
    `extra_args["audex_tts_skip_paged_logits_eval"] = true`, allowing the
    paged path to avoid an extra full-vocab logits materialization before the
    native MLX sampler consumes the logits.
  - Instrumented 30B no-CFG probe, full-vocab sampler, 96 token cap:
    `generation_seconds=14.743`, `skipped_logits_eval=49` by sampler count 50,
    and the token stream matches the known-good full-vocab distribution prefix.
  - Instrumented 30B no-CFG probe with the compact speech-codec window was
    rejected: `generation_seconds=25.464` for the same 96-token cap and a
    different sampled token stream. Do not use codec-window restriction as a
    production no-CFG optimization.
  - Four parallel no-CFG diagnostic requests improved aggregate throughput to
    `22.345` generated tokens/sec, but a real four-sentence `./start.sh`
    attempt produced unintelligible audio under the MLX-Audio oracle after batching
    independent TTS segments. Do not route production TTS through batched
    no-CFG segments without a passing speech-transcription check.
  - Restored production single-stream no-CFG run:
    `.audex/runs/speech-output-vllm-20260708-200312.wav` passed
    `scripts/evaluate_tts_wav.py` with `ratio=1.0`, but the corresponding
    speech-output log measured `stream_finished_seconds=40.172`,
    `generated_token_count=212`, and only `5.277` generated tokens/sec on the
    30B model.
  - Conclusion: the current passing no-CFG path is intelligible but not
    near-real-time on 30B. The next viable performance direction is not
    codec-window sampling or segmented no-CFG; it is either correct paired CFG
    continuous batching, a model-side speculative/multi-token path, or a much
    smaller model for the no-CFG baseline.
- Final-output async TTS update:
  - Apple-Silicon-CPU-Optimization-Guide.pdf points at the right class of
    bottleneck for the remaining 30B issue: synchronization, shared-queue
    contention, and per-token CPU scheduling should be measured with
    Instruments Time Profiler/CPU Counters rather than guessed from Activity
    Monitor alone.
  - The old async CLI path paid per-token stream queue overhead even though it
    decoded only after token generation completed. `AudexAsyncVllmRuntime`
    now has a true final-output path that consumes the async vLLM generator
    internally and returns only the final result. The default CLI TTS path uses
    that final-output no-CFG generation and logs
    `streaming=false`, `vllm_token_streaming=false`,
    `decoder_after_token_stream=true`.
  - Diagnostic comparison on 30B for `A Python list is a mutable ordered
    collection of items separated by commas.`:
    - async token stream no-CFG: 223 tokens, `generation_seconds=64.539`,
      `aggregate_tokens_per_second=3.455`.
    - sync final no-CFG: same 223 token IDs, `generation_seconds=53.593`,
      `aggregate_tokens_per_second=4.161`.
    - sync-final WAV `.audex/runs/tts-probe-vllm-20260708-202405.wav`
      passed `scripts/evaluate_tts_wav.py` with `ratio=1.0`.
  - Product smoke after switching the async CLI to final-output no-CFG:
    `./start.sh --model audex-30b-a3b --new-conversation --input-wav
    .audex/fixtures/codex-python-list-smoke-16k-mono.wav --no-play
    --response-max-tokens 64`.
    The run produced transcript `please explain python lists in one sentence`,
    response `A python list is a mutable ordered collection of items indexed
    from zero`, 204 generated speech tokens, 203 codec frames,
    `stream_finished_seconds=47.842`, and `first_audio_ready_seconds=48.128`.
    `scripts/evaluate_tts_wav.py` passed with `ratio=0.9796`, actual
    transcript `A Python list is a mutable ordered collection of items indexed
    from zero to`.
  - Conclusion: final-output TTS is a quality-safe cleanup and measurable
    speedup, but it does not solve near-real-time 30B TTS. The remaining
    performance problem is still inside vLLM-Metal's single-request per-token
    forward/scheduler/synchronization loop.
- Streaming no-CFG async TTS update:
  - After switching the default vLLM Metal path to non-paged attention, async
    no-CFG token streaming became viable again. A direct 30B probe for
    `A python list is a mutable ordered collection of items indexed from zero`
    produced the same 206 generated token IDs as sync-final generation.
  - Comparison on that prompt:
    - async no-CFG stream:
      `.audex/runs/tts-probe-vllm-20260708-205756.json`, first token at
      `0.210s`, 75 codec frames at `3.012s`, `generation_seconds=6.992`,
      and `aggregate_tokens_per_second=29.462`.
    - sync-final no-CFG:
      `.audex/runs/tts-probe-vllm-20260708-205837.json`,
      `generation_seconds=5.214`, `aggregate_tokens_per_second=39.509`, but
      no audio can be decoded until generation completes.
  - The async CLI now streams codec-frame deltas from
    `AudexAsyncVllmRuntime.stream_tts_codec_frames(...)`, feeds the MLX causal
    speech decoder during generation, and logs
    `streaming=true`, `vllm_token_streaming=true`,
    `decoder_streaming=true`, and `decoder_after_token_stream=false`.
  - `AudexAsyncVllmRuntime.stream_tts_codec_frames(...)` now uses a
    single-request stream loop directly over `engine.generate(...)` instead of
    routing through `stream_many(...)`; this avoids an extra asyncio queue hop
    and avoids rebuilding every generic `VllmStreamDelta` field for TTS-only
    codec-frame deltas. A direct 30B probe after this change,
    `.audex/runs/tts-probe-vllm-20260708-212246.json`, produced the same 206
    generated token IDs with `generation_seconds=6.321` and
    `aggregate_tokens_per_second=32.590`.
  - The async vLLM no-CFG TTS output path now keeps decoded audio as packed
    PCM16 chunks after each MLX decoder emission instead of accumulating every
    float sample in Python and repacking the full output at the end. The same
    packed bytes are queued directly to `_ContinuousPcmPlayer.enqueue_pcm(...)`
    when playback is enabled, so each chunk is packed once. Speech-output logs
    now include `pcm_pack_seconds`, `player_enqueue_seconds`, and
    `wav_write_seconds` to make host-side audio plumbing visible in live runs.
  - No-CFG runtime TTS now passes Audex's compact speech-token sampler window
    (`audex_tts_codec_min_id`, `audex_tts_codec_max_id`, and
    `audex_tts_speechgen_end_id`) by default, alongside the existing
    `audex_tts_skip_paged_logits_eval` hint. The same 30B probe above recorded
    `codec_window=false`, so it forced sampling across the full Audex
    vocabulary and should not be treated as final evidence for the optimized
    no-CFG runtime path. `scripts/probe_vllm_tts_decode.py` now defaults
    `--codec-window` to enabled and exposes `--no-codec-window` only for A/B
    diagnostics.
  - `AudexAsyncVllmRuntime.stream_many(...)` now preserves its cumulative
    `token_ids` contract when an individual request uses vLLM
    `RequestOutputKind.DELTA`; it accumulates the incoming deltas internally
    and exposes the real per-event token delta as `new_token_ids`. This lets
    no-CFG probes and TTS batch diagnostics avoid cumulative vLLM output
    overhead without changing callers that expect cumulative request state.
    The built-in no-CFG TTS batch diagnostic now reports
    `codec_window=true` and `output_kind=DELTA` in its JSON.
  - `stream_many(...)` now detects DELTA output by either the local string or a
    vLLM enum-like object whose `.name == "DELTA"`, matching the normalization
    path used when constructing `SamplingParams`. Keep this guard when vLLM
    changes `RequestOutputKind`; a false negative here makes diagnostics
    accumulate already-cumulative outputs and invalidates timing evidence.
  - Native sampler detail timing (`build_sample_logits`, `sample_eval`,
    `tolist`, and optional `materialize_decode_logits`) is now collected only
    when `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1`. The previous code paid
    per-token `perf_counter()` and Python dict-update overhead even in normal
    speech runs where the detail summary was never printed. Keep aggregate
    behavior and token sampling unchanged; enable the debug env for benchmark
    runs that need `native_detail_ms`.
  - `audex_mac.vllm_diagnostics` no longer forces
    `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1` for STS smoke or TTS batch probes.
    Use `./start.sh --diagnose-vllm-metal --diagnose-vllm-sts-smoke
    --diagnose-vllm-native-sampling-debug` only when the benchmark needs
    EngineCore stderr timing; leave it off for normal fast-path smoke evidence.
  - The same debug mode now instruments vLLM-Metal's non-paged decode loop,
    emitting `Audex vLLM Metal: nonpaged decode timing ...` checkpoints.
    This matters because Audex-Mac's default fast path runs with
    `VLLM_METAL_USE_PAGED_ATTENTION=0`; paged-only timing left the current
    single-stream no-CFG bottleneck invisible even though native MLX sampling
    was active. The diagnostic parser records these checkpoints under
    `vllm_metal_timing.latest_non_paged_decode`.
  - The no-CFG TTS batch diagnostic now uses
    `AudexAsyncVllmRuntime.stream_tts_codec_frames(...)`, matching the
    interactive CLI's specialized no-CFG streaming path. CFG diagnostics still
    use paired `stream_many(...)` requests. This avoids benchmarking generic
    cumulative-token extraction when the human-mouth-and-ears path uses
    codec-frame deltas.
  - Apple-Silicon-CPU-Optimization-Guide.pdf was added locally as performance
    evidence. Diagnostics now record Apple-recommended `sysctl` topology data
    (`hw.nperflevels`, per-perflevel core/cache sizes, page size, cache-line
    size, Rosetta status) under
    `parent_process.apple_silicon_topology`. This is diagnostic evidence only:
    it should help interpret host-side scheduler/queue/synchronization pressure
    and should not be used to excuse MLX CPU fallback.
  - Rejected sampler micro-optimization: replacing the native sampler's
    `mx.argpartition` top-k path with an `mx.topk` threshold mask produced the
    same token sequence but regressed the same prompt to
    `generation_seconds=9.033` and `aggregate_tokens_per_second=22.805`.
    Keep the argpartition path until a faster top-k implementation is proven
    inside the full vLLM-Metal generation loop, not only a standalone
    microbenchmark.
  - Decoder chunk tuning:
    - 75-frame chunks were safe but first decoded audio remained late
      (`first_audio_ready_seconds=6.956`) because the causal decoder did not
      emit on the first chunk.
    - 40-frame chunks improved first decoded audio to `5.044s` and passed
      `scripts/evaluate_tts_wav.py` with `ratio=0.9931`.
    - 24-frame chunks improved first decoded audio to `1.852s` on a no-play
      fixture run and passed `scripts/evaluate_tts_wav.py` with `ratio=0.9931`.
    - 16-frame chunks improved decoded-audio timing in isolation but actual
      playback smoke `.audex/runs/speech-output-vllm-20260708-211902.json`
      reported `device_underflow_count=2` and `queue_underrun_count=2`; do
      not use 16-frame chunks as the default.
  - Playback tuning:
    - 24-frame chunks with the original `0.8s` prebuffer generated a valid WAV
      but actual speaker playback reported `device_underflow_count=5` and
      `queue_underrun_count=4`; this is the choppy-audio failure mode.
    - 24-frame chunks with `1.5s` prebuffer still reported
      `device_underflow_count=3` and `queue_underrun_count=1`.
    - 24-frame chunks with `1.75s` prebuffer still reported
      `device_underflow_count=2` and `queue_underrun_count=1`.
    - 24-frame chunks with `2.0s` prebuffer became the first stable default.
      Playback
      smoke `.audex/runs/speech-output-vllm-20260708-210906.json` reported
      `first_audio_ready_seconds=1.899`, `first_playback_started_seconds=5.996`,
      `device_underflow_count=0`, `queue_underrun_count=0`,
      `queue_overrun_count=1`, and `queue_high_water_seconds=2.4`.
      The corresponding WAV passed `scripts/evaluate_tts_wav.py` with
      `ratio=0.9931`.
      A later smoke using the specialized TTS stream loop,
      `.audex/runs/speech-output-vllm-20260708-212329.json`, also reported
      `device_underflow_count=0`, `queue_underrun_count=0`, and
      `queue_overrun_count=1`, and passed `scripts/evaluate_tts_wav.py` with
      `ratio=0.9931`.
    - After decoder priming and streaming-WAV writes, 24-frame chunks with
      `1.0s` prebuffer became the safe default. Playback smoke
      `.audex/runs/speech-output-vllm-20260709-021823.json` reported
      `first_audio_ready_seconds=3.263`,
      `first_playback_started_seconds=5.063`, `device_underflow_count=0`,
      `queue_underrun_count=0`, `queue_overrun_count=0`,
      `queue_high_water_seconds=1.98`, and full `28.72s` playback written.
    - The decoder chunk default was reduced to 18 frames with the same 1.0s
      prebuffer. Playback smoke
      `.audex/runs/speech-output-vllm-20260709-024203.json` reported
      `first_audio_ready_seconds=3.006`,
      `first_playback_started_seconds=4.628`, `device_underflow_count=0`,
      `queue_underrun_count=0`, `queue_overrun_count=0`,
      `queue_high_water_seconds=1.42`, and full `28.72s` playback written.
      Neighbor probes were worse: 20 frames was stable but slower
      (`4.869s` first playback), 17 frames was stable but slower (`4.867s`),
      and 16 frames reintroduced `device_underflow_count=1`.
    - The continuous PCM player now requests `sounddevice` latency `"low"`
      instead of `"high"`. Playback smoke
      `.audex/runs/speech-output-vllm-20260709-024718.json` kept
      `device_underflow_count=0`, `queue_underrun_count=0`, full `28.72s`
      playback written, and recorded `first_playback_started_seconds=4.577`.
      The logged write-start metric only moves slightly, but lower device
      latency should reduce actual speaker-side buffering.
    - The current vLLM playback prebuffer is 0.8s, matching the shared PCM
      player default. Playback smoke
      `.audex/runs/speech-output-vllm-20260709-025121.json` kept
      `device_underflow_count=0`, `queue_underrun_count=0`, full `28.72s`
      playback written, and recorded `first_playback_started_seconds=4.571`.
      Since the logged write-start barely changed from 1.0s, the next latency
      bottleneck is producer cadence rather than nominal prebuffer length.
    - vLLM TTS run logs now include `codec_frames_per_second`,
      `audio_realtime_ratio`, `first_playback_after_audio_seconds`, and
      `first_decoder_wait_after_codec_seconds` so future changes can tell
      whether they improved generation throughput, decoder gating, or playback
      buffering without re-parsing raw token/frame logs.
    - Rejected first-substantial-sentence release on 2026-07-09. It improved
      no-play first decoded audio (`2.722s` in
      `.audex/runs/speech-output-vllm-20260709-022234.json`) but real speaker
      playback introduced device underflows with 1.0s, 1.25s, and 1.35s
      prebuffers. Raising the prebuffer to 1.5s cleared underflows in
      `.audex/runs/speech-output-vllm-20260709-022748.json` but delayed first
      playback to `5.359s`, slower than the current safe 2-sentence default.
  - Interpretation: the 30B no-CFG path is still slower than real-time audio
    generation, so audible playback must buffer enough PCM to avoid underruns.
    The CLI now exposes fast decoded audio for logs/files while using a larger
    vLLM-specific playback prebuffer to avoid speaker dropouts.
- Tests:
  - `tests/test_vllm_runtime.py` covers single-request no-CFG TTS generation
    and streaming, while retaining CFG request-shape coverage.
  - `tests/test_vllm_sts_cli.py` asserts the default STS speech path uses the
    no-CFG stream and records `tts_cfg_enabled=false`.
- Live verification:
  - Synthetic input was generated with macOS `say`, converted to 16 kHz WAV,
    and passed through:
    `./start.sh --model audex-30b-a3b --new-conversation --input-wav
    .audex/runs/startsh-e2e-nocfg/input.wav --no-play --response-max-tokens 64
    --speech-max-tokens 512`
  - The run produced transcript `my name is matt please answer in one short
    sentence what is a python list`, response `A python list is a mutable
    ordered collection of items separated by commas.`, and output WAV
    `.audex/runs/speech-output-vllm-20260708-185811.wav`.
  - `scripts/evaluate_tts_wav.py` using the in-repository MLX-Audio oracle
    passed with `ratio=1.0`, actual transcript `A Python list is a mutable
    ordered collection of items separated by commas.`
- Reapply notes:
  - Do not restore vLLM Metal CFG as the default audible path until a local
    TTS transcript evaluator passes on a generated WAV.
  - When repairing CFG, use the existing no-CFG probe as the green control and
    compare against both Audex-Mac MLX decoding and NVIDIA's bundled PyTorch
    decoder before blaming playback.

### vLLM Metal Non-Paged Audex Multimodal Prefill

- Purpose: make the fast vLLM Metal non-paged path usable for the default
  speech-to-speech CLI, including raw Audex audio input, instead of limiting
  non-paged execution to text/TTS-only probes.
- Upstream target:
  - Repository: `https://github.com/vllm-project/vllm-metal`
  - Commit: `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`
  - File/symbols:
    `vllm_metal.v1.model_runner.MetalModelRunner._reject_scheduled_encoder_inputs`,
    `MetalModelRunner._handle_new_requests`, and the non-paged prefill branch
    around `_prefill_single`.
- Finding:
  - `VLLM_METAL_USE_PAGED_ATTENTION=0` is dramatically faster for no-CFG
    Audex speech generation than the paged path, but the pinned runner rejected
    all scheduled multimodal encoder inputs in non-paged mode with
    `Multimodal requests require the paged attention backend`.
  - A first non-paged splice patch removed that exception but returned the
    wrong KV cache after Audex's adapter replaced the prompt cache with its
    hybrid cache. That produced corrupted ASR text on the Python-list fixture.
  - Returning the same cache object actually used by the Audex text model fixed
    the ASR corruption.
- Audex-Mac patch:
  - File: `audex_mac/patches/vllm_metal_audex_adapter.py`
  - Startup hook: `audex_mac.patches.runtime.apply_audex_runtime_patches`
  - Patch function:
    `_patch_vllm_metal_non_paged_multimodal_prefill`
  - The patch is guarded by
    `_audex_mac_non_paged_mm_prefill_patch`.
  - In non-paged mode, scheduled encoder inputs are allowed only when the
    active adapter is `AudexMultimodalAdapter` and `forward_ready=True`.
  - For Audex multimodal new requests, the patch embeds text tokens, splices
    projected audio embeddings with vLLM Metal's existing
    `merge_multimodal_embeddings`, calls the Audex text model with
    `input_embeddings`/`inputs_embeds`, and stores the exact hybrid cache used
    by that forward pass for subsequent decode steps.
- Default policy update:
  - `start.sh` and `audex_mac.metal_policy.REQUIRED_METAL_ENV` now require
    `VLLM_METAL_USE_PAGED_ATTENTION=0` for the default fast STS path.
  - Diagnostics now treat `use_paged_attention=True` as a default-path
    performance regression rather than the required state.
- Live verification:
  - Failed control before cache fix:
    `VLLM_METAL_USE_PAGED_ATTENTION=0 ... --input-wav
    .audex/fixtures/codex-python-list-smoke-16k-mono.wav --no-play` no longer
    raised the old paged-only exception, but transcribed the fixture as
    `{t}' is not a valid integer and will be ignored.", end='`.
  - Paged baseline on the same fixture transcribed correctly:
    `please explain python lists in one sentence`, but the run took
    `elapsed_seconds=70.755`, with `asr_elapsed_seconds=8.804` and
    `tts_first_audio_ready_seconds=56.905`.
  - Corrected non-paged direct CLI run:
    `.audex/runs/sts-turn-vllm-20260708-205135.json` transcribed
    `please explain python lists in one sentence`, responded
    `A python list is a mutable ordered collection of items indexed from zero`,
    and measured `elapsed_seconds=12.533`, `asr_elapsed_seconds=1.735`,
    `text_elapsed_seconds=1.139`, and `tts_first_audio_ready_seconds=9.461`.
  - User-facing `./start.sh` run:
    `.audex/runs/sts-turn-vllm-20260708-205356.json` transcribed the same
    fixture correctly, responded coherently, and measured
    `elapsed_seconds=13.434`, `asr_elapsed_seconds=2.678`,
    `text_elapsed_seconds=1.832`, and `tts_first_audio_ready_seconds=8.711`.
  - `scripts/evaluate_tts_wav.py` over
    `.audex/runs/speech-output-vllm-20260708-205347.wav` passed with
    `ratio=0.9931`, actual transcript:
    `A Python list is a mutable ordered collection of items indexed from zero.`
- Tests:
  - `tests/test_audex_patches.py` covers the non-paged runner hook allowing
    encoder dispatch only through the Audex adapter path.
  - `tests/test_metal_policy.py`, `tests/test_vllm_diagnostics.py`, and
    `tests/test_start_sh.py` cover the non-paged default policy.
- Reapply notes:
  - Keep this patch until upstream vLLM Metal supports non-paged multimodal
    prefill with adapter-provided embeddings and request-correct cache
    ownership.
  - If upstream changes `MetalModelRunner._handle_new_requests`, re-check this
    patch against both a paged baseline and a non-paged ASR fixture before
    trusting a green unit test.
  - Do not set the default back to paged attention merely to make multimodal
    input work; the measured near-real-time path depends on non-paged prefill
    plus correct Audex cache handoff.

### vLLM Metal No-CFG Spoken TTS Stabilization

- Purpose: keep the fast no-CFG vLLM Metal TTS path intelligible on longer
  spoken answers without re-enabling the currently broken CFG audio path.
- Finding:
  - Human smoke testing showed intelligible short greeting/turn audio, but
    longer responses could become choppy, repetitive, or cut off before the
    printed transcript finished.
  - The relevant run log showed `tts_hit_max_tokens=false` and
    `tts_reached_end_token=true`, so the cut-off was not caused by a speech
    token cap.
  - The old default text prompt included an explicit Audex/NVIDIA/Nemotron
    biography and a `[CRITICAL]` formatting line; those strings leaked into
    spoken responses.
- Audex-Mac patch:
  - File: `audex_mac/vllm_sts_requests.py`
  - `DEFAULT_VLLM_TEXT_PROMPT` now asks for concise spoken prose without naming
    the model/vendor/architecture in the prompt text itself.
  - The local response policy is appended to the system message instead of
    being prepended to the latest user message. This keeps prompt-control text
    out of user-visible content and reduces the chance that Audex recites the
    policy or treats it as the topic.
  - File: `audex_mac/vllm_sts_cli.py`
  - Async no-CFG TTS now splits response text by non-empty lines and then by up
    to three sentences per chunk, preserving spoken order.
  - Three-sentence chunks are additionally capped at 360 characters so a long
    paragraph cannot become one giant no-CFG TTS request. The cap is logged as
    `tts_max_chars_per_chunk`.
  - Each chunk is sent through the existing vLLM Metal no-CFG
    `stream_tts_codec_frames(...)` path, then the MLX causal speech decoder is
    flushed and reset before the next chunk.
  - A short silence is inserted between chunks, multi-chunk playback uses a
    slightly larger prebuffer, and `mx.clear_cache()` is called once during
    speech-output teardown when available. Do not clear the global MLX cache
    between spoken chunks; that can create playback gaps without improving
    token throughput.
  - Run logs now include `tts_sentence_chunk_size`, `tts_segment_texts`,
    `tts_segment_codec_frame_counts`, `tts_segment_token_counts`,
    `tts_segment_wall_seconds`, `tts_max_chars_per_chunk`, decoder
    push/flush/reset seconds, and MLX cache-clear timing so long-answer
    failures are segment-local instead of opaque.
- Tests:
  - `tests/test_vllm_sts_requests.py` asserts the default text prompt no
    longer contains the leaked NVIDIA/Nemotron/`[CRITICAL]` strings and that
    the spoken-response policy is carried by the system role, not the user
    role.
  - `tests/test_vllm_sts_cli.py` covers newline/three-sentence chunking and
    verifies long no-CFG TTS calls the vLLM Metal stream once per spoken chunk.
    It also covers the character ceiling for long three-sentence chunks.
- Live verification:
  - Human checkpoint from an attached investigation note:
    greeting was intelligible and the first real ASR -> response -> TTS turn
    was intelligible. Longer utterances could still become choppy, repetitive,
    and occasionally cut off before the printed transcript completed. Memory
    climbed rapidly over multiple turns. This is not a final latency/stability
    checkpoint.
  - A no-play 30B vLLM Metal fixture run on
    `.audex/runs/ptt-input-20260708-223521.wav` produced
    `.audex/runs/sts-turn-vllm-20260708-225147.json` and
    `.audex/runs/speech-output-vllm-20260708-225058.json`.
  - The response stayed on topic about Go-to-Rust migration and did not include
    the previous Audex/NVIDIA biography leakage.
  - The speech log recorded `tts_observed_segments=2`, segment end tokens true
    for both chunks, `tts_hit_max_tokens=false`, `first_audio_ready_seconds=1.235`,
    and 24 seconds of finite 16 kHz audio.
  - `.venv/bin/python scripts/evaluate_tts_wav.py` passed with
    `ratio=0.5675`; the oracle heard the complete response, with the main
    mismatch being `AWS Go SD` for `AWS Go SDK`.
  - The same run still measured slow no-play TTS throughput
    (`stream_finished_seconds=49.689`), so this patch is a stability and
    intelligibility checkpoint rather than the final latency fix.
  - Follow-up serial chunking run
    `.audex/runs/speech-output-vllm-20260708-225633.json` moved cache clearing
    to one end-of-turn call and recorded `mlx_clear_cache_seconds=0.020751`,
    `decoder_push_seconds=1.788457`, `decoder_flush_seconds=0.054413`, and
    `stream_finished_seconds=45.451`. This rules out cache clearing, WAV
    writing, PCM packing, and the MLX speech decoder as the main no-CFG TTS
    throughput bottleneck for that run.
  - Native-sampling debug run
    `.audex/runs/sts-turn-vllm-20260708-230030.json` confirmed the TTS path is
    using the native MLX sampler, but EngineCore stderr reported cumulative
    `native_sample_ms=36556.0` for `native_sampled_rows=899` at the 1000-step
    checkpoint. The remaining bottleneck is therefore still around per-step
    sampling/logit evaluation in vLLM Metal, not local audio plumbing.
  - An experiment that submitted the two no-CFG TTS chunks concurrently through
    vLLM continuous batching made this same fixture worse:
    `.audex/runs/speech-output-vllm-20260708-230427.json` recorded
    `first_audio_ready_seconds=5.753` and `stream_finished_seconds=52.183`.
    Do not switch the default no-CFG spoken chunks to independent concurrent
    TTS requests unless a later vLLM Metal scheduler change proves it faster
    on the same fixture.
  - Failed performance experiment: a temporary TTS-window lm-head/logit
    projection patch produced the same kind of output but regressed the
    fixture to `.audex/runs/speech-output-vllm-20260708-231434.json` with
    `stream_finished_seconds=79.089`. Do not reintroduce that patch without a
    new hypothesis and same-fixture proof.
  - Follow-up cached TTS-window projection A/B also lost on the full CLI
    fixture and was backed out:
    - Batch probe with the cached projection looked superficially better on the
      actual app path,
      `.audex/runs/vllm-metal-diagnostic-20260708-233455.json`:
      `elapsed_seconds=13.371`, `codec_frames_per_second=30.14`.
    - The full no-play fixture contradicted that narrow signal. With cached
      projection enabled,
      `.audex/runs/speech-output-vllm-20260708-233632.json` took
      `stream_finished_seconds=64.876` for 1,317 codec frames across three
      TTS segments. With the projection disabled on the same fixture,
      `.audex/runs/speech-output-vllm-20260708-233913.json` took
      `stream_finished_seconds=51.401` for the same 1,317 codec frames and
      three segments. EngineCore cumulative native sample time at the
      1000-step checkpoint was also worse with the projection
      (`40799.9ms/884 rows`) than without it (`32607.6ms/884 rows`).
    - Conclusion: cached partial-head projection is still not the right
      optimization for the current vLLM Metal / MLX path. Keep the full-head
      native-sampling path as the green no-CFG control until a stronger
      same-fixture benchmark proves otherwise.
  - Materialized-logits diagnostic
    `.audex/runs/vllm-metal-diagnostic-20260708-231929.json` recorded
    `codec_frames_per_second=22.544`. At the 200-token checkpoint,
    `native_sample_ms=8251.4`, with `materialize_decode_logits=6821.6ms`,
    `sample_eval=1415.4ms`, `build_sample_logits=4.4ms`, and `tolist=0.3ms`.
    The current 30B no-CFG bottleneck is mostly decode-logit materialization,
    with categorical sampling second.
  - Follow-up speculative-decode inspection found that vLLM Metal has
    `draft_model`, `ngram`, and Gemma4 MTP proposer support, but the verifier
    is explicitly greedy-only: `SpeculativeDecodeController._validate_greedy_sampling`
    rejects positive temperature, positive top-k, top-p values other than 1.0,
    penalties, logprobs, allowed token ids, bad-word ids, and non-argmax
    logits processors. Audex no-CFG TTS deliberately uses NVIDIA's sampled
    audio-token defaults (`temperature=0.8`, `top_p=1.0`, `top_k=0`) plus the
    Audex codec-token guard in `SamplingParams.extra_args`, so the current
    upstream speculative path cannot speed this green TTS path without a
    sampled-speculative verifier.
  - The no-CFG chunked TTS path now budgets speech tokens per spoken chunk
    instead of passing the whole utterance budget to every chunk. Single-chunk
    utterances keep the original `_speech_max_tokens_for_text(...)` budget so
    long startup/user-visible utterances can still scale with text length, but
    multi-chunk responses use `max(512, chunk_text_tokens * 64)` per chunk.
    This prevents a short sentence chunk from receiving the old 2400-token
    minimum and then drifting into long sampled repetition tails. Run logs
    include `tts_segment_max_tokens` so cutoffs/repetition can be tied to the
    exact segment budget.
  - The native sampler now slices Audex TTS rows to
    `[speechgen_end, speechcodec_min..speechcodec_max]` before the sampler's
    float32 row conversion. This keeps the patch from carrying/casting unused
    text-token logits through `_build_native_sample_logits(...)`. The real
    model still computes the full vLLM logits row before this point, so this is
    not expected to solve the main throughput problem by itself.
    `tests/test_vllm_cfg.py::test_vllm_metal_cfg_builds_tts_window_rows_before_float32_cast`
    pins the order so future edits do not accidentally restore full-row
    sampler casts.
  - Verification after the sampler-window patch on the same explicit
    1024-token no-play fixture:
    `.audex/runs/speech-output-vllm-20260708-235417.json` recorded
    `stream_finished_seconds=52.063`, `first_audio_ready_seconds=1.222`, and
    1,317 codec frames across three TTS segments. A same-session pre-patch run
    `.audex/runs/speech-output-vllm-20260708-235010.json` recorded
    `stream_finished_seconds=54.184`, `first_audio_ready_seconds=1.888`, and
    the same 1,317 codec frames. Earlier baseline
    `.audex/runs/speech-output-vllm-20260708-233913.json` was still
    `stream_finished_seconds=51.401`, so treat this as a small sampler hygiene
    patch, not a proven throughput breakthrough.
  - Follow-up continuous-batching experiment: `AUDEX_VLLM_CONCURRENT_TTS_CHUNKS=1`
    routes the existing no-CFG spoken chunks through an ordered multi-request
    stream so vLLM Metal can schedule independent TTS chunks together. The
    default remains serial because the fixture lost despite true batching:
    `.audex/runs/speech-output-vllm-20260709-000319.json` recorded
    `tts_concurrent_segments=true`, `stream_finished_seconds=59.774`, and
    `first_audio_ready_seconds=7.693`. EngineCore showed the intended
    continuous-batching shape during TTS (`decode_reqs=3`, `cached_reqs=3`,
    `batched=1`), but wall time and first audio were worse than serial.
    The same code with the flag off produced
    `.audex/runs/speech-output-vllm-20260709-000540.json`:
    `tts_concurrent_segments=false`, `stream_finished_seconds=49.488`,
    `first_audio_ready_seconds=1.226`, and the expected single-request decode
    shape (`decode_reqs=1`, `batched=0`). Keep chunk concurrency opt-in for
    diagnostics only; do not make it the default unless a future scheduler or
    request-shaping change beats the same fixture.
  - Text-stream latency probe: a temporary vLLM Metal probe on the same
    30B Go-to-Rust prompt measured `first_event_seconds=0.649`,
    `first_sentence_seconds=2.001`, and `final_seconds=5.327` for the text
    response. This means an async text-to-TTS interleaver has a real
    time-to-first-audio opportunity, unlike independent TTS chunk batching.
    `AudexAsyncVllmRuntime.stream_text_response_from_messages(...)` now exposes
    a cleaned cumulative text stream for that follow-up without changing the
    default CLI behavior yet.
- Reapply notes:
  - Preserve the no-CFG path as the green control until CFG audio passes a
    transcript-based TTS evaluator.
  - Do not restore vendor/model biography text to the default spoken prompt;
    negative instructions containing those exact words are still prompt tokens
    the model can echo.
  - If replacing the chunking heuristic, keep sentence/newline boundaries in
    the run log so choppiness and cut-off reports can be tied to the failing
    segment.

### Audex Causal Speech Decoder

- Purpose: port or wrap NVIDIA's `audex_causal_speech_decoder` so it runs on
  Mac without CUDA calls.
- NVIDIA decoder source evidence:
  - `audex_causal_speech_decoder/streaming_utils.py` loads the decoder through
    `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` and
    accepts a `device` parameter, but NVIDIA's unified server defaults it to
    CUDA.
  - `modeling_audex_causal_speech_decoder.py` implements
    `AudexSpeechTokenEmbedder`, a lookahead causal Vocos-style transformer, and
    a 320-sample `PatchHead`.
- Current Audex-Mac code:
  - `audex_mac.audio_contract.preflight_decoder` verifies the decoder directory,
    config, `model.safetensors`, remote-code files, 16 kHz sample rate,
    lookahead steps, and codebook size.
  - `audex_mac.speech_decoder` ports the decoder execution path to MLX:
    speech-codec frame embedding with NVIDIA's codebook decomposition, the
    project-out bias, lookahead depthwise convolution, causal self-attention
    with RoPE, RMSNorm, SiLU MLPs, and the tanh patch head.
  - `./start.sh --preflight-speech-decoder --model audex-2b` loads the real
    Audex-2B decoder float32 safetensors and verifies 8 codec frames decode on
    `Device(gpu, 0)` to 2560 finite float32 waveform samples at 16 kHz.
  - `audex_mac.speech_output` combines the MLX full-vocab speech-token
    generator with the MLX decoder and writes a local PCM16 WAV plus JSON run
    log under `.audex/runs/`.
- Expected guard: `.cuda()`/CUDA-only paths are not reachable in the Mac demo;
  decoder execution evidence must name `backend=mlx` and `Device(gpu, 0)`.
- Tests: `tests/test_audio_contract.py` covers decoder artifact/config
  preflight. `tests/test_speech_decoder.py` covers decoder config parsing,
  required tensor keys, project-out bias, and smoke-result readiness. The fast
  Gherkin suite covers finite 16 kHz waveform output with 320 samples per codec
  frame and local speech-output artifacts.

## 2026-07-09: vLLM Metal TTS Stability Checkpoint

- Purpose: address the first human-mouth/ear checkpoint after no-CFG vLLM
  Metal TTS became intelligible but still showed prompt leakage, long-utterance
  choppiness, occasional repetition/cutoff, and fast memory growth across
  turns.
- Code changes:
  - `audex_mac.vllm_runtime.extract_spoken_answer` now scrubs the observed
    generated prompt/template leakage before text is persisted or sent to TTS:
    NVIDIA/Audex self-biography lines and the leaked `[CRITICAL] ...` policy
    line from the 2026-07-08 transcript.
  - `audex_mac.vllm_sts_cli.generate_speech_output_streaming_from_async_runtime`
    now clears MLX cache after each completed TTS segment and again at utterance
    shutdown. This does not unload the vLLM engine; it releases transient MLX
    decoder/projector arrays at segment boundaries.
  - TTS logs now include `tts_segment_hit_max_tokens` so cutoff reports can be
    tied to a specific sentence/newline segment instead of an utterance-wide
    guess.
  - Playback prebuffer is length-aware for static multi-segment utterances and
    explicitly logged as the actual value used.
  - A text-to-TTS interleaving path consumes completed sentence/newline chunks
    from `AudexAsyncVllmRuntime.stream_text_response_from_messages(...)` while
    the final text response is still streaming. As of the follow-up latency
    checkpoint, this is the default non-thinking vLLM STS path; set
    `AUDEX_VLLM_STREAM_TEXT_TO_TTS=0` to force serial text-then-TTS for
    diagnostics.
- Measurement:
  - Interleaving ON fixture:
    `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 ./start.sh --model audex-30b-a3b --new-conversation --input-wav .audex/runs/ptt-input-20260708-223521.wav --no-play --response-max-tokens 128 --speech-max-tokens 1024`
    produced `.audex/runs/sts-turn-vllm-20260709-002550.json` and
    `.audex/runs/speech-output-vllm-20260709-002445.json`.
    Result: on-topic response, no Audex/NVIDIA boilerplate, 4 TTS segments, all
    segments reached end, no max-token hits, `mlx_clear_cache_count=5`,
    `text_elapsed_seconds=12.898`, `tts_first_audio_ready_seconds=10.442`,
    total turn `71.564s`.
  - Interleaving OFF fixture:
    `AUDEX_VLLM_STREAM_TEXT_TO_TTS=0 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 ./start.sh --model audex-30b-a3b --new-conversation --input-wav .audex/runs/ptt-input-20260708-223521.wav --no-play --response-max-tokens 128 --speech-max-tokens 1024`
    produced `.audex/runs/sts-turn-vllm-20260709-002805.json` and
    `.audex/runs/speech-output-vllm-20260709-002711.json`.
    Result: on-topic response, no Audex/NVIDIA boilerplate, 3 TTS segments, all
    segments reached end, no max-token hits, `mlx_clear_cache_count=4`,
    `text_elapsed_seconds=8.474`, `tts_first_audio_ready_seconds=2.613`, total
    turn `68.187s`.
- Original decision: keep text-to-TTS interleaving opt-in. In this fixture it
  did not materially improve total no-play runtime.
- Updated decision: enable text-to-TTS interleaving by default for non-thinking
  interactive STS because the user-facing bottleneck is time-to-first-audio
  after the response text has already started streaming. The serial chunked
  no-CFG TTS path remains available with `AUDEX_VLLM_STREAM_TEXT_TO_TTS=0`.
- Tests:
  - `tests/test_vllm_runtime.py` covers the prompt-leak scrubber.
  - `tests/test_vllm_sts_cli.py` covers stable streamed text chunk extraction,
    default non-thinking interleaving, diagnostic opt-out, per-segment cache
    clearing, and per-segment max-token diagnostics.
  - Validation: `.venv/bin/python -m pytest -q` passed with
    `325 passed, 3 skipped`.

## 2026-07-09: Non-Paged No-CFG TTS Window Decode

- Purpose: make concrete progress on the active goal, "make the passing no-CFG
  path fast on vLLM Metal, without reintroducing the CFG garbage-audio
  failure."
- Tight feedback loop:
  - Added `scripts/bench_vllm_metal_sampler.py`, a MLX/Metal microbenchmark
    for Audex-sized no-CFG TTS sampling. It compares raw codec-window sampling,
    pending full-vocab projection plus sampling, and pending codec-window
    projection plus sampling.
  - Red-capable command:
    `PYTHONPATH="$PWD/.audex/vendor/vllm-metal" VLLM_LOGGING_LEVEL=ERROR TRANSFORMERS_VERBOSITY=error .audex/vendor/vllm-metal/.venv-vllm-metal/bin/python scripts/bench_vllm_metal_sampler.py --iterations 3 --warmup 1 --fail-if-pending-full-ms-over 5`
    failed on the current full-projection path with
    `pending_full_projection_then_window_sample avg_ms 12.031 > 5.000`.
  - Full-shape benchmark artifact:
    `.audex/runs/vllm-metal-sampler-bench-20260709-windowdecode.json`.
    Result: raw codec-window sampling `0.237ms`, pending full projection plus
    window sampling `12.249ms`, pending window projection plus sampling
    `4.131ms`.
- Code changes:
  - `audex_mac.patches.vllm_metal_cfg` now patches
    `vllm_metal.v1.model_runner.MetalModelRunner._sequential_decode` for a
    narrow non-paged, single-request, no-CFG Audex TTS shape.
  - Eligible requests are identified only by Audex TTS codec-window metadata in
    `SamplingParams.extra_args`; CFG roles/pair IDs, generators, mixed batches,
    top-p fallback, unsupported logits processors, constraints, penalties, and
    non-random/non-greedy mixed batches fall back to the original vLLM Metal
    path.
  - The patch calls the model backbone to get hidden states, projects only
    `[<speechgen_end>] + [<speechcodec_*>]` rows from `lm_head.weight`, samples
    with NVIDIA's existing sampler settings, maps the local codec-window index
    back to the full tokenizer ID, and updates the request state.
  - Existing CFG sampling code is not used by this path. This is deliberate:
    CFG remains rejected here until a separate paired CFG path produces
    transcript-verified intelligible audio.
  - Native debug output now includes `tts_window_decode_count` in the non-paged
    decode timing line.
- Measurement:
  - Short non-paged 30B no-CFG TTS probe:
    `PYTHONPATH="$PWD:$PWD/.audex/vendor/vllm-metal" AUDEX_MAC_AUTO_PATCHES=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 VLLM_METAL_USE_PAGED_ATTENTION=0 VLLM_LOGGING_LEVEL=ERROR TRANSFORMERS_VERBOSITY=error .audex/vendor/vllm-metal/.venv-vllm-metal/bin/python scripts/probe_vllm_tts_decode.py --model audex-30b-a3b --text "One short sentence about local speech." --max-tokens 64 --no-cfg --chunk-frames 24`
    produced `.audex/runs/tts-probe-vllm-20260709-004602.wav`.
    Result: `generation_seconds=2.275`, `generated_token_count=64`,
    `aggregate_tokens_per_second=28.132`, `hit_max_tokens=true` because this
    was an intentionally capped 64-token probe. Debug output showed
    `tts_window_decode_count=49` at the 50-step checkpoint while native sampler
    sampled rows stayed at the prefill row, proving the window path handled the
    decode tokens.
  - Same short probe on the paged path, before forcing
    `VLLM_METAL_USE_PAGED_ATTENTION=0`, generated 64 tokens in
    `generation_seconds=10.811` (`5.92 tokens/sec`). That comparison is useful
    only as a path sanity check; the CLI's green no-CFG path is non-paged.
  - Full 30B no-play STS fixture after the window-decode patch:
    `.audex/runs/sts-turn-vllm-20260709-004218.json` and
    `.audex/runs/speech-output-vllm-20260709-004125.json`.
    Result: `stream_finished_seconds=52.672` for 1,380 codec frames across
    three TTS segments, no segment max-token hits, and clean end tokens. The
    prior serial no-interleave control
    `.audex/runs/speech-output-vllm-20260709-002711.json` took `53.486s` for
    1,317 codec frames. This is a real but modest full-turn improvement because
    backbone decode still dominates after removing most full-head sampler work.
- Tests:
  - `tests/test_vllm_cfg.py` covers that no-CFG non-paged TTS uses the backbone
    path instead of the full logits model call, and that CFG-shaped requests
    are rejected before model access.
  - Validation: `.venv/bin/python -m pytest -q` passed with
    `327 passed, 3 skipped` before this ledger update.
- Decision:
  - Keep this patch. It is aligned with the no-CFG green path, does not change
    sampler parameters, and has a measurable short-probe speedup.
  - Do not declare the active performance goal complete. The full 30B CLI turn
    is still far from near-real-time; the next bottleneck is likely backbone
    decode/KV/cache scheduling rather than categorical sampling.

### 2026-07-09 Stability Checkpoint: Window-Head Cache, Playback Drain, and Long-TTS Segments

- Purpose: make the current vLLM Metal no-CFG path more stable for human
  evaluation without changing NVIDIA sampler parameters or reintroducing CFG.
- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_cfg._cached_tts_window_head(...)` caches the
    speechgen-end plus speech-codec output head window on the vLLM Metal runner.
    This avoids rebuilding the same 65k-row window head on every no-CFG TTS
    decode token. Native debug output now includes
    `tts_window_weight_cache_hits` and `tts_window_weight_cache_misses`.
  - `audex_mac.sts_cli._ContinuousPcmPlayer` now waits for its estimated
    buffered PCM tail to drain before closing the `sounddevice` stream. This
    addresses the case where the WAV/run log contains the full utterance but
    speaker playback cuts off the final buffered audio.
  - `audex_mac.vllm_sts_cli.VllmSpeechToSpeechSession` calls `mx.clear_cache()`
    after ASR and text generation, and the existing TTS segment path continues
    clearing after each segment/final cleanup. This is intentionally scoped to
    dead MLX temporaries; it does not unload the persistent vLLM model.
  - `audex_mac.vllm_runtime.extract_spoken_answer(...)` remains the prompt-leak
    scrubber. Regression tests now cover observed `Audex is created by NVIDIA`
    and `Audex is a conversational partner created by NVIDIA` leakage from the
    2026-07-08 human transcript.
- Measurement:
  - Cached-window short non-paged 30B no-CFG TTS probe:
    `.audex/runs/tts-probe-vllm-20260709-005010.json`.
    Result: `generation_seconds=2.198`, `generated_token_count=64`,
    `aggregate_tokens_per_second=29.117`, `hit_max_tokens=true` because this
    was an intentionally capped 64-token probe. This is only a small improvement
    over the prior `2.275s` short probe, but it removes repeated head-window
    construction from the decode loop.
  - Current no-play 30B fixture:
    `.audex/runs/sts-turn-vllm-20260709-005756.json` and
    `.audex/runs/speech-output-vllm-20260709-005654.json`.
    Result: on-topic text with no Audex/NVIDIA/Nemotron preamble, three TTS
    chunks, all chunks reached end tokens, no max-token hits, and
    `mlx_clear_cache_count=4`.
  - Remaining performance issue: the fixture still spent
    `stream_finished_seconds=62.177` in TTS for 1,380 codec frames. This
    checkpoint is better for intelligibility/stability testing, not yet a
    near-real-time success.
- Tests:
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_sts_cli.py tests/test_vllm_runtime.py tests/test_vllm_sts_cli.py tests/test_vllm_cfg.py -q`
    passed with `110 passed`.
  - Lint validation:
    `python -m ruff check audex_mac/sts_cli.py audex_mac/vllm_sts_cli.py audex_mac/patches/vllm_metal_cfg.py tests/test_sts_cli.py tests/test_vllm_runtime.py tests/test_vllm_cfg.py`
    passed.
  - Full validation:
    `.venv/bin/python -m pytest -q` passed with
    `330 passed, 3 skipped`, and `python -m ruff check .` passed.
- Decision:
  - Keep this checkpoint. It directly addresses the latest human-observed
    cutoff, prompt leakage, and turn-to-turn memory pressure concerns while
    preserving the currently intelligible no-CFG path.

### 2026-07-09 Latency Checkpoint: Earlier Streaming TTS Release

- Purpose: improve time-to-first-utterance on the currently intelligible
  no-CFG vLLM Metal path without changing NVIDIA sampler settings or
  reintroducing CFG.
- Audex-Mac code:
  - `audex_mac.vllm_sts_requests.VllmSamplingPlan` can now pass
    `detokenize=False` through to vLLM. TTS requests set it because the Audex
    speech path consumes token IDs and codec frames, not decoded
    `<speechcodec_...>` text. ASR/text requests leave vLLM defaults unchanged.
  - `audex_mac.vllm_sts_cli._stable_streaming_tts_cut_index(...)` now permits
    a first completed sentence to feed TTS once it reaches the substantial
    streaming floor. The floor is now `120` chars, so short fragments still
    wait, avoiding the earlier one-word/robotic prosody failure.
- Measurement:
  - Detokenization alone was stable but not a meaningful speed win on the
    30B fixture: codec throughput stayed essentially flat
    (`44.614` -> `44.664` codec fps).
  - Same 30B no-play fixture after earlier first-sentence release:
    `.audex/runs/sts-turn-vllm-20260709-030625.json` and
    `.audex/runs/speech-output-vllm-20260709-030553.json`.
    Result: `first_tts_chunk_ready_seconds` improved from `1.752` to `1.210`,
    `first_audio_ready_seconds` improved from `3.020` to `2.453`, no
    max-token hits, clean end tokens, and the response stayed on topic.
  - Same 30B playback fixture:
    `.audex/runs/sts-turn-vllm-20260709-030811.json` and
    `.audex/runs/speech-output-vllm-20260709-030737.json`.
    Result: `first_audio_ready_seconds=2.467`,
    `first_playback_started_seconds=4.066`, `device_underflow_count=0`,
    `queue_underrun_count=0`, `queue_overrun_count=0`, and full `28.52s`
    playback written.
- Tests:
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py tests/test_vllm_runtime.py -q`
    passed with `70 passed`.
  - Lint validation:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py audex_mac/vllm_sts_requests.py tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py tests/test_vllm_runtime.py`
    passed.
- Decision:
  - Keep this checkpoint. It does not make full no-CFG TTS realtime, but it
    measurably reduces first-audio latency while preserving intelligibility and
    clean playback diagnostics.

### 2026-07-09 Speed Checkpoint: Cached TTS Projection Matrix

- Purpose: reduce sustained no-CFG vLLM Metal TTS decode time without changing
  NVIDIA sampler parameters or enabling the rejected CFG path.
- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_cfg._cached_tts_window_head(...)` now caches
    the projection-ready `transpose(window_weight).astype(float32)` matrix for
    Audex's `[speechgen_end, speechcodec_min..speechcodec_max]` output window.
    The prior cache stored the sliced window rows but still repeated the large
    transpose and float32 cast every generated speech token.
  - The TTS fast path still uses the same backbone hidden state, the same
    allowed speech-token window, and the same sampler settings. This is a
    representation/cache optimization, not a sampling-policy change.
- Measurement:
  - Short isolated 30B no-CFG TTS probe:
    `.audex/runs/tts-probe-vllm-20260709-031239.wav`.
    Result: `generation_seconds=1.284`, `generated_token_count=64`,
    `aggregate_tokens_per_second=49.844`. The previous cached-window probe in
    this ledger was `2.198s` / `29.117 tokens/sec`, so the isolated decode
    path improved by about `1.71x`.
  - Full same-fixture 30B no-play STS run:
    `.audex/runs/sts-turn-vllm-20260709-031351.json` and
    `.audex/runs/speech-output-vllm-20260709-031322.json`.
    Result: `stream_finished_seconds=29.336` for `28.52s` of audio,
    `audio_realtime_ratio=0.972`, and `codec_frames_per_second=47.927`.
    The previous no-play checkpoint was `31.704s`, `0.900`, and `44.348`
    respectively for the same segment/token counts.
  - Full same-fixture 30B playback run:
    `.audex/runs/sts-turn-vllm-20260709-031517.json` and
    `.audex/runs/speech-output-vllm-20260709-031444.json`.
    Result: `device_underflow_count=0`, `queue_underrun_count=0`, full
    `28.52s` playback written, and clean end tokens. `queue_overrun_count=32`
    reflected the producer getting ahead of the speaker with a `3.5s` high
    water buffer; it is not a sounddevice underrun/dropout.
- Memory note:
  - This intentionally keeps a float32 transposed projection matrix in the
    vLLM EngineCore process for the speech-token window. The CLI-side MLX
    decoder memory snapshots were unchanged in the fixture logs, but the
    EngineCore tradeoff should be considered if future multi-turn memory
    pressure returns.
- Tests:
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q`
    passed with `33 passed`.
  - Lint validation:
    `.venv/bin/python -m ruff check audex_mac/patches/vllm_metal_cfg.py tests/test_vllm_cfg.py`
    passed.
- Decision:
  - Keep this patch. It materially improves both the isolated TTS path and the
    full no-CFG STS fixture while preserving intelligible output and clean
    speaker underrun diagnostics.

### 2026-07-09 Speed Checkpoint: Split Primer and Steady Decoder Chunks

- Purpose: preserve low first-audio latency while reducing steady-state MLX
  causal speech decoder overhead on the green no-CFG vLLM Metal path.
- Audex-Mac code:
  - `audex_mac.vllm_sts_cli` now keeps the first decoder push at
    `DEFAULT_VLLM_STREAM_DECODER_CHUNK_FRAMES=18` plus decoder lookahead, then
    uses `DEFAULT_VLLM_STREAM_DECODER_STEADY_CHUNK_FRAMES=48` for later pushes.
  - Explicit diagnostic/test overrides for `decoder_chunk_frames` keep uniform
    chunking unless a separate steady value is supplied, so small synthetic
    tests and probes remain predictable.
  - Speech-output run logs now include `decoder_steady_chunk_frames`.
- Measurement:
  - Baseline after the projection-cache patch:
    `.audex/runs/speech-output-vllm-20260709-031322.json` recorded
    `stream_finished_seconds=29.336`, `audio_realtime_ratio=0.972`,
    `codec_frames_per_second=47.927`, `decoder_push_seconds=2.377410`, and
    `decoded_chunk_count=79`.
  - 24-frame steady candidate:
    `.audex/runs/speech-output-vllm-20260709-032206.json` recorded
    `28.696s`, `0.994`, `48.996`, `decoder_push_seconds=1.731909`, and
    `decoded_chunk_count=61`.
  - 32-frame steady candidate:
    `.audex/runs/speech-output-vllm-20260709-032407.json` recorded
    `28.645s`, `0.996`, `49.084`, `decoder_push_seconds=1.341299`, and
    `decoded_chunk_count=46`.
  - 48-frame steady candidate, kept:
    `.audex/runs/speech-output-vllm-20260709-032605.json` recorded
    `28.533s` for `28.52s` of audio, `audio_realtime_ratio=1.000`,
    `codec_frames_per_second=49.276`, `decoder_push_seconds=0.900957`, and
    `decoded_chunk_count=31`.
  - 64-frame steady candidate, rejected:
    `.audex/runs/speech-output-vllm-20260709-032746.json` lowered decoder work
    further (`decoder_push_seconds=0.670500`, `decoded_chunk_count=24`) but
    total wall time regressed to `28.601s` / `0.997`, likely because delayed
    larger decoder pushes stopped overlapping as well with token generation.
  - 48-frame playback validation:
    `.audex/runs/speech-output-vllm-20260709-032930.json` recorded
    `device_underflow_count=0`, `queue_underrun_count=0`, full `28.52s`
    playback written, and clean end tokens. First playback moved later
    (`4.062s` -> `4.566s`) because the prebuffer waits for larger PCM chunks;
    this is an explicit latency/throughput tradeoff to revisit if human
    testing prefers earlier playback over steadier generation.
- Tests:
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q`
    passed with `24 passed`.
  - Lint validation:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py tests/test_vllm_sts_cli.py`
    passed.
- Decision:
  - Keep the 48-frame steady default. It is the best same-fixture no-play
    throughput point found so far and keeps playback underrun-free. Do not
    keep the 64-frame candidate.

### 2026-07-09 Human Checkpoint: Long-Turn Stability Follow-Up

- Human result:
  - Startup greeting is intelligible.
  - First real ASR -> response -> TTS turn is intelligible.
  - Longer responses can still become choppy after playback has started,
    repeat a phrase, or stop audibly before the full text response has been
    spoken.
  - Memory pressure climbs too quickly across turns on the 30B bf16 model.
  - Some responses still try to talk about Audex/NVIDIA instead of the user's
    topic.
- Audex-Mac code:
  - `audex_mac.vllm_cfg.MIN_CFG_NUM_SEQS` is reduced from NVIDIA's
    server-oriented `128` to `8` for the local Mac CLI. The SLC default runs
    one interactive user, one ASR request, one text request, and serial no-CFG
    TTS chunks; it does not need a 128-sequence scheduler/KV budget. Keep
    `AUDEX_VLLM_CFG_MAX_NUM_SEQS` for explicit diagnostics.
  - `personas/assistant.md` and `DEFAULT_VLLM_TEXT_PROMPT` avoid naming
    Audex-Mac or asking the model not to describe its implementation. Those
    words were giving the model convenient handles for the observed
    Audex/NVIDIA/Nemotron boilerplate.
  - Interleaved text-to-TTS playback now starts with a `2.0s` prebuffer instead
    of the default `0.8s`. This deliberately trades some first-playback latency
    for fewer underruns when long no-CFG TTS generation is only barely faster
    than realtime.
  - The default spoken TTS chunk character cap is `240` chars, still up to
    two sentences or a newline boundary, but less likely to hand no-CFG TTS a
    large prompt that degrades over time.
- Rejected experiment:
  - Temperature-folding into the cached TTS projection matrix was tested and
    removed. A 64-token 30B no-CFG probe measured `1.288s` /
    `49.689 tokens/sec`, slightly worse than the kept projection-cache
    checkpoint at `1.284s` / `49.844 tokens/sec`, while requiring a second
    cache variant. Do not reintroduce it without a better benchmark.

### 2026-07-09 Speed Checkpoint: Delta Text Streaming for Interleaved TTS

- Purpose: reduce Python/vLLM output churn before TTS starts without changing
  the green no-CFG TTS sampler path.
- Audex-Mac code:
  - `AudexAsyncVllmRuntime.stream_text_response_from_messages(...)` now requests
    `RequestOutputKind.DELTA` for streamed text responses and accumulates text
    locally before applying `extract_spoken_answer(...)`. The public iterator
    still yields cumulative cleaned text, so `streamed_tts_chunks_from_text(...)`
    and existing chunking behavior are preserved.
  - vLLM STS turn logs now include `text_to_tts_streaming` diagnostics:
    `text_stream_event_count`, first/last text delta times, first/last emitted
    TTS chunk times, emitted TTS chunk count, and emitted chunk character sizes.
    This is intended to distinguish "text did not stream early" from "TTS was
    slow after the first stable chunk was available" in human runs.
- Expected effect:
  - This should reduce repeated cumulative text payload handling during long
    responses and give the next real Metal run sharper timing evidence. It does
    not change ASR, no-CFG TTS token sampling, speech decoder chunking, or
    playback buffering.

### 2026-07-09 Speed Checkpoint: TTS Window Sampler Without SamplingBatch

- Purpose: reduce per-token Python/Torch-side overhead in the no-CFG Audex TTS
  window decode path without changing logits, masks, `temperature`, `top_k`, or
  token sampling semantics.
- Audex-Mac code:
  - `_try_sequential_tts_window_decode(...)` and
    `_try_batched_tts_window_decode(...)` now call
    `_sample_tts_window_logits_from_params_if_supported(...)` directly instead
    of constructing vLLM Metal's generic `SamplingBatch` on every speech token.
  - The direct helper derives the same native-sampler guards from
    `sampling_params`: no logprobs, no penalties, top-p disabled, top-k absent
    or uniformly positive, no allowed/bad-word constraints, supported/inert
    logits processors only, and all-greedy or all-random sampling.
  - The generic `_sample_native_mlx_if_supported(...)` path still uses
    `SamplingBatch`; this change is scoped to the Audex TTS window fast path
    where the caller already has the exact sampling params and allowed codec
    window.
- Expected effect:
  - This removes one vLLM/Torch-oriented object construction from each generated
    speech-token step. It should help long no-CFG TTS turns where CPU overhead
    contributes to speaker underruns, but it still needs a real Metal run to
    measure tokens/sec because the Codex sandbox cannot access a Metal device.

### 2026-07-09 Stability Checkpoint: Smaller Mac vLLM Batch Token Budget

- Purpose: reduce unified-memory pressure on the 30B bf16 local CLI path so
  long no-CFG speech turns are less likely to fall into compression/paging.
- Audex-Mac code:
  - `configure_audex_vllm_cfg(...)` now defaults `max_num_batched_tokens` to
    `8192` instead of `DEFAULT_CFG_NUM_SEQS * max_model_len` (`10240` for the
    current `5120` context). This still stays above vLLM Metal's non-paged
    requirement that `max_num_batched_tokens >= max_model_len`.
  - `AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS` can raise the value for diagnostics or
    NVIDIA-style paired/server-capacity experiments. Values below
    `max_model_len` are clamped to `max_model_len` so a diagnostic override
    cannot silently make prefill unschedulable.
  - `AUDEX_VLLM_CFG_MAX_NUM_SEQS` remains the separate sequence-count override.
- Expected effect:
  - This does not change no-CFG TTS sampling, logits, audio chunks, or response
    text. It reduces the scheduler/KV admission budget for the single-user Mac
    CLI, which should help avoid the memory-pressure cliff reported during
    multi-turn 30B tests.

### 2026-07-09 Stability Checkpoint: Avoid Duplicate TTS Frame Lists

- Purpose: reduce Python-side memory churn during long no-CFG TTS turns.
- Audex-Mac code:
  - `generate_speech_output_streaming_from_async_runtime(...)` no longer stores
    a second per-segment list of every generated codec frame. It now maintains
    `tts_segment_codec_frame_counts` incrementally while keeping the global
    `generated_codec_frames` list used by the return value and run log.
- Expected effect:
  - This does not change generation, decoding, playback, or run-log count
    fields. It removes duplicate frame storage on long segmented utterances,
    which is small compared with model memory but still aligned with keeping
    the interactive 30B path away from avoidable Python memory growth.

### 2026-07-09 Startup/Memory Checkpoint: Lazy CLI Audio Projection Loading

- Purpose: avoid duplicating Audex audio encoder/projector weights in the CLI
  process during the default vLLM Metal STS path.
- Audex-Mac code:
  - `VllmSpeechToSpeechSession.__init__(...)` no longer eagerly loads
    `audio_encoder` and `audio_projector` weights. The default raw-audio ASR
    path sends audio to vLLM Metal, where the Audex adapter projects it inside
    EngineCore.
  - `project_wav_audio(...)` now calls
    `_ensure_audio_projection_components_loaded()` before using the local
    projected-audio diagnostic path, preserving that helper without paying the
    memory/startup cost for normal `./start.sh`.
- Expected effect:
  - This does not change ASR prompts, text generation, no-CFG TTS sampling,
    speech decoding, or playback. It removes an avoidable CLI-process memory
    allocation from the interactive vLLM path and should slightly reduce
    startup work while keeping projected-audio diagnostics available.

### 2026-07-09 EngineCore Memory Checkpoint: Replace Used Hybrid Cache Cleanly

- Purpose: reduce EngineCore-side MLX memory growth across repeated ASR/text/TTS
  prefills on Audex's hybrid Nemotron-H cache path.
- vLLM Metal Audex adapter patch:
  - `AudexMultimodalAdapter._prepare_hybrid_cache(...)` now tracks whether its
    adapter-owned hybrid cache has actually been used. The first prefill can use
    the cache created while probing for `ArraysCache` layers instead of
    immediately allocating a second cache.
  - When a later prefill (`seq_len > 1`) needs a fresh hybrid cache, the adapter
    drops the old cache reference, runs Python GC, and calls `mx.clear_cache()`
    inside EngineCore before allocating the replacement cache.
- Expected effect:
  - This does not change vLLM request scheduling, token sampling, no-CFG TTS
    logits, or the speech decoder. It targets the turn-to-turn memory-pressure
    cliff observed in Activity Monitor by making the EngineCore cache handoff
    explicit instead of relying on eventual Python/MLX cleanup.

### 2026-07-09 No-CFG Fast-Path Checkpoint: Make CFG Engine Wiring Explicit

- Purpose: remove a global CFG tax from the default intelligible no-CFG vLLM
  Metal STS path.
- Audex-Mac code:
  - `AudexVllmRuntime.from_model_path(...)` and
    `AudexAsyncVllmRuntime.from_model_path(...)` now pass `cfg_scale=0.0` to
    `configure_audex_vllm_cfg(...)` unless
    `AUDEX_VLLM_ENABLE_CFG_WIRING=1` is set.
  - Default `./start.sh` therefore keeps `enable_prefix_caching=True` from
    `_base_engine_kwargs(...)` and avoids installing CFG logits processors for
    ordinary no-CFG ASR/text/TTS turns.
  - CFG batch diagnostics set `AUDEX_VLLM_ENABLE_CFG_WIRING=1` explicitly when
    `use_cfg=True`; no-CFG diagnostics and STS smoke clear it.
- Expected effect:
  - This does not change NVIDIA sampler values, no-CFG TTS request prompts,
    codec-token masks, or decoder/playback behavior. It aligns engine
    construction with the current product reality: CFG audio remains rejected
    for audible STS until proven, while the green no-CFG path should not carry
    CFG scheduler/logits-processor overhead or lose prefix caching.

### 2026-07-09 Human Checkpoint: Scrub and Shorten Spoken TTS Chunks

- Human result:
  - The startup greeting and first ASR -> response -> TTS turn are intelligible.
  - Longer TTS can still become choppy or repetitive, and attached logs from
    the earlier long-turn run show single-segment TTS plus queue/device
    underruns on the longest utterances.
  - The visible response text can still include Audex/NVIDIA/Nemotron
    boilerplate before the useful answer.
- Audex-Mac code:
  - `VllmSpeechToSpeechSession.run_turn_from_wav(...)` now scrubs model
    self-description/prompt leakage before printing response text, persisting
    the assistant turn, and sending text to TTS in both sync and async paths.
  - The default no-CFG spoken chunk policy is now three sentences or `240`
    characters per TTS request, whichever boundary comes first, while still
    preferring newlines. Single overlong sentences are split at word/phrase
    boundaries (`140` characters for unpunctuated text, `160` for punctuated
    text) so a run-on model response does not become one long TTS prompt.
  - Streaming text-to-TTS no longer resets the emitted-text cursor to zero when
    prompt-leak scrubbing makes the cumulative text shorter. It clamps the
    cursor to the current cleaned text length, preventing already-spoken text
    from being queued again.
  - The vLLM text response policy now explicitly asks for ordinary sentence
    punctuation and capitalization. This is a prompt-shape nudge only; sampler
    settings remain NVIDIA-shaped.
  - STS turn logs now include `process_memory` snapshots for the CLI process
    and visible `VLLM::EngineCore` RSS so the next interactive run can verify
    whether the multi-turn memory climb is in EngineCore rather than the MLX
    speech decoder.
- Evidence:
  - Short Audex-30B-A3B fixture
    `.audex/runs/ptt-input-20260708-222951.wav`, run with
    `--response-max-tokens 128 --speech-max-tokens 1024 --no-play`, produced
    `.audex/runs/sts-turn-vllm-20260709-045746.json` and
    `.audex/runs/speech-output-vllm-20260709-045736.json`.
  - The model still emitted uncapitalized run-on text, but TTS chunking split it
    into phrase-aware chunks:
    `hey matt thanks for joining in let's dive into go routines` and
    `maybe start with how you launch one with go func...`.
  - The fixture reported `tts_observed_segments=2`,
    `tts_interleaved_tail_batched=True`, `first_audio_ready_seconds=2.461`,
    `stream_finished_seconds=10.092`, `audio_duration_seconds=7.920`,
    `audio_realtime_ratio=0.785`, and `codec_frames_per_second=38.248`.
  - Memory snapshots in that one-shot run showed CLI `max_rss_bytes` moving from
    `4111499264` to `4121608192`; `VLLM::EngineCore` RSS was `13923155968`.
    This does not reproduce the reported 122 GB interactive climb, but it gives
    the next human run per-turn evidence for where memory is accumulating.
- Expected effect:
  - This does not change sampler settings, max speech-token policy, CFG state,
    decoder frame sizing, or playback transport. It reduces long no-CFG TTS
    degradation by making the default vLLM TTS requests smaller and keeps
    boilerplate out of the text that gets spoken and resumed in conversation
    history.

### 2026-07-09 No-CFG Speed Checkpoint: Batch Interleaved TTS Tail

- Purpose: keep the low first-audio latency from text-to-TTS interleaving while
  using vLLM Metal continuous batching for the remaining spoken TTS chunks.
- Evidence on the same cached Audex-30B-A3B no-play fixture
  `.audex/runs/ptt-input-20260708-223521.wav` with
  `--response-max-tokens 128 --speech-max-tokens 1024`:
  - Serial interleaved baseline
    `.audex/runs/speech-output-vllm-20260709-042338.json`:
    `stream_finished_seconds=31.558`, `audio_realtime_ratio=0.874`,
    `codec_frames_per_second=42.747`, `first_audio_ready_seconds=2.412`,
    total turn `elapsed_seconds=36.299`.
  - Completed-text concurrent comparison
    `.audex/runs/speech-output-vllm-20260709-042545.json`:
    `stream_finished_seconds=22.720`, `audio_realtime_ratio=1.307`,
    `codec_frames_per_second=63.600`, but it waits until text generation is
    complete before starting TTS.
  - Kept background-tail default
    `.audex/runs/speech-output-vllm-20260709-043310.json`:
    `stream_finished_seconds=27.082`, `audio_realtime_ratio=1.021`,
    `codec_frames_per_second=49.590`, `first_audio_ready_seconds=2.390`,
    total turn `elapsed_seconds=31.828`.
- Audex-Mac code:
  - Interleaved no-CFG TTS still starts the first spoken chunk immediately.
  - Once the first chunk starts, a background task collects the remaining
    completed text chunks and submits them through
    `stream_tts_segmented_codec_frames(...)`, buffering events until the first
    chunk has been decoded.
  - Decoding and playback remain ordered through the existing
    `AudexSpeechDecoderSession`; CFG remains disabled and sampler settings are
    unchanged.
  - `AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL=0` restores the older strictly
    serial interleaved path for diagnostics.
- Known separate issue:
  - One-shot fixture shutdown still prints the PyTorch MPS allocator traceback
    from the `EngineCore` subprocess. A parent-process cleanup patch was tested
    and rejected because it does not affect the child process. Treat that as a
    separate vLLM-Metal shutdown patch, not part of this speed checkpoint.

### 2026-07-09 No-CFG Speed Checkpoint: Coalesce Ready TTS Chunks

- Purpose: avoid serializing the first TTS chunk when the text stream has
  already produced every spoken chunk. This keeps the default text-to-TTS
  interleaving behavior for genuinely long responses, but lets short finalized
  answers use vLLM Metal's segmented no-CFG batch path immediately.
- Audex-Mac code:
  - Interleaved text-to-TTS now feeds chunks through `_QueuedTtsChunkSource`
    instead of a bare `asyncio.Queue`.
  - Before starting first-chunk serial TTS, the speech side waits a small
    `DEFAULT_VLLM_INTERLEAVED_READY_BATCH_WINDOW_SECONDS=0.05` coalescing
    window. If the text side has queued all chunks and the end sentinel, the
    speech side calls `stream_tts_segmented_codec_frames(...)` for the whole
    utterance.
  - If the text stream is not finished after that small window, the previous
    behavior remains: speak the first chunk, then collect and batch the tail in
    the background.
  - Run logs now include `tts_interleaved_all_ready_batched` to distinguish the
    new path from the older `tts_interleaved_tail_batched` path.
- Evidence on the short cached Audex-30B-A3B no-play fixture
  `.audex/runs/ptt-input-20260708-222951.wav` with
  `--response-max-tokens 128 --speech-max-tokens 1024`:
  - Previous default phrase-aware interleaved run
    `.audex/runs/speech-output-vllm-20260709-045736.json`:
    `stream_finished_seconds=10.092`, `audio_realtime_ratio=0.785`,
    `codec_frames_per_second=38.248`, total turn `elapsed_seconds=12.201`.
  - Pre-window probe
    `.audex/runs/speech-output-vllm-20260709-051056.json`:
    `tts_interleaved_all_ready_batched=False`; the second chunk arrived about
    22 ms after the first chunk was ready, so the older tail path still fired.
  - Coalesced ready-chunk run
    `.audex/runs/speech-output-vllm-20260709-051234.json`:
    `tts_interleaved_all_ready_batched=True`,
    `tts_interleaved_tail_batched=False`,
    `stream_finished_seconds=9.909`, `audio_realtime_ratio=0.799`,
    `codec_frames_per_second=38.954`, total turn `elapsed_seconds=12.003`.
- Expected effect:
  - This does not change NVIDIA sampler settings, no-CFG logits, CFG state,
    response text, decoder frame sizing, or playback transport. It removes an
    avoidable serial-first-chunk step for short responses whose full text is
    already available, while preserving streaming for responses that are still
    being generated.

### 2026-07-09 No-CFG Speed Checkpoint: Install Worker Patch Hook

- Purpose: make the default no-CFG vLLM Metal path actually install the
  worker-local Audex Metal patch hook. Before this change, the no-CFG runtime
  called `configure_audex_vllm_cfg(..., cfg_scale=0.0)`, returned early, and
  did not add `AudexMetalCFGTokenSyncInstaller` to `engine_kwargs`. Parent
  process probes could still report that the patch symbols existed, but the
  spawned `VLLM::EngineCore` process did not reliably receive the no-CFG
  TTS-window decode patch.
- Audex-Mac code:
  - `audex_mac.vllm_cfg.configure_audex_vllm_cfg(...)` now adds the inert
    `AudexMetalCFGTokenSyncInstaller` logits processor even when CFG is
    disabled. This does not add NVIDIA's `CFGLogitsProcessor`, does not disable
    prefix caching, and does not enable CFG. It only gives the spawned worker a
    safe constructor hook that calls `apply_vllm_metal_cfg_patches()`.
  - CFG mode still appends NVIDIA's `CFGLogitsProcessor` followed by
    `AudexMetalCFGTokenSyncInstaller`, preserving the prior CFG processor
    ordering.
  - `audex_mac.vllm_diagnostics` now parses and records
    `tts_window_decode_count`, `tts_window_weight_cache_hits`, and
    `tts_window_weight_cache_misses` from no-CFG nonpaged decode timing lines.
- Evidence:
  - Pre-fix no-CFG batch probe
    `.audex/runs/vllm-metal-diagnostic-20260709-052233.json`:
    `elapsed_seconds=8.045`, `codec_frames_per_second=46.613`, no captured
    Audex worker timing lines beyond the shutdown traceback.
  - Post-fix no-CFG batch probe
    `.audex/runs/vllm-metal-diagnostic-20260709-052654.json`:
    `elapsed_seconds=7.055`, `codec_frames_per_second=53.154`.
    EngineCore stderr showed the worker patch active:
    `tts_window_decode_count=197`, `tts_window_weight_cache_hits=98`,
    `tts_window_weight_cache_misses=1`, and `batched=1`.
  - Same short no-play CLI fixture
    `.audex/runs/ptt-input-20260708-222951.wav`, with
    `--response-max-tokens 128 --speech-max-tokens 1024`:
    - Pre-fix `.audex/runs/speech-output-vllm-20260709-051234.json`:
      `stream_finished_seconds=9.909`, `codec_frames_per_second=38.954`,
      total turn `elapsed_seconds=12.003`.
    - Post-fix `.audex/runs/speech-output-vllm-20260709-052901.json`:
      `stream_finished_seconds=9.058`, `codec_frames_per_second=43.166`,
      total turn `elapsed_seconds=11.102`.
- Tests:
  - `tests/test_vllm_cfg.py` asserts the no-CFG config installs only
    `AudexMetalCFGTokenSyncInstaller` and leaves prefix caching/model limits
    untouched.
  - `tests/test_vllm_diagnostics.py` asserts the parser keeps the TTS window
    decode/cache counters from nonpaged timing output.

### 2026-07-09 CFG Correctness Checkpoint: Restore NVIDIA CFG Sampler and Codec Window

- Purpose: move the next research target from "acceptable no-CFG fallback" to
  "correct CFG first, fast CFG second" without destabilizing the no-CFG vLLM
  Metal path.
- Audex-Mac code:
  - `audex_mac.audio_contract` now records separate current defaults for the
    stable no-CFG fallback (`temperature=0.8`, `top_p=1.0`, `top_k=0`) and the
    NVIDIA model-card CFG audio-generation path (`temperature=1.0`,
    `top_p=1.0`, `top_k=80`, `cfg_scale=3.0`, `cfg_pairs_per_batch=2`).
  - `audex_mac.vllm_sts_requests.build_tts_cfg_requests(...)` uses the CFG
    sampler constants while `build_tts_request(...)` keeps the no-CFG fallback
    sampler constants.
  - `audex_mac.vllm_runtime.AudexVllmRuntime.build_tts_cfg_pair(...)` and
    `AudexAsyncVllmRuntime.build_tts_cfg_pair(...)` now pass
    `codec_min_id`/`codec_max_id` into the CFG request builder. This gives the
    vLLM Metal native sampler the same `audex_tts_codec_min_id`,
    `audex_tts_codec_max_id`, and `audex_tts_speechgen_end_id` metadata already
    used by the passing no-CFG path.
  - `audex_mac.speech_generation` now reports and uses the CFG-specific sampler
    values for its direct MLX CFG smoke instead of borrowing no-CFG TTS values.
  - `scripts/probe_vllm_tts_decode.py --no-codec-window` now explicitly builds
    CFG requests without codec-window metadata. The default probe path keeps
    using codec-window metadata, but the diagnostic A/B flag remains meaningful
    after the runtime CFG builder was corrected.
- Why this matters:
  - The native Metal sampler patch reads the codec-window metadata from
    `SamplingParams.extra_args` before restricting logits to
    `<speechgen_end>` plus `<speechcodec_*>`. The no-CFG path supplied that
    metadata; the CFG runtime path accepted the fields but failed to pass them.
    That made CFG easier to desynchronize from the speech-token contract and
    was consistent with "valid-looking speechcodec tokens, bad audio" symptoms.
  - This checkpoint is not a throughput optimization. It is a correctness
    guardrail before the next CFG speed work.
- Tests:
  - `tests/test_audio_contract.py` records both no-CFG and CFG sampler defaults.
  - `tests/test_vllm_sts_requests.py` asserts CFG requests use
    `temperature=1.0`, `top_p=1.0`, `top_k=80`, and `cfg_scale=3.0` while
    no-CFG requests keep the no-CFG sampler.
  - `tests/test_vllm_runtime.py` asserts sync and async vLLM CFG requests carry
    codec-window metadata plus CFG roles/pair IDs.
- Local evidence:
  - 2B CFG token-only diagnostic:
    `.audex/runs/vllm-metal-diagnostic-20260709-080123.json` reported
    `cfg_enabled=True`, `batch_size=2`, `request_count=4`,
    `codec_frames=256`, and `codec_fps=51.231`.
  - 2B decoded CFG probe:
    `scripts/probe_vllm_tts_decode.py --model audex-2b --max-tokens 256`
    wrote `.audex/runs/tts-probe-vllm-20260709-080225.wav`, reached
    `<speechgen_end>`, generated 189 codec frames, and reported
    `first_request_tokens_per_second=32.39`.
  - The MLX-Audio STT intelligibility check for the 2B WAV passed with
    ratio `0.9929`, transcribing
    `"Audx CFG test. This should be intelligible speech from the metal path."`
    for expected text
    `"Audex CFG test. This should be intelligible speech from the Metal path."`
  - 30B decoded CFG probe:
    `scripts/probe_vllm_tts_decode.py --model audex-30b-a3b --max-tokens 192`
    wrote `.audex/runs/tts-probe-vllm-20260709-080355.wav`, reached
    `<speechgen_end>`, generated 190 codec frames, and reported
    `first_request_tokens_per_second=15.496`.
  - The MLX-Audio STT intelligibility check for the 30B WAV passed with
    ratio `0.9859`, transcribing
    `"Audix CFG test. This should be intelligible speech from the metal path."`
    for the same expected text.

### 2026-07-09 CFG CLI Checkpoint: Opt-In Product Path and Speakable TTS Prompts

- Purpose: make the corrected vLLM Metal CFG TTS path reachable from the actual
  STS CLI without changing the default no-CFG fallback.
- Audex-Mac code:
  - `AUDEX_VLLM_TTS_CFG=1` now enables the CFG TTS product path before the
    async vLLM engine is constructed and sets `AUDEX_VLLM_ENABLE_CFG_WIRING=1`
    so the NVIDIA CFG processors and Metal CFG token-sync patch are installed
    early enough.
  - When CFG TTS is enabled, `VllmSpeechToSpeechSession` routes static TTS
    chunks through `AudexAsyncVllmRuntime.stream_tts_cfg_codec_frames(...)`.
    The no-CFG path remains the default and still uses
    `stream_tts_codec_frames(...)`.
  - CFG TTS currently disables text-to-TTS interleaving for that run. This
    keeps the path honest while CFG quality and latency are being tuned, rather
    than silently falling back to the faster no-CFG streamer.
  - Speech-output JSON now records `tts_cfg_enabled=true` and
    `text_to_tts_interleaved=false` for these runs.
  - `prepare_text_for_tts(...)` prepares only the spoken TTS prompt, not the
    persisted response text. It strips backticks, maps Python dunders such as
    `__enter__`/`__exit__` to speakable words, and preserves newline chunk
    boundaries for sentence/paragraph splitting.
- Tests:
  - `tests/test_vllm_sts_cli.py` covers the CFG env switch, early CFG wiring
    env propagation, routing to `stream_tts_cfg_codec_frames(...)`, run-log
    fields, and the TTS-only sanitizer.
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_runtime.py tests/test_vllm_sts_requests.py -q`
    passed with `94 passed`.
  - `scripts/lint.sh` passed.
  - Full validation: `.venv/bin/python -m pytest -q` passed with
    `379 passed, 3 skipped`.
- Local evidence:
  - 30B CFG CLI fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-081912.wav` and
    `.audex/runs/speech-output-vllm-20260709-081912.json`.
  - The speech JSON reported `tts_cfg_enabled=true`,
    `text_to_tts_interleaved=false`, `reached_end_token=true`,
    `hit_max_tokens=false`, `first_audio_ready_seconds=2.872`,
    `stream_finished_seconds=93.371`, `audio_duration_seconds=14.18`,
    `audio_realtime_ratio=0.152`, and `codec_frames_per_second=7.486`.
  - The spoken segments show the sanitizer was used for TTS while the response
    text retained code formatting:
    `"A context manager is an object that defines enter and exit, ..."` and
    `"The with statement runs those methods, ... explicit try-finally blocks."`
  - The MLX-Audio STT intelligibility check for the generated 30B CFG WAV passed
    with ratio `0.9223`. It transcribed the output as:
    `"A context manager is an object that defines enter and exit, letting Python
    acquire a resource when you enter it and release it automatically when you
    leave. The width statement runs those methods, guaranteeing cleanup even if
    an exception is raised, so you don't need explicit try-finally blocks."`
- Current interpretation:
  - This checkpoint fixes CFG CLI reachability and the earlier code-token TTS
    quality problem. It does not solve CFG throughput. 30B CFG remains well
    below realtime, so the next performance target is paired CFG scheduling /
    continuous-batch utilization and the small-batch decode/logits boundary.

### 2026-07-09 CFG Throughput Checkpoint: Batch Static CFG TTS Segments

- Purpose: stop underfeeding vLLM Metal during product CLI CFG TTS by submitting
  static spoken chunks as multiple active CFG pairs instead of generating each
  chunk through a separate sequential conditional/unconditional pair.
- Audex-Mac code:
  - `AudexAsyncVllmRuntime.stream_tts_cfg_segments_codec_frames(...)` now
    accepts explicit sanitized TTS segments plus per-segment token budgets.
    This is the CFG counterpart to the existing no-CFG segmented stream and
    avoids re-splitting product CLI chunks inside the runtime.
  - The explicit CFG segmented path sets `output_kind="DELTA"` on both
    conditional and unconditional requests and calls `stream_many(...,
    include_cumulative_token_ids=False)`. This keeps per-step processing on
    new token deltas instead of repeatedly extracting codec frames from
    cumulative token histories.
  - CFG segmented events now report `generated_token_ids` and
    `reached_end_token` per segment, matching the no-CFG segmented contract and
    keeping max-token/repetition diagnostics segment-local.
  - Static multi-chunk CLI TTS with `AUDEX_VLLM_TTS_CFG=1` now routes through
    `stream_tts_cfg_segments_codec_frames(...)` once and records
    `tts_concurrent_segments=true`. Single-chunk CFG TTS still uses
    `stream_tts_cfg_codec_frames(...)`. The default no-CFG fallback is
    unchanged.
- Tests:
  - `tests/test_vllm_runtime.py` covers explicit CFG segments, output-kind
    delta requests, per-segment token IDs, and per-segment reached-end flags.
  - `tests/test_vllm_sts_cli.py` asserts a multi-sentence static CFG utterance
    makes exactly one `tts-cfg-segmented-stream` runtime call with the CLI's
    sanitized chunks and per-segment token budgets.
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_runtime.py tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py -q`
    passed with `96 passed`.
  - `scripts/lint.sh` passed.
- Local evidence:
  - Same 30B CFG CLI fixture as the previous checkpoint:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-083038.wav` and
    `.audex/runs/speech-output-vllm-20260709-083038.json`.
  - Before this change, the sequential CFG CLI run
    `.audex/runs/speech-output-vllm-20260709-081912.json` reported
    `first_audio_ready_seconds=2.872`, `stream_finished_seconds=93.371`,
    `audio_realtime_ratio=0.152`, and `codec_frames_per_second=7.486`.
  - After batching explicit CFG segments, the new run reported
    `tts_concurrent_segments=true`, `reached_end_token=true`,
    `hit_max_tokens=false`, `first_audio_ready_seconds=3.843`,
    `stream_finished_seconds=59.156`, `audio_duration_seconds=14.38`,
    `audio_realtime_ratio=0.243`, and `codec_frames_per_second=11.985`.
  - The MLX-Audio STT intelligibility check for the new 30B CFG WAV passed with
    ratio `0.8893`, transcribing:
    `"A context manager is an object that implements enter and exit, letting
    Python acquire a resource when you enter it and release it automatically
    when you leave. The with statement runs those methods, guaranteeing cleanup
    even if an exception is raised so you don't need explicit try-finally
    blocks."`
- Current interpretation:
  - This is a meaningful throughput improvement and proves CFG product CLI TTS
    can use paired-request continuous batching without reverting to garbage
    audio. It is still not near realtime on 30B. First-audio latency regressed
    slightly because the static batched CFG path waits for ordered segment
    availability before segment 0 frames are emitted. The next step is reducing
    that first-audio penalty while keeping multiple CFG pairs active.

### 2026-07-09 CFG Scheduling Checkpoint: Stage Tail Pairs After Segment 0 Starts

- Purpose: A/B whether handing all CFG segment pairs to vLLM Metal up front is
  the best tradeoff for product CLI speech, or whether priming segment 0 first
  improves the perceived latency/throughput balance.
- Audex-Mac code:
  - At this checkpoint,
    `AudexAsyncVllmRuntime.stream_tts_cfg_segments_codec_frames(...)` set
    `prime_first_segment=True` by default. It started the first segment's
    conditional/unconditional CFG pair first, then submitted the remaining
    segment pairs after the first conditional delta was observed.
  - This default is superseded by the later "CFG Scheduling Default: Start All
    Segment Pairs Up Front" section below.
  - The CLI passes this flag for static multi-segment CFG TTS and records
    `tts_cfg_prime_first_segment` in speech-output JSON.
  - At this checkpoint, `AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT=0` disabled
    this staging for A/B diagnostics and restored full upfront CFG segment
    submission.
  - The default no-CFG fallback and single-segment CFG path are unchanged.
- Tests:
  - At this checkpoint, `tests/test_vllm_sts_cli.py` covered the then-default
    diagnostic flag, env-based disable behavior, and the run-log field for
    batched CFG chunks.
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_runtime.py tests/test_vllm_sts_cli.py tests/test_vllm_sts_requests.py -q`
    passed with `96 passed`.
  - `scripts/lint.sh` passed.
- Local evidence:
  - Fixture created with:
    `scripts/create-test-utterance.sh --basename four-sentence-context-question --text 'Please explain Python context managers in exactly four short sentences.'`
  - Staged CFG final-code run:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-084828.wav` and
    `.audex/runs/speech-output-vllm-20260709-084828.json`.
  - The final-code staged JSON reported `tts_cfg_enabled=true`,
    `tts_cfg_prime_first_segment=true`, `tts_concurrent_segments=true`,
    `reached_end_token=true`, `hit_max_tokens=false`,
    `first_audio_ready_seconds=4.654`, `stream_finished_seconds=64.654`,
    `audio_duration_seconds=14.98`, `audio_realtime_ratio=0.232`, and
    `codec_frames_per_second=11.43`.
  - The MLX-Audio STT intelligibility check for the staged WAV passed with ratio
    `0.9967`, transcribing the four expected sentences with only
    `try finally` vs `try-finally` punctuation normalization.
  - Upfront-batch A/B with the same fixture and staging disabled in code during
    the experiment wrote `.audex/runs/speech-output-vllm-20260709-084419.wav`
    and `.audex/runs/speech-output-vllm-20260709-084419.json`. It reported
    `first_audio_ready_seconds=3.557`, `stream_finished_seconds=78.798`,
    `audio_realtime_ratio=0.194`, and `codec_frames_per_second=9.569`.
- Current interpretation:
  - Staging tail CFG pairs did not reduce first-audio latency; it made it
    worse on this fixture. It did improve total stream throughput versus full
    upfront batching on the same four-sentence prompt at that point in the
    patch series. A later same-day rerun on the current code reversed this
    tradeoff; see "CFG Scheduling Default: Start All Segment Pairs Up Front"
    below.
  - The next target is not more Python-level request staging. The evidence now
    points back at vLLM Metal decode/sampling cost and scheduler behavior for
    small CFG batches.

### 2026-07-09 CFG Scheduling Default: Start All Segment Pairs Up Front

- Purpose: favor vLLM Metal continuous batching for multi-segment CFG TTS after
  rerunning the same fixture on the current code.
- Audex-Mac code:
  - `_vllm_tts_cfg_prime_first_segment_enabled()` now defaults to `False`.
  - `AudexAsyncVllmRuntime.stream_tts_cfg_segments_codec_frames(...)` now also
    defaults `prime_first_segment=False`.
  - `AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT=1` opts into the old staged
    segment-0-first experiment. `0` and `false` keep the new default behavior.
- Tests:
  - `tests/test_vllm_sts_cli.py` now covers the default-off env behavior and
    verifies multi-segment CFG run logs record `tts_cfg_prime_first_segment`
    as `false` by default.
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_runtime.py`
    passed with `84 passed`.
- Local evidence:
  - Full upfront CFG segment submission:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_TTS_CFG_PRIME_FIRST_SEGMENT=0 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-094146.wav`,
    `.audex/runs/speech-output-vllm-20260709-094146.json`, and
    `.audex/runs/speech-output-vllm-20260709-094146.eval.json`.
  - Upfront result:
    `tts_cfg_prime_first_segment=false`, `tts_concurrent_segments=true`,
    `first_audio_ready_seconds=5.39`, `stream_finished_seconds=50.013`,
    `audio_duration_seconds=14.52`, `audio_realtime_ratio=0.29`,
    `codec_frames_per_second=14.316`, `generated_codec_frame_count=716`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - Upfront MLX-Audio oracle:
    passed with ratio `0.9982425307557118`.
  - Prime-first comparison:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-094337.wav`,
    `.audex/runs/speech-output-vllm-20260709-094337.json`, and
    `.audex/runs/speech-output-vllm-20260709-094337.eval.json`.
  - Prime-first result:
    `tts_cfg_prime_first_segment=true`, `tts_concurrent_segments=true`,
    `first_audio_ready_seconds=3.65`, `stream_finished_seconds=65.016`,
    `audio_duration_seconds=14.84`, `audio_realtime_ratio=0.228`,
    `codec_frames_per_second=11.259`, `generated_codec_frame_count=732`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - Prime-first MLX-Audio oracle:
    passed with ratio `0.9257950530035336`.
- Interpretation:
  - Prime-first improved first-audio latency by `1.74s` in this run, but cost
    about `15s` of total stream time and reduced codec throughput from `14.316`
    to `11.259` frames/sec. The default now prioritizes continuous-batching
    throughput and longer-utterance stability. Keep the env flag for future
    first-audio experiments.

### 2026-07-09 CFG TTS-Window Decode Experiment: Correct but Slower

- Purpose: test whether CFG TTS can use the existing compact speech-window
  decode path, projecting only `<speechgen_end>` plus `<speechcodec_*>` logits
  instead of the full Audex vocabulary during decode.
- Audex-Mac code:
  - `_sample_tts_window_logits_from_params_if_supported(...)` can now sample a
    CFG pair from compact speech-window logits by blending the conditional and
    unconditional rows, sampling once, and expanding the sampled token back to
    both request slots.
  - `_try_batched_tts_window_decode(...)` can execute CFG pairs through that
    compact path, but only when `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1` is set.
  - CFG compact-window decode is disabled by default because the real 30B smoke
    below regressed throughput. The no-CFG TTS-window fast path remains
    available as before.
- Tests:
  - `tests/test_vllm_cfg.py` covers the default CFG rejection, opt-in CFG
    batched TTS-window decode, and sampler expansion of one CFG sampled token
    into both conditional/unconditional slots.
  - Focused validation:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py tests/test_vllm_diagnostics.py -q`
    passed with `80 passed`.
  - `scripts/lint.sh` passed.
- Local evidence:
  - Experimental CFG-window decode run:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-085734.wav` and
    `.audex/runs/speech-output-vllm-20260709-085734.json` before the env guard
    was added. This is the same compact-window CFG path now available via
    `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1`.
  - Native debug proved the compact path fired:
    `tts_window_decode_count=1438`, `tts_window_weight_cache_hits=396`,
    `tts_window_weight_cache_misses=1`, and `batched=1`.
  - The same timing line showed the regression source:
    `tts_window_batch_sample=27304.5/397`; compact-window sampling, not
    projection, dominated.
  - The speech JSON reported `first_audio_ready_seconds=5.682`,
    `stream_finished_seconds=109.209`, `audio_realtime_ratio=0.138`, and
    `codec_frames_per_second=6.831`, significantly worse than the previous
    staged CFG baseline.
  - The MLX-Audio STT intelligibility check still passed with ratio `0.9966`, so this
    was a performance rejection, not an audio-quality rejection.
  - Guarded default rerun without `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1` wrote
    `.audex/runs/speech-output-vllm-20260709-090142.wav` and
    `.audex/runs/speech-output-vllm-20260709-090142.json`; it reported
    `first_audio_ready_seconds=5.205`, `stream_finished_seconds=71.767`,
    `audio_realtime_ratio=0.197`, and `codec_frames_per_second=9.712`.
    the MLX-Audio oracle passed with ratio `0.9947`.
- Current interpretation:
  - Compact CFG-window decode is correct and useful for diagnostics, but it is
    not the current product path. The bottleneck moves into
    `tts_window_batch_sample`, so the next real optimization is a faster
    top-k/categorical sampler for the compact speech window or a lower-level
    Metal kernel, not merely avoiding the full LM head projection.

### 2026-07-09 CFG TTS-Window Diagnostics: Nonpaged Bottleneck Assessment

- Purpose: make the CFG-window regression self-identifying in structured
  diagnostics instead of relying on hand-reading EngineCore stderr.
- Audex-Mac code:
  - `audex_mac.vllm_diagnostics._assess_sts_timing(...)` now includes
    `latest_non_paged_decode` timing in `sts_timing_assessment`, including
    `nonpaged_decode_avg_ms`, `nonpaged_native_sample_ms_per_step`,
    `tts_window_decode_count`, `tts_window_weight_cache_hit_rate`,
    `nonpaged_native_detail_ms_per_step_by_category`, and
    `dominant_nonpaged_native_detail_category`.
  - When `tts_window_sample` or `tts_window_batch_sample` dominates the
    nonpaged native-detail timing, `likely_bottleneck` becomes
    `pending_graph_eval_during_tts_window_sampling`.
- Tests:
  - `tests/test_vllm_diagnostics.py` covers parsing
    `tts_window_batch_forward`, `tts_window_batch_project`, and
    `tts_window_batch_sample` native-detail timing from nonpaged decode logs.
  - It also covers the derived CFG-window sampling bottleneck assessment using
    the real slow-run shape: `tts_window_batch_sample=27304.5/397`.
- Interpretation:
  - The next optimization should target the compact-window top-k/categorical
    sample/eval path or a native Metal sampler kernel. If a future diagnostic
    still reports `pending_graph_eval_during_tts_window_sampling`, scheduler
    staging and Python request batching are not the primary bottleneck.

### 2026-07-09 CFG Sampler Benchmark Shape

- Purpose: make the sampler microbenchmark exercise the actual NVIDIA CFG TTS
  shape instead of only the no-CFG baseline.
- Audex-Mac code:
  - `scripts/bench_vllm_metal_sampler.py` now accepts `--cfg-pairs`,
    `--temperature`, `--top-k`, and `--cfg-scale`.
  - When `--cfg-pairs` is set, the benchmark builds conditional/unconditional
    row pairs, applies `_build_cfg_pair_sampling_plan(...)`, blends CFG logits
    through `_build_native_sample_logits(...)`, and reports both request rows
    and sampled rows.
  - The raw-window and pending-window benchmark paths both use the same CFG
    blend as runtime sampling before calling `_sample_random_tokens_mlx(...)`.
- Tests:
  - `tests/test_vllm_sampler_bench.py` verifies the NVIDIA-shaped benchmark
    case: `--cfg-pairs 2 --temperature 1.0 --top-k 80 --cfg-scale 3.0` creates
    four request rows, two sampled rows, paired output slots, uniform top-k 80,
    and `cfg_scale=3.0`.
  - The same test file verifies the no-CFG benchmark default still creates one
    sampled row per request row with top-k disabled.
- Local evidence:
  - Command:
    `PYTHONPATH="$PWD/.audex/vendor/vllm-metal" VLLM_LOGGING_LEVEL=ERROR TRANSFORMERS_VERBOSITY=error .audex/vendor/vllm-metal/.venv-vllm-metal/bin/python scripts/bench_vllm_metal_sampler.py --cfg-pairs 2 --temperature 1.0 --top-k 80 --cfg-scale 3.0 --iterations 5 --warmup 2 --json .audex/runs/vllm-metal-sampler-bench-20260709-cfg-topk.json`
  - Device: `Device(gpu, 0)`.
  - Results: raw CFG window sample averaged `0.472 ms`,
    pending full-vocab projection plus window sample averaged `13.929 ms`, and
    pending compact-window projection plus sample averaged `4.639 ms`.
- Interpretation:
  - The standalone CFG top-k/categorical sample is not the 30B runtime
    bottleneck by itself. The real run's `tts_window_batch_sample=27304.5/397`
    is more consistent with delayed graph evaluation or runtime scheduling
    around the integrated TTS-window sample/eval boundary. Keep the benchmark as
    a guard before attempting native Metal sampler work.

### 2026-07-09 TTS-Window Lazy Eval Split

- Purpose: separate MLX lazy graph costs from the misleading
  `tts_window_batch_sample` bucket during CFG compact-window diagnostics.
- Audex-Mac code:
  - Added `AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES=1` as a diagnostic-only
    switch. It only has an effect when `AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1` is
    also enabled.
  - When enabled, sequential TTS-window decode records
    `tts_window_forward_eval` and `tts_window_project_eval` after forcing
    `mx.eval(hidden_states)` and `mx.eval(logits)`.
  - Batched TTS-window decode records `tts_window_batch_forward_eval` and
    `tts_window_batch_project_eval` the same way.
- Tests:
  - `tests/test_vllm_cfg.py` verifies the batched window path calls `mx.eval`
    for hidden states and logits only when the sync env is enabled, and records
    the new native-detail timing categories.
- Interpretation:
  - Use this only for diagnostic attribution. It intentionally adds
    synchronization points and should not be enabled for product latency runs.
    A synced CFG-window run can now distinguish "actual sampler/top-k is slow"
    from "forward/project work was lazily forced inside the sample bucket."
  - 2026-07-09 follow-up run:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_VLLM_DEBUG_SYNC_TTS_WINDOW_STAGES=1 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-092544.json` and
    `.audex/runs/sts-turn-vllm-20260709-092719.json`.
  - Result:
    `stream_finished_seconds=94.85`, `first_audio_ready_seconds=5.556`,
    `audio_realtime_ratio=0.155`, `codec_frames_per_second=7.633`, and
    `tts_reached_end_token=true`.
  - Native detail:
    the synced checkpoint at 200 decode steps showed
    `tts_window_batch_forward_eval=7278.4/101` (about `72.06 ms/step`),
    `tts_window_batch_project_eval=1223.0/101` (about `12.11 ms/step`), and
    `tts_window_batch_sample=1347.8/101` (about `13.34 ms/step`). This corrects
    the earlier working theory: the compact-window CFG bottleneck is dominated
    by model forward eval or vLLM-Metal scheduling/cache behavior, not by the
    top-k/categorical sampler alone.
- Audex-Mac code:
  - `sts_timing_assessment` now reports
    `likely_bottleneck=model_forward_eval_during_tts_window_decode` when the
    synced `tts_window_forward_eval`/`tts_window_batch_forward_eval` bucket
    dominates, and
    `likely_bottleneck=projection_eval_during_tts_window_decode` when projection
    eval dominates.
- Tests:
  - `tests/test_vllm_diagnostics.py` covers parsing the synced eval categories
    and assessing the forward-eval-dominated compact-window CFG shape.

### 2026-07-09 mlx-audio-Inspired Adaptive Playback Prebuffer

- Purpose: reduce time to first audible output on longer/generated-in-chunks TTS
  while preserving underrun diagnostics.
- Attribution:
  - This change was inspired by the local MIT-licensed mlx-audio project at
    `Blaizzy/mlx-audio`, especially
    `mlx_audio/tts/audio_player.py` and the TTS docs under
    `docs/models/tts/`.
  - No mlx-audio source code was copied wholesale into Audex-Mac in this patch.
    The implementation remains Audex-Mac-specific and keeps our existing
    `sounddevice.RawOutputStream` playback, queue diagnostics, and PCM16 path.
- Audex-Mac code:
  - `_ContinuousPcmPlayer` now tracks PCM arrival rate with a short EMA window
    and uses that to adapt the initial prebuffer target.
  - The configured prebuffer remains a ceiling; the adaptive target cannot drop
    below `DEFAULT_PLAYBACK_ADAPTIVE_MIN_PREBUFFER_SECONDS`.
  - Playback diagnostics now include `adaptive_prebuffer`,
    `prebuffer_target_seconds`, `actual_prebuffer_seconds`, and
    `arrival_rate_audio_realtime_ratio`.
- Why this matches the mlx-audio lesson:
  - mlx-audio starts playback after enough actual generated audio has arrived
    relative to observed arrival rate, rather than using only a static
    text-shape-derived delay. Audex-Mac now applies the same idea to the shared
    continuous PCM player without changing model generation, CFG sampling, or
    decoder output.
- Tests:
  - `tests/test_sts_cli.py` covers adaptive target capping, minimum floor, and
    the ability to disable adaptive prebuffering.

### 2026-07-09 Single-Segment CFG Delta Streaming

- Purpose: remove avoidable Python-side token-history copying from the
  single-segment CFG TTS path without changing sampler settings, CFG math, token
  masks, or decoder flow.
- Audex-Mac code:
  - `AudexAsyncVllmRuntime.stream_tts_cfg_codec_frames(...)` now requests
    vLLM `output_kind="DELTA"` for both conditional and unconditional requests,
    calls `stream_many(..., include_cumulative_token_ids=False)`, and keeps the
    conditional generated-token history locally.
  - Intermediate TTS events now carry only newly generated codec frames. The
    full conditional token history is emitted on the final event for logs.
- Why this matters:
  - The segmented CFG path already used delta output. The single-segment path
    still consumed cumulative token IDs and re-extracted the whole codec-frame
    sequence on every delta, which is O(n^2) host overhead for longer TTS
    utterances. This does not solve the 30B Metal forward-eval bottleneck, but
    it removes unnecessary CPU work and makes both CFG TTS paths consistent.
- Tests:
  - `tests/test_vllm_runtime.py` verifies the CFG requests use
    `output_kind="DELTA"` and only emit the full generated-token history on the
    final event.
- Validation:
  - Real vLLM Metal run:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`.
  - Artifacts:
    `.audex/runs/speech-output-vllm-20260709-093617.wav`,
    `.audex/runs/speech-output-vllm-20260709-093617.json`, and
    `.audex/runs/sts-turn-vllm-20260709-093701.json`.
  - Speech timing:
    `first_audio_ready_seconds=1.926`, `stream_finished_seconds=43.48`,
    `audio_duration_seconds=8.62`, `audio_realtime_ratio=0.198`,
    `codec_frames_per_second=9.913`, `generated_codec_frame_count=431`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - MLX-Audio oracle:
    `.venv/bin/python scripts/evaluate_tts_wav.py .audex/runs/speech-output-vllm-20260709-093617.wav --expected-from-speech-log .audex/runs/speech-output-vllm-20260709-093617.json --json-out .audex/runs/speech-output-vllm-20260709-093617.eval.json --min-ratio 0.55`
    passed with ratio `0.9899665551839465`. The oracle heard "context" as
    "convex" once, but otherwise matched the intended TTS text.

### 2026-07-09 TTS WAV Oracle Sidecar Output

- Purpose: make TTS intelligibility checks reproducible without manually
  copying expected text from the speech-output JSON.
- Audex-Mac code:
  - `scripts/evaluate_tts_wav.py` now accepts
    `--expected-from-speech-log <speech-output.json>` and uses its
    `tts_segment_texts` as the expected transcript.
  - The same helper accepts `--json-out <path>` to write the full oracle
    payload beside generated run artifacts.
- Tests:
  - `tests/test_evaluate_tts_wav.py` verifies segment-order extraction,
    explicit expected-text precedence, and expected-source validation.

### 2026-07-09 TTS WAV Oracle Repetition Gate

- Purpose: catch the audible phrase-loop failure mode in generated TTS artifacts
  instead of relying only on transcript similarity.
- Audex-Mac code:
  - `scripts/evaluate_tts_wav.py` now computes a consecutive n-gram repetition
    summary over the STT oracle transcript.
  - The oracle payload includes a `repetition` object with `excessive`,
    `max_consecutive_repetitions`, `max_allowed_repetitions`, `ngram_size`, and
    `ngram`.
  - The evaluator now fails if transcript similarity passes but the generated
    audio transcript contains an excessive consecutive phrase loop.
  - Defaults are conservative: `--repetition-ngram-size 4` and
    `--max-consecutive-repetitions 3`.
- Validation:
  - Re-ran the latest upfront-batched CFG artifact:
    `.venv/bin/python scripts/evaluate_tts_wav.py .audex/runs/speech-output-vllm-20260709-094146.wav --expected-from-speech-log .audex/runs/speech-output-vllm-20260709-094146.json --json-out .audex/runs/speech-output-vllm-20260709-094146.eval.json --min-ratio 0.55`
  - Result: passed with ratio `0.9982425307557118` and
    `repetition.excessive=false`.
- Tests:
  - `tests/test_evaluate_tts_wav.py` covers normal reused words, a pathological
    repeated phrase loop, and the threshold boundary.

### 2026-07-09 CFG Playback Prebuffer for Slow Metal TTS

- Purpose: prevent audible dropouts when CFG TTS generation is much slower than
  realtime on 30B vLLM Metal.
- Problem evidence:
  - Playback-enabled CFG fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-095223.json`.
  - That run was intelligible, but reported `device_underflow_count=8`,
    `queue_underrun_count=8`, `queue_overrun_count=0`,
    `playback_prebuffer_seconds=0.8`, `actual_prebuffer_seconds=1.32`, and
    `arrival_rate_audio_realtime_ratio=0.163`.
- Audex-Mac code:
  - `_ContinuousPcmPlayer` adaptive prebuffer now grows when observed audio
    arrival is slower than realtime, bounded by
    `DEFAULT_PLAYBACK_ADAPTIVE_MAX_PREBUFFER_SECONDS=8.0`.
  - vLLM CFG playback now uses `DEFAULT_VLLM_CFG_PLAYBACK_PREBUFFER_SECONDS=8.0`
    while no-CFG keeps `DEFAULT_VLLM_PLAYBACK_PREBUFFER_SECONDS=0.8`.
  - Queue overrun diagnostics now ignore intentional initial prebuffer and
    treat overrun as queue growth beyond the configured prebuffer plus the
    normal overrun margin after playback starts. `queue_high_water_seconds`
    still records the actual buffer size.
- Validation:
  - Final playback-enabled CFG fixture wrote
    `.audex/runs/speech-output-vllm-20260709-100701.wav`,
    `.audex/runs/speech-output-vllm-20260709-100701.json`, and
    `.audex/runs/speech-output-vllm-20260709-100701.eval.json`.
  - It reported `playback_prebuffer_seconds=8.0`,
    `first_audio_ready_seconds=5.005`,
    `first_playback_started_seconds=63.848`,
    `first_playback_after_audio_seconds=58.843`,
    `device_underflow_count=0`, `queue_underrun_count=0`,
    `queue_overrun_count=0`, `queue_high_water_seconds=8.04`,
    `playback_written_audio_seconds=8.62`, `reached_end_token=true`, and
    `hit_max_tokens=false`.
  - `scripts/evaluate_tts_wav.py` passed with ratio `0.9899665551839465` and
    `repetition.excessive=false`.
- Tradeoff:
  - This deliberately favors smooth playback over early playback for slow CFG
    TTS. It does not make CFG near-realtime; it prevents the speaker path from
    making slow generation sound broken while lower-level vLLM Metal throughput
    work continues.
- Tests:
  - `tests/test_sts_cli.py` covers slow-arrival adaptive prebuffer growth,
    max cap, minimum floor, disabled adaptive prebuffer, and overrun
    diagnostics around intentional prebuffering.
  - `tests/test_vllm_sts_cli.py` verifies no-CFG short answers keep the 0.8s
    prebuffer while CFG short answers use the 8.0s prebuffer.

### 2026-07-09 CFG Research Checkpoint: Measure Nonpaged Cache Copy Cost

- Purpose: turn the current vLLM Metal CFG throughput hypothesis into
  structured evidence before changing cache lifetime or decode scheduling.
- Upstream symbols touched by monkey patch:
  - `vllm_metal.v1.model_runner._merge_kv_caches`
  - `vllm_metal.v1.model_runner._extract_kv_cache`
- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_cfg._patch_model_runner_cache_timing(...)`
    wraps the imported `model_runner` helper symbols, preserving their original
    behavior and recording debug-only native detail categories:
    `nonpaged_kv_cache_merge` and `nonpaged_kv_cache_extract`.
  - `audex_mac.vllm_diagnostics._assess_sts_timing(...)` now reports
    `likely_bottleneck=nonpaged_kv_cache_copy` when either cache-copy category
    dominates, and reports `likely_bottleneck=pending_graph_eval_during_sampling`
    when the default non-window path is dominated by `sample_eval`.
- Guard/reapply notes:
  - The wrappers are sentinel-guarded with `_audex_mac_cache_timing_patch`.
  - If upstream removes or renames the imported model-runner helper symbols, the
    patch should remain a no-op and the diagnostic will simply omit these
    categories until the seam is updated.
  - This is measurement only: it does not alter CFG math, sampler parameters,
    token masks, scheduling, KV contents, or decoder behavior.
- GPU-visible validation:
  - Command:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --diagnose-vllm-native-sampling-debug --diagnose-vllm-tts-batch-size 2 --diagnose-vllm-tts-batch-max-tokens 64 --diagnose-vllm-tts-batch-text 'Please explain Python context managers in two concise sentences.'`
  - Report:
    `.audex/runs/vllm-metal-diagnostic-20260709-101507.json`
  - Result: `ready=True`, `Device(gpu, 0)`, `request_count=4`,
    `codec_frames_per_second=16.708`, `hit_max_token_count=2`.
  - Latest nonpaged timing at `count=50`: `avg_ms=114.4`,
    `sample_eval=2638.6/52` (~50.742 ms/step),
    `nonpaged_kv_cache_extract=3666.6/192` (~19.097 ms/extract), and
    `nonpaged_kv_cache_merge=15.5/48` (~0.323 ms/merge).
  - Interpretation: cache extraction is a real cost, but this short default
    CFG batch is still dominated by MLX graph work forced at the sampling
    boundary. Persistent batched cache remains a plausible improvement, but
    async/nonpaged decode pipelining is still the larger measured target.
- Tests:
  - `tests/test_vllm_diagnostics.py` parses the new timing categories and
    covers both cache-copy-dominant and sampling-eval-dominant assessments.

### 2026-07-09 CFG Speed Checkpoint: Async-Submit Nonpaged Decode Logits

- Purpose: reduce the default nonpaged CFG decode stall where the MLX forward
  graph was first forced inside native sampling's `mx.eval(tokens)` boundary.
- Upstream symbol touched by monkey patch:
  - `vllm_metal.v1.model_runner.MetalModelRunner._batched_decode`
- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_cfg._try_batched_decode_with_async_eval(...)`
    mirrors the pinned upstream batched nonpaged decode implementation, but
    calls `mx.async_eval(next_token_logits)` immediately after extracting the
    decode logits and before building `SamplingBatch`.
  - The patch is reached from the existing batched decode wrapper after the
    compact TTS-window path declines the request, so the default full-vocab CFG
    path benefits without enabling the experimental CFG TTS-window decoder.
  - `AUDEX_VLLM_NONPAGED_ASYNC_EVAL=0` disables this path and falls back to the
    pinned upstream `_batched_decode` behavior.
  - Debug timing adds `nonpaged_decode_logits_async_submit`; diagnostics report
    `likely_bottleneck=nonpaged_async_graph_submit` if that category dominates.
- Guard/reapply notes:
  - This does not change CFG math, NVIDIA sampler settings, token masks, request
    scheduling, or decoder behavior.
  - Re-check against upstream if `_batched_decode` changes its `SamplingBatch`
    construction, generator handling, cache merge/extract protocol, or
    `_SamplingResult` shape.
  - The env kill switch should remain until a longer audible CFG validation
    proves no regressions in real STS turns.
- GPU-visible validation:
  - Async enabled command:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --diagnose-vllm-native-sampling-debug --diagnose-vllm-tts-batch-size 2 --diagnose-vllm-tts-batch-max-tokens 64 --diagnose-vllm-tts-batch-text 'Please explain Python context managers in two concise sentences.'`
  - Report:
    `.audex/runs/vllm-metal-diagnostic-20260709-102231.json`
  - Result: `ready=True`, `Device(gpu, 0)`, `codec_frames_per_second=19.119`,
    `nonpaged_decode_avg_ms=95.2`, `nonpaged_native_sample_ms_per_step=22.47`,
    `sample_eval=1117.7/52` (~21.494 ms/step),
    `nonpaged_decode_logits_async_submit=1492.7/48` (~31.098 ms/submit), and
    `nonpaged_kv_cache_extract=2726.9/192` (~14.203 ms/extract).
  - Same-checkout kill-switch comparison:
    `AUDEX_VLLM_NONPAGED_ASYNC_EVAL=0 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --diagnose-vllm-native-sampling-debug --diagnose-vllm-tts-batch-size 2 --diagnose-vllm-tts-batch-max-tokens 64 --diagnose-vllm-tts-batch-text 'Please explain Python context managers in two concise sentences.'`
    wrote `.audex/runs/vllm-metal-diagnostic-20260709-102355.json` and reported
    `codec_frames_per_second=17.028`, `nonpaged_decode_avg_ms=98.1`,
    `nonpaged_native_sample_ms_per_step=52.192`, and
    `sample_eval=2605.2/52` (~50.1 ms/step).
  - Interpretation: early async submission moves much of the forced graph work
    out of the sampling boundary and improves this short CFG batch by about
    12% over the kill-switch run, but it is not sufficient for near-realtime
    CFG TTS on 30B.
- Tests:
  - `tests/test_vllm_cfg.py` covers async submit, state/cache/token updates,
    generator preservation, debug timing, and the env kill switch.
  - `tests/test_vllm_diagnostics.py` covers the new async-submit bottleneck
    assessment.

### 2026-07-09 CFG Speed Checkpoint: Reuse Stable Nonpaged Batch KV Cache

- Purpose: remove the repeated per-step `_extract_kv_cache(...)` cost for
  stable nonpaged CFG decode batches. The previous async-submit checkpoint
  showed extraction remained a meaningful steady-state cost after the sampling
  boundary improved.
- Upstream symbols touched by monkey patch:
  - `vllm_metal.v1.model_runner.MetalModelRunner._batched_decode`
  - `vllm_metal.v1.model_runner.MetalModelRunner._cleanup_finished_requests`
- Audex-Mac code:
  - `audex_mac.patches.vllm_metal_cfg._nonpaged_batch_cache_for_decode(...)`
    stores the merged batch cache on the runner when the exact request ID tuple
    and request-state object identities stay stable across decode steps.
  - Per-request cache extraction is skipped while the batch is stable; token IDs
    and generated-token counts are still updated every step.
  - `_flush_persistent_nonpaged_batch_cache(...)` extracts the latest per-request
    caches before batch membership changes or before finished request states are
    evicted by `_cleanup_finished_requests`.
  - `AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE=0` disables this optimization.
  - Nonpaged timing now reports `nonpaged_persistent_cache_hits`,
    `nonpaged_persistent_cache_misses`, `nonpaged_persistent_cache_flushes`, and
    derived `nonpaged_persistent_cache_hit_rate`.
- Guard/reapply notes:
  - This patch assumes a stable nonpaged batch cache remains valid while the
    scheduler presents the same request IDs in the same order. If upstream
    changes nonpaged cache mutation semantics or request-state ownership, recheck
    this patch before bumping the vLLM Metal pin.
  - The compact TTS-window decode experiment still performs per-step extraction;
    this checkpoint only changes the default full-vocab nonpaged decode path.
  - This does not change CFG math, NVIDIA sampler settings, token masks, request
    scheduling, or decoder behavior.
- GPU-visible validation:
  - Persistent cache enabled:
    `AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --diagnose-vllm-native-sampling-debug --diagnose-vllm-tts-batch-size 2 --diagnose-vllm-tts-batch-max-tokens 64 --diagnose-vllm-tts-batch-text 'Please explain Python context managers in two concise sentences.'`
    wrote `.audex/runs/vllm-metal-diagnostic-20260709-103256.json`.
  - Result: `ready=True`, `Device(gpu, 0)`, `codec_frames_per_second=40.468`,
    `elapsed_seconds=3.163`, `nonpaged_decode_avg_ms=36.9`,
    `nonpaged_persistent_cache_hits=47`,
    `nonpaged_persistent_cache_misses=1`, `nonpaged_persistent_cache_flushes=0`,
    `nonpaged_persistent_cache_hit_rate=0.979`, and no steady-state
    `nonpaged_kv_cache_extract` timing category.
  - Same-checkout kill-switch comparison:
    `AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE=0 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --diagnose-vllm-native-sampling-debug --diagnose-vllm-tts-batch-size 2 --diagnose-vllm-tts-batch-max-tokens 64 --diagnose-vllm-tts-batch-text 'Please explain Python context managers in two concise sentences.'`
    wrote `.audex/runs/vllm-metal-diagnostic-20260709-103417.json` and reported
    `codec_frames_per_second=17.133`, `elapsed_seconds=7.471`,
    `nonpaged_decode_avg_ms=106.1`, and
    `nonpaged_kv_cache_extract=17.174 ms/extract`.
  - CFG STS fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-103520.wav` and
    `.audex/runs/sts-turn-vllm-20260709-103538.json`. It reported
    `first_audio_ready_seconds=1.445`, `stream_finished_seconds=17.787`,
    `codec_frames_per_second=24.231`, `audio_realtime_ratio=0.485`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - The CFG WAV passed the MLX-Audio oracle:
    `.audex/runs/speech-output-vllm-20260709-103520.eval.json` reported
    `ratio=0.9164086687306502` and `repetition.excessive=false`.
  - No-CFG fallback fixture using the same input wrote
    `.audex/runs/speech-output-vllm-20260709-103720.wav` and
    `.audex/runs/sts-turn-vllm-20260709-103750.json`; its oracle
    `.audex/runs/speech-output-vllm-20260709-103720.eval.json` passed with
    `ratio=0.9259259259259259` and `repetition.excessive=false`.
- Tests:
  - `tests/test_vllm_cfg.py` covers stable batch-cache reuse, delayed flush,
    state/cache/token updates, and the persistent-cache kill switch path.
  - `tests/test_vllm_diagnostics.py` covers parsing the persistent-cache
    counters and deriving the hit rate.

### 2026-07-09 CFG Scheduling Checkpoint: Sentence-Level Static CFG Chunks

- Purpose: feed vLLM Metal's CFG pair batcher more consistently during product
  CLI TTS. The research note showed continuous batching is under-occupied when
  normal answers become only one or two large TTS chunks; CFG quality still
  depends on paired requests, so the smallest product-side scheduling lever is
  to split static CFG TTS at sentence boundaries while leaving no-CFG chunking
  unchanged.
- Audex-Mac code:
  - `audex_mac.vllm_sts_cli.DEFAULT_VLLM_CFG_TTS_SENTENCES_PER_CHUNK`
    separates the CFG static chunk policy from the existing no-CFG
    three-sentence policy.
  - `audex_mac.vllm_sts_cli.split_cfg_spoken_tts_chunks(...)` delegates to the
    existing spoken chunker with a one-sentence CFG limit.
  - `VllmSpeechToSpeechSession.generate_speech_output_streaming_from_async_runtime(...)`
    chooses the CFG chunker only when `AUDEX_VLLM_TTS_CFG=1` and the TTS input
    is static. Streaming text-to-TTS and the default no-CFG fallback keep their
    existing chunk behavior.
- Guard/reapply notes:
  - This is product scheduling only. It does not touch vLLM Metal internals,
    CFG math, NVIDIA sampler settings, token masks, speech decoder chunking, or
    playback prebuffer policy.
  - If upstream vLLM Metal gains a stronger paired-CFG scheduler or Audex-Mac
    adds adaptive CFG segment sizing, keep the no-CFG and CFG chunk policies
    separate so the intelligible no-CFG fallback remains a clean control.
- GPU-visible validation:
  - CFG STS fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-104424.wav`,
    `.audex/runs/speech-output-vllm-20260709-104424.json`, and
    `.audex/runs/sts-turn-vllm-20260709-104439.json`.
  - Result: `tts_cfg_enabled=true`, `tts_concurrent_segments=true`,
    `tts_observed_segments=2`, `first_audio_ready_seconds=2.297`,
    `stream_finished_seconds=15.058`, `codec_frames_per_second=27.427`,
    `audio_realtime_ratio=0.562`, `generated_codec_frame_count=413`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - Compared with the previous persistent-cache CFG fixture, throughput moved
    from `24.231` to `27.427 codec_frames_per_second` and total stream time
    moved from `17.787s` to `15.058s`; first audio regressed from `1.445s` to
    `2.297s` because the static batched path waits for ordered segment
    readiness before first decoded PCM.
  - The new CFG WAV passed the MLX-Audio oracle:
    `.audex/runs/speech-output-vllm-20260709-104424.eval.json` reported
    `ratio=0.9259259259259259` and `repetition.excessive=false`.
- Tests:
  - `tests/test_vllm_sts_cli.py` covers the CFG env switch, sentence-level CFG
    static chunks, and that static CFG TTS submits each sentence as a concurrent
    segment with per-segment decoder flush/reset accounting.

### 2026-07-09 Decoder Hot-Loop Checkpoint: Avoid MLX Scalar Diagnostics

- Purpose: reduce decoder/playback-side stalls while vLLM Metal is generating
  CFG speech tokens. The streaming decoder path previously computed
  `mx.max(...).item()` and `mx.isfinite(...).item()` for every decoded waveform
  chunk before packing PCM. Those scalar reads force MLX work to synchronize on
  the host and compete with the same Metal device used by the vLLM engine.
- Audex-Mac code:
  - `audex_mac.speech_output.pcm16_bytes_peak_abs(...)` derives the normalized
    peak from already-packed PCM16 bytes on the CPU.
  - `VllmSpeechToSpeechSession.generate_speech_output_streaming_from_async_runtime(...)`
    updates `peak_abs` from packed PCM bytes and no longer calls MLX scalar
    reduction diagnostics in the hot streaming waveform path.
  - The MLX fast PCM pack path records `finite=true` when packing succeeds;
    the Python fallback path still checks the materialized sample values for
    NaN before writing PCM.
- Guard/reapply notes:
  - This does not change vLLM requests, CFG math, NVIDIA sampler settings,
    speech-token masks, decoder frames, audio bytes, chunking, or playback
    prebuffer policy. It only changes how streaming output diagnostics are
    collected.
  - If stronger waveform validation is needed, add an explicit diagnostic mode
    rather than restoring per-chunk MLX scalar reads to the default playback
    loop.
- GPU-visible validation:
  - CFG STS fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-105115.wav`,
    `.audex/runs/speech-output-vllm-20260709-105115.json`, and
    `.audex/runs/sts-turn-vllm-20260709-105128.json`.
  - Result: `tts_cfg_enabled=true`, `tts_concurrent_segments=true`,
    `tts_observed_segments=2`, `first_audio_ready_seconds=1.971`,
    `stream_finished_seconds=12.717`, `codec_frames_per_second=32.476`,
    `audio_realtime_ratio=0.665`, `generated_codec_frame_count=413`,
    `pcm_pack_fast_path_count=10`, `pcm_pack_fallback_count=1`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - Compared with the previous static CFG sentence-chunk fixture, throughput
    moved from `27.427` to `32.476 codec_frames_per_second`, total stream time
    moved from `15.058s` to `12.717s`, and first audio moved from `2.297s` to
    `1.971s`.
  - The CFG WAV passed the MLX-Audio oracle:
    `.audex/runs/speech-output-vllm-20260709-105115.eval.json` reported
    `ratio=1.0` and `repetition.excessive=false`.
  - No-CFG fallback fixture using the same input wrote
    `.audex/runs/speech-output-vllm-20260709-105252.wav` and
    `.audex/runs/sts-turn-vllm-20260709-105325.json`; its oracle
    `.audex/runs/speech-output-vllm-20260709-105252.eval.json` passed with
    `ratio=1.0` and `repetition.excessive=false`.
- Tests:
  - `tests/test_speech_output.py` covers `pcm16_bytes_peak_abs(...)`.
  - `tests/test_vllm_sts_cli.py` exercises the streaming output log fields
    through the fake async vLLM/decoder path.

### 2026-07-09 CFG TTS-Window Diagnostic: Reuse Persistent Batch Cache

- Purpose: retest the compact CFG speech-window decoder after the stable
  nonpaged batch-cache checkpoint. The earlier compact-window CFG experiment
  was correct but slower, and the suspected reason was that it still paid
  per-step KV merge/extract cost while the default full-vocab path no longer
  did.
- Audex-Mac code:
  - `_try_batched_tts_window_decode(...)` now calls
    `_nonpaged_batch_cache_for_decode(...)` instead of directly merging request
    caches every step.
  - When `AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE` remains enabled, the
    compact-window path updates token IDs/generated-token counts but skips
    per-request cache extraction until the existing flush hook runs.
  - `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1` is still required for CFG compact
    speech-window decode. The default CFG product path remains the full-vocab
    nonpaged decode path because it is faster in the validation below.
- Guard/reapply notes:
  - This does not change CFG math, NVIDIA sampler settings, token masks,
    scheduler behavior, default product routing, no-CFG routing, or decoder
    behavior.
  - If upstream changes nonpaged cache ownership, recheck both
    `_try_batched_decode_with_async_eval(...)` and
    `_try_batched_tts_window_decode(...)`; they now intentionally share the
    persistent batch-cache helper.
- GPU-visible validation:
  - Opt-in compact-window CFG fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_CFG_TTS_WINDOW_DECODE=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-110815.wav` and
    `.audex/runs/sts-turn-vllm-20260709-110837.json`.
  - Result: `first_audio_ready_seconds=2.016`,
    `stream_finished_seconds=21.19`, `codec_frames_per_second=19.773`,
    `audio_realtime_ratio=0.405`, `generated_codec_frame_count=419`,
    `reached_end_token=true`, and `hit_max_tokens=false`.
  - The WAV passed the MLX-Audio oracle:
    `.audex/runs/speech-output-vllm-20260709-110815.eval.json` reported
    `ratio=1.0` and `repetition.excessive=false`.
  - Interpretation: persistent cache reuse makes the diagnostic path structurally
    cleaner, but compact CFG speech-window decode is still slower than the
    current default full-vocab CFG path
    (`.audex/runs/speech-output-vllm-20260709-105115.json` reported
    `codec_frames_per_second=32.476`). Keep
    `AUDEX_VLLM_CFG_TTS_WINDOW_DECODE` opt-in.
- Tests:
  - `tests/test_vllm_cfg.py` covers the old extraction path with
    `AUDEX_VLLM_NONPAGED_PERSISTENT_BATCH_CACHE=0` and the new persistent-cache
    compact-window path with one merge, no per-step extracts, and stable cache
    reuse across decode steps.

## CFG Playback Prebuffer Reduction for Current Fast Path

Added on 2026-07-09 after the no-CFG scalar-sync fix made the product CFG path
substantially faster than the earlier slow CFG playback experiments.

- Upstream repository/commit: not an upstream vLLM Metal patch; this is
  Audex-Mac CLI playback policy around vLLM Metal CFG output.
- Audex-Mac files:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
- Purpose:
  - The earlier stable CFG playback checkpoint used
    `DEFAULT_VLLM_CFG_PLAYBACK_PREBUFFER_SECONDS=8.0` because CFG generation was
    only about `0.12x` realtime at that point, and shorter buffers produced
    audible underruns.
  - Current CFG generation is faster, so the fixed 8s minimum became a
    user-visible latency bug: even when first decoded audio was ready near 2s,
    audible playback could wait many more seconds before starting.
- Change:
  - CFG short answers, including ordinary one- or two-segment sentence chunks,
    now use a configured 2.0s playback prebuffer.
  - Three-or-more CFG chunks still use a 4.0s guard.
  - Very long CFG utterances still use a 5.0s guard.
  - The existing `_ContinuousPcmPlayer` adaptive prebuffer remains enabled and
    can expand the actual initial buffer when audio arrives slower than
    realtime.
- Guard/reapply notes:
  - This does not change CFG generation, sampler settings, token masks,
    decoder chunking, waveform packing, or no-CFG playback.
  - If CFG generation throughput regresses again, compare
    `playback_diagnostics.queue_underrun_count`,
    `device_underflow_count`, `prebuffer_target_seconds`, and
    `actual_prebuffer_seconds` before raising this default back up.
- Validation:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `52 passed`.
  - Adaptive player tests:
    `.venv/bin/python -m pytest tests/test_sts_cli.py::test_continuous_pcm_player_adaptive_prebuffer_expands_for_slow_arrival tests/test_sts_cli.py::test_continuous_pcm_player_adaptive_prebuffer_keeps_configured_realtime_target tests/test_sts_cli.py::test_continuous_pcm_player_adaptive_prebuffer_caps_very_slow_arrival -q`
    reported `3 passed`.
  - Baseline comparison with the same 30B CFG playback fixture and a 3.0s
    interim prebuffer wrote `.audex/runs/speech-output-vllm-20260709-112120.json`
    and reported `first_audio_ready_seconds=2.477`,
    `first_playback_started_seconds=13.517`,
    `first_playback_after_audio_seconds=11.04`,
    `playback_prebuffer_seconds=3.0`,
    `actual_prebuffer_seconds=6.22`,
    `queue_underrun_count=0`, and `device_underflow_count=0`.
  - Final 2.0s short-answer CFG playback fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-112349.wav` and
    `.audex/runs/speech-output-vllm-20260709-112349.json`.
  - Result: `playback_prebuffer_seconds=2.0`,
    `first_audio_ready_seconds=2.562`,
    `first_playback_started_seconds=10.084`,
    `first_playback_after_audio_seconds=7.522`,
    `actual_prebuffer_seconds=4.3`, `prebuffer_target_seconds=3.615`,
    `queue_underrun_count=0`, `device_underflow_count=0`,
    `stream_finished_seconds=19.736`, `codec_frames_per_second=20.926`,
    `audio_realtime_ratio=0.429`, `reached_end_token=true`, and
    `hit_max_tokens=false`.
  - The generated WAV passed the MLX-Audio oracle:
    `.venv/bin/python scripts/evaluate_tts_wav.py .audex/runs/speech-output-vllm-20260709-112349.wav --expected-from-speech-log .audex/runs/speech-output-vllm-20260709-112349.json --json-out .audex/runs/speech-output-vllm-20260709-112349.eval.json`
    reported `ratio=1.0` and `repetition.excessive=false`.

## Nonpaged CFG Async-Eval Target Diagnostic

Added on 2026-07-09 to keep the CFG throughput investigation reproducible.

- Upstream repository/commit: `https://github.com/vllm-project/vllm-metal` at
  pinned commit `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`.
- Upstream file/symbol touched:
  - Audex-Mac monkey patch around `vllm_metal.v1.model_runner._batched_decode`
  - Audex-Mac monkey patch around `vllm_metal.v1.sampling_batch.sample_from_logits`
- Audex-Mac files:
  - `audex_mac/patches/vllm_metal_cfg.py`
  - `tests/test_vllm_cfg.py`
- Purpose:
  - Test whether async-evaluating the smaller CFG-blended speech-token sample
    window is faster than the current default async-evaluation of full
    `next_token_logits`.
  - Preserve the current default unless evidence proves a better target.
- Change:
  - Added optional `AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET`.
  - Default / `logits`: current product behavior, async-submit full
    `next_token_logits` before native sampling.
  - `sample_logits`: skip the full-logits async submit and async-submit the
    native sampler's CFG-blended sample logits instead.
  - `none` / `off`: skip the async submit while still using the patched
    nonpaged decode path.
- Guard/reapply notes:
  - Default behavior is unchanged.
  - This does not change CFG math, NVIDIA sampler settings, speech-token masks,
    scheduler behavior, persistent batch-cache behavior, decoder behavior, or
    no-CFG fallback routing.
  - If upstream vLLM Metal changes where logits are extracted or sampled,
    recheck `_try_batched_decode_with_async_eval(...)` and
    `_sample_native_mlx_if_supported(...)`.
- Validation:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q` reported
    `47 passed`.
  - Ruff:
    `.venv/bin/python -m ruff check audex_mac/patches/vllm_metal_cfg.py tests/test_vllm_cfg.py`
    reported `All checks passed`.
  - Default full-logits async baseline remains the best known no-play CFG
    fixture:
    `.audex/runs/speech-output-vllm-20260709-105115.json` reported
    `first_audio_ready_seconds=1.971`, `stream_finished_seconds=12.717`,
    `codec_frames_per_second=32.476`, and `audio_realtime_ratio=0.665`.
  - Sample-logits async diagnostic:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET=sample_logits AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-113053.json` and reported
    `first_audio_ready_seconds=2.069`, `stream_finished_seconds=13.473`,
    `codec_frames_per_second=30.654`, and `audio_realtime_ratio=0.628`.
    Native timing moved work into `native_sample_logits_async_submit`; it did
    not beat the default.
  - No-async diagnostic:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_NONPAGED_ASYNC_EVAL_TARGET=none AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 64 --speech-max-tokens 768`
    wrote `.audex/runs/speech-output-vllm-20260709-113256.json` and reported
    `first_audio_ready_seconds=3.137`, `stream_finished_seconds=16.293`,
    `codec_frames_per_second=25.348`, and `audio_realtime_ratio=0.519`.
  - Interpretation: keep the product default at full-logits async submit.
    The next throughput target is not merely changing the async-eval target;
    it is reducing or pipelining the model-forward / sampling-bound graph
    evaluation itself.

## 2026-07-09 - Explicit CFG Segment Probe and 30B Concurrency Sweep

- Files patched:
  - `scripts/probe_vllm_tts_decode.py`
- What changed:
  - Added `--cfg-segment`, repeatable, to submit explicit TTS CFG segment
    pairs directly through the async vLLM Metal runtime.
  - The diagnostic now records runtime CFG wiring in its JSON payload:
    `cfg_enabled`, `cfg_scale`, `max_num_seqs`, `max_num_batched_tokens`, and
    `max_model_len`.
  - The script now adds the repository root to `sys.path`, matching the other
    standalone benchmark scripts, so it can run from a plain checkout path.
- Why:
  - The earlier one-at-a-time concurrency tests could not answer whether 30B
    CFG throughput is capped by `max_num_seqs`, vLLM scheduler admission, or
    nonpaged KV/SSM cache copy overhead.
  - A first attempt at an 8-segment probe accidentally ran with
    `cfg_enabled=false` and `cfg_scale=0.0`; it reported about 57 aggregate
    codec fps, but that is not valid CFG evidence. Future probes must inspect
    the JSON `runtime` block before drawing CFG conclusions.
- Validation:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q` reported
    `47 passed`.
  - Ruff:
    `.venv/bin/python -m ruff check scripts/probe_vllm_tts_decode.py`
    reported `All checks passed`.
  - False/no-CFG diagnostic, included only as a guardrail:
    `.audex/runs/tts-probe-vllm-20260709-114941.json` reported
    `cfg_enabled=false`, `cfg_scale=0.0`, `max_num_seqs=null`, and
    `1435 codec frames / 45.740s = 31.4 fps`; do not use this as CFG data.
  - Valid 30B CFG sweep with 12 explicit segments and NVIDIA CFG scale:
    - `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_CFG_MAX_NUM_SEQS=8 ... scripts/probe_vllm_tts_decode.py ...`
      wrote `.audex/runs/tts-probe-vllm-20260709-115109.json`:
      `cfg_enabled=true`, `cfg_scale=3.0`, `max_num_seqs=8`,
      `max_num_batched_tokens=8192`, `1446 frames / 35.171s = 41.113 fps`.
    - `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16` wrote
      `.audex/runs/tts-probe-vllm-20260709-115229.json`:
      `1446 frames / 34.546s = 41.857 fps`.
    - `AUDEX_VLLM_CFG_MAX_NUM_SEQS=24` wrote
      `.audex/runs/tts-probe-vllm-20260709-115359.json`:
      `1446 frames / 41.792s = 34.600 fps`.
    - `AUDEX_VLLM_CFG_MAX_NUM_SEQS=24 AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS=32768`
      wrote `.audex/runs/tts-probe-vllm-20260709-115547.json`:
      `max_num_batched_tokens=32768`, `1446 frames / 35.416s = 40.829 fps`.
- Interpretation:
  - Raising `max_num_seqs` alone does not unlock >2 active CFG segment pairs
    for the real CFG path. The logged event order advances segments in pairs:
    `0/1`, then `2/3`, then `4/5`, and so on.
  - Raising `max_num_batched_tokens` to `32768` did not change that admission
    pattern in this probe.
  - NVIDIA's `CFGLogitsProcessor` can blend every complete pair already in the
    batch; the observed bottleneck is therefore earlier in request admission,
    scheduler/pair-hold behavior, or vLLM Metal's nonpaged batched decode/cache
    handling.
  - The next plausible combo patch is still persistent nonpaged batched cache
    plus compact/window sampling, but it must be measured with JSON artifacts
    showing `cfg_enabled=true` and `cfg_scale=3.0`.
- Guard/reapply notes:
  - Keep `--cfg-segment` diagnostic-only; it should not change the interactive
    STS product path.
  - When using this probe directly, set both `AUDEX_VLLM_TTS_CFG=1` and
    `AUDEX_VLLM_ENABLE_CFG_WIRING=1`; the STS CLI normally sets CFG wiring
    before engine construction, but the standalone probe does not go through
    that CLI setup path.
  - Recheck this script if `AudexAsyncVllmRuntime.build_tts_cfg_pair(...)`,
    `stream_many(...)`, or vLLM's `output_kind=DELTA` behavior changes.

## 2026-07-09 - CFG Row-Count Instrumentation and Product Decoder Bisect

- Files patched:
  - `audex_mac/patches/vllm_metal_cfg.py`
  - `audex_mac/vllm_diagnostics.py`
  - `audex_mac/vllm_sts_cli.py`
  - `scripts/probe_vllm_tts_decode.py`
  - `tests/test_vllm_cfg.py`
  - `tests/test_vllm_diagnostics.py`
  - `tests/test_vllm_sts_cli.py`
  - `tests/test_probe_vllm_tts_decode.py`
- What changed:
  - Native vLLM Metal timing stderr now records actual CFG decode membership:
    `cfg_cond_reqs`, `cfg_uncond_reqs`, and `cfg_complete_pairs` for each
    nonpaged decode timing line.
  - `audex_mac.vllm_diagnostics` parses those new fields so reports can tell
    configured scheduler limits apart from actual active CFG rows.
  - `scripts/probe_vllm_tts_decode.py --cfg-segment ...` now fails loudly when
    `runtime.cfg_enabled` is false or `cfg_scale` is not NVIDIA's recipe
    `3.0`. Invalid no-CFG artifacts should no longer be mistaken for CFG
    throughput evidence.
  - The probe JSON now records `request_count`, `cfg_segment_count`, and
    `aggregate_codec_frames_per_second`.
  - Added diagnostic-only `AUDEX_VLLM_SKIP_SPEECH_DECODER=1` for the product
    STS path. It discards generated codec frames instead of calling the causal
    speech decoder, skips playback/WAV PCM emission, and suppresses
    per-segment MLX cache clears so the product path can isolate generation
    throughput from decoder/cache-clear overhead.
- Why:
  - The direct CFG probe had reached roughly 41 aggregate codec fps while the
    product path was materially slower. The largest known loss needed an
    apples-to-apples product-path bisect before deeper vLLM Metal engine work.
  - `AUDEX_VLLM_CFG_MAX_NUM_SEQS` is only a configuration cap; without actual
    decode row counts it was easy to assume a run tested wider CFG concurrency
    when the scheduler may still have admitted only two CFG pairs.
- Validation:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_probe_vllm_tts_decode.py tests/test_vllm_cfg.py tests/test_vllm_diagnostics.py -q`
    reported `147 passed`.
  - Ruff:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py scripts/probe_vllm_tts_decode.py audex_mac/patches/vllm_metal_cfg.py audex_mac/vllm_diagnostics.py tests/test_vllm_sts_cli.py tests/test_probe_vllm_tts_decode.py tests/test_vllm_cfg.py tests/test_vllm_diagnostics.py`
    reported `All checks passed`.
  - Product CFG baseline:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-120848.json` and
    `.audex/runs/sts-turn-vllm-20260709-120915.json`:
    `733 frames / 26.299s = 27.872 fps`, `first_audio_ready_seconds=2.644`,
    `decoder_push_seconds=0.948181`, `decoder_flush_seconds=0.098489`,
    `mlx_clear_cache_seconds=0.013614`, and `decoded_chunk_count=18`.
  - Product CFG with decoder skipped:
    same command plus `AUDEX_VLLM_SKIP_SPEECH_DECODER=1` wrote
    `.audex/runs/speech-output-vllm-20260709-121026.json` and
    `.audex/runs/sts-turn-vllm-20260709-121045.json`:
    `733 frames / 19.053s = 38.471 fps`, `speech_decoder_skipped=true`,
    `decoded_chunk_count=0`, and zero decoder/cache-clear timings.
  - Direct CFG row-count probe:
    `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16` with 12 `--cfg-segment` values wrote
    `.audex/runs/tts-probe-vllm-20260709-121302.json`:
    `max_num_seqs=16`, `1151 frames / 28.338s = 40.652 fps`; native timing
    lines consistently showed `decode_reqs=4`, `cfg_cond_reqs=2`,
    `cfg_uncond_reqs=2`, and `cfg_complete_pairs=2`.
  - Larger token-budget admission check:
    `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16 AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS=32768`
    with 8 `--cfg-segment` values wrote
    `.audex/runs/tts-probe-vllm-20260709-121447.json`:
    `512 frames / 12.913s = 39.650 fps`; native timing still showed only two
    active CFG pairs.
- Interpretation:
  - The product decoder/cache-clear path costs about 38% throughput on this
    fixture: 27.9 fps with decoder work versus 38.5 fps when codec frames are
    discarded.
  - The engine/product generation gap is now much smaller than the original
    10-12 fps observation on this fixture. The current product path without
    decoder work is close to the direct CFG probe ceiling.
  - The prior `max_num_seqs=8/16/24` sweep did not prove 4 or 8 active CFG
    pairs. The actual decode batch stayed at two conditional and two
    unconditional requests even with `max_num_seqs=16` and
    `max_num_batched_tokens=32768`.
  - Next code-level target: find why request admission advances CFG segments in
    two-pair waves despite all requests being submitted, then decide whether a
    diagnostic `cfg_pairs_per_batch` override is safe. Separately, move causal
    speech decoding/cache clearing off the hot event loop or pipeline it after
    generation because it is now a proven product-path cost.
- Guard/reapply notes:
  - `AUDEX_VLLM_SKIP_SPEECH_DECODER` is diagnostic only. Do not enable it in
    `./start.sh` defaults; it intentionally writes an empty WAV and cannot be
    used for intelligibility validation.
  - Keep the CFG probe guard. Any artifact with `runtime.cfg_enabled=false` or
    `runtime.cfg_scale != 3.0` must be treated as a no-CFG/non-recipe artifact.
  - Recheck native timing parsing if the debug string in
    `_record_non_paged_decode_timing(...)` changes.

## 2026-07-09 Nonpaged CFG KV Capacity Override

- Files changed:
  - `audex_mac/patches/runtime.py`
  - `audex_mac/patches/vllm_metal_cfg.py`
  - `audex_mac/vllm_cfg.py`
  - `audex_mac/vllm_runtime.py`
  - `scripts/probe_vllm_tts_decode.py`
  - `start.sh`
  - `README.md`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
  - `tests/test_audex_patches.py`
  - `tests/test_start_sh.py`
  - `tests/test_vllm_cfg.py`
  - `tests/test_vllm_runtime.py`
- What changed:
  - Added native vLLM Metal scheduler admission debug that logs scheduled,
    running, and waiting CFG request/pair counts.
  - Added KV-allocation rejection debug around
    `kv_cache_manager.allocate_slots(...)`, including request role, pair id,
    free blocks, prompt token counts, and scheduler reservation flags.
  - Added one-time scheduler capacity debug that logs `num_blocks`,
    `block_size`, `max_model_len`, `pool_bytes_per_block`, and
    `estimated_max_concurrency`.
  - Added diagnostic CFG engine overrides:
    `AUDEX_VLLM_CFG_MAX_MODEL_LEN` and
    `AUDEX_VLLM_CFG_SCHEDULER_RESERVE_FULL_ISL`.
  - Added the early runtime patch
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS`. When set, it multiplies vLLM
    Metal's non-paged `single_sequence_estimate` scheduler-visible capacity by
    that many max-length-sequence equivalents. It does not affect paged
    attention mode.
  - `./start.sh` now defaults
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS:-4}"`
    for the vLLM Metal launch path.
  - Probe JSON now records `runtime.nonpaged_kv_capacity_seqs`.
- Why:
  - The latest evidence proved the two-pair CFG wave was a cache-admission
    capacity problem, not a request submission or `max_num_seqs` problem.
    vLLM Metal's non-paged MLX path reports one max-length sequence to the
    scheduler. For short Audex CFG TTS segments this under-admits requests and
    hides the actual 30B concurrency curve.
- Validation and evidence:
  - Default `max_model_len=5120`, `AUDEX_VLLM_CFG_MAX_NUM_SEQS=16`, and
    `AUDEX_VLLM_CFG_MAX_BATCHED_TOKENS=32768` logged:
    `num_blocks=5`, `block_size=1072`, `estimated_max_concurrency=1.0`,
    `scheduled_reqs=4`, `scheduled_complete_pairs=2`, `waiting_reqs=12`, and
    repeated `free_blocks=0` allocation rejections. The run wrote
    `.audex/runs/tts-probe-vllm-20260709-123208.json` with 30.944 aggregate
    codec fps on the short 32-token fixture.
  - `AUDEX_VLLM_CFG_MAX_MODEL_LEN=2048` was a negative result. It logged
    `num_blocks=2` and could not admit a complete CFG pair; the run wrote
    `.audex/runs/tts-probe-vllm-20260709-123100.json`.
  - `AUDEX_VLLM_GPU_MEMORY_UTILIZATION=0.85` was also a negative result for
    this admission problem. It still logged `num_blocks=5` and
    `scheduled_complete_pairs=2`; the run wrote
    `.audex/runs/tts-probe-vllm-20260709-123308.json`.
  - With `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=4`, the 8-segment CFG probe
    logged `num_blocks=20`, `scheduled_reqs=16`,
    `scheduled_complete_pairs=8`, `waiting_reqs=0`, and nonpaged decode timing
    with `decode_reqs=16`, `cfg_cond_reqs=8`, `cfg_uncond_reqs=8`, and
    `cfg_complete_pairs=8`.
  - The comparable 64-token 8-pair run wrote
    `.audex/runs/tts-probe-vllm-20260709-123825.json`: 512 aggregate codec
    frames in 9.104 seconds, or 56.239 aggregate codec fps.
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_audex_patches.py tests/test_vllm_cfg.py tests/test_vllm_runtime.py tests/test_probe_vllm_tts_decode.py -q`
    reported `122 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/patches/runtime.py tests/test_audex_patches.py audex_mac/vllm_cfg.py audex_mac/vllm_runtime.py scripts/probe_vllm_tts_decode.py tests/test_vllm_cfg.py tests/test_vllm_runtime.py tests/test_probe_vllm_tts_decode.py`
    reported `All checks passed`.
- Interpretation:
  - The decisive 8-pair 30B CFG datapoint now exists. It clears the 50
    codec-fps real-time target on the direct probe fixture, though the product
    path still needs decoder overlap/pipelining work to keep long utterances
    smooth.
  - `max_model_len` and `AUDEX_VLLM_GPU_MEMORY_UTILIZATION` should not be
    presented as fixes for this specific admission wall.
- Guard/reapply notes:
  - Keep `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` explicit and logged. Raising it
    beyond `4` must be backed by Activity Monitor memory observations and
    native `cfg_complete_pairs` evidence.
  - If upstream vLLM Metal changes `WorkerCachePlanner.determine_available_memory`
    or the `single_sequence_estimate` mode name, this patch must fail loudly
    and be re-evaluated against the new cache policy.

## 2026-07-09 CFG Capacity Default and Admission Arithmetic

- Files changed:
  - `audex_mac/patches/vllm_metal_cfg.py`
  - `audex_mac/vllm_cfg.py`
  - `start.sh`
  - `tests/test_start_sh.py`
  - `tests/test_vllm_cfg.py`
- What changed:
  - Raised the CFG engine floor from `max_num_seqs=8` to `max_num_seqs=16` so
    vLLM Metal can schedule eight conditional/unconditional CFG pairs when
    admission allows it.
  - `./start.sh` now defaults
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS:-8}"`
    for the vLLM Metal launch path.
  - The one-time CFG KV capacity log now prints
    `max_length_blocks_per_request`, `max_length_bytes_per_request`, and
    `inferred_request_capacity` so the admission limit is checkable by hand.
- Why:
  - The current product path previously raised `max_num_seqs` in some probes
    while `./start.sh` still reserved only four max-length nonpaged sequences.
    That made product runs look scheduler-limited even after the engine was
    configured for wider CFG scheduling.
  - The explicit arithmetic matters for hybrid-model debugging. With the
    default product context, native debug now shows
    `num_blocks=40`, `block_size=1072`, `max_model_len=5120`,
    `max_length_blocks_per_request=5`, and
    `inferred_request_capacity=8.0`.
- Validation and evidence:
  - Default product CFG fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=600 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/four-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 96 --speech-max-tokens 1024`
    wrote `.audex/runs/speech-output-vllm-20260709-130130.json`.
  - That run logged `inferred_request_capacity=8.0`, scheduled the four
    available sentence chunks as `scheduled_reqs=8` and
    `scheduled_complete_pairs=4`, and generated 720 codec frames at 36.362
    codec fps with first audio ready at 3.166 seconds.
  - The MLX-Audio oracle wrote
    `.audex/runs/speech-output-vllm-20260709-130130.eval.json` and passed:
    ratio `0.8841354723707665`, no excessive repetition.
  - Diagnostic `AUDEX_VLLM_CFG_MAX_MODEL_LEN=4096` plus
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=8` wrote
    `.audex/runs/speech-output-vllm-20260709-125830.json`. It logged
    `num_blocks=32`, `max_length_blocks_per_request=4`, and
    `inferred_request_capacity=8.0`, generated 720 codec frames at 41.891
    codec fps, and passed the same oracle ratio/repetition gate.
  - Diagnostic `AUDEX_VLLM_CFG_MAX_MODEL_LEN=2048` and `3072` both failed
    before response generation because the product prompt validated at
    `max_model_len + 1` tokens. These are context-size negative results, not
    CFG quality failures.
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_cfg.py -q` reported
    `53 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/patches/vllm_metal_cfg.py audex_mac/vllm_cfg.py tests/test_vllm_cfg.py`
    reported `All checks passed`.
- Interpretation:
  - The four-sentence product fixture is not a valid eight-pair throughput
    measurement because it only produces four TTS chunks. It is a product
    safety/intelligibility check for the wider default capacity.
  - `max_model_len=4096` is promising for this fixture but is not promoted to
    default yet because the STS demo promises long utterances; shortening the
    engine context needs a separate conversation-history and long-answer
    validation pass.
- Guard/reapply notes:
  - Keep `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` user-overridable. If Activity
    Monitor shows memory pressure on the 30B path, rerun with
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=4` before blaming CFG sampling.
  - Do not claim a concurrency win from `max_num_seqs=16` unless native debug
    shows actual `cfg_complete_pairs` above the old admitted width.

## 2026-07-09 Nonpaged Capacity Headroom Guard and 8-Pair Product Evidence

- Files changed:
  - `audex_mac/patches/runtime.py`
  - `tests/test_audex_patches.py`
  - `PATCH.md`
  - `docs/engineering/patches.md`
- What changed:
  - The `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` patch now checks MLX/Metal
    headroom before it reports widened non-paged capacity to vLLM's scheduler.
    It logs the Metal working-set limit, current active/cache memory, full
    Metal headroom, `gpu_memory_utilization` headroom, and the requested
    max-length KV worst case.
  - If the requested worst case exceeds the current MLX working-set headroom,
    the patch raises before returning the widened scheduler-visible capacity.
  - Removed the experimental `AUDEX_VLLM_CFG_TTS_MAX_CHARS_PER_CHUNK`
    diagnostic before documenting it as a real product lever. The experiment
    created 9 chunks for an 8-pair-capacity run and therefore measured a
    second admission wave, not balanced chunking.
- Why:
  - The previous default `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=8` widened
    scheduler admission by assertion. That was enough to prove true 8-pair CFG
    decode, but it did not prove the advertised max-length reservation fit
    inside the current MLX working-set budget.
  - The correct long-term fix is per-request non-paged reservation based on
    the actual Audex TTS segment budget. Until that exists, the override must
    at least fail loudly when its conservative max-length arithmetic exceeds
    the visible MLX working-set headroom.
- Validation and evidence:
  - Eight-sentence product-path CFG run:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=900 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/eight-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 192 --speech-max-tokens 1536`
    wrote `.audex/runs/speech-output-vllm-20260709-130630.json`.
  - That run logged real product-path 8-pair scheduling:
    `scheduled_reqs=16`, `scheduled_complete_pairs=8`, `waiting_reqs=0`, and
    nonpaged decode timing with `decode_reqs=16`, `cfg_cond_reqs=8`,
    `cfg_uncond_reqs=8`, and `cfg_complete_pairs=8`.
  - The same run generated 1286 codec frames at 23.276 codec fps with
    `first_audio_ready_seconds=7.941` and passed the MLX-Audio oracle:
    `.audex/runs/speech-output-vllm-20260709-130630.eval.json` ratio
    `0.9705304518664047`, no excessive repetition.
  - Decoder-skip A/B at the same true 8-pair width wrote
    `.audex/runs/speech-output-vllm-20260709-131630.json`: 1301 codec frames
    at 23.852 codec fps. That ~2.5% lift means the inline waveform decoder is
    not the binding product bottleneck on this fixture.
  - `AUDEX_VLLM_CFG_TTS_MAX_CHARS_PER_CHUNK=80` wrote
    `.audex/runs/speech-output-vllm-20260709-131105.json`: 9 observed
    segments, 16.444 codec fps, and `first_audio_ready_seconds=7.687`. This
    is a negative result for over-producing chunks, not a negative result for
    capacity-aware balanced chunking.
  - `AUDEX_VLLM_CFG_MAX_MODEL_LEN=4096` on the eight-sentence fixture wrote
    `.audex/runs/speech-output-vllm-20260709-131349.json`: 8 segments,
    18.203 codec fps, and `first_audio_ready_seconds=5.456`. Do not promote
    4096 from the earlier four-sentence result without a separate long-context
    validation pass.
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_audex_patches.py tests/test_vllm_sts_cli.py -q`
    reported `86 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/patches/runtime.py audex_mac/vllm_sts_cli.py tests/test_audex_patches.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - Short 30B product smoke:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=420 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/normal-answer-question-16k-mono.wav --no-play --response-max-tokens 32 --speech-max-tokens 64`
    completed and wrote
    `.audex/runs/speech-output-vllm-20260709-133252.wav`.
  - That smoke printed the new guard line:
    `metal_limit=118.11 GB active=63.95 GB cache=5.74 GB metal_headroom=48.42 GB gpu_memory_utilization=0.60 gpu_utilization_headroom=1.18 GB requested_worst_case=2.28 GB`.
    A prior hard-fail version that used the `gpu_memory_utilization` headroom
    as the allocator limit was rejected because it blocked this known-working
    non-paged `VLLM_METAL_MEMORY_FRACTION=auto` path.
- Interpretation:
  - The 30B product path now has valid evidence for true 8-pair CFG admission,
    but the product path still trails the direct generation-side probe. The
    first bottleneck to chase is not the waveform decoder on this fixture; it
    is generation scheduling/tail behavior after segments finish at uneven
    lengths.
  - Capacity-aware chunking should choose a chunk count no larger than the
    admitted pair capacity and balance sentence groups by text length. A
    char-cap that accidentally creates one extra chunk is expected to regress.
- Guard/reapply notes:
  - If this guard trips on a real machine, do not bypass it by raising
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS`. Lower capacity or implement
    actual per-request reservation sizing.
  - If vLLM Metal changes `gpu_memory_utilization` plumbing or MLX memory APIs,
    revalidate the headroom calculation before trusting the override.

## 2026-07-09 CFG TTS Target-Width Chunk Guard

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
  - `PATCH.md`
  - `docs/engineering/patches.md`
  - `docs/engineering/vllm-metal.md`
- What changed:
  - Static CFG TTS now passes `tts_target_segments` into
    `split_cfg_spoken_tts_chunks(...)`.
  - The vLLM STS default target segment count is now 8, matching the proven
    product CFG admission width (`max_num_seqs=16` -> eight CFG pairs).
  - If sentence-level CFG chunking produces more chunks than the target, the
    CLI merges neighboring ordered chunks by approximate text length instead
    of submitting an avoidable extra scheduler wave.
- Why:
  - The `AUDEX_VLLM_CFG_TTS_MAX_CHARS_PER_CHUNK=80` experiment produced 9
    chunks against an 8-pair-capacity run. The ninth chunk forced a second
    admission wave and regressed throughput to 16.444 codec fps.
  - The product path should not create extra waves merely because sentence or
    character splitting crossed the admitted pair count by one chunk.
- Validation and evidence:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `55 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - Local splitter sanity check:
    `split_cfg_spoken_tts_chunks("One. Two. Three. Four. Five. Six. Seven. Eight. Nine.", target_segments=8)`
    returned 8 chunks and round-tripped the text without duplication.
  - 30B product fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=900 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/eight-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 192 --speech-max-tokens 1536`
    wrote `.audex/runs/speech-output-vllm-20260709-134035.json`.
  - That run logged `scheduled_complete_pairs=8`, `waiting_reqs=0` at TTS
    startup and observed 8 TTS segments. It generated 1300 codec frames at
    23.187 codec fps with `first_audio_ready_seconds=6.806`.
  - The MLX-Audio oracle wrote
    `.audex/runs/speech-output-vllm-20260709-134035.eval.json` and passed:
    ratio `0.9950738916256158`, no excessive repetition.
- Interpretation:
  - This is a correctness/scheduler guard against accidental over-segmentation,
    not a throughput win on the eight-sentence fixture. That fixture already
    had 8 chunks.
  - The measured remaining bottleneck is tail collapse from uneven generated
    audio lengths: the same run's segment frame counts were
    `{0: 165, 1: 252, 2: 181, 3: 169, 4: 205, 5: 136, 6: 128, 7: 64}`.
    Text-length balancing alone is too weak a proxy for generated codec-frame
    duration.
- Guard/reapply notes:
  - Do not reintroduce character caps that can create a ninth initial CFG
    chunk when the admitted pair width is eight.
  - The next likely product fix is staged widening or a better duration
    predictor for segment grouping, not decoder overlap and not more chunks.

## 2026-07-09 CFG Short Tail Merge

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/patches.md`
- What changed:
  - Static CFG TTS now merges a very short chunk into a neighboring chunk when
    the initial chunk list is already at or above the target pair width.
  - The default minimum is
    `DEFAULT_VLLM_CFG_TTS_MIN_CHARS_PER_CHUNK = 40`.
  - All-short responses are left alone so terse answers do not collapse into
    one large prompt.
  - Speech run logs now include `tts_cfg_min_chars_per_chunk`.
- Why:
  - The eight-sentence fixture was capped to 8 segments, but the final segment
    generated only 64 codec frames. It then ran alone at the end of the batch,
    dragging product throughput down even though admission was correct.
  - Merging only the tiny tail keeps ordered playback and reduces low-width
    tail time without changing NVIDIA CFG sampler settings.
- Validation and evidence:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `57 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - Splitter sanity check on the prior fixture response produced 7 segments,
    merging only `"This pattern is useful"` into the previous chunk.
  - 30B product fixture:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=900 ./start.sh --model audex-30b-a3b --input-wav .audex/fixtures/eight-sentence-context-question-16k-mono.wav --no-play --response-max-tokens 192 --speech-max-tokens 1536`
    wrote `.audex/runs/speech-output-vllm-20260709-134831.json`.
  - Compared with the preceding 8-segment run
    `.audex/runs/speech-output-vllm-20260709-134035.json`, the short-tail
    merge changed:
    - observed segments: `8 -> 7`
    - stream time: `56.066s -> 40.190s`
    - codec fps: `23.187 -> 32.321`
    - first audio: `6.806s -> 6.515s`
    - generated codec frames stayed effectively equal: `1300 -> 1299`
  - The MLX-Audio oracle wrote
    `.audex/runs/speech-output-vllm-20260709-134831.eval.json` and passed:
    ratio `0.9950738916256158`, no excessive repetition.
- Interpretation:
  - This is the first product-path tail reduction after the 8-pair admission
    fix. It does not make 30B CFG real-time yet, but it removes an avoidable
    one-pair tail on the current fixture.
  - The remaining tail still drops through 2 pairs. The next improvement should
    use a better duration predictor or staged pair admission; do not add more
    initial chunks.
- Guard/reapply notes:
  - Keep this heuristic conservative. If future fixtures show prosody damage
    from merged chunks, prefer a duration-aware grouping model over raising the
    minimum chunk length globally.

## 2026-07-09 CFG Capacity-Aware Target and Long-Phrase Refill

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/patches.md`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- What changed:
  - Added diagnostic env `AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS` for fixed-width
    CFG chunk experiments.
  - Static CFG TTS now caps the requested target segment count to
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS` when that capacity override is set,
    so lowering scheduler capacity does not leave the product path asking for
    more upfront CFG chunks than can be admitted.
  - Speech run logs now record both `tts_requested_target_segments` and the
    effective `tts_target_segments`.
  - After the short-tail merge collapses a tiny segment, CFG chunking can refill
    unused target capacity by splitting the longest oversized chunk once on a
    phrase/word boundary. This is intentionally gated on a prior short-tail
    merge so ordinary two-sentence answers are not fragmented just because the
    target width is eight.
- Why:
  - The earlier `target_segments=6` diagnostic proved that blindly reducing
    segment count can create a long solo tail. On the context-manager fixture it
    generated 6 segments and ended with a 344-codec-frame segment.
  - The better shape is not "fewer chunks"; it is "no tiny tail, no second
    admission wave, and use the admitted capacity when a long phrase can be
    split cleanly."
- Validation and evidence:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `60 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - Decoder-skip A/B on the current 7-segment product fixture:
    - decoder skipped: `.audex/runs/speech-output-vllm-20260709-135805.json`,
      `1299` codec frames at `32.059` codec fps.
    - decoder enabled: `.audex/runs/speech-output-vllm-20260709-140007.json`,
      `1299` codec frames at `28.375` codec fps, first audio `4.486s`.
    - Interpretation: decoder still costs throughput, but it is not the old
      10-fps wall under this product shape. Segment shape remains the larger
      bottleneck.
  - `AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS=6` diagnostic with decoder skipped
    wrote `.audex/runs/speech-output-vllm-20260709-140217.json`: 6 observed
    segments, `1319` codec frames at `33.691` codec fps, but frame counts were
    `{0:159, 1:248, 2:183, 3:183, 4:202, 5:344}`. This is not a product win
    because the tail is badly unbalanced.
  - A full ASR->text->TTS fixture after the patch returned only
    `"A context manager is an"` from text generation and therefore scheduled
    one CFG pair. That run is invalid as chunk-planner performance evidence.
  - Fixed-text vLLM Metal CFG probe using the chunker-shaped eight segments
    admitted all 8 real pairs:
    `scheduled_reqs=16 scheduled_complete_pairs=8 waiting_reqs=0`.
    It wrote `.audex/runs/tts-probe-vllm-20260709-140727.wav` and reported
    aggregate `1304` codec frames at `49.792` aggregate codec fps.
- Interpretation:
  - The fixed-text probe shows the chunk planner can occupy the full 8-pair
    scheduler width without a tiny tail and nearly reaches realtime. It does
    not prove the product path is realtime yet, because the full conversational
    path still depends on answer text shape and decoder/playback behavior.
  - The next likely product win is still duration-aware chunk planning or
    staged widening. Decoder overlap is worth keeping on the list, but the
    latest A/B says it is secondary after the short-tail merge.
- Guard/reapply notes:
  - Keep `AUDEX_VLLM_CFG_TTS_TARGET_SEGMENTS` diagnostic-only. The normal CLI
    should stay simple and derive the effective target from scheduler capacity.
  - Do not split short ordinary answers just to hit target width. Refill should
    only happen after a short-tail merge created spare capacity.
  - Any future duration predictor should be validated against generated
    `tts_segment_codec_frame_counts`, not text length alone.

## 2026-07-09 Product-Path Fixed-Text TTS Probe

- Files changed:
  - `audex_mac/cli.py`
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_start_sh.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/patches.md`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- What changed:
  - Added diagnostic CLI inputs:
    - `--diagnose-vllm-tts-text TEXT`
    - `--diagnose-vllm-tts-text-file PATH`
  - Added `run_vllm_tts_text_probe(...)`, which loads the same
    `VllmSpeechToSpeechSession` used by product STS and calls
    `generate_speech_output(...)` directly with fixed text.
  - The diagnostic respects the normal product CFG switch:
    `AUDEX_VLLM_TTS_CFG=1` enables CFG; otherwise it measures the no-CFG
    fallback.
  - The diagnostic uses the session's normal speech-token budget calculation
    unless `--speech-max-tokens` is provided.
  - CLI summary output now reads `generated_codec_frame_count`,
    `codec_frames_per_second`, `first_audio_ready_seconds`, and
    `hit_max_tokens` from the speech-output JSON log. This avoids printing
    `0` codec frames for streaming results where heavy frame arrays are
    intentionally not retained in the returned object.
- Why:
  - Full ASR->LLM->TTS fixture runs can be invalid as TTS/chunking evidence when
    the text response drifts. One such run produced only
    `"A context manager is an"` and scheduled a single CFG pair.
  - The existing `scripts/probe_vllm_tts_decode.py` is useful, but it exercises
    the low-level async runtime directly. We also need a deterministic probe for
    the product speech-output path, including decoder, segment run log, and
    product chunking.
- Validation and evidence:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_start_sh.py tests/test_vllm_sts_cli.py -q`
    reported `72 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/cli.py audex_mac/vllm_sts_cli.py tests/test_start_sh.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - Product-path fixed-text CFG probe:
    `AUDEX_VLLM_TTS_CFG=1 AUDEX_VLLM_ENABLE_CFG_WIRING=1 AUDEX_VLLM_NATIVE_SAMPLING_DEBUG=1 AUDEX_STS_SMOKE_TIMEOUT_SECONDS=900 ./start.sh --model audex-30b-a3b --diagnose-vllm-tts-text "<context-manager text>" --no-play`
    wrote `.audex/runs/tts-text-probe-vllm-20260709-141524.json`.
  - That run logged `scheduled_complete_pairs=8`, `waiting_reqs=0`, 8 observed
    TTS segments, `generated_codec_frame_count=1304`,
    `codec_frames_per_second=49.953`, `first_audio_ready_seconds=5.758`, and
    `hit_max_tokens=false`.
- Interpretation:
  - The fixed-text product probe is now the preferred TTS/chunking benchmark
    before running full speech-to-speech fixture turns. It proves the shaped
    CFG TTS batch can drive the product path to just under realtime without
    relying on a lucky text response.
  - This does not complete the main goal: the interactive CLI still needs
    product-path realtime margin, playback stability, intelligibility checks,
    and a human listening checkpoint.
- Guard/reapply notes:
  - Keep this path diagnostic-only. Do not make normal `./start.sh` ask for text
    or expose extra configuration before entering microphone/speaker STS.
  - When streaming results retain no heavy frame arrays, prefer the JSON run log
    for summary counts.

## 2026-07-09 CFG Linear-Partition Chunk Planner and Fixture Validity Gate

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `audex_mac/vllm_diagnostics.py`
  - `tests/test_vllm_sts_cli.py`
  - `tests/test_vllm_diagnostics.py`
  - `docs/engineering/patches.md`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- What changed:
  - Replaced the static CFG TTS heuristic stack with CFG atomization plus an
    exact contiguous linear-partition planner. The planner splits long CFG
    sentence atoms at about 80 characters, caps group count to the effective
    target/admitted capacity, and minimizes the maximum character-cost group.
  - Removed the stale `tts_cfg_min_chars_per_chunk` run-log field and replaced
    it with `tts_cfg_partition_cost="chars"`.
  - Speech run logs now include `tts_cfg_atom_max_chars`.
  - STS smoke diagnostics now record `response_word_count`,
    `min_response_words`, and `valid_response_length`.
  - The STS evidence gate rejects timing artifacts when the generated text
    response is below the response-length sanity threshold.
- Why:
  - The previous short-tail merge plus longest-phrase refill were useful
    experiments, but they were approximating one deterministic partitioning
    problem. The DP is smaller, easier to reason about, and directly represents
    the constraint that CFG chunks must remain ordered and must not exceed
    admitted pair capacity.
  - A full ASR->LLM->TTS fixture previously emitted only
    `"A context manager is an"`. That kind of truncated text should invalidate
    performance evidence instead of masquerading as a chunk-planner regression.
- Validation:
  - Focused splitter and evidence-gate tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py tests/test_vllm_diagnostics.py -q`
    reported `106 passed`.
  - Focused ruff check:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py audex_mac/vllm_diagnostics.py tests/test_vllm_sts_cli.py tests/test_vllm_diagnostics.py`
    reported `All checks passed`.
  - A sentence-only DP attempt was measured and rejected. It wrote
    `.audex/runs/tts-text-probe-vllm-20260709-142532.json`: 8 admitted
    segments, decoder enabled, but the 87-character second sentence generated
    a 253-frame straggler and throughput fell to `40.627` fps.
  - Corrected CFG atomization plus linear partition was measured on the
    fixed-text product path:
    - `.audex/runs/tts-text-probe-vllm-20260709-142835.json`: `1304` frames at
      `48.793` fps, first audio `5.986s`, native debug confirmed
      `scheduled_complete_pairs=8` and `waiting_reqs=0`.
    - `.audex/runs/tts-text-probe-vllm-20260709-142956.json`: `1304` frames at
      `53.264` fps, first audio `5.402s`.
    - `.audex/runs/tts-text-probe-vllm-20260709-143110.json`: `1304` frames at
      `51.063` fps, first audio `5.714s`.
  - All three corrected runs logged `speech_decoder_skipped=false`,
    `tts_cfg_enabled=true`, 8 observed segments, `tts_cfg_atom_max_chars=80`,
    `hit_max_tokens=false`, and identical segment frame counts:
    `{0:172, 1:124, 2:128, 3:172, 4:174, 5:207, 6:128, 7:199}`.
  - MLX-Audio oracle on the latest WAV:
    `.venv/bin/python scripts/evaluate_tts_wav.py .audex/runs/tts-text-probe-vllm-20260709-143110.wav --expected-from-speech-log .audex/runs/tts-text-probe-vllm-20260709-143110.json --json-out .audex/runs/tts-text-probe-vllm-20260709-143110.eval.json --min-ratio 0.55`
    passed with ratio `1.0` and no excessive repetition.
- Reapply notes:
  - Keep the planner exact and deterministic. Do not reintroduce ad hoc
    short-tail or refill heuristics unless generated
    `tts_segment_codec_frame_counts` prove character-cost partitioning is the
    wrong proxy.
  - If the cost proxy changes from characters to a duration predictor, keep the
    same ordered linear-partition objective and update the run-log cost label.
  - Keep too-short generated responses structurally invalid in diagnostics; do
    not rely on manual interpretation of fixture logs.

## 2026-07-09 Visible Text Context Overflow Guard

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/patches.md`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- What changed:
  - Added a text-context preflight before vLLM text generation. The CLI builds
    the same Audex text prompt the runtime will submit, counts prompt tokens,
    and records `text_context` metadata in successful STS turn logs.
  - If `prompt_tokens + response_max_tokens` exceeds the current engine
    `max_model_len`, Audex-Mac now raises a clear `RuntimeError` instead of
    letting vLLM fail later with a raw validation exception.
  - An earlier local experiment briefly pruned history to fit the context; that
    was rejected because it silently forgot conversation history. The committed
    behavior does not prune, summarize, or drop history.
- Why:
  - `5120` is Audex-Mac's current vLLM CFG engine reservation arithmetic, not
    Audex's model context capability. It comes from
    `max(4096, asr_max_tokens + 1024, text_max_tokens + 1024,
    tts_max_tokens + 1024)`.
  - On the vendored non-paged Metal path, increasing `max_model_len` is not a
    free long-context knob because each admitted request reserves state against
    the full engine length. Blindly raising it would destroy the 8-pair CFG TTS
    capacity win or exceed memory headroom.
- Validation and evidence:
  - Focused tests:
    `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `63 passed`.
  - Focused ruff:
    `.venv/bin/python -m ruff check audex_mac/vllm_sts_cli.py tests/test_vllm_sts_cli.py`
    reported `All checks passed`.
  - The test
    `test_validate_text_prompt_messages_rejects_over_context_without_pruning`
    asserts that over-budget history raises and records
    `fits=false`, without dropping old messages.
  - After the rejected pruning experiment had shortened the local conversation,
    the same 30B CFG fixture completed and wrote
    `.audex/runs/sts-turn-vllm-20260709-144256.json`. It logged
    `text_context` with `fits=true`, `messages_before=108`,
    `prompt_tokens=4952`, `prompt_token_budget=5024`,
    `context_token_limit=5120`, and `response_max_tokens=96`.
  - The corresponding TTS log
    `.audex/runs/speech-output-vllm-20260709-144242.json` had
    `speech_decoder_skipped=false`, `tts_cfg_enabled=true`, 3 observed
    segments, 416 codec frames at `28.872` fps, first audio `3.981s`, and no
    max-token hit.
  - MLX-Audio oracle:
    `.venv/bin/python scripts/evaluate_tts_wav.py .audex/runs/speech-output-vllm-20260709-144242.wav --expected-from-speech-log .audex/runs/speech-output-vllm-20260709-144242.json --json-out .audex/runs/speech-output-vllm-20260709-144242.eval.json --min-ratio 0.55`
    passed with ratio `1.0` and no excessive repetition.
- Reapply notes:
  - Do not reintroduce silent history pruning as a fallback.
  - Raising text context should come after request-scoped KV/state reservation,
    so short TTS CFG segments keep reserving short request budgets rather than
    the full engine max.
  - Long-running conversation support should be designed around recurrent state
    carryover or incremental prefill, not ever-growing transcript re-prefill.

## 2026-07-09 Text Prompt Prefix Contract for vLLM State Carryover

- Files touched:
  - `audex_mac/vllm_sts_requests.py`
  - `tests/test_vllm_sts_requests.py`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- Why:
  - The old MLX-native path can load a persisted conversation prompt cache and
    feed only suffix tokens, but the current vLLM Metal STS path persists only
    the transcript and re-prefills the rendered conversation every turn.
  - Before patching vLLM Metal to carry request cache/state across turns, the
    prompt rendering contract must be explicit: committed history for turn N
    must be an exact token prefix of the generation prompt for turn N+1.
- What changed:
  - Added `build_text_messages_history_prompt`, which renders the same
    system-policy-adjusted text conversation with
    `add_generation_prompt=False`.
  - Added `build_text_messages_generation_prompt`, used by
    `build_text_messages_response_request`, which renders the next vLLM text
    request with `add_generation_prompt=True`.
  - Added
    `test_text_messages_history_prompt_is_prefix_of_next_generation_prompt`,
    which tokenizes both renderings and asserts the committed history tokens are
    an exact prefix of the next generation prompt tokens.
  - Documented that future vLLM Metal state injection should be a
    single-conversation append-only state path, not generic prefix caching or
    ever-growing transcript re-prefill.
- Reapply notes:
  - Any future patch that injects saved state into vLLM Metal must use this
    history-vs-generation prefix contract as its first correctness gate.
  - If the Audex chat template or local response-policy insertion changes, the
    prefix test must remain green before a cache/state snapshot is reused.

## 2026-07-09 Single-Snapshot State Carryover Rationale

- Files touched:
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
  - `tests/test_vllm_sts_cli.py`
- Why:
  - The generic vLLM prefix-cache mechanism is not the right target for Audex
    30B on vLLM Metal. In the vendored fork, prefix-cache support is tied to
    the paged backend; Nemotron-H Audex 30B uses the non-paged contiguous path,
    where that subsystem is absent rather than merely disabled.
  - Mamba2 recurrent state is not per-token-block addressable in the same way
    transformer KV is. Generic block-hash prefix caching for a hybrid would
    require complete recurrent snapshots at multiple block boundaries plus
    eviction policy, while the CLI needs one append-only conversation snapshot:
    the state at the end of the last committed turn.
- What changed:
  - Replaced tentative wording in `docs/engineering/vllm-metal.md` with the concrete
    backend and Mamba2-state reasons for a single-snapshot design.
  - Added the invalidation rule to `PATCH.md`: any history rewrite invalidates
    the saved snapshot, and mismatch handling must fail loudly or rebuild rather
    than silently reuse stale state.
  - Added
    `test_resumed_history_sanitization_invalidates_stale_state_snapshot`, which
    exercises the real vLLM STS session constructor path: resumed history
    sanitizer rewrites an old system prompt, persists the sanitized history, and
    deletes the stale `<conversation_id>.kv.safetensors` snapshot.
- Reapply notes:
  - Do not port generic paged prefix caching into the non-paged Nemotron-H path
    as the first conversation-state solution. Implement the append-only
    single-snapshot path first.
  - Keep snapshot invalidation coupled to every persisted-history rewrite,
    including persona/system-prompt refresh and assistant-response scrubbing.

## 2026-07-09 Text Conversation State Hint Plumbing

- Files touched:
  - `audex_mac/vllm_sts_requests.py`
  - `audex_mac/vllm_runtime.py`
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_requests.py`
  - `tests/test_vllm_runtime.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- Why:
  - The vLLM Metal fork needs metadata to decide whether a saved
    single-conversation state snapshot can be reused, but changing vLLM's
    public `generate(prompt, sampling_params, request_id)` call would be a
    larger and less upstream-friendly seam.
  - `SamplingParams.extra_args` is already the Audex-Mac control channel for
    CFG pairing, speech-token masks, and related Metal patches.
- What changed:
  - Text conversation requests can now carry Audex-owned state keys in
    `SamplingParams.extra_args`:
    - `audex_text_state_key`
    - `audex_text_state_mode=append`
    - `audex_text_state_prefix_token_count`
    - `audex_text_state_prefix_token_hash`
  - `VllmSpeechToSpeechSession` computes the prefix metadata from the committed
    text history prompt rendered with `add_generation_prompt=False`, not from
    the pending user turn.
  - Sync, async, and streaming text response paths pass the hint through to the
    request builder when a persisted conversation id exists.
- Validation:
  - `test_text_messages_response_request_can_carry_conversation_state_hint`
    asserts the request-builder `extra_args` shape.
  - `test_async_vllm_runtime_passes_text_conversation_state_hint` asserts the
    async runtime hands those fields to `SamplingParams`.
  - `test_text_conversation_state_kwargs_use_committed_history_prefix` asserts
    the CLI computes the count/hash from the committed history prompt.
- Reapply notes:
  - These fields are intentionally inert until the non-paged vLLM Metal runner
    consumes them. The next fork patch should capture state immediately before
    `_cleanup_finished_requests` deletes `RequestState.cache`, then compare
    count/hash before injecting that snapshot into a later append-mode text
    request.

## 2026-07-09 Non-Paged Text State Snapshot Metadata Hook

- Files touched:
  - `audex_mac/patches/vllm_metal_cfg.py`
  - `tests/test_audex_patches.py`
  - `docs/engineering/vllm-metal.md`
- Why:
  - The vLLM Metal non-paged runner owns live text request state in
    `_request_states[req_id]` and deletes it in `_cleanup_finished_requests`.
    The single-snapshot conversation path needs a hook before that deletion.
  - This slice proves the request lifecycle seam without retaining cache objects
    or changing generation behavior.
- What changed:
  - Extended the existing `_cleanup_finished_requests` wrapper used for
    persistent non-paged batch-cache flushing.
  - Before delegating to the original cleanup, the wrapper now scans evicted
    request states for `audex_text_state_*` append-mode hints in
    `SamplingParams.extra_args`.
  - Matching text requests record metadata in
    `runner._audex_text_state_snapshots[state_key]`: request id, mode, prefix
    token count/hash, prompt length, token count, generated token count, and
    whether a cache existed.
  - Snapshot metadata is stamped with `boundary=raw_generation_state`,
    `committed_boundary_verified=false`, and `reuse_eligible=false` because
    cleanup-time `RequestState.token_ids` is raw `prompt + sampled tokens`,
    while STS persists `scrub_spoken_answer(text.text)` as the committed
    assistant message.
  - Added `audex_text_state_boundary=committed_history_prefill`. In this mode,
    metadata uses `token_ids[:prompt_len]` as the boundary and marks the record
    reusable only if its count/hash match the supplied committed-history prefix,
    the request generated no more than the first sampled token, and a cache
    exists.
  - The record intentionally does not retain `RequestState.cache` yet.
- Validation:
  - `test_vllm_cfg_patch_records_text_state_snapshot_before_cleanup` creates a
    fake `MetalModelRunner`, applies the patch, and asserts metadata is captured
    before the original cleanup removes `_request_states`.
  - `test_vllm_cfg_text_state_snapshot_ignores_missing_or_wrong_mode` asserts
    non-append/missing hints do not produce snapshot records.
  - `test_vllm_cfg_text_state_snapshot_marks_committed_prefill_reusable`
    asserts committed-history prefill metadata can become reuse-eligible without
    relying on raw assistant text.
- Reapply notes:
  - The next patch should replace metadata-only capture with one retained
    single-conversation state snapshot only after proving the retained state is
    at the committed-history boundary, then compare prefix count/hash before
    using it to prefill only suffix tokens.
  - Keep this hook coupled to non-paged cleanup; the paged backend's prefix
    cache subsystem is still not the target for Nemotron-H Audex 30B.

## 2026-07-09 STS Segment-Balance Evidence Gate

- Files changed:
  - `audex_mac/vllm_diagnostics.py`
  - `tests/test_vllm_diagnostics.py`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
- Why:
  - The current 30B CFG product bottleneck is not just aggregate codec fps; it
    is tail collapse from uneven generated codec-frame counts across ordered
    TTS segments. A short final segment can force a low-width final admission
    wave and make a planner experiment look better or worse for the wrong
    reason.
  - The speech-output JSON already records `tts_segment_codec_frame_counts`,
    but the STS diagnostic result did not surface or gate on that shape.
- What changed:
  - `_probe_vllm_sts_default_runtime(...)` now copies
    `tts_observed_segments`, `tts_target_segments`, and
    `tts_segment_codec_frame_counts` from the product speech-output run log
    into `speech_streaming`.
  - `_assess_sts_timing(...)` now computes `tts_segment_count`,
    segment min/max/mean codec frames, max/min ratio, final-tail codec frames,
    final-tail/mean ratio, and `tts_tail_underfilled`.
  - `_sts_smoke_evidence_failures(...)` now reports a chunk-planner failure
    when a multi-segment CFG STS smoke run leaves the final segment below half
    the mean codec-frame count.
- Validation:
  - `.venv/bin/python -m pytest tests/test_vllm_diagnostics.py -q` reported
    `47 passed`.
  - `test_assess_sts_timing_reports_cfg_segment_tail_imbalance` uses the prior
    measured 30B bad-tail fixture shape `{165, 252, 181, 169, 205, 136, 128,
    64}` and asserts `tts_tail_to_mean_ratio=0.394`.
  - `test_sts_smoke_evidence_failures_reject_underfilled_tts_tail` asserts the
    evidence gate names the underfilled final segment rather than accepting the
    run as clean throughput evidence.
- Reapply notes:
  - Keep this as a diagnostic/evidence guard. It does not change NVIDIA sampler
    settings, CFG request pairing, speech-token masks, or decoder flow.
  - Future chunk-planner changes should compare this assessment before and
    after the change, alongside MLX-Audio/human intelligibility.

## 2026-07-09 CFG Underfilled Tail Merge at Capacity

- Files changed:
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/vllm-metal.md`
  - `docs/engineering/patches.md`
- Why:
  - The segment-balance evidence gate made the tail-collapse failure explicit:
    when CFG atomization already produces exactly the admitted pair width, the
    prior chunker returned every atom as-is. A tiny final sentence could then
    decode as a low-width tail even though merging it into the previous chunk
    would preserve ordered speech and avoid an avoidable final admission wave.
- What changed:
  - Added `DEFAULT_VLLM_CFG_TTS_MIN_TAIL_CHARS = 40`.
  - `split_cfg_spoken_tts_chunks(...)` now runs
    `_merge_underfilled_cfg_tail(...)` when atomization is at or below the
    target width and again after linear partitioning.
  - The merge is intentionally narrow: it only applies when there are at least
    four chunks, the chunk count is at capacity, the final chunk is below the
    threshold, and the response is not all-short chunks.
  - Speech-output JSON now records `tts_cfg_min_tail_chars`.
- Validation:
  - `.venv/bin/python -m pytest tests/test_vllm_sts_cli.py -q` reported
    `65 passed`.
  - `test_split_cfg_spoken_tts_chunks_merges_underfilled_tail_at_capacity`
    asserts a normal multi-sentence response with a final `"Useful."` atom
    merges that tail into the previous chunk.
  - `test_split_cfg_spoken_tts_chunks_keeps_all_short_chunks` continues to
    assert terse all-short responses keep separate chunks.
  - The exact tiny-tail fixture passed through the product vLLM Metal path and
    wrote `.audex/runs/tts-text-probe-vllm-20260709-153044.json`: all 7 CFG
    pairs were admitted up front with no waiting wave, the decoder remained
    enabled, and 837 codec frames completed at 47.518 codec fps with first
    audio ready in 4.928 seconds.
  - The resulting segment frame counts were
    `{164, 120, 115, 108, 112, 98, 120}`. The final segment is 0.97 times the
    segment mean, so the known underfilled-tail failure is absent.
  - The MLX-Audio oracle wrote
    `.audex/runs/tts-text-probe-vllm-20260709-153044.eval.json` and passed with
    ratio `0.9709677419354839` and no excessive repetition.
- Reapply notes:
  - This changes only CFG text segmentation before paired TTS requests are
    submitted. It does not change CFG scale, CFG pair math, speech-token masks,
    top-k, temperature, or decoder flow.
  - Keep the rule conservative. If a future fixture shows prosody damage, tune
    the threshold with `tts_segment_codec_frame_counts` and oracle/human
    evidence rather than raising it blindly.

## 2026-07-09 Static CFG Scheduler KV Reset Before TTS

- Files changed:
  - `audex_mac/vllm_runtime.py`
  - `audex_mac/vllm_sts_cli.py`
  - `tests/test_vllm_runtime.py`
  - `tests/test_vllm_sts_cli.py`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
  - `docs/engineering/patches.md`
- Why:
  - The prior fixed-text/end-to-end comparison was contaminated by Codex GUI
    GPU activity. A clean CLI run measured 90.58% GPU idle residency at the
    lowest 338 MHz state before starting the model loop.
  - Three clean exact tiny-tail product probes established a stable fresh TTS
    control:
    - `.audex/runs/tts-text-probe-vllm-20260709-154759.json`: 837 frames,
      55.261 codec fps, first audio 4.712s.
    - `.audex/runs/tts-text-probe-vllm-20260709-154918.json`: 837 frames,
      54.129 codec fps, first audio 4.816s.
    - `.audex/runs/tts-text-probe-vllm-20260709-155024.json`: 840 frames,
      54.805 codec fps, first audio 4.901s.
    Mean throughput was 54.732 codec fps with a 2.07% min/max spread. All
    three oracle reports passed at ratio `0.9709677419354839` with no excessive
    repetition.
  - Two clean end-to-end controls generated the same eight-sentence response
    and 929 frames but missed realtime:
    - `.audex/runs/vllm-metal-diagnostic-20260709-155305.json`: 49.368 codec
      fps to the last frame; speech log full-stream rate 46.539 codec fps.
    - `.audex/runs/vllm-metal-diagnostic-20260709-160007.json`: 48.772 codec
      fps to the last frame; speech log full-stream rate 45.758 codec fps.
  - Fresh TTS-only replays of the exact generated response produced 51.473
    codec fps without native debug and 55.057 codec fps with native debug.
    This proved answer shape explains part of the tiny-tail difference, but
    native debug does not explain the repeatable shared-session regression.
- What changed:
  - `AudexAsyncVllmRuntime.reset_prefix_cache()` exposes the pinned vLLM
    `AsyncLLM.reset_prefix_cache()` API.
  - After ASR/text requests have finished and immediately before static CFG
    TTS, the STS CLI resets scheduler KV/prefix-cache bookkeeping. It records
    the result as `tts_prefix_cache_reset` in the turn JSON and fails loudly if
    vLLM refuses the reset.
  - The reset defaults on only when `AUDEX_VLLM_TTS_CFG=1` has selected the
    static CFG path. No-CFG/interleaved TTS keeps the previous behavior.
    `AUDEX_VLLM_RESET_PREFIX_CACHE_BEFORE_TTS=0` is the diagnostic A/B escape
    hatch.
  - This does not change NVIDIA sampler defaults, CFG pairing/math, segment
    planning, token masks, speech decoding, or the no-CFG fallback.
- Validation:
  - Scheduler-reset A/B run 1 wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-160329.json` and passed the
    realtime gate at 54.977 codec fps / 1.100 realtime ratio. Its speech-output
    log reported 51.320 codec fps through full stream completion.
  - Scheduler-reset A/B run 2 wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-160542.json` and passed at
    52.977 codec fps / 1.060 realtime ratio. Its full-stream rate was 49.588
    codec fps.
  - Both A/B WAVs passed the MLX-Audio oracle at ratio `0.9973474801061007` with no
    excessive repetition. Both retained eight complete CFG pairs, 929 total
    frames, no max-token hit, and a healthy final segment.
  - The final no-override product proof wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-161211.json`. Its turn log
    records `tts_prefix_cache_reset=true`; the diagnostic passed at 56.574
    codec fps / 1.131 realtime ratio and the speech log reported 52.700 codec
    fps through full stream completion. The MLX-Audio oracle passed at ratio
    `0.9973474801061007` with no excessive repetition.
  - Focused runtime/STS tests reported `108 passed`; the final full suite
    reported `460 passed, 3 skipped`. Lint, `bash -n start.sh`, and
    `git diff --check` passed.
- Reapply notes:
  - The upstream seam is `vllm.v1.engine.async_llm.AsyncLLM.reset_prefix_cache`
    -> `EngineCore.reset_prefix_cache` -> `Scheduler.reset_prefix_cache`.
    Recheck that the call remains async and returns a success boolean when the
    vLLM pin changes.
  - Keep the reset after text completion and before static CFG submission. Do
    not move it into interleaved TTS, where text/TTS requests may still be live.
  - Continue reporting both diagnostic last-frame throughput and speech-log
    full-stream throughput; they use different timing boundaries.

## 2026-07-09 Scope CFG Capacity Default Away From No-CFG

- Files changed:
  - `start.sh`
  - `tests/test_start_sh.py`
  - `docs/engineering/vllm-metal.md`
  - `PATCH.md`
  - `docs/engineering/patches.md`
- Why:
  - The no-CFG regression gate exposed that `start.sh` unconditionally exported
    `AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS=8`, even though that capacity was
    designed for CFG's 5120-token engine.
  - Without CFG wiring, the longer-context 30B engine correctly refused the
    unsafe reservation. `.audex/runs/vllm-metal-diagnostic-20260709-161400.json`
    reported 111.89 GB requested worst-case KV state against 48.42 GB of
    MLX/Metal headroom and failed before inference.
- What changed:
  - `start.sh` preserves any explicit user capacity override.
  - In the absence of an override, it supplies capacity `8` only when
    `AUDEX_VLLM_TTS_CFG` is truthy. Ordinary no-CFG startup passes an empty
    value, which the existing runtime patch treats as no override.
  - The MLX/Metal headroom guard remains unchanged and still fails loudly for
    unsafe explicit values.
- Validation:
  - `bash -n start.sh`, lint, and the focused startup/runtime/patch/STS suite
    passed with `155 passed`.
  - The fixed no-CFG product run wrote
    `.audex/runs/vllm-metal-diagnostic-20260709-161742.json` and passed at
    134.961 last-frame codec fps / 2.699 realtime ratio with first audio at
    2.486s. Its interleaved full-stream metric was 38.796 codec fps because the
    timer includes overlapping text/TTS work.
  - The no-CFG turn log records `tts_prefix_cache_reset=false` and
    `text_to_tts_interleaved=true`. The WAV passed the MLX-Audio oracle at ratio
    `0.9973474801061007` with no excessive repetition.
- Reapply notes:
  - Keep the CFG capacity default coupled to the product CFG switch, not merely
    to the presence of a vLLM Metal engine.
  - Preserve explicit user overrides so capacity/headroom diagnostics remain
    reproducible.

## 2026-07-09 Required Seeded Compact TTS-Window Decode

- Upstream repository and pin:
  - `https://github.com/vllm-project/vllm-metal`
  - `cd72e7d6d5c3eec452afe2693c3a45a0564d7650`
- Upstream seams patched:
  - `vllm_metal.v1.model_runner.MetalModelRunner._sequential_decode`
  - `vllm_metal.v1.model_runner.MetalModelRunner._batched_decode`
- Audex-Mac owner:
  - `audex_mac/patches/vllm_metal_cfg.py`
- Why:
  - The controlled CFG quality matrix must prove that all three arms use the
    compact speech-codec token-window projection. Merely requesting the path
    was insufficient because the patch could reject an unsupported request
    and silently delegate to vLLM Metal's full-vocabulary decode.
  - vLLM Metal creates a Torch per-request generator when `SamplingParams.seed`
    is set. The compact MLX sampler previously rejected such generators, which
    made fixed-seed quality runs fall through to the stock path and introduced
    a CPU-side sampling bridge.
- What changed:
  - Quality requests add
    `extra_args["audex_tts_require_compact_window_decode"] = True`.
  - The sequential and batched wrappers raise a fatal `RuntimeError` when a
    required request cannot use their compact path. Ordinary interactive
    requests do not carry the flag and retain the existing fallback behavior.
  - Seeded compact requests derive an MLX PRNG key from the configured seed and
    codec-step index. CFG uses the conditional row's key and expands the sampled
    token to both pair members, preserving lockstep sampling without invoking
    Torch's generator.
- API-shape guard:
  - Patch installation still requires callable `_sequential_decode` and
    `_batched_decode` symbols and marks each wrapper with its existing sentinel.
  - The audit flag is read only from `SamplingParams.extra_args`; missing flags
    leave product behavior unchanged.
- Validation and evidence:
  - Targeted tests cover sequential/batched fail-loud behavior, request-flag
    propagation, and deterministic per-step key derivation.
  - The full suite passed with `479 passed, 3 skipped`; formatting and Ruff
    passed.
  - Controlled 30B runs for `plain-reference`, `nvidia-tts-cfg`, and
    `audex-cfg3` all completed six one-segment passages with the requirement
    enabled. A pre-fix plain run failed on its first decode, demonstrating that
    the guard detects the previously silent seeded fallback.
- Limitations:
  - This proves compact decode remained eligible throughout each successful
    run; it does not attribute the earlier Activity Monitor CPU/GPU split to a
    single phase. Prefill, decoder, finalization, and per-recipe profiling remain
    a separate investigation.
- Reapply/update notes:
  - Recheck request-state generator creation, `SamplingParams.seed`, and the
    sequential/batched decode signatures whenever the vLLM Metal pin moves.
  - If upstream adds native seeded MLX sampling, remove the Audex key bridge
    only after proving identical CFG lockstep behavior and keeping the
    required-path assertion for controlled experiments.
  - Do not enable the fatal requirement on normal interactive STS requests.

## 2026-07-09 CFG3 Default With 256K Conversation Context

- Files and seams changed:
  - `start.sh` CFG default and non-paged capacity selection.
  - `audex_mac/vllm_cfg.py` engine length/batched-token configuration.
  - `audex_mac/vllm_runtime.py` explicit engine limit and runtime stats.
  - `audex_mac/vllm_sts_cli.py` context guard.
  - `audex_mac/conversations.py` persisted demo context policy.
  - `audex_mac/cli.py` parser bounds and visible startup recipe/status.
  - `personas/assistant.md` default spoken persona wording.
- Why:
  - Blind listening preferred CFG3 in four of six groups and nearly five after
    reconsideration, so quality mode should be the normal demo behavior.
  - A resumed 5,025-token prompt was falsely rejected against a hard-coded
    5,120 fallback even though the no-CFG engine was configured at 262,144.
  - The model card advertises up to one million tokens, but 262,144 is the
    deliberately bounded full-precision Mac demo target for this release.
- What changed:
  - `start.sh` defaults `AUDEX_VLLM_TTS_CFG` to `1` while preserving explicit
    `0` as the no-CFG speed path.
  - Plain and CFG engines both receive `max_model_len=262144`; the runtime
    exposes that exact limit to prompt validation instead of using 5,120.
  - CFG defaults to two worst-case max-length non-paged sequences rather than
    eight. At 256K this requested 27.97 GB against 48.42 GB of measured Metal
    headroom and still provides enough short blocks for product CFG pairs.
  - Existing one-million-token conversation metadata migrates to the 262,144
    release policy on automatic or explicit-ID resume. No history is compacted
    or silently removed.
  - Explicit `--max-context-tokens` values outside `1..262144` fail at the CLI
    boundary instead of configuring a limit the release has not validated.
  - Resume identity inference now lets the latest explicit "call me" statement
    replace stale stored metadata. The Gilroy reproduction exposed a name that
    had been polluted by a prior benchmark prompt; focused tests cover the
    precedence rule.
  - The stock persona now names Audex explicitly and asks for fewer than five
    spoken sentences by default, preserving room for the user in conversation.
  - The vLLM CLI no longer prints a `.kv.safetensors` path it does not create.
    Resumed transcripts are re-prefilled in full.
- Validation evidence:
  - The release suite passed with `484 passed, 3 skipped`; formatting, Ruff,
    shell syntax, local Markdown links, and staged-diff whitespace checks also
    passed.
  - The exact previously failing Gilroy fixture completed with
    `prompt_tokens=5029`, `prompt_token_budget=262048`, CFG3 enabled, and a clean
    speech end token.
  - `.audex/runs/sts-turn-vllm-20260709-191251.json` records the 256K context
    and 5,047 committed conversation tokens.
  - `.audex/runs/speech-output-vllm-20260709-191225.json` records two CFG
    segments, first audio at 2.675 seconds, 40.411 codec fps, and no max-token
    hit. The deliberately 96-token text diagnostic ended mid-sentence, so its
    WAV is not release-quality listening evidence.
- Limitations and reapply notes:
  - Full transcript prefill latency will grow with conversation length because
    vLLM state persistence/reuse is not implemented in this release.
  - Do not restore the old 5,120 CFG cap or eight-max-length-sequence capacity
    without re-running the conversation regression and Metal headroom guard.
  - The current model checkpoint config exposes 262,144 positions even though
    the model card advertises one million. Raising the Mac demo limit requires a
    separate memory, correctness, and long-context validation pass.

## Out-of-Date vLLM Metal Prompt

If upstream vLLM Metal has moved beyond the pinned commit, `./start.sh` should
print and log a copy/paste prompt like this:

```text
Role: You are a senior Python/MLX/vLLM Metal coding agent working in the Audex-Mac repository.

# Goal
Update Audex-Mac's vLLM Metal monkey patches from pinned commit <OLD_COMMIT> to upstream commit <NEW_COMMIT>, while preserving the project goal: native local Mac Audex speech-to-speech inference from ./start.sh.

# Success criteria
- Audex-Mac still pins vLLM Metal to an explicit commit.
- docs/engineering/patches.md is updated with every changed patch, upstream symbol/file touched, reason, and reapply notes.
- Patch guards fail loudly if the upstream API shape changes again.
- Existing tests pass, and any new patch behavior has targeted tests.
- ./start.sh can still detect cached Audex models and reach the CLI startup path.
- Do not silently remove the monkey-patch mechanism unless Audex-Mac no longer needs it and the replacement is proven.

# Constraints
- Keep Audex-Mac as the owner of demo integration code; do not fork vLLM Metal into this repo.
- Do not change NVIDIA sampler settings creatively.
- Do not add separate STT/TTS/VAD models.
- Preserve MIT licensing for Audex-Mac code and NVIDIA license notices for model/code artifacts.
- Prefer small, explicit edits over broad refactors.

# Context
- Current pinned vLLM Metal commit: <OLD_COMMIT>
- Upstream vLLM Metal HEAD: <NEW_COMMIT>
- Patch ledger: docs/engineering/patches.md
- Startup path: ./start.sh
- Primary validated model: nvidia/Nemotron-Labs-Audex-30B-A3B
- Minimally tested smaller model: nvidia/Nemotron-Labs-Audex-2B

# Validation
After changes, run the most relevant fast tests first, then a startup smoke test. Report any model-download or full-inference steps that were skipped because they are slow or require local cached weights.
```
## 2026-07-09 Direct Audio Response Latency Checkpoint

- The spoken latency path can now ask Audex to answer raw audio directly,
  stream that text into Audex TTS, and defer full-width Audex ASR until codec
  generation releases the engine. The exact ASR transcript remains the durable
  conversation history; no Whisper or external speech model is involved.
- `start.sh` now defaults to no-CFG for conversational throughput. CFG3 remains
  available with `AUDEX_VLLM_TTS_CFG=1` as the human-preferred quality mode.
- The speech decoder defaults to MLX CPU so it does not contend with the 30B
  EngineCore on the GPU. A deterministic CPU/GPU comparison differed only at
  numerical-noise scale.
- Short direct requests trim only padded audio-tower output before 30B prefill.
  Deferred ASR still uses the full 750 embeddings. The multimodal cache key
  includes embedding count so the two representations cannot alias.
- On `.audex/fixtures/codex-python-list-smoke-16k-mono.wav`, eager audio
  component loading plus 73 direct embeddings produced:
  - first response token: `0.819s` wall;
  - first coherent TTS chunk: `1.060s` wall;
  - first decoded PCM: `1.711s` from submit;
  - exact deferred transcript: `please explain python lists in one sentence`.
  Evidence:
  `.audex/runs/sts-turn-vllm-20260709-232338.json` and
  `.audex/runs/speech-output-vllm-20260709-232329.json`.
- This is not a sub-second-audio result. The first decoded PCM remains about
  `0.71s` outside the gate before CoreAudio buffering. Run logs deliberately
  report first token, coherent text, PCM, device write, and estimated audible
  time separately.
- Decoder retuning kept 32 steady frames: 16 frames reduced device-write delay
  but caused repeated queue/device underruns; 32 frames with tail decode delayed
  until first PCM had zero application-queue underruns in the measured run.

## 2026-07-10 Exact History Reuse and Native-Rate Playback Checkpoint

- Direct spoken requests now prime and verify the exact tokenized conversation
  prefix inside EngineCore. Reuse is accepted only when the requested prefix
  token tuple exactly matches the captured multimodal prefill boundary; a hash
  or reconstructed text match is not sufficient.
- Short direct requests crop padded NV-Whisper feature frames before the MLX
  audio tower and expose only the proportional Audex embedding count to the
  30B prompt. Deferred ASR deliberately retains the full 750-embedding recipe.
- Pre-submit staging now measures quiet audio captured by the microphone rather
  than wall time between callbacks. A 180 ms default rejects the observed
  mid-utterance pauses, and cancellation now cleans up child text/TTS tasks on
  `asyncio.CancelledError`. The 3.21-second replay has only about 130 ms after
  final voiced energy, so no pre-submit latency credit is claimed for it.
- No-CFG playback now uses a one-frame causal decoder primer, eight-frame steady
  chunks, and an 80 ms adaptive prebuffer floor. Audex still decodes at 16 kHz;
  a stateful integer linear interpolator feeds a 192 kHz PortAudio/Core Audio
  stream, avoiding the device's high-latency 16 kHz path without adding NumPy
  to the playback or inference loop. A 512-frame PortAudio block is required on
  the tested device: smaller 96/192 kHz blocks reported an underflow on every
  pre-generated write, while 192 kHz/512 reported none in the isolated probe.
- The output stream is imported and constructed while model audio is still
  being prepared. It is not entered or written until the semantic audio buffer
  and any pre-submit validation gate are ready.
- Full-precision BF16 hardware evidence on
  `.audex/runs/sts-turn-vllm-20260710-004711.json` records:
  - first decoded PCM at `1.539s` after submit;
  - first 48 kHz device write at `1.785s`;
  - estimated first DAC sample at `1.966s`;
  - output latency `0.181646s`, down from `0.512938s` at 16 kHz;
  - four device underflows and three application-queue underruns.
- The repository suite passes with `544 passed, 3 skipped`; Black, Ruff, shell
  syntax, Markdown-link coverage, and staged-diff whitespace checks are clean.
- The completed local NVFP4 artifact passes the semantic-audio gate twice on
  the same full utterance without pre-submit staging:
  - `.audex/runs/sts-turn-vllm-20260710-010030.json`: PCM `0.705s`, device write
    `0.905s`, estimated DAC onset `0.959s`, `gate_passed=true`;
  - `.audex/runs/sts-turn-vllm-20260710-010136.json`: PCM `0.706s`, device write
    `0.905s`, estimated DAC onset `0.959s`, `gate_passed=true`;
  - both runs generated the coherent model answer “Testing, testing—loud and
    clear,” had no initial device underflow or application queue underrun, and
    sustained about 64.9 codec fps; each reported one later device underflow.
- The BF16 Audex-2B checkpoint passes the same semantic-audio fixture twice in
  cold processes:
  - `.audex/runs/sts-turn-vllm-20260710-043555.json`: PCM `0.427s`, device write
    `0.561s`, estimated DAC onset `0.578s`, `gate_passed=true`;
  - `.audex/runs/sts-turn-vllm-20260710-043702.json`: PCM `0.402s`, device write
    `0.536s`, estimated DAC onset `0.553s`, `gate_passed=true`;
  - both runs transcribed the same fixture and generated the same coherent
    answer, “Got it, testing.”
- Engine context reservation is checkpoint-aware: Audex-2B is clamped to its
  declared 131,072-token limit while Audex-30B retains the 262,144-token Mac
  demo ceiling. CFG configuration preserves that clamp.
- No-CFG model audio is now eligible for the semantic gate. CFG3 remains the
  human-preferred quality recipe, but it is not required by the latency metric
  after the explicit product decision to use no-CFG for performance.
- Failed experiments were removed rather than left as product flags:
  recording-time GPU keepalive did not speed the encoder, fixed-width/cropped
  and masked fixed-width encoders hurt quality or performance, `mx.compile`
  recompiled dynamic utterance shapes, four-frame steady decoder chunks
  increased decoder overhead, and canceling text generation at the first
  semantic phrase regressed onset.
- Reapply notes:
  - preserve the exact prefix tuple comparison when vLLM Metal state handling
    changes;
  - keep deferred ASR full-width unless a separate transcript-quality gate
    proves proportional embeddings;
  - remeasure 16/48/96/192 kHz device latency and clean block size after
    audio-device or PortAudio changes;
  - the 30B BF16 compute gap and a Core ML/ANE audio-encoder prototype remain
    separate investigations; the current sub-second evidence covers NVFP4 30B
    and BF16 2B.
- Long-form interleaved playback now treats every underrun as audible and
  therefore requires zero initial/device/application-queue underruns. The
  Audex-owned scheduler holds ordinary running requests while TTS is running or
  waiting, restores them without freeing request state, and supports the
  `AUDEX_VLLM_SPEECH_FIRST_SCHEDULING=0` rollback. Interleaved tail batching now
  defaults off in `start.sh`; `AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL=1` remains
  diagnostic-only. On the same eight-sentence NVFP4 30B response, batched tails
  produced 24 device and 23 queue underruns over 69.7 seconds, while sequential
  speech-first playback produced 67.5 seconds with all three underrun indicators
  at zero in `.audex/runs/speech-output-vllm-20260710-052928.json`.
- Resumed startup greetings no longer depend on `kv_cache_loaded`, which was
  always false on the vLLM path and incorrectly selected the first-run “I'm
  Audex” introduction. New conversations retain the fixed introduction;
  resumed conversations with substantive history append an ephemeral 96-token
  greeting instruction after the committed history prefix, reuse the existing
  conversation-state key/hash, limit the result to two sentences, and never
  persist that synthetic exchange. The NVFP4 hardware smoke resumed the
  transistor conversation with: “Hello, it's great to continue our chat about
  the history of the transistor and integrated circuit. What would you like to
  discuss today?”

## 2026-07-10 In-Repository MLX-Audio TTS Oracle

- `audex_mac.tts_oracle.MlxAudioTranscriber` now owns the small MLX-Audio STT
  adapter used by both TTS evaluators. It loads one model lazily, reuses it
  across a quality manifest, and normalizes sentence text, timestamps, and
  elapsed time without depending on a sibling source checkout.
- The optional `oracle` project extra pins the locally proven
  `mlx-audio==0.3.1` runtime. Ordinary Audex installation and the product
  speech-to-speech path do not install or invoke this separate evaluation
  model.
- `scripts/evaluate_tts_wav.py` and
  `scripts/evaluate_tts_quality_manifest.py` import the adapter directly from
  Audex-Mac. Install the evaluator from a fresh checkout with
  `.venv/bin/python -m pip install -e '.[oracle]'`.
- A clean staged-tree export installed `.[oracle]` into a new virtual
  environment, loaded `mlx-community/parakeet-tdt-0.6b-v3`, and transcribed a
  prior TTS WAV at ratio `1.0` and word error rate `0.0`; model load took
  `0.457s` and transcription took `0.534s`.
- Validation passes with `563 passed`; Black, Ruff, shell syntax, Markdown-link
  coverage, staged-diff whitespace checks, and repository hygiene scans are
  clean.

## 2026-07-10 Behavioral-Test Correctness Pass

- `start.sh` now executes the pinned vLLM-Metal patch guards before installing
  Audex runtime shims; incompatible API shapes stop startup with the missing
  symbol and this patch ledger, while upstream movement writes an update prompt.
- Fast Gherkin scenarios now drive production routing, persistence, PCM/WAV
  preparation, decoder buffering, and run-log paths through controlled runtime
  seams instead of assigning the expected result into the test context.
- CI and the pre-commit hook run `pytest -m fast`. The two model-backed Gherkin
  scenarios are genuinely slow/local, skip when their explicit prerequisites
  are absent, and no longer fabricate passing model or audio evidence.
- The text benchmark now has a deterministic acceptance gate recorded in its
  run log and returned by the CLI. A real ten-turn Audex-2B direct-MLX run on
  Metal failed honestly because turn 9 returned the wrong final chunk; the
  gate reported `turn 9 does not produce [[3, 1, 4], [1, 5, 9], [2]]`.
- Black, Ruff, shell syntax, whitespace checks, and the final fast suite pass;
  the Metal-enabled run completed with `569 passed, 2 deselected`.
