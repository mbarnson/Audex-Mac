# Audex-Mac Demo Scope

Audex-Mac demonstrates a complete local speech-to-speech conversation with
NVIDIA Audex on Apple Silicon:

```sh
./start.sh
```

The user can type text directly or press Enter on an empty editor to start and
stop recording. Audex consumes typed text or the recorded audio
directly, and Audex speaks its locally generated response.

## Product Contract

- CLI-only typed or push-to-talk conversation with spoken responses.
- One persistent vLLM Metal engine on MLX/Metal.
- Audex audio input, text reasoning, speech-token output, and causal decoder.
- Non-thinking instruct mode by default; thinking is explicit.
- Faster no-CFG speech generation by default, with explicit CFG3 quality mode.
- Persistent readable transcripts and a 262,144-token active-context policy.
- No silent history pruning or compaction.

## Validated Model

The release-quality target is `nvidia/Nemotron-Labs-Audex-30B-A3B` on Apple
Silicon. The repository recognizes and can load Audex-2B, but 2B has received
only minimal local end-to-end validation. See the
[runbook](../operations/runbook.md#hardware-assumptions) for memory guidance.

When both models are cached, 30B is preferred. When neither is cached, startup
offers 2B as the smaller first download and explains the NVIDIA license.

## Runtime Policy

- vLLM Metal is an exact pinned external dependency, not a vendored fork.
- Audex-Mac owns explicit guarded monkey patches recorded in the
  [patch ledger](../engineering/patches.md).
- Conflicting CPU-mode environment settings fail before model launch.
- Generated audio, models, caches, logs, and conversations remain under ignored
  local paths.

## Semantic Boundary

The speech path must not use separate semantic helpers:

- no Whisper or separate STT
- no Kokoro or separate TTS
- no Silero VAD
- no external or cloud LLM

PCM conversion, microphone/speaker I/O, local codec handling, and
NVIDIA-provided Audex components are allowed.

## Acceptance Criteria

- A fresh supported Mac can reach the demo through `./start.sh`.
- The 30B model can carry a multi-turn spoken conversation without dropping
  committed history inside the configured 256K window.
- Audio input and output remain Audex-native.
- No-CFG is the visible default and CFG3 is an explicit quality override.
- Patch guards, lint, and fast tests pass.
- The README and docs state tested hardware, model scope, limitations, and
  licensing accurately.

## Non-Goals

- conversation compaction or summarization
- million-token Mac inference in this release
- VAD, endpointing, barge-in, or continuous microphone streaming
- GUI, LiveKit, or server API for the conversational demo
- exact token/logit parity with NVIDIA CUDA inference
- committing model weights, WAVs, or other generated binaries
