# Audex-Mac Documentation

The root [README](../README.md) is the public demo landing page. Detailed
engineering and operating material lives here.

## Project

- [Demo scope](project/scope.md) — supported experience, validated model, and
  explicit non-goals.
- [Development workflow](project/development.md) — public development, review,
  CI, and artifact discipline.
- [Licensing and local artifacts](project/licensing.md) — what the MIT license
  covers, what NVIDIA licenses separately, and what must stay out of Git.

## Operations

- [Runbook](operations/runbook.md) — installation, model selection, diagnostics,
  fixtures, benchmarks, and quality evaluation.

## Engineering

- [Viability and evidence](engineering/viability.md) — current findings,
  measured local runs, and remaining limitations.
- [vLLM Metal history](engineering/vllm-metal.md) — the investigation and
  implementation record for the Mac runtime.
- [Patch ledger](engineering/patches.md) — every Audex-Mac-owned runtime patch,
  its upstream seam, validation, and reapplication notes.
- [NVFP4 conversion](engineering/nvfp4-quantization.md) — reproducible,
  quality-first Audex-30B routed-expert quantization.
- [Autonomous audio-capability evaluation](engineering/autonomous-audio-evaluation.md)
  — engineering design and implementation foundation for non-speech audio
  understanding and text-to-audio generation.
- [Audex Sound Lab](engineering/audex-sound-lab.md) — product and implementation
  design for conversational sound generation, live audio understanding, blind
  auditioning, and a local capability catalog.
- [Browser interface](engineering/browser-interface.md) — local HTTP/UI
  architecture, modes, cache identity, audio transport, and operating contract.
- [Browser low-latency speech streaming](engineering/browser-low-latency-streaming.md)
  — automatic incremental speech playback, live-turn events, PCM transport, and
  verification contract.

The stable implementation contract remains at [PATCH.md](../PATCH.md). 
I try to describe there the kinds of changes required
to keep this Rube Goldberg machine running.
You should be able to throw PATCH.md at your favorite model
(I tried with both Opus 4.8 and GPT5.5 High; they did fine)
how to reproduce this from the vllm-metal checkpoint.
