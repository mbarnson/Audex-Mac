# Audex-Mac Documentation

The root [README](../README.md) is the public demo landing page. Detailed
engineering and operating material lives here.

## Project

- [Demo scope](project/scope.md) — supported experience, validated model, and
  explicit non-goals.
- [Development workflow](project/development.md) — public development, review,
  CI, and artifact discipline.

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

The stable implementation contract remains at [PATCH.md](../PATCH.md), as
requested by the project owner.
