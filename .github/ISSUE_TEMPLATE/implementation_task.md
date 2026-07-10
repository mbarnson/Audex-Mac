---
name: Implementation task
about: Track one Codex-driven implementation outcome
title: ""
labels: implementation
assignees: ""
---

## Goal

State the concrete outcome this issue should deliver.

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Relevant docs/features updated

## Constraints

- Follow `docs/project/scope.md`.
- Do not add separate STT/TTS/VAD models.
- Do not change NVIDIA sampler settings creatively.
- Do not commit model weights, downloaded checkpoints, local audio captures, or venv/cache files.
- Keep vLLM Metal pinned; document monkey patches in
  `docs/engineering/patches.md`.

## Relevant Files

- `docs/project/scope.md`
- `docs/engineering/patches.md`
- `docs/operations/runbook.md`
- `docs/engineering/viability.md`
- `features/`

## Validation

List the expected fast tests, BDD scenarios, smoke tests, or local model runs.

## Notes

Add links, observations, or local hardware constraints.
