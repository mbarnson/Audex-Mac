# Development Out Loud

Audex-Mac is intended to be developed publicly as a Codex-driven project.

The public repository is expected to live at:

```text
https://github.com/mbarnson/Audex-Mac
```

## Working Agreement

- Human maintainers define goals, review outcomes, and run local hardware tests
  when required.
- Codex agents perform implementation through GitHub issues and pull requests.
- Every material change should be traceable to an issue, a PR, or an explicit
  local run note.
- PRs should explain what changed, why it changed, how it was verified, and
  what remains unknown.
- Failed attempts are useful evidence. Record them in
  `docs/engineering/viability.md` when they change our understanding of Mac
  viability.

## Public Narrative

Use these files as the public technical narrative:

- `docs/project/scope.md`: product scope and acceptance contract.
- `docs/engineering/viability.md`: running technical status and evidence.
- `docs/engineering/patches.md`: exact vLLM Metal monkey-patch ledger.
- `docs/operations/runbook.md`: local usage and operational behavior.
- `features/*.feature`: executable behavioral intent.

## Issue Discipline

Create small issues with one clear outcome. Prefer issue titles like:

- `Bootstrap one-command start.sh skeleton`
- `Implement Hugging Face cache model selection`
- `Pin vLLM Metal and add patch guards`
- `Prove Audex-2B text-only generation`

Each issue should include:

- goal
- success criteria
- constraints
- relevant files
- validation expectations

## Pull Request Discipline

Each PR should include:

- summary
- linked issue
- implementation notes
- validation run
- artifacts or logs, if relevant
- known limitations

Do not merge a PR that weakens these invariants:

- no committed model weights
- no separate STT/TTS/VAD models in the speech-to-speech path
- no unpinned vLLM Metal dependency
- no silent sampler changes away from NVIDIA recommendations
- no unclear licensing around NVIDIA artifacts
- no bypassing local hooks with `git commit --no-verify`

## CI Expectations

GitHub CI should run fast tests only:

- unit tests
- fast BDD scenarios
- formatting/linting once configured

Local commits should run the same cheap gates through `.githooks/pre-commit`.
Install them with:

```sh
scripts/install-hooks.sh
```

If a hook fails, fix the failure and recommit. Do not use `--no-verify`.

CI is not expected to prove full local speech-to-speech because that requires:

- Apple Silicon hardware
- local audio devices
- large Hugging Face model snapshots
- substantial unified memory

Full inference evidence belongs in local run logs summarized in
`docs/engineering/viability.md`. Do not commit large raw logs or audio captures.

## Initial Public Issue Set

1. Bootstrap repo and one-command startup skeleton.
2. Add BDD test harness and make `@fast` scenarios executable.
3. Implement Hugging Face cache detection and model selection.
4. Pin vLLM Metal and implement patch guard framework.
5. Prove Audex-2B text-only load/generation path.
6. Add 10-turn text benchmark and run logging.
7. Port Audex audio input path.
8. Port speech-token generation.
9. Port Audex causal speech decoder to Mac.
10. Wire push-to-talk CLI.
