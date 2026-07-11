"""Stage completed generated WAVs into the pinned OpenL3 corpus layout."""

from __future__ import annotations

import json
from pathlib import Path

from .audio_evaluation import AudioEvaluationRun, EvaluationTrack

_OPENL3_CATEGORIES = frozenset({"audiocaps", "song-describer"})


def stage_openl3_corpora(run: AudioEvaluationRun) -> dict[str, int]:
    """Hardlink enhanced metric WAVs by stable dataset row identifier."""

    outputs = _generation_outputs(run.run_dir)
    staged_files: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for case in run.cases:
        if (
            case.track is not EvaluationTrack.GENERATION
            or case.category not in _OPENL3_CATEGORIES
        ):
            continue
        output = outputs.get(case.case_id, {})
        source_text = str(output.get("enhanced_wav_path") or "").strip()
        if not source_text:
            raise RuntimeError(f"{case.case_id} has no enhanced metric WAV for OpenL3")
        source = Path(source_text)
        if not source.is_file():
            raise FileNotFoundError(
                f"{case.case_id} enhanced metric WAV does not exist: {source}"
            )
        filename = _metric_filename(case.source_row_id)
        destination = run.run_dir / "media" / "openl3" / case.category / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"OpenL3 staged path already exists: {destination}")
        destination.hardlink_to(source)
        counts[case.category] = counts.get(case.category, 0) + 1
        staged_files.append(
            {
                "case_id": case.case_id,
                "dataset": case.category,
                "source": str(source),
                "destination": str(destination),
            }
        )
    payload = {
        "schema_version": 1,
        "counts": dict(sorted(counts.items())),
        "files": staged_files,
    }
    path = run.run_dir / "generation" / "openl3-staging.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dict(sorted(counts.items()))


def _generation_outputs(run_dir: Path) -> dict[str, dict[str, object]]:
    path = run_dir / "generation" / "outputs.jsonl"
    outputs: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("generation output must be a JSON object")
        case_id = str(payload.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("generation output has no case_id")
        if case_id in outputs:
            raise ValueError(f"duplicate generation output case_id: {case_id}")
        outputs[case_id] = payload
    return outputs


def _metric_filename(source_row_id: str) -> str:
    raw = source_row_id.strip()
    if not raw or Path(raw).name != raw or raw in {".", ".."}:
        raise ValueError(f"unsafe OpenL3 filename source row ID: {source_row_id!r}")
    return raw if raw.lower().endswith(".wav") else f"{raw}.wav"
