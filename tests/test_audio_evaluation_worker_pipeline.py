from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    EvaluationTrack,
)
from audex_mac.audio_evaluation_worker_pipeline import (
    GenerationWorkerConfig,
    run_generation_worker_pipeline,
)

pytestmark = pytest.mark.fast


def _generation_case(case_id: str) -> AudioEvaluationCase:
    return AudioEvaluationCase(
        case_id=case_id,
        track=EvaluationTrack.GENERATION,
        dataset_id="fixture/audio",
        dataset_revision="revision",
        dataset_config="default",
        dataset_split="test",
        source_row_id=case_id,
        source_row_hash=f"hash-{case_id}",
        license="fixture",
        category="audiocaps",
        prompt=f"Caption {case_id}",
        caption=f"Caption {case_id}",
        hard_foil_caption=f"Foil {case_id}",
    )


def _write_request(path: Path, case_ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "pipeline",
                "requests": [{"case_id": case_id} for case_id in case_ids],
            }
        ),
        encoding="utf-8",
    )


def test_generation_worker_pipeline_runs_and_ingests_qualified_metrics(
    tmp_path: Path,
) -> None:
    cases = (_generation_case("gen-a"), _generation_case("gen-b"))
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="pipeline",
        tier="standard",
        master_seed=7,
        cases=cases,
        manifest_metadata={},
    )
    _write_request(run.run_dir / "generation" / "clap-request.json", ("gen-a", "gen-b"))
    _write_request(run.run_dir / "generation" / "ast-request.json", ("gen-a",))
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> int:
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        module = command[command.index("-m") + 1]
        if module.endswith("clap_worker"):
            payload = {
                "schema_version": 1,
                "status": "PASS",
                "qualification": {"qualified": True},
                "per_case": [
                    {
                        "case_id": "gen-a",
                        "caption_similarity": 0.8,
                        "hard_foil_win": True,
                        "hard_foil_margin": 0.3,
                        "retrieval_rank": 1,
                    },
                    {
                        "case_id": "gen-b",
                        "caption_similarity": 0.6,
                        "hard_foil_win": False,
                        "hard_foil_margin": -0.1,
                        "retrieval_rank": 2,
                    },
                ],
            }
        else:
            payload = {
                "schema_version": 1,
                "status": "PASS",
                "qualification": {"qualified": True},
                "per_case": [
                    {
                        "case_id": "gen-a",
                        "expected_label_hit": True,
                        "forbidden_label_false_positive": False,
                    }
                ],
            }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    result = run_generation_worker_pipeline(
        run,
        config=GenerationWorkerConfig(
            semantic_python=Path("/opt/audio-eval/bin/python"),
            semantic_device="mps",
        ),
        command_runner=fake_run,
    )

    assert result.qualified is True
    assert result.failures == ()
    assert len(commands) == 2
    assert all("mps" in command for command in commands)
    metrics = [
        json.loads(line)
        for line in (run.run_dir / "generation" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(metric["case_id"], metric["oracle"]) for metric in metrics] == [
        ("gen-a", "clap"),
        ("gen-b", "clap"),
        ("gen-a", "ast"),
    ]
    qualification = json.loads(
        (run.run_dir / "generation" / "oracle_qualification.json").read_text(
            encoding="utf-8"
        )
    )
    assert qualification["qualified"] is True
    assert set(qualification["oracle_results"]) == {"ast", "clap"}


def test_generation_worker_pipeline_fails_closed_on_case_coverage_mismatch(
    tmp_path: Path,
) -> None:
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="pipeline",
        tier="standard",
        master_seed=7,
        cases=(_generation_case("gen-a"), _generation_case("gen-b")),
        manifest_metadata={},
    )
    _write_request(run.run_dir / "generation" / "clap-request.json", ("gen-a", "gen-b"))
    _write_request(run.run_dir / "generation" / "ast-request.json", ("gen-a",))

    def fake_run(command: tuple[str, ...]) -> int:
        output = Path(command[command.index("--output") + 1])
        module = command[command.index("-m") + 1]
        case_ids = ("gen-a",) if module.endswith("clap_worker") else ("gen-a",)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "qualification": {"qualified": True},
                    "per_case": [{"case_id": case_id} for case_id in case_ids],
                }
            ),
            encoding="utf-8",
        )
        return 0

    result = run_generation_worker_pipeline(
        run,
        config=GenerationWorkerConfig(
            semantic_python=Path("/opt/audio-eval/bin/python"),
            semantic_device="mps",
        ),
        command_runner=fake_run,
    )

    assert result.qualified is False
    assert result.failures == (
        "clap_worker_case_coverage_mismatch:missing=['gen-b']:unexpected=[]",
    )
    assert (
        not (run.run_dir / "generation" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )


def test_generation_worker_pipeline_fails_loud_when_worker_is_unqualified(
    tmp_path: Path,
) -> None:
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="pipeline",
        tier="standard",
        master_seed=7,
        cases=(_generation_case("gen-a"),),
        manifest_metadata={},
    )
    _write_request(run.run_dir / "generation" / "clap-request.json", ("gen-a",))
    _write_request(run.run_dir / "generation" / "ast-request.json", ("gen-a",))

    def fake_run(command: tuple[str, ...]) -> int:
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "UNSCORED",
                    "reason": "oracle_not_qualified",
                    "qualification": {"qualified": False},
                    "per_case": [{"case_id": "gen-a"}],
                }
            ),
            encoding="utf-8",
        )
        return 2

    result = run_generation_worker_pipeline(
        run,
        config=GenerationWorkerConfig(
            semantic_python=Path("/opt/audio-eval/bin/python"),
            semantic_device="mps",
        ),
        command_runner=fake_run,
    )

    assert result.qualified is False
    assert result.failures == (
        "clap_worker_failed:exit=2:status=UNSCORED:reason=oracle_not_qualified",
        "ast_worker_failed:exit=2:status=UNSCORED:reason=oracle_not_qualified",
    )


def test_generation_worker_pipeline_records_worker_launch_failure(
    tmp_path: Path,
) -> None:
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="pipeline",
        tier="standard",
        master_seed=7,
        cases=(_generation_case("gen-a"),),
        manifest_metadata={},
    )
    _write_request(run.run_dir / "generation" / "clap-request.json", ("gen-a",))
    _write_request(run.run_dir / "generation" / "ast-request.json", ("gen-a",))

    def unavailable(_command: tuple[str, ...]) -> int:
        raise FileNotFoundError("isolated interpreter is missing")

    result = run_generation_worker_pipeline(
        run,
        config=GenerationWorkerConfig(
            semantic_python=Path("/missing/python"),
            semantic_device="mps",
        ),
        command_runner=unavailable,
    )

    assert result.qualified is False
    assert result.failures == (
        "clap_worker_launch_failed:FileNotFoundError: isolated interpreter is missing",
        "ast_worker_launch_failed:FileNotFoundError: isolated interpreter is missing",
    )


def test_generation_worker_pipeline_ingests_qualified_openl3_dataset_metrics(
    tmp_path: Path,
) -> None:
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="pipeline",
        tier="standard",
        master_seed=7,
        cases=(_generation_case("gen-a"),),
        manifest_metadata={},
    )
    _write_request(run.run_dir / "generation" / "clap-request.json", ("gen-a",))
    _write_request(run.run_dir / "generation" / "ast-request.json", ("gen-a",))
    (run.run_dir / "generation" / "openl3-request.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "pipeline",
                "requests": [
                    {"dataset": "audiocaps"},
                    {"dataset": "song-describer"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command: tuple[str, ...]) -> int:
        output = Path(command[command.index("--output") + 1])
        module = command[command.index("-m") + 1]
        if module.endswith("openl3_worker"):
            payload = {
                "schema_version": 2,
                "status": "PASS",
                "qualification": {"qualified": True},
                "fd_openl3_by_dataset": {
                    "audiocaps": 70.0,
                    "song-describer": 65.0,
                },
                "per_dataset": [
                    {"dataset": "audiocaps", "fd_openl3": 70.0},
                    {"dataset": "song-describer", "fd_openl3": 65.0},
                ],
            }
        else:
            payload = {
                "schema_version": 1,
                "status": "PASS",
                "qualification": {"qualified": True},
                "per_case": [{"case_id": "gen-a"}],
            }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    result = run_generation_worker_pipeline(
        run,
        config=GenerationWorkerConfig(
            semantic_python=Path("/opt/audio-eval/bin/python"),
            semantic_device="mps",
            openl3_python=Path("/opt/openl3/bin/python"),
            openl3_implementation_file=Path("/opt/stable/openl3_fd.py"),
        ),
        command_runner=fake_run,
    )

    assert result.qualified is True
    metrics = [
        json.loads(line)
        for line in (run.run_dir / "generation" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert metrics[-1] == {
        "fd_openl3_by_dataset": {
            "audiocaps": 70.0,
            "song-describer": 65.0,
        },
        "oracle": "openl3",
    }
