from __future__ import annotations

import hashlib
import json
import re
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from audex_mac import audio_evaluation_cli, cli
from audex_mac.audio_evaluation import (
    AudioEvaluationCase,
    AudioEvaluationRun,
    EvaluationTrack,
)
from audex_mac.audio_evaluation_datasets import MaterializedAudio
from audex_mac.audio_evaluation_esc50 import ESC50_HARD_NEGATIVES
from audex_mac.audio_evaluation_hf import DatasetPin
from audex_mac.vllm_runtime import VllmRequestResult

pytestmark = pytest.mark.fast


def _mmau_row(row_id: str, task: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "audio": [{"src": f"https://example.invalid/{row_id}.wav"}],
        "question": f"What is heard in {row_id}?",
        "choices": ["A dog", "A bell"],
        "answer": "A dog",
        "task": task,
        "category": "Reasoning",
    }


def _esc_row(filename: str, category: str) -> dict[str, Any]:
    return {"filename": filename, "category": category, "fold": 1, "target": 0}


def _caption_row(row_id: int, caption: str) -> dict[str, Any]:
    return {"audiocap_id": row_id, "caption": caption}


def _song_row(row_id: int, caption: str) -> dict[str, Any]:
    return {"caption_id": row_id, "track_id": row_id + 1000, "caption": caption}


@pytest.mark.parametrize("first_arg", ["eval-audio-capabilities"])
def test_top_level_cli_delegates_audio_evaluation_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    first_arg: str,
) -> None:
    calls: list[list[str]] = []

    def fake_eval_main(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(cli, "run_audio_evaluation_cli", fake_eval_main)

    assert cli.main([first_arg, "--tier", "smoke"]) == 7
    assert calls == [["--tier", "smoke"]]


def test_audio_evaluation_cli_materializes_smoke_manifest_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = {
        "TwinkStart/MMAU": tuple(
            _mmau_row(f"{task}-{index}", task)
            for task in ("sound", "music")
            for index in range(8)
        ),
        "ashraq/esc50": tuple(
            _esc_row(f"{category}-{index}.wav", category)
            for category in ("dog", "rooster", "rain", "clock_tick")
            for index in range(4)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(4)
        ),
    }

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    monkeypatch.setenv("HF_TOKEN", "hf_should_not_be_recorded")

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "smoke-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert "smoke-test" in capsys.readouterr().out
    run_dir = tmp_path / "runs" / "smoke-test"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "smoke"
    assert manifest["case_count"] == 32
    assert "OpenSound/AudioCaps" in {
        dataset["repo_id"] for dataset in manifest["datasets"]
    }
    assert manifest["model"]["size"] == "30b"
    assert manifest["model"]["context"] == {
        "checkpoint_max_position_embeddings": None,
        "configured_demo_max_tokens": 262144,
        "effective_engine_max_model_len": None,
        "model_card_max_tokens": 1_000_000,
    }
    assert manifest["generation_recipe"]["name"] == "audex_tta_cfg3_xcodec1"
    assert manifest["generation_recipe"]["cfg_scale"] == 3.0
    assert manifest["oracle_registry"]["signal"]["status"] == "implemented"
    assert (
        manifest["oracle_registry"]["clap"]["status"]
        == "implemented_worker_boundary_unqualified"
    )
    assert manifest["oracle_registry"]["clap"]["repo_id"] == "laion/clap-htsat-unfused"
    assert (
        manifest["oracle_registry"]["ast"]["status"]
        == "implemented_worker_boundary_unqualified"
    )
    assert (
        manifest["oracle_registry"]["ast"]["revision"]
        == "f826b80d28226b62986cc218e5cec390b1096902"
    )
    assert (
        manifest["oracle_registry"]["openl3_fd"]["status"]
        == "implemented_external_worker_unqualified"
    )
    assert manifest["understanding_protocol"]["scoring"].startswith("exact")
    assert environment["hf_token_present"] is True
    assert environment["git"]["available"] is True
    assert environment["git"]["commit"]
    assert environment["host"]["python"]
    assert "transformers" in environment["dependencies"]
    assert "hf_should_not_be_recorded" not in json.dumps(manifest)
    assert "hf_should_not_be_recorded" not in json.dumps(environment)


def test_audio_evaluation_cli_reads_hf_token_from_dotenv_without_recording_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows_by_repo = _smoke_rows()
    seen_client_headers: list[bool] = []

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        seen_client_headers.append(
            getattr(client, "_headers", {}).get("Authorization")
            == "Bearer hf_dotenv_secret"
        )
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_dotenv_secret\n", encoding="utf-8")

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "dotenv-token",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert all(seen_client_headers)
    run_dir = tmp_path / "runs" / "dotenv-token"
    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    environment_text = (run_dir / "environment.json").read_text(encoding="utf-8")
    assert "hf_dotenv_secret" not in manifest_text
    assert "hf_dotenv_secret" not in environment_text
    assert json.loads(environment_text)["hf_token_present"] is True


def test_audio_evaluation_cli_can_explicitly_skip_esc50(
    tmp_path: Path,
) -> None:
    rows_by_repo = _smoke_rows()
    fetched_repos: list[str] = []

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        fetched_repos.append(pin.repo_id)
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--skip-esc50",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "skip-esc",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert "ashraq/esc50" not in fetched_repos
    run_dir = tmp_path / "runs" / "skip-esc"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 24
    assert manifest["omitted_datasets"] == [
        {"reason": "explicit --skip-esc50", "repo_id": "ashraq/esc50"}
    ]


def test_audio_evaluation_cli_can_explicitly_skip_optional_song_describer(
    tmp_path: Path,
) -> None:
    rows_by_repo = _smoke_rows()
    fetched_repos: list[str] = []

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        fetched_repos.append(pin.repo_id)
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--skip-song-describer",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "skip-song",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert "renumics/song-describer-dataset" not in fetched_repos
    run_dir = tmp_path / "runs" / "skip-song"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 28
    assert manifest["omitted_datasets"] == [
        {
            "reason": "explicit --skip-song-describer",
            "repo_id": "renumics/song-describer-dataset",
        }
    ]


def test_audio_evaluation_cli_materializes_standard_manifest(
    tmp_path: Path,
) -> None:
    rows_by_repo = _standard_rows()
    fetched_repos: list[str] = []

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        fetched_repos.append(pin.repo_id)
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "standard",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "standard-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    assert fetched_repos == [
        "TwinkStart/MMAU",
        "ashraq/esc50",
        "d0rj/audiocaps",
        "renumics/song-describer-dataset",
    ]
    run_dir = tmp_path / "runs" / "standard-test"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "standard"
    assert manifest["case_count"] == 652
    generation_cases = (run_dir / "generation" / "cases.jsonl").read_text(
        encoding="utf-8"
    )
    assert generation_cases.count("\n") == 152
    assert "audex-mac/ualm-inspired-controls" in generation_cases
    assert not (run_dir / "generation" / "openl3-request.json").exists()


def test_audio_evaluation_cli_requires_isolated_workers_for_full_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        audio_evaluation_cli.main(
            [
                "--tier",
                "full",
                "--run-root",
                str(tmp_path / "runs"),
            ],
            fetch_rows=lambda *_args, **_kwargs: (),
            materialize_audio=lambda _row: MaterializedAudio(
                path="fixture.wav",
                sha256="sha",
                sample_rate=16_000,
                duration_seconds=1.0,
            ),
        )

    assert "full execution requires --semantic-worker-python" in capsys.readouterr().err


def test_full_openl3_execution_requires_exact_paper_corpus_counts() -> None:
    audio_evaluation_cli._require_full_openl3_corpus_counts(
        {"audiocaps": 4_875, "song-describer": 746}
    )
    with pytest.raises(RuntimeError, match="exact paper corpora"):
        audio_evaluation_cli._require_full_openl3_corpus_counts(
            {"audiocaps": 4_874, "song-describer": 746}
        )


def test_audio_evaluation_cli_executes_standard_from_prepared_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "prepared-audio.wav"
    _write_silent_wav(audio_path)
    understanding = AudioEvaluationCase(
        case_id="sound-prepared",
        track=EvaluationTrack.UNDERSTANDING,
        dataset_id="fixture/mmau",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="standard",
        source_row_id="sound-prepared",
        source_row_hash="hash-sound-prepared",
        license="fixture",
        category="sound",
        prompt="What is heard?",
        expected_answer="A",
        audio_path=str(audio_path),
        choices=("A", "B"),
        tags=("dataset:mmau",),
    )
    generation = AudioEvaluationCase(
        case_id="control-quantity-01",
        track=EvaluationTrack.GENERATION,
        dataset_id="audex-mac/ualm-inspired-controls",
        dataset_revision="2026-07-10",
        dataset_config="ualm-inspired",
        dataset_split="standard",
        source_row_id="quantity-01",
        source_row_hash="hash-control-quantity-01",
        license="local-synthetic-evaluation-prompts",
        category="structured-control",
        prompt="Three dogs bark one after another in a small room.",
        caption="Three dogs bark one after another in a small room.",
        hard_foil_caption="Rain falls softly.",
        tags=("control:quantity", "generation:structured-control"),
    )
    caption_generation = AudioEvaluationCase(
        case_id="audiocaps-123",
        track=EvaluationTrack.GENERATION,
        dataset_id="d0rj/audiocaps",
        dataset_revision="rev1",
        dataset_config="default",
        dataset_split="standard",
        source_row_id="123",
        source_row_hash="hash-audiocaps-123",
        license="fixture",
        category="audiocaps",
        prompt="A dog barks twice.",
        caption="A dog barks twice.",
        hard_foil_caption="Rain falls softly.",
        tags=("generation:caption",),
    )
    prepared = AudioEvaluationRun.create(
        root=tmp_path / "prepared",
        run_id="standard-prepared",
        tier="standard",
        master_seed=20260710,
        cases=(understanding, generation, caption_generation),
        manifest_metadata={"model": {"repo_id": "fixture/audex"}},
    )

    def fake_worker(command: tuple[str, ...]) -> int:
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        module = command[command.index("-m") + 1]
        if module.endswith("openl3_worker"):
            payload = {
                "schema_version": 2,
                "status": "PASS",
                "qualification": {"qualified": True},
                "fd_openl3_by_dataset": {"audiocaps": 70.0},
                "per_dataset": [{"dataset": "audiocaps", "fd_openl3": 70.0}],
            }
        else:
            payload = {
                "schema_version": 1,
                "status": "PASS",
                "qualification": {"qualified": True},
                "per_case": [
                    {"case_id": item["case_id"]} for item in request["requests"]
                ],
            }
        output_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return 0

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "standard",
            "--cases-from-run",
            str(prepared.run_dir),
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "standard-execute",
            "--semantic-worker-python",
            "/opt/audio-eval/bin/python",
            "--semantic-worker-device",
            "mps",
            "--openl3-reference-stats-root",
            str(tmp_path / "reference-stats"),
            "--openl3-worker-python",
            "/opt/openl3/bin/python",
            "--openl3-implementation-file",
            "/opt/stable/openl3_fd.py",
        ],
        runtime_factory=lambda model_path, profile: FakeAudioEvalRuntime(),
        decoder_factory=lambda _config: _decode_to_tone_wav,
        worker_command_runner=fake_worker,
        case_manifest_validator=lambda _cases, _tier: None,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Verdict: CHARACTERIZED" in output
    run_dir = tmp_path / "runs" / "standard-execute"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "CHARACTERIZED"
    assert summary["completed_cases"] == 3
    assert summary["generation"]["completed_cases"] == 2
    assert set(summary["generation"]["semantic_metrics"]) == {
        "ast",
        "clap",
        "openl3",
    }
    assert summary["generation"]["semantic_metrics"]["openl3"] == {
        "fd_openl3_by_dataset": {"audiocaps": 70.0}
    }


def test_audio_evaluation_cli_materializes_full_manifest(
    tmp_path: Path,
) -> None:
    rows_by_repo = _full_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "full",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "full-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    run_dir = tmp_path / "runs" / "full-test"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "full"
    assert manifest["case_count"] == 42
    assert (run_dir / "understanding" / "cases.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 10
    assert (run_dir / "generation" / "cases.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 32
    assert not (run_dir / "generation" / "openl3-request.json").exists()


def test_audio_evaluation_cli_writes_openl3_request_when_reference_stats_are_configured(
    tmp_path: Path,
) -> None:
    rows_by_repo = _full_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF fixture")
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=5.0,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "full",
            "--materialize-only",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "full-openl3",
            "--openl3-reference-stats-root",
            str(tmp_path / "reference-stats"),
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert exit_code == 0
    run_dir = tmp_path / "runs" / "full-openl3"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["openl3_reference_stats_root"] == str(tmp_path / "reference-stats")
    openl3_request = json.loads(
        (run_dir / "generation" / "openl3-request.json").read_text(encoding="utf-8")
    )
    assert openl3_request["run_id"] == "full-openl3"
    assert openl3_request["schema_version"] == 2
    assert [request["dataset"] for request in openl3_request["requests"]] == [
        "audiocaps",
        "song-describer",
    ]
    assert openl3_request["requests"][0]["expected_file_count"] == 4875
    assert openl3_request["requests"][1]["expected_file_count"] == 746


def test_audio_evaluation_cli_executes_smoke_run_with_unqualified_generation_oracles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = _smoke_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    runtime = FakeAudioEvalRuntime()

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--generation-oracles",
            "unqualified",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "execute-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
        runtime_factory=lambda model_path, profile: runtime,
        decoder_factory=lambda _config: _decode_to_silent_wav,
    )

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "Verdict: PROTOCOL_FAIL" in output
    assert "generation_oracles_not_qualified" in output
    run_dir = tmp_path / "runs" / "execute-test"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_cases"] == 32
    assert summary["verdict"] == "PROTOCOL_FAIL"
    assert "required_oracle_qualification_failed" in summary["protocol_failures"]
    generation_output_lines = (
        (run_dir / "generation" / "outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(generation_output_lines) == 8
    first_generation = json.loads(generation_output_lines[0])
    assert first_generation["structurally_valid"] is True
    assert first_generation["signal_metrics"]["nonempty"] is True
    assert runtime.one_final_calls == 24
    assert runtime.many_final_calls == 8


def test_audio_evaluation_cli_signal_oracle_characterizes_smoke_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = _smoke_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "signal-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
        runtime_factory=lambda model_path, profile: FakeAudioEvalRuntime(),
        decoder_factory=lambda _config: _decode_to_tone_wav,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Verdict: CHARACTERIZED" in output
    run_dir = tmp_path / "runs" / "signal-test"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "CHARACTERIZED"
    metric = json.loads(
        (run_dir / "generation" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert metric["oracle"] == "signal_sanity"
    assert metric["verdict"] == "PASS"
    generation_output = json.loads(
        (run_dir / "generation" / "outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    enhanced_path = Path(generation_output["enhanced_wav_path"])
    assert enhanced_path.is_file()
    assert enhanced_path.parent == run_dir / "media" / "enhanced"
    with wave.open(str(enhanced_path), "rb") as enhanced:
        assert enhanced.getframerate() == 48_000
        assert enhanced.getnchannels() == 2
    clap_request = json.loads(
        (run_dir / "generation" / "clap-request.json").read_text(encoding="utf-8")
    )
    assert clap_request["run_id"] == "signal-test"
    assert len(clap_request["requests"]) == 8
    assert len(clap_request["qualification_requests"]) == 8
    assert {request["generated_wav_path"] for request in clap_request["requests"]} == {
        json.loads(line)["enhanced_wav_path"]
        for line in (run_dir / "generation" / "outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }


def test_audio_evaluation_cli_writes_ast_request_for_labeled_controls(
    tmp_path: Path,
) -> None:
    case = AudioEvaluationCase(
        case_id="control-quantity-01",
        track=EvaluationTrack.GENERATION,
        dataset_id="audex-mac/ualm-inspired-controls",
        dataset_revision="2026-07-10",
        dataset_config="ualm-inspired",
        dataset_split="standard",
        source_row_id="quantity-01",
        source_row_hash="hash-control-quantity-01",
        license="local-synthetic-evaluation-prompts",
        category="structured-control",
        prompt="Three dogs bark one after another in a small room.",
        caption="Three dogs bark one after another in a small room.",
        hard_foil_caption="A piano plays one quiet note.",
        tags=("control:quantity", "generation:structured-control"),
    )
    run = AudioEvaluationRun.create(
        root=tmp_path,
        run_id="completed-control",
        tier="standard",
        master_seed=20260710,
        cases=(case,),
        manifest_metadata={"model": {"repo_id": "fixture/audex"}},
    )
    enhanced_path = run.run_dir / "media" / "enhanced" / "control-quantity-01.wav"
    enhanced_path.write_bytes(b"RIFF fixture")
    run.record_output(
        case_id=case.case_id,
        payload={
            "enhanced_wav_path": str(enhanced_path),
            "raw_wav_path": None,
            "structurally_valid": True,
            "signal_metrics": {"finite": True, "nonempty": True},
        },
    )

    audio_evaluation_cli._write_completed_generation_worker_requests(run)

    ast_request = json.loads(
        (run.run_dir / "generation" / "ast-request.json").read_text(encoding="utf-8")
    )
    assert ast_request["run_id"] == "completed-control"
    assert ast_request["requests"] == [
        {
            "case_id": "control-quantity-01",
            "expected_labels": ["Dog", "Bark"],
            "forbidden_labels": ["Speech", "Music"],
            "generated_wav_path": str(enhanced_path),
        }
    ]
    assert (run.run_dir / "generation" / "clap-request.json").is_file()


def test_audio_evaluation_cli_passes_explicit_capability_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = _smoke_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--skip-esc50",
            "--skip-song-describer",
            "--capability-target",
            "technical_failure_rate_max=0.0",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "target-pass",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
        runtime_factory=lambda model_path, profile: FakeAudioEvalRuntime(),
        decoder_factory=lambda _config: _decode_to_tone_wav,
    )

    assert exit_code == 0
    assert "Verdict: PASS" in capsys.readouterr().out
    run_dir = tmp_path / "runs" / "target-pass"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "PASS"
    assert summary["capability_targets"] == {"technical_failure_rate_max": 0.0}
    assert manifest["capability_targets"] == {"technical_failure_rate_max": 0.0}


def test_audio_evaluation_cli_reports_capability_failure_for_missed_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_by_repo = _smoke_rows()

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--skip-esc50",
            "--skip-song-describer",
            "--capability-target",
            "accuracy_min=1.1",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "target-fail",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
        runtime_factory=lambda model_path, profile: FakeAudioEvalRuntime(),
        decoder_factory=lambda _config: _decode_to_tone_wav,
    )

    assert exit_code == 2
    assert "Verdict: CAPABILITY_FAIL" in capsys.readouterr().out
    summary = json.loads(
        (tmp_path / "runs" / "target-fail" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["verdict"] == "CAPABILITY_FAIL"
    assert summary["capability_failures"] == ["accuracy_min:1<1.1"]


def test_audio_evaluation_cli_executes_from_materialized_case_run(
    tmp_path: Path,
) -> None:
    rows_by_repo = _smoke_rows()
    fetch_count = 0

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        nonlocal fetch_count
        del client
        fetch_count += 1
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    materialize_exit = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--materialize-only",
            "--skip-esc50",
            "--skip-song-describer",
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-id",
            "prepared",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
    )

    assert materialize_exit == 0
    assert fetch_count == 2

    execute_exit = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--cases-from-run",
            str(tmp_path / "runs" / "prepared"),
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "from-prepared",
        ],
        fetch_rows=fake_fetch,
        runtime_factory=lambda model_path, profile: FakeAudioEvalRuntime(),
        decoder_factory=lambda _config: _decode_to_tone_wav,
    )

    assert execute_exit == 0
    assert fetch_count == 2
    manifest = json.loads(
        (tmp_path / "runs" / "from-prepared" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["case_count"] == 20
    assert manifest["source_cases_run"] == str(tmp_path / "runs" / "prepared")


def test_audio_evaluation_cli_resolves_cached_model_path_for_execution(
    tmp_path: Path,
) -> None:
    rows_by_repo = _smoke_rows()
    resolved_paths: list[tuple[str, str]] = []
    runtime_paths: list[Path | None] = []
    config_text = json.dumps({"max_position_embeddings": 131_072})

    def fake_fetch(pin: DatasetPin, *, client: object) -> tuple[Mapping[str, Any], ...]:
        del client
        return rows_by_repo[pin.repo_id]

    def fake_materialize(row: Mapping[str, Any]) -> MaterializedAudio:
        row_id = str(row.get("id") or row.get("filename"))
        path = tmp_path / "cache" / f"{row_id}.wav"
        _write_silent_wav(path)
        return MaterializedAudio(
            path=str(path),
            sha256=f"sha-{row_id}",
            sample_rate=16_000,
            duration_seconds=0.1,
        )

    def fake_resolver(model: str, profile: str) -> tuple[Path, str]:
        resolved_paths.append((model, profile))
        checkpoint = tmp_path / "checkpoint_folder_full"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(config_text, encoding="utf-8")
        (checkpoint / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 131_072}),
            encoding="utf-8",
        )
        return checkpoint, "fixture/audex"

    def fake_runtime_factory(model_path: Path | None, profile: str) -> Any:
        del profile
        runtime_paths.append(model_path)
        return FakeAudioEvalRuntime()

    exit_code = audio_evaluation_cli.main(
        [
            "--tier",
            "smoke",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "resolved-model-test",
        ],
        fetch_rows=fake_fetch,
        materialize_audio=fake_materialize,
        runtime_factory=fake_runtime_factory,
        decoder_factory=lambda _config: _decode_to_tone_wav,
        model_path_resolver=fake_resolver,
    )

    assert exit_code == 0
    assert resolved_paths == [("30b", "bf16")]
    assert runtime_paths == [tmp_path / "checkpoint_folder_full"]
    manifest = json.loads(
        (tmp_path / "runs" / "resolved-model-test" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["model"]["repo_id"] == "fixture/audex"
    assert manifest["model"]["path"] == str(tmp_path / "checkpoint_folder_full")
    assert manifest["model"]["file_hashes"]["config.json"] == {
        "bytes": len(config_text.encode("utf-8")),
        "sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
    }
    assert manifest["model"]["file_hashes"]["tokenizer_config.json"]["bytes"] > 0
    assert manifest["model"]["context"] == {
        "checkpoint_max_position_embeddings": 131_072,
        "configured_demo_max_tokens": 262144,
        "effective_engine_max_model_len": 131_072,
        "model_card_max_tokens": 1_000_000,
    }
    environment = json.loads(
        (tmp_path / "runs" / "resolved-model-test" / "environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert environment["audex_eval"]["model_repo"] == "fixture/audex"
    assert environment["audex_eval"]["model_path_exists"] is True


def test_audio_evaluation_cli_rejects_nvfp4_2b_selection(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        audio_evaluation_cli.main(
            [
                "--tier",
                "smoke",
                "--model",
                "2b",
                "--profile",
                "nvfp4",
                "--run-root",
                str(tmp_path / "runs"),
            ],
            fetch_rows=lambda *_args, **_kwargs: (),
            materialize_audio=lambda _row: MaterializedAudio(
                path="fixture.wav",
                sha256="sha",
                sample_rate=16_000,
                duration_seconds=1.0,
            ),
        )


def _smoke_rows() -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {
        "TwinkStart/MMAU": tuple(
            _mmau_row(f"{task}-{index}", task)
            for task in ("sound", "music")
            for index in range(8)
        ),
        "ashraq/esc50": tuple(
            _esc_row(f"{category}-{index}.wav", category)
            for category in ("dog", "rooster", "rain", "clock_tick")
            for index in range(4)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(4)
        ),
    }


def _standard_rows() -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {
        "TwinkStart/MMAU": tuple(
            _mmau_row(f"{task}-{index}", task)
            for task in ("sound", "music")
            for index in range(130)
        ),
        "ashraq/esc50": tuple(
            _esc_row(f"{category}-{index}.wav", category)
            for category in ESC50_HARD_NEGATIVES
            for index in range(5)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(70)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(70)
        ),
    }


def _full_rows() -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {
        "TwinkStart/MMAU": tuple(
            _mmau_row(f"{task}-{index}", task)
            for task in ("sound", "music", "speech")
            for index in range(3)
        ),
        "ashraq/esc50": tuple(
            _esc_row(f"{category}-{index}.wav", category)
            for category in ("dog", "rain")
            for index in range(2)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(5)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(3)
        ),
    }


class FakeAudioEvalRuntime:
    def __init__(self) -> None:
        self.tokenizer = FakeAudioEvalTokenizer()
        self.one_final_calls = 0
        self.many_final_calls = 0

    async def generate_one_final(self, request: Any) -> VllmRequestResult:
        del request
        self.one_final_calls += 1
        return VllmRequestResult(
            text="A",
            token_ids=(11, 12),
            elapsed_seconds=0.1,
            finish_reason="stop",
            request_debug_name="understanding",
        )

    async def generate_many_final(
        self, requests: tuple[Any, ...]
    ) -> tuple[VllmRequestResult, ...]:
        self.many_final_calls += 1
        end = self.tokenizer.get_vocab()["<audiogen_end>"]
        return (
            VllmRequestResult(
                text="",
                token_ids=tuple(
                    self.tokenizer.codec_token_id(index)
                    for index in _phase_valid_codec_ids(2000)
                )
                + (end,),
                elapsed_seconds=0.3,
                finish_reason="stop",
                request_debug_name=requests[0].debug_name,
            ),
            VllmRequestResult(
                text="",
                token_ids=(end,),
                elapsed_seconds=0.3,
                finish_reason="stop",
                request_debug_name=requests[1].debug_name,
            ),
        )


class FakeAudioEvalTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self._vocab = {
            "<audiogen_start>": 100,
            "<audiogen_end>": 101,
            "<sound>": 102,
            "<|audio_bos|>": 103,
            "<|audio_eos|>": 104,
        }
        self._vocab.update(
            {f"<audiocodec_{index}>": 1000 + index for index in range(4096)}
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def codec_token_id(self, codec_id: int) -> int:
        return self._vocab[f"<audiocodec_{codec_id}>"]

    def encode(self, text: str) -> list[int]:
        tokens = re.findall(r"<unk>|[^\s]+", text)
        return [abs(hash(part)) % 500 + 200 for part in tokens]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )


def _phase_valid_codec_ids(count: int) -> tuple[int, ...]:
    return tuple((index % 4) * 1024 + (index // 4) % 1024 for index in range(count))


def _decode_to_silent_wav(inspection: Any, destination: Path, case: Any) -> None:
    del inspection, case
    _write_silent_wav(destination)


def _decode_to_tone_wav(inspection: Any, destination: Path, case: Any) -> None:
    del inspection, case
    _write_tone_wav(destination)


def _write_silent_wav(path: Path) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1600)


def _write_tone_wav(path: Path) -> None:
    import math
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(8000):
        sample = int(8000 * math.sin(2.0 * math.pi * 440.0 * index / 16_000))
        frames.extend(sample.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(bytes(frames))
