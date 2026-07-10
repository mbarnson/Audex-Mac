from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from audex_mac import audio_evaluation_cli, cli
from audex_mac.audio_evaluation_datasets import MaterializedAudio
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
            for category in ("dog", "rooster", "rain", "clock")
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
    assert manifest["model"]["size"] == "30b"
    assert environment["hf_token_present"] is True
    assert "hf_should_not_be_recorded" not in json.dumps(manifest)
    assert "hf_should_not_be_recorded" not in json.dumps(environment)


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


def test_audio_evaluation_cli_execution_requires_explicit_model_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        audio_evaluation_cli.main(
            [
                "--tier",
                "smoke",
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
            for category in ("dog", "rooster", "rain", "clock")
            for index in range(4)
        ),
        "d0rj/audiocaps": tuple(
            _caption_row(index, f"AudioCaps caption {index}") for index in range(4)
        ),
        "renumics/song-describer-dataset": tuple(
            _song_row(index, f"SongDescriber caption {index}") for index in range(4)
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


def _write_silent_wav(path: Path) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1600)
