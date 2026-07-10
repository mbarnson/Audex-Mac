from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from audex_mac.tts_oracle import MlxAudioTranscriber

pytestmark = pytest.mark.fast


def test_mlx_audio_transcriber_loads_once_and_transcribes_a_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated_paths: list[str] = []

    class FakeModel:
        def generate(self, wav_path: str) -> object:
            generated_paths.append(wav_path)
            return SimpleNamespace(
                sentences=[
                    SimpleNamespace(start=0.0, end=0.5, text="  Hello "),
                    SimpleNamespace(start=0.5, end=1.0, text="world.  "),
                ]
            )

    loaded_models: list[str] = []

    def fake_load(model_name: str) -> FakeModel:
        loaded_models.append(model_name)
        return FakeModel()

    mlx_audio = ModuleType("mlx_audio")
    stt = ModuleType("mlx_audio.stt")
    utils = ModuleType("mlx_audio.stt.utils")
    utils.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)

    wav_path = tmp_path / "sample.wav"
    wav_path.touch()
    transcriber = MlxAudioTranscriber("example/parakeet")

    assert transcriber.load() >= 0.0
    result = transcriber.transcribe_file(wav_path)

    assert loaded_models == ["example/parakeet"]
    assert generated_paths == [str(wav_path)]
    assert result["text"] == "Hello world."
    assert result["segments"] == [
        {"start": 0.0, "end": 0.5, "text": "Hello"},
        {"start": 0.5, "end": 1.0, "text": "world."},
    ]
    assert float(result["elapsed"]) >= 0.0


@pytest.mark.parametrize(
    "script",
    ["evaluate_tts_wav.py", "evaluate_tts_quality_manifest.py"],
)
def test_tts_evaluator_does_not_require_a_legacy_checkout(script: str) -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MLX-Audio STT model" in result.stdout
    assert "speak" + "scribe" not in result.stdout.lower()
    assert "/Users/" + "patbarnson" + "/devel" not in result.stdout


def test_oracle_extra_installs_the_proven_mlx_audio_runtime() -> None:
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["oracle"] == ["mlx-audio==0.3.1"]


def test_repository_has_no_personal_checkout_or_legacy_oracle_references() -> None:
    repository = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = ("/Users/" + "patbarnson" + "/devel", "speak" + "scribe")
    matches: list[str] = []
    for relative_path in listed.stdout.splitlines():
        path = repository / relative_path
        if not path.is_file():
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for value in forbidden:
            if value.lower() in contents.lower():
                matches.append(f"{relative_path}: {value}")

    assert matches == []
