from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.web import cli

pytestmark = pytest.mark.fast


def test_web_cli_help_describes_local_browser_and_sound_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "browser chat and sound-generation" in output
    assert "--no-open" in output
    assert "--xcodec1-path" in output


def test_web_environment_enables_sound_generation_without_forcing_speech_cfg() -> None:
    env: dict[str, str] = {}

    cli._configure_web_environment(env)

    assert env == {
        "AUDEX_VLLM_ENABLE_CFG_WIRING": "1",
        "AUDEX_VLLM_CFG_MAX_MODEL_LEN": "8192",
        "AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS": "4",
    }
    assert "AUDEX_VLLM_TTS_CFG" not in env


def test_web_environment_respects_explicit_capacity_override() -> None:
    env = os.environ.copy()
    env["AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"] = "7"

    cli._configure_web_environment(env)

    assert env["AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"] == "7"


def test_web_environment_promotes_start_sh_empty_capacity_for_sound_cfg() -> None:
    env = {"AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS": ""}

    cli._configure_web_environment(env)

    assert env["AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"] == "4"


def test_web_cli_builds_local_application_without_loading_model_until_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    decoder_path = model_path / "audex_causal_speech_decoder"
    decoder_path.mkdir(parents=True)
    selected = SimpleNamespace(repo_id="nvidia/Nemotron-Labs-Audex-2B")
    monkeypatch.setattr(
        cli,
        "_resolve_model",
        lambda **_kwargs: (selected, model_path),
    )
    monkeypatch.setattr(
        cli,
        "preflight_audio_runtime",
        lambda _model: SimpleNamespace(
            ready=True,
            decoder_path=decoder_path,
            missing_items=(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_persona",
        lambda _persona: SimpleNamespace(
            persona_id="assistant",
            path=tmp_path / "assistant.md",
            system_prompt="System.",
        ),
    )
    served: list[tuple[object, dict[str, object]]] = []

    def fake_serve(application, **kwargs) -> None:
        served.append((application, kwargs))

    result = cli.main(
        ["--no-open", "--port", "0"],
        serve_fn=fake_serve,
    )

    assert result == 0
    application, kwargs = served[0]
    assert application.coordinator.runtime_factory.loaded is False
    assert kwargs == {"host": "127.0.0.1", "port": 0, "on_ready": None}


def test_web_model_resolution_prompts_before_first_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MissingProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            assert readiness == "speech"
            return False

    downloads: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", MissingProbe)
    monkeypatch.setattr(
        cli,
        "download_model_snapshot",
        lambda model, readiness: downloads.append((model.repo_id, readiness)),
    )
    monkeypatch.setattr(
        cli,
        "preflight_audio_runtime",
        lambda _model: SimpleNamespace(model_path=tmp_path / "downloaded"),
    )

    selected, resolved = cli._resolve_model(
        model="2b",
        model_path=None,
        yes_download=False,
        input_func=lambda prompt: "yes" if prompt == "Download now? [y/N] " else "",
    )

    assert selected.repo_id == "nvidia/Nemotron-Labs-Audex-2B"
    assert resolved == tmp_path / "downloaded"
    assert downloads == [(selected.repo_id, "speech")]
    assert "NVIDIA's model license" in capsys.readouterr().out
