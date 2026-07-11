from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac import cli
from audex_mac.cli import DEFAULT_STS_BACKEND, DEFAULT_TEXT_BACKEND
from audex_mac.speech_output import SpeechOutputSmokeResult
from audex_mac.sts_cli import SpeechToSpeechTurnResult
from audex_mac.text_gate import TextBenchmarkAssessment, TextQualityObservation
from audex_mac.text_generation import run_text_benchmark

pytestmark = pytest.mark.fast


def test_start_sh_guards_empty_args_with_set_u() -> None:
    script = Path("start.sh").read_text(encoding="utf-8")

    assert "ARGS_COUNT=0" in script
    assert "exec_vllm_metal_cli()" in script
    assert "exec_project_cli()" in script
    assert (
        'exec env PYTHONPATH="${VLLM_METAL_PYTHONPATH}" "${VLLM_METAL_VENV_DIR}/bin/python" -m audex_mac.cli "${ARGS[@]}"'
        not in script
    )
    assert 'exec "${VENV_DIR}/bin/python" -m audex_mac.cli "${ARGS[@]}"' not in script


def test_start_sh_defaults_vllm_logs_to_error_for_cli_stdout() -> None:
    script = Path("start.sh").read_text(encoding="utf-8")

    assert 'VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"' in script
    assert 'TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"' in script
    assert "AUDEX_MAC_AUTO_PATCHES=1" in script
    assert 'export AUDEX_VLLM_TTS_CFG="${AUDEX_VLLM_TTS_CFG:-0}"' in script
    assert (
        "export AUDEX_VLLM_DIRECT_AUDIO_RESPONSE="
        '"${AUDEX_VLLM_DIRECT_AUDIO_RESPONSE:-1}"' in script
    )
    assert (
        "export AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL="
        '"${AUDEX_VLLM_INTERLEAVED_TTS_BATCH_TAIL:-0}"' in script
    )
    assert (
        "export AUDEX_VLLM_EAGER_AUDIO_COMPONENTS="
        '"${AUDEX_VLLM_EAGER_AUDIO_COMPONENTS:-1}"' in script
    )
    assert "require_metal_env VLLM_METAL_MEMORY_FRACTION auto" in script
    assert "vllm_nonpaged_kv_capacity_seqs()" in script
    assert 'case "${AUDEX_VLLM_TTS_CFG:-}" in' in script
    assert 'echo "2"' in script
    assert 'echo "${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS}"' in script
    assert (
        'AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${nonpaged_kv_capacity_seqs}"' in script
    )
    assert (
        'AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS="${AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS:-8}"'
        not in script
    )


def test_start_sh_installs_audex_deps_into_vllm_metal_runtime() -> None:
    script = Path("start.sh").read_text(encoding="utf-8")

    assert 'VLLM_METAL_DEPS_STAMP="${STATE_DIR}/vllm-metal-deps.stamp"' in script
    assert "NEEDS_AUDIO_EVAL_DEPS=0" in script
    assert "ensure_vllm_metal_audex_deps()" in script
    assert "import audex_mac, huggingface_hub, prompt_toolkit, sounddevice" in script
    assert "soundfile, torch, transformers" in script
    assert 'install_target="${ROOT_DIR}[audio-eval]"' in script
    assert 'install_target="${ROOT_DIR}"' in script
    assert '"${python_bin}" -m pip install -e "${install_target}"' in script
    assert "ensure_vllm_metal_audex_deps" in script
    assert "sound-lab|web)" in script


def test_start_sh_reinstalls_audex_deps_after_runtime_rebuild(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    root = tmp_path / "project"
    state = tmp_path / "state"
    python_log = tmp_path / "python.log"
    (runtime / "bin").mkdir(parents=True)
    root.mkdir()
    state.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    stamp = state / "vllm-metal-deps.stamp"
    stamp.touch()
    runtime_python = runtime / "bin" / "python"
    runtime_python.write_text(
        "#!/bin/sh\n" 'printf \'%s\\n\' "$*" >> "$FAKE_PYTHON_LOG"\n' "exit 0\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o755)

    start_sh = Path("start.sh").resolve()
    command = f"""
source {start_sh!s}
VLLM_METAL_VENV_DIR={runtime!s}
VLLM_METAL_DEPS_STAMP={stamp!s}
STATE_DIR={state!s}
ROOT_DIR={root!s}
VLLM_METAL_PYTHONPATH={tmp_path!s}
VLLM_METAL_INSTALL_REQUIRED=1
ensure_vllm_metal_audex_deps
"""
    env = os.environ | {"FAKE_PYTHON_LOG": str(python_log)}

    result = subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    invocation = python_log.read_text(encoding="utf-8")
    assert f"-m pip install -e {root!s}" in invocation


def test_start_sh_installs_audio_eval_extra_for_eval_command(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    root = tmp_path / "project"
    state = tmp_path / "state"
    python_log = tmp_path / "python.log"
    (runtime / "bin").mkdir(parents=True)
    root.mkdir()
    state.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    stamp = state / "vllm-metal-deps.stamp"
    runtime_python = runtime / "bin" / "python"
    runtime_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_PYTHON_LOG"\n'
        'case "$*" in\n'
        "  *'scipy, sounddevice, soundfile, torch, transformers'*) exit 1 ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o755)

    start_sh = Path("start.sh").resolve()
    command = f"""
source {start_sh!s}
VLLM_METAL_VENV_DIR={runtime!s}
VLLM_METAL_DEPS_STAMP={stamp!s}
STATE_DIR={state!s}
ROOT_DIR={root!s}
VLLM_METAL_PYTHONPATH={tmp_path!s}
VLLM_METAL_INSTALL_REQUIRED=0
NEEDS_AUDIO_EVAL_DEPS=1
ensure_vllm_metal_audex_deps
"""
    env = os.environ | {"FAKE_PYTHON_LOG": str(python_log)}

    result = subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    invocation = python_log.read_text(encoding="utf-8")
    assert f"-m pip install -e {root!s}[audio-eval]" in invocation


def test_start_sh_ensures_managed_checkout_before_reusing_runtime() -> None:
    script = Path("start.sh").read_text(encoding="utf-8")

    checkout = script.index("\nensure_vllm_metal_checkout\n")
    fast_path = script.index(
        'if [[ "${REFRESH_DEPS}" == "0" && '
        '"${VLLM_METAL_INSTALL_REQUIRED}" == "0" && '
        '-x "${VLLM_METAL_VENV_DIR}/bin/python" ]]'
    )

    assert checkout < fast_path


def test_start_sh_replaces_unmanaged_vendor_checkout(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor" / "vllm-metal"
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    git_log = tmp_path / "git.log"
    (vendor / ".venv-vllm-metal" / "bin").mkdir(parents=True)
    (vendor / "foreign-marker").write_text("orphaned runtime\n", encoding="utf-8")
    bin_dir.mkdir()

    pin_python = bin_dir / "pin-python"
    pin_python.write_text(
        "#!/bin/sh\n"
        'case "$3" in\n'
        "  repo) echo https://github.com/vllm-project/vllm-metal ;;\n"
        "  pinned_commit) echo pin ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_GIT_LOG"\n'
        'if [ "$1" = "clone" ]; then\n'
        '  mkdir -p "$3/.git"\n'
        "  exit 0\n"
        "fi\n"
        'case "$*" in\n'
        "  *'rev-parse HEAD'*) echo pin ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    for executable in (pin_python, fake_git):
        executable.chmod(0o755)

    start_sh = Path("start.sh").resolve()
    command = f"""
source {start_sh!s}
VLLM_METAL_VENDOR_DIR={vendor!s}
STATE_DIR={state!s}
PYTHON_BIN={pin_python!s}
VLLM_METAL_INSTALL_REQUIRED=0
ensure_vllm_metal_checkout
echo install-required=$VLLM_METAL_INSTALL_REQUIRED
"""
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_GIT_LOG": str(git_log),
    }

    result = subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "Moving unmanaged vLLM Metal runtime" in result.stdout
    assert "install-required=1" in result.stdout
    backups = list((state / "runtime-backups").glob("vllm-metal-*"))
    assert len(backups) == 1
    assert (backups[0] / "foreign-marker").read_text(encoding="utf-8") == (
        "orphaned runtime\n"
    )
    invocation = git_log.read_text(encoding="utf-8")
    assert f"clone https://github.com/vllm-project/vllm-metal {vendor!s}" in invocation


def test_start_sh_enforces_patch_guards_before_installing_runtime_shims() -> None:
    script = Path("start.sh").read_text(encoding="utf-8")

    assert "run_vllm_metal_patch_guards()" in script
    assert script.count("\n    run_vllm_metal_patch_guards\n") == 1
    assert script.count("\n  run_vllm_metal_patch_guards\n") == 1
    assert script.count("-m audex_mac.patch_guards") == 1
    assert script.count("-m audex_mac.patches.install") == 2
    assert (
        "run_vllm_metal_patch_guards\n"
        '    PYTHONPATH="${VLLM_METAL_PYTHONPATH}" '
        '"${VLLM_METAL_VENV_DIR}/bin/python" '
        "-m audex_mac.patches.install" in script
    )


def test_start_sh_patch_guard_function_propagates_guard_failure(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    vendor = tmp_path / "vendor"
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    log_path = tmp_path / "python.log"
    (runtime / "bin").mkdir(parents=True)
    vendor.mkdir()
    bin_dir.mkdir()
    runtime_python = runtime / "bin" / "python"
    runtime_python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$FAKE_GUARD_LOG"\n'
        'exit "${FAKE_GUARD_STATUS:-0}"\n',
        encoding="utf-8",
    )
    pin_python = bin_dir / "pin-python"
    pin_python.write_text("#!/bin/sh\necho pin\n", encoding="utf-8")
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *'refs/remotes/origin/main'*) echo upstream ;;\n"
        "  *'rev-parse HEAD'*) echo pin ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    for executable in (runtime_python, pin_python, fake_git):
        executable.chmod(0o755)

    start_sh = Path("start.sh").resolve()
    command = f"""
source {start_sh!s}
VLLM_METAL_VENV_DIR={runtime!s}
VLLM_METAL_VENDOR_DIR={vendor!s}
STATE_DIR={state!s}
PYTHON_BIN={pin_python!s}
VLLM_METAL_PYTHONPATH={tmp_path!s}
run_vllm_metal_patch_guards
echo should-not-reach
"""
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_GUARD_LOG": str(log_path),
        "FAKE_GUARD_STATUS": "7",
    }

    result = subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 7
    assert "should-not-reach" not in result.stdout
    invocation = log_path.read_text(encoding="utf-8")
    assert "-m audex_mac.patch_guards" in invocation
    assert "--installed-commit pin" in invocation
    assert "--pinned-commit pin" in invocation


def test_text_benchmark_defaults_to_vllm_backend() -> None:
    assert DEFAULT_TEXT_BACKEND == "vllm"
    assert run_text_benchmark.__kwdefaults__["backend"] == "vllm"


def test_text_benchmark_cli_reports_quality_miss_without_failing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            return True

    run_log = tmp_path / "text.json"
    run_log.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(
        cli,
        "run_text_benchmark",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_log_path=run_log,
            transcript=[{"assistant": "bad answer"}],
            assessment=TextBenchmarkAssessment(
                compatibility_failures=(),
                quality_observations=(
                    TextQualityObservation(
                        name="contextual_chunking_answer",
                        satisfied=False,
                        detail="turn 9 reasoning was incorrect",
                    ),
                ),
            ),
        ),
    )

    result = cli.main(["--model", "audex-2b", "--run-text-benchmark"])

    assert result == 0
    stdout = capsys.readouterr().out
    assert "Text runtime compatibility: passed" in stdout
    assert (
        "Model quality observation: contextual_chunking_answer: "
        "turn 9 reasoning was incorrect" in stdout
    )


def test_text_benchmark_cli_returns_nonzero_for_runtime_incompatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            return True

    run_log = tmp_path / "text.json"
    run_log.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(
        cli,
        "run_text_benchmark",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_log_path=run_log,
            transcript=[{"assistant": ""}],
            assessment=TextBenchmarkAssessment(
                compatibility_failures=("turn 1 generated no text",),
                quality_observations=(),
            ),
        ),
    )

    result = cli.main(["--model", "audex-2b", "--run-text-benchmark"])

    assert result == 2
    stdout = capsys.readouterr().out
    assert "Text runtime compatibility: failed" in stdout
    assert "Compatibility failure: turn 1 generated no text" in stdout


def test_cli_context_budget_is_bounded_to_demo_limit() -> None:
    assert cli._demo_context_tokens("262144") == 262_144

    with pytest.raises(ValueError, match="between 1 and 262144"):
        cli._demo_context_tokens("1000000")


def test_explicit_conversation_resume_migrates_old_context_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    _patch_fixture_cli(tmp_path, monkeypatch, events)
    conversation = SimpleNamespace(
        root=tmp_path,
        conversation_id="old-conversation",
        transcript_path=tmp_path / "old-conversation.md",
        persona_id="assistant",
        messages=[{"role": "system", "content": "System."}],
        token_count=0,
        max_context_tokens=1_000_000,
        user_name=None,
    )
    saved: list[int] = []

    class ExplicitResumeStore:
        def load(self, conversation_id: str):
            assert conversation_id == conversation.conversation_id
            return conversation

        def save(self, resumed) -> None:
            saved.append(resumed.max_context_tokens)

        def set_current(self, conversation_id: str) -> None:
            assert conversation_id == conversation.conversation_id

    monkeypatch.setattr(cli, "ConversationStore", ExplicitResumeStore)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--conversation-id",
            conversation.conversation_id,
            "--input-wav",
            str(input_wav),
            "--no-play",
        ]
    )

    assert result == 0
    assert conversation.max_context_tokens == 262_144
    assert saved == [262_144]
    assert events == ["vllm"]


def _patch_fixture_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    class FakeProbe:
        def is_cached(self, _model, _readiness="speech") -> bool:
            return True

    class FakeStore:
        def resume_current_or_create(self, **_kwargs):
            conversation = SimpleNamespace(
                root=tmp_path,
                conversation_id="conversation",
                transcript_path=tmp_path / "conversation.md",
                persona_id="assistant",
                messages=[{"role": "system", "content": "System."}],
                token_count=0,
                max_context_tokens=1_000_000,
                user_name=None,
            )
            return conversation, False

    def fake_preflight(_model):
        return SimpleNamespace(
            ready=True,
            model_path=tmp_path / "model",
            decoder_path=tmp_path / "decoder",
            missing_items=(),
            audio_components=None,
        )

    def fake_result(input_wav_path: Path) -> SpeechToSpeechTurnResult:
        output_wav = tmp_path / "output.wav"
        output_wav.write_bytes(b"RIFF")
        run_log = tmp_path / "turn.json"
        run_log.write_text("{}\n", encoding="utf-8")
        return SpeechToSpeechTurnResult(
            transcript="hello",
            response_text="Hi.",
            input_wav_path=input_wav_path,
            output_wav_path=output_wav,
            run_log_path=run_log,
            played=False,
        )

    def fake_vllm_fixture(**kwargs):
        events.append("vllm")
        return fake_result(kwargs["input_wav_path"])

    def fake_mlx_fixture(**kwargs):
        events.append("mlx")
        return fake_result(kwargs["input_wav_path"])

    class FakeMlxSession:
        def __init__(self, **_kwargs) -> None:
            pass

        @property
        def stats(self):
            return SimpleNamespace(
                model_load_seconds=1.0,
                audio_component_load_seconds=1.0,
                decoder_load_seconds=1.0,
                speech_warmup_seconds=1.0,
            )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "ConversationStore", lambda: FakeStore())
    monkeypatch.setattr(cli, "preflight_audio_runtime", fake_preflight)
    monkeypatch.setattr(cli, "AudexSpeechToSpeechSession", FakeMlxSession)
    monkeypatch.setattr(
        cli,
        "load_persona",
        lambda _name: SimpleNamespace(
            persona_id="assistant",
            path=tmp_path / "assistant.md",
            system_prompt="System.",
        ),
    )
    monkeypatch.setattr(cli, "run_vllm_fixture_turn", fake_vllm_fixture)
    monkeypatch.setattr(cli, "run_fixture_turn", fake_mlx_fixture)


def test_sts_defaults_to_vllm_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    _patch_fixture_cli(tmp_path, monkeypatch, events)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--input-wav",
            str(input_wav),
            "--no-play",
        ]
    )

    assert DEFAULT_STS_BACKEND == "vllm"
    assert result == 0
    assert events == ["vllm"]


def test_sts_mlx_backend_requires_explicit_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"RIFF")
    _patch_fixture_cli(tmp_path, monkeypatch, events)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--input-wav",
            str(input_wav),
            "--no-play",
            "--sts-backend",
            "mlx",
        ]
    )

    assert result == 0
    assert events == ["mlx"]


def test_vllm_sts_smoke_diagnostic_uses_speech_readiness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readiness_calls: list[str] = []
    diagnostic_kwargs: dict[str, object] = {}

    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            readiness_calls.append(readiness)
            return True

    def fake_diagnostic(_model, **kwargs):
        diagnostic_kwargs.update(kwargs)
        return SimpleNamespace(
            run_log_path=Path("/tmp/vllm-metal-diagnostic.json"),
            report={
                "vllm_metal": {
                    "platform": {
                        "device_type_facade": "cpu",
                        "device_name": "Apple Silicon",
                    },
                    "config": {"use_paged_attention": False},
                    "mlx": {},
                },
                "platform_resolution_probe": {},
                "spawn_probe": {},
                "generation_probe": {"enabled": False},
                "sts_probe": {
                    "enabled": True,
                    "ready": True,
                    "engine_class": "fake.AsyncLLMEngine",
                    "elapsed_seconds": 1.0,
                    "vllm_metal_timing": {
                        "latest_paged_sample": {
                            "count": 50,
                            "avg_ms": 98.5,
                            "last_ms": 14.2,
                            "native_sample_ms": 123.4,
                            "native_sampled_rows": 50,
                            "native_output_rows": 100,
                            "skipped_logits_eval": 42,
                            "mx_eval_ms": {
                                "logits": {"milliseconds": 456.7, "count": 50}
                            },
                        }
                    },
                    "sts_timing_assessment": {
                        "codec_frames_per_second": 3.278,
                        "audio_realtime_ratio": 0.066,
                        "paged_sample_avg_ms": 98.5,
                        "skipped_logits_eval": 42,
                        "native_sample_ms_per_step": 2.468,
                        "native_sample_ms_per_sampled_row": 2.468,
                        "dominant_mx_eval_per_step_category": "logits",
                        "native_sampling_row_ratio": 0.5,
                        "likely_bottleneck": "logits_eval",
                    },
                    "speech_streaming": {
                        "vllm_token_streaming": False,
                        "decoder_streaming": False,
                        "first_audio_ready_seconds": 0.5,
                        "generated_codec_frame_count": 4,
                    },
                },
                "verdict": {"ready": True, "failures": []},
            },
        )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "run_vllm_metal_diagnostics", fake_diagnostic)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--diagnose-vllm-sts-smoke",
            "--diagnose-vllm-native-sampling-debug",
            "--diagnose-vllm-sts-speech-max-tokens",
            "256",
        ]
    )

    assert result == 0
    assert diagnostic_kwargs["native_sampling_debug"] is True
    stdout = capsys.readouterr().out
    assert "vLLM Metal TTS timing: count=50" in stdout
    assert "native_sample_ms=123.4" in stdout
    assert "native_sampled_rows=50" in stdout
    assert "native_output_rows=100" in stdout
    assert "skipped_logits_eval=42" in stdout
    assert "vLLM STS timing assessment: codec_fps=3.278" in stdout
    assert "paged_avg_ms=98.5" in stdout
    assert "native_ms_per_step=2.468" in stdout
    assert "native_ms_per_sample=2.468" in stdout
    assert "dominant_eval_per_step=logits" in stdout
    assert "native_row_ratio=0.5" in stdout
    assert "likely_bottleneck=logits_eval" in stdout
    assert readiness_calls
    assert set(readiness_calls) == {"speech"}
    assert diagnostic_kwargs["run_sts_smoke"] is True
    assert diagnostic_kwargs["sts_speech_max_tokens"] == 256


def test_vllm_sts_play_diagnostic_implies_smoke_and_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_kwargs: dict[str, object] = {}

    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            return True

    def fake_diagnostic(_model, **kwargs):
        diagnostic_kwargs.update(kwargs)
        return SimpleNamespace(
            run_log_path=Path("/tmp/vllm-metal-diagnostic.json"),
            report={
                "vllm_metal": {
                    "platform": {
                        "device_type_facade": "cpu",
                        "device_name": "Apple Silicon",
                    },
                    "config": {"use_paged_attention": False},
                    "mlx": {},
                },
                "platform_resolution_probe": {},
                "spawn_probe": {},
                "generation_probe": {"enabled": False},
                "sts_probe": {
                    "enabled": True,
                    "ready": True,
                    "engine_class": "fake.AsyncLLMEngine",
                    "elapsed_seconds": 1.0,
                    "speech_streaming": {
                        "vllm_token_streaming": False,
                        "decoder_streaming": False,
                        "playback_transport": "sounddevice_raw_output_stream",
                        "first_audio_ready_seconds": 0.5,
                        "first_playback_started_seconds": 0.75,
                        "generated_codec_frame_count": 4,
                        "playback_diagnostics": {
                            "device_underflow_count": 0,
                            "queue_underrun_count": 0,
                            "queue_overrun_count": 0,
                            "chunks_written": 2,
                        },
                    },
                },
                "verdict": {"ready": True, "failures": []},
            },
        )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "run_vllm_metal_diagnostics", fake_diagnostic)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--diagnose-vllm-sts-play",
        ]
    )

    assert result == 0
    assert diagnostic_kwargs["run_sts_smoke"] is True
    assert diagnostic_kwargs["sts_play_audio"] is True


def test_vllm_tts_batch_diagnostic_uses_speech_readiness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readiness_calls: list[str] = []
    diagnostic_kwargs: dict[str, object] = {}

    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            readiness_calls.append(readiness)
            return True

    def fake_diagnostic(_model, **kwargs):
        diagnostic_kwargs.update(kwargs)
        return SimpleNamespace(
            run_log_path=Path("/tmp/vllm-metal-diagnostic.json"),
            report={
                "vllm_metal": {
                    "platform": {
                        "device_type_facade": "cpu",
                        "device_name": "Apple Silicon",
                    },
                    "config": {"use_paged_attention": False},
                    "mlx": {},
                },
                "platform_resolution_probe": {},
                "spawn_probe": {},
                "generation_probe": {"enabled": False},
                "sts_probe": {"enabled": False},
                "tts_batch_probe": {
                    "enabled": True,
                    "ready": True,
                    "batch_size": 4,
                    "cfg_enabled": False,
                    "request_count": 8,
                    "elapsed_seconds": 12.5,
                    "total_codec_frame_count": 512,
                    "codec_frames_per_second": 40.96,
                    "min_codec_frames_per_request": 120,
                    "max_codec_frames_per_request": 128,
                    "reached_end_count": 1,
                    "hit_max_token_count": 3,
                },
                "verdict": {"ready": True, "failures": []},
            },
        )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "run_vllm_metal_diagnostics", fake_diagnostic)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--diagnose-vllm-tts-batch-size",
            "4",
            "--diagnose-vllm-tts-batch-max-tokens",
            "128",
            "--diagnose-vllm-tts-batch-text",
            "hello",
            "--diagnose-vllm-tts-batch-no-cfg",
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert "vLLM TTS batch probe: ready=True batch_size=4" in stdout
    assert "cfg_enabled=False" in stdout
    assert "codec_fps=40.96" in stdout
    assert "min_frames=120" in stdout
    assert "max_frames=128" in stdout
    assert "reached_end=1" in stdout
    assert "hit_max=3" in stdout
    assert set(readiness_calls) == {"speech"}
    assert diagnostic_kwargs["tts_batch_size"] == 4
    assert diagnostic_kwargs["tts_batch_max_tokens"] == 128
    assert diagnostic_kwargs["tts_batch_text"] == "hello"
    assert diagnostic_kwargs["tts_batch_cfg"] is False


def test_vllm_tts_text_diagnostic_uses_product_speech_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readiness_calls: list[str] = []
    probe_kwargs: dict[str, object] = {}

    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            readiness_calls.append(readiness)
            return True

    def fake_preflight(_model):
        return SimpleNamespace(
            ready=True,
            model_path=tmp_path / "model",
            decoder_path=tmp_path / "decoder",
            missing_items=(),
            audio_components=None,
        )

    def fake_tts_probe(**kwargs):
        probe_kwargs.update(kwargs)
        wav_path = tmp_path / "tts.wav"
        run_log_path = tmp_path / "tts.json"
        wav_path.write_bytes(b"RIFF")
        run_log_path.write_text("{}\n", encoding="utf-8")
        return SpeechOutputSmokeResult(
            backend="vllm",
            device="Device(gpu, 0)",
            prompt_tokens=0,
            generated_token_ids=(1, 2, 3),
            generated_codec_frames=(1, 2),
            reached_end_token=True,
            hit_max_tokens=False,
            waveform_shape=(640,),
            sample_rate=16_000,
            hop_length=320,
            finite=True,
            peak_abs=0.1,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=True,
            first_audio_ready_seconds=0.5,
        )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "preflight_audio_runtime", fake_preflight)
    monkeypatch.setattr(cli, "run_vllm_tts_text_probe", fake_tts_probe)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--diagnose-vllm-tts-text",
            "hello there",
            "--speech-max-tokens",
            "123",
            "--no-play",
        ]
    )

    assert result == 0
    assert set(readiness_calls) == {"speech"}
    assert probe_kwargs["text"] == "hello there"
    assert probe_kwargs["play"] is False
    assert probe_kwargs["speech_max_tokens"] == 123
    stdout = capsys.readouterr().out
    assert "vLLM TTS text probe WAV:" in stdout
    assert "codec_frames=2" in stdout
    assert "first_audio_ready_seconds=0.5" in stdout


def test_vllm_tts_text_diagnostic_reads_text_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_path = tmp_path / "tts.txt"
    text_path.write_text("file text", encoding="utf-8")
    probe_kwargs: dict[str, object] = {}

    class FakeProbe:
        def is_cached(self, _model, readiness="speech") -> bool:
            return True

    def fake_preflight(_model):
        return SimpleNamespace(
            ready=True,
            model_path=tmp_path / "model",
            decoder_path=tmp_path / "decoder",
            missing_items=(),
            audio_components=None,
        )

    def fake_tts_probe(**kwargs):
        probe_kwargs.update(kwargs)
        wav_path = tmp_path / "tts.wav"
        run_log_path = tmp_path / "tts.json"
        wav_path.write_bytes(b"RIFF")
        run_log_path.write_text("{}\n", encoding="utf-8")
        return SpeechOutputSmokeResult(
            backend="vllm",
            device="Device(gpu, 0)",
            prompt_tokens=0,
            generated_token_ids=(1,),
            generated_codec_frames=(1,),
            reached_end_token=True,
            hit_max_tokens=False,
            waveform_shape=(320,),
            sample_rate=16_000,
            hop_length=320,
            finite=True,
            peak_abs=0.1,
            wav_path=wav_path,
            run_log_path=run_log_path,
            streaming=True,
        )

    monkeypatch.setattr(cli, "HuggingFaceSnapshotProbe", FakeProbe)
    monkeypatch.setattr(cli, "preflight_audio_runtime", fake_preflight)
    monkeypatch.setattr(cli, "run_vllm_tts_text_probe", fake_tts_probe)

    result = cli.main(
        [
            "--model",
            "audex-2b",
            "--diagnose-vllm-tts-text-file",
            str(text_path),
            "--no-play",
        ]
    )

    assert result == 0
    assert probe_kwargs["text"] == "file text"
