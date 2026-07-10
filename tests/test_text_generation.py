from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audex_mac.patches.runtime import AudexPatchReport
from audex_mac.text_benchmark import TextBenchmark
from audex_mac.text_chat import (
    complete_text_assistant_turn,
    ensure_audex_chat_template,
    render_text_chat_prompt,
)
from audex_mac.text_generation import (
    _build_run_log,
    _contains_stop_marker,
    _evaluate_assessment,
    _sampler_config,
    _vllm_turn_record,
    clean_generation,
    run_text_benchmark,
)

pytestmark = pytest.mark.fast


def test_clean_generation_trims_stop_markers() -> None:
    assert clean_generation("hello<|im_end|>ignored") == "hello"
    assert clean_generation("hello<|end_of_text|>ignored") == "hello"
    assert clean_generation("hello<|eot_id|>ignored") == "hello"


def test_contains_stop_marker_detects_chat_end() -> None:
    assert _contains_stop_marker("hello<|im_end|>") is True
    assert _contains_stop_marker("hello") is False


def test_sampler_config_preserves_benchmark_values() -> None:
    benchmark = TextBenchmark(
        name="test",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": "q"} for _ in range(10)],
        pass_criteria=["coherent"],
        sampler_reference="test",
    )

    assert _sampler_config(benchmark) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 100,
        "max_tokens": 4096,
    }


def test_vllm_turn_record_captures_generation_throughput() -> None:
    request_output = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    completion_output = SimpleNamespace(
        token_ids=(10, 11, 12, 13),
        finish_reason="length",
        stop_reason=None,
    )

    record = _vllm_turn_record(
        turn=1,
        user="question",
        assistant="answer",
        elapsed_seconds=2.0,
        request_output=request_output,
        completion_output=completion_output,
    )

    assert record == {
        "turn": 1,
        "user": "question",
        "assistant": "answer",
        "assistant_raw": "answer",
        "elapsed_seconds": 2.0,
        "prompt_tokens": 3,
        "generation_tokens": 4,
        "generation_tps": 2.0,
        "finish_reason": "length",
        "stop_reason": None,
    }


def test_run_log_records_audex_patch_report() -> None:
    benchmark = TextBenchmark(
        name="test",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": False,
        },
        turns=[{"role": "user", "content": "q"} for _ in range(10)],
        pass_criteria=["coherent"],
        sampler_reference="test",
    )
    patch_report = AudexPatchReport(
        transformers_local_dynamic_modules=True,
        mlx_lm_nemotron_dense=True,
        mlx_lm_nemotron_h_audex=True,
        vllm_metal_platform_repair=True,
        vllm_metal_device_info_api=True,
        vllm_metal_nonpaged_capacity=True,
        vllm_nemotron_dense=True,
        vllm_metal_audex_adapter=True,
    )
    preflight = SimpleNamespace(
        model=SimpleNamespace(repo_id="model"),
        model_path="/tmp/model",
        patch_report=patch_report,
    )
    metal_policy = SimpleNamespace(
        env={"VLLM_MLX_DEVICE": "gpu"},
        mlx_metal_available=True,
        mlx_default_device="Device(gpu, 0)",
    )

    run_log = _build_run_log(
        preflight,
        backend="vllm",
        benchmark=benchmark,
        sampler=_sampler_config(benchmark),
        thinking_enabled=False,
        metal_policy=metal_policy,
        started_at=0,
        transcript=[],
    )

    assert run_log["audex_patches"] == {
        "transformers_local_dynamic_modules": True,
        "mlx_lm_nemotron_dense": True,
        "mlx_lm_nemotron_h_audex": True,
        "vllm_metal_platform_repair": True,
        "vllm_metal_device_info_api": True,
        "vllm_metal_nonpaged_capacity": True,
        "vllm_nemotron_dense": True,
        "vllm_metal_audex_adapter": True,
    }


def test_run_text_benchmark_preflights_the_selected_backend(monkeypatch) -> None:
    observed: dict[str, object] = {}
    preflight = SimpleNamespace(ready=True, model_path="/tmp/model")

    def fake_preflight(model, *, backend):
        observed["model"] = model
        observed["backend"] = backend
        return preflight

    monkeypatch.setattr(
        "audex_mac.text_generation.preflight_text_runtime", fake_preflight
    )
    monkeypatch.setattr(
        "audex_mac.text_generation._run_text_benchmark_from_preflight",
        lambda actual, **kwargs: (actual, kwargs),
    )

    result = run_text_benchmark(SimpleNamespace(repo_id="model"), backend="mlx")

    assert observed["backend"] == "mlx"
    assert result[0] is preflight
    assert result[1]["backend"] == "mlx"


def test_limited_text_benchmark_does_not_claim_a_full_assessment() -> None:
    assessment = _evaluate_assessment(
        SimpleNamespace(turns=[{} for _ in range(10)]),
        [{"assistant": "one diagnostic turn"}],
        limit_turns=1,
    )

    assert assessment.full_benchmark_evaluated is False
    assert assessment.compatible is True
    assert assessment.quality_observations == ()


class RecordingChatTokenizer:
    def __init__(self) -> None:
        self.chat_template = "official-template"
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append((messages, kwargs))
        mode = "<think>\n" if kwargs["enable_thinking"] else "<think></think>"
        return f"rendered:{mode}"


def test_text_prompt_is_rendered_by_the_model_chat_template() -> None:
    tokenizer = RecordingChatTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    prompt = render_text_chat_prompt(
        tokenizer,
        messages,
        model_path="/model/checkpoint_folder_textonly",
        thinking_enabled=True,
    )

    assert prompt == "rendered:<think>\n"
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        )
    ]


def test_missing_textonly_template_is_loaded_from_the_full_checkpoint(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "snapshot" / "checkpoint_folder_textonly"
    model_path.mkdir(parents=True)
    template_path = (
        tmp_path / "snapshot" / "checkpoint_folder_full" / "chat_template.jinja"
    )
    template_path.parent.mkdir()
    template_path.write_text("official shipped template", encoding="utf-8")
    inner_tokenizer = SimpleNamespace(chat_template=None)
    tokenizer = SimpleNamespace(
        chat_template=None,
        _tokenizer=inner_tokenizer,
        _chat_template=None,
        has_chat_template=False,
    )

    ensure_audex_chat_template(tokenizer, model_path)

    assert tokenizer.chat_template == "official shipped template"
    assert inner_tokenizer.chat_template == "official shipped template"
    assert tokenizer.has_chat_template is True


def test_template_is_installed_on_hugging_face_fast_tokenizer_not_rust_backend(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "snapshot" / "checkpoint_folder_textonly"
    model_path.mkdir(parents=True)
    template_path = (
        tmp_path / "snapshot" / "checkpoint_folder_full" / "chat_template.jinja"
    )
    template_path.parent.mkdir()
    template_path.write_text("official shipped template", encoding="utf-8")
    rust_backend = SimpleNamespace()
    tokenizer = SimpleNamespace(chat_template=None, _tokenizer=rust_backend)

    ensure_audex_chat_template(tokenizer, model_path)

    assert tokenizer.chat_template == "official shipped template"
    assert not hasattr(rust_backend, "chat_template")


@pytest.mark.parametrize(
    ("thinking_enabled", "generated", "raw_content", "answer"),
    [
        (
            True,
            "private reasoning\n</think>\nPublic answer.<|im_end|>",
            "<think>\nprivate reasoning\n</think>\nPublic answer.",
            "Public answer.",
        ),
        (
            False,
            "Public answer.<|im_end|>",
            "<think></think>Public answer.",
            "Public answer.",
        ),
    ],
)
def test_completed_assistant_turn_reconstructs_the_consumed_template_prefix(
    thinking_enabled: bool,
    generated: str,
    raw_content: str,
    answer: str,
) -> None:
    turn = complete_text_assistant_turn(
        generated,
        thinking_enabled=thinking_enabled,
    )

    assert turn.raw_content == raw_content
    assert turn.answer == answer


@pytest.mark.parametrize("thinking_enabled", [False, True])
def test_ten_turn_vllm_benchmark_uses_template_valid_assistant_history(
    thinking_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = TextBenchmark(
        name="template-history",
        system="system",
        generation={
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 100,
            "max_tokens": 4096,
            "thinking_enabled": thinking_enabled,
        },
        turns=[
            {"role": "user", "content": f"question {index}"} for index in range(1, 11)
        ],
        pass_criteria=[],
        sampler_reference="test",
    )

    class TemplateTokenizer:
        chat_template = "official-template"

        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

        def apply_chat_template(self, messages, **kwargs) -> str:
            copied = [dict(message) for message in messages]
            self.calls.append((copied, dict(kwargs)))
            rendered: list[str] = []
            for message in copied:
                content = message["content"]
                if message["role"] == "assistant":
                    if "<think>" in content and "</think>" in content:
                        content = (
                            "<think></think>"
                            + content.rsplit("</think>", 1)[-1].strip()
                        )
                    elif "<think>" not in content and "</think>" not in content:
                        content = "<think></think>" + content
                rendered.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
            prefix = "<think>\n" if kwargs["enable_thinking"] else "<think></think>"
            return "".join(rendered) + "<|im_start|>assistant\n" + prefix

    tokenizer = TemplateTokenizer()

    class FakeLLM:
        instances: list[FakeLLM] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.prompts: list[str] = []
            self.__class__.instances.append(self)

        def get_tokenizer(self):
            return tokenizer

        def generate(self, prompts, _sampling_params):
            self.prompts.extend(prompts)
            turn = len(self.prompts)
            text = (
                f"private reasoning {turn}\n</think>\nAnswer {turn}."
                if thinking_enabled
                else f"Answer {turn}."
            )
            output = SimpleNamespace(
                text=text,
                token_ids=(turn,),
                finish_reason="stop",
                stop_reason="<|im_end|>",
            )
            return [SimpleNamespace(outputs=[output], prompt_token_ids=(1, 2))]

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    model_path = tmp_path / "snapshot" / "checkpoint_folder_textonly"
    model_path.mkdir(parents=True)
    preflight = SimpleNamespace(
        ready=True,
        missing_items=(),
        model_path=model_path,
        model=SimpleNamespace(repo_id="test/audex"),
        benchmark=benchmark,
        patch_report=None,
    )
    metal_policy = SimpleNamespace(
        ready=True,
        env={"VLLM_MLX_DEVICE": "gpu"},
        mlx_metal_available=True,
        mlx_default_device="Device(gpu, 0)",
    )
    monkeypatch.setattr(
        "audex_mac.text_generation.preflight_text_runtime",
        lambda _model, *, backend: preflight,
    )
    monkeypatch.setattr(
        "audex_mac.text_generation.inspect_metal_runtime", lambda: metal_policy
    )
    monkeypatch.setattr(
        "audex_mac.text_generation.apply_audex_runtime_patches", lambda: None
    )
    monkeypatch.setattr("audex_mac.text_generation.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    result = run_text_benchmark(
        SimpleNamespace(repo_id="test/audex"),
        backend="vllm",
        thinking_enabled=thinking_enabled,
    )

    assert len(tokenizer.calls) == 10
    assert all(
        call_kwargs
        == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": thinking_enabled,
        }
        for _, call_kwargs in tokenizer.calls
    )
    first_history_content = tokenizer.calls[1][0][2]["content"]
    expected_history = (
        "<think>\nprivate reasoning 1\n</think>\nAnswer 1."
        if thinking_enabled
        else "<think></think>Answer 1."
    )
    assert first_history_content == expected_history
    assert "<think></think>Answer 1." in FakeLLM.instances[0].prompts[1]
    assert "private reasoning 1" not in FakeLLM.instances[0].prompts[1]
    assert [turn["assistant"] for turn in result.transcript] == [
        f"Answer {index}." for index in range(1, 11)
    ]
    assert result.transcript[0]["assistant_raw"] == expected_history
    assert result.assessment.compatible is True
