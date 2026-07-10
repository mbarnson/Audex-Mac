from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pytest_bdd import given, scenarios, then, when

from audex_mac import sts_cli
from audex_mac.audio_contract import (
    NVIDIA_TTS_CFG_SCALE,
    NVIDIA_TTS_TEMPERATURE,
    NVIDIA_TTS_TOP_K,
    NVIDIA_TTS_TOP_P,
    SOUND_END_TOKEN,
    SOUND_START_TOKEN,
    SOUND_TOKEN,
    build_audio_chat_prompt,
    build_audio_prompt_plan,
)
from audex_mac.audio_encoder import (
    expected_audio_encoder_weight_keys,
    load_audio_encoder_config,
    resolve_audio_encoder_shards,
)
from audex_mac.audio_features import extract_audex_input_features
from audex_mac.audio_pcm import prepare_audex_pcm_clips
from audex_mac.audio_projector import (
    load_audio_projector_config,
    resolve_audio_projector_shards,
)
from audex_mac.audio_splice import validate_audio_splice_plan
from audex_mac.bootstrap import BootstrapState, model_download_notice, plan_bootstrap
from audex_mac.cli import DEFAULT_STS_BACKEND
from audex_mac.conversations import ConversationStore
from audex_mac.interactive_input import InputKind, classify_submission
from audex_mac.model_select import select_model
from audex_mac.models import AUDEX_2B_REPO, AUDEX_30B_NVFP4_REPO, AUDEX_30B_REPO
from audex_mac.patch_guards import PatchTarget, VllmMetalState, run_patch_guards
from audex_mac.personas import load_persona
from audex_mac.speech_decoder import SpeechDecoderSmokeResult
from audex_mac.speech_generation import SpeechTokenGenerationSmokeResult
from audex_mac.speech_output import SpeechOutputSmokeResult
from audex_mac.speech_policy import (
    ALLOWED_AUDIO_PLUMBING,
    NON_THINKING_PREFIX,
    assistant_prefix,
    validate_no_forbidden_models,
)
from audex_mac.text_generation import run_text_benchmark
from audex_mac.vllm_diagnostics import _interpret_expected_cpu_facade
from audex_mac.vllm_runtime import AudexVllmRuntime
from audex_mac.vllm_sts_requests import (
    build_asr_projected_embeddings_request,
    build_asr_request,
    build_text_response_request,
    build_tts_cfg_requests,
)

FEATURE_DIR = Path(__file__).resolve().parents[1] / "features"

scenarios(str(FEATURE_DIR / "model_selection.feature"))
scenarios(str(FEATURE_DIR / "patch_guards.feature"))
scenarios(str(FEATURE_DIR / "no_extra_models.feature"))
scenarios(str(FEATURE_DIR / "startup.feature"))
scenarios(str(FEATURE_DIR / "licensing.feature"))
scenarios(str(FEATURE_DIR / "speech_to_speech_cli.feature"))
scenarios(str(FEATURE_DIR / "text_benchmark.feature"))


@pytest.fixture
def ctx() -> dict:
    return {}


class FakeProbe:
    def __init__(self, cached: dict[str, bool]) -> None:
        self.cached = cached

    def is_cached(self, model, readiness: str = "speech") -> bool:
        return self.cached.get(
            f"{readiness}:{model.repo_id}",
            self.cached.get(model.repo_id, False),
        )


class FakeProvider:
    def __init__(self, missing: bool = False) -> None:
        self.missing = missing

    def import_module(self, module_name: str) -> ModuleType:
        module = ModuleType(module_name)
        if self.missing:
            return module

        if module_name.endswith("model_adapter"):
            module.DefaultModelAdapter = type("DefaultModelAdapter", (), {})
        elif module_name.endswith("model_lifecycle"):
            module.ModelLifecycle = type(
                "ModelLifecycle",
                (),
                {"_load_generation_model": lambda self: None},
            )
        elif module_name.endswith("model_runner"):
            module.MetalModelRunner = type(
                "MetalModelRunner",
                (),
                {"load_model": lambda self: None},
            )
        return module


# Model selection


@given("the Audex 30B-A3B snapshot is fully present in the Hugging Face cache")
def cached_30b(ctx: dict) -> None:
    ctx.setdefault("cached", {})[AUDEX_30B_REPO] = True


@given("the Audex 30B-A3B NVFP4 snapshot is fully present in the Hugging Face cache")
def cached_30b_nvfp4(ctx: dict) -> None:
    ctx.setdefault("cached", {})[AUDEX_30B_NVFP4_REPO] = True


@given("the Audex 2B snapshot is fully present in the Hugging Face cache")
def cached_2b(ctx: dict) -> None:
    ctx.setdefault("cached", {})[AUDEX_2B_REPO] = True


@given("the Audex 30B-A3B speech snapshot is fully present in the Hugging Face cache")
def cached_30b_speech(ctx: dict) -> None:
    ctx.setdefault("cached", {})[f"speech:{AUDEX_30B_REPO}"] = True


@given(
    "the Audex 30B-A3B text checkpoint is not fully present in the Hugging Face cache"
)
def uncached_30b_text(ctx: dict) -> None:
    ctx.setdefault("cached", {})[f"text:{AUDEX_30B_REPO}"] = False


@given("the Audex 2B text checkpoint is fully present in the Hugging Face cache")
def cached_2b_text(ctx: dict) -> None:
    ctx.setdefault("cached", {})[f"text:{AUDEX_2B_REPO}"] = True


@given("the Audex 30B-A3B snapshot is not fully present in the Hugging Face cache")
def uncached_30b(ctx: dict) -> None:
    ctx.setdefault("cached", {})[AUDEX_30B_REPO] = False


@given("no supported Audex snapshot is fully present in the Hugging Face cache")
def no_supported_snapshot(ctx: dict) -> None:
    ctx["cached"] = {AUDEX_30B_REPO: False, AUDEX_2B_REPO: False}


@when("start.sh resolves the model to launch")
def resolve_model(ctx: dict) -> None:
    ctx["selection"] = select_model(FakeProbe(ctx.get("cached", {})))


@when("start.sh resolves the model for a text command")
def resolve_text_model(ctx: dict) -> None:
    ctx["selection"] = select_model(FakeProbe(ctx.get("cached", {})), readiness="text")


@then("it selects nvidia/Nemotron-Labs-Audex-30B-A3B")
def selected_30b(ctx: dict) -> None:
    assert ctx["selection"].selected.repo_id == AUDEX_30B_REPO


@then("it selects txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx")
def selected_30b_nvfp4(ctx: dict) -> None:
    assert ctx["selection"].selected.repo_id == AUDEX_30B_NVFP4_REPO


@then("it selects nvidia/Nemotron-Labs-Audex-2B")
def selected_2b(ctx: dict) -> None:
    assert ctx["selection"].selected.repo_id == AUDEX_2B_REPO


@then("it logs that 30B-A3B was selected because it was already cached")
def logs_30b_cached(ctx: dict) -> None:
    assert any("30B-A3B" in message for message in ctx["selection"].log_messages)
    assert any("cached" in message for message in ctx["selection"].log_messages)


@then("it tells the user that 2B is the default first-run model")
def tells_user_2b_default(ctx: dict) -> None:
    assert any("Audex-2B" in message for message in ctx["selection"].user_messages)
    assert ctx["selection"].defaulted is True


@then("it mentions nvidia/Nemotron-Labs-Audex-30B-A3B as the higher-reasoning option")
def mentions_30b_option(ctx: dict) -> None:
    assert any(AUDEX_30B_REPO in message for message in ctx["selection"].user_messages)


# Patch guards


@given("the installed vLLM Metal package matches the pinned commit")
def vllm_matches_pin(ctx: dict) -> None:
    ctx["vllm_state"] = VllmMetalState(
        installed_commit="old",
        pinned_commit="old",
        upstream_head="old",
    )


@given("upstream vLLM Metal HEAD differs from the pinned commit")
def upstream_moved(ctx: dict) -> None:
    current = ctx["vllm_state"]
    ctx["vllm_state"] = VllmMetalState(
        installed_commit=current.installed_commit,
        pinned_commit=current.pinned_commit,
        upstream_head="new",
    )


@given("the installed vLLM Metal package does not expose a required patched symbol")
def missing_patched_symbol(ctx: dict) -> None:
    ctx["vllm_state"] = VllmMetalState(
        installed_commit="old",
        pinned_commit="old",
        upstream_head="old",
    )
    ctx["missing_symbol"] = True


@when("start.sh runs patch guards")
def run_guards(ctx: dict) -> None:
    ctx["patch_result"] = run_patch_guards(
        ctx["vllm_state"],
        provider=FakeProvider(missing=ctx.get("missing_symbol", False)),
    )


@then("startup continues")
def startup_continues(ctx: dict) -> None:
    assert ctx["patch_result"].startup_allowed is True


@then("a loud advisory warning is shown")
def advisory_warning(ctx: dict) -> None:
    assert ctx["patch_result"].warnings


@then("the generated coding-agent update prompt is written to the log")
def update_prompt_logged(ctx: dict) -> None:
    assert ctx["patch_result"].update_prompt is not None
    assert "docs/engineering/patches.md" in ctx["patch_result"].update_prompt


@then("startup stops before model launch")
def startup_stops(ctx: dict) -> None:
    assert ctx["patch_result"].startup_allowed is False


@then("the error names the missing symbol")
def error_names_symbol(ctx: dict) -> None:
    assert ctx["patch_result"].missing_symbol


@then("the error points to docs/engineering/patches.md")
def error_points_to_patches(ctx: dict) -> None:
    assert ctx["patch_result"].startup_allowed is False


@given("vLLM Metal reports its compatibility CPU facade")
def vllm_metal_reports_cpu_facade(ctx: dict) -> None:
    ctx["vllm_metal_diagnostic"] = {
        "platform": {"device_type_facade": "cpu"},
        "interpretation": _interpret_expected_cpu_facade(),
    }


@given("MLX reports a GPU default device")
def mlx_reports_gpu_default_device(ctx: dict) -> None:
    ctx["vllm_metal_diagnostic"]["mlx"] = {
        "default_device": "Device(gpu, 0)",
        "probe_array_device": "Device(gpu, 0)",
    }


@when("Audex-Mac evaluates the vLLM Metal diagnostic report")
def evaluate_vllm_metal_diagnostic_report(ctx: dict) -> None:
    diagnostic = ctx["vllm_metal_diagnostic"]
    interpretation = diagnostic["interpretation"]
    ctx["cpu_facade_expected"] = (
        diagnostic["platform"]["device_type_facade"] == "cpu"
        and interpretation["vllm_device_type_cpu_can_be_expected"] is True
    )
    ctx["mlx_gpu_required"] = (
        "VLLM_MLX_DEVICE is not gpu" in interpretation["cpu_fallback_indicators"]
        and diagnostic["mlx"]["default_device"] == "Device(gpu, 0)"
    )


@then("the report treats the CPU facade as expected")
def report_treats_cpu_facade_as_expected(ctx: dict) -> None:
    assert ctx["cpu_facade_expected"] is True


@then("the report treats MLX GPU evidence as required")
def report_treats_mlx_gpu_evidence_as_required(ctx: dict) -> None:
    assert ctx["mlx_gpu_required"] is True


# Startup


@given("no local virtual environment exists")
def no_venv(ctx: dict) -> None:
    ctx["bootstrap_state"] = BootstrapState(
        venv_exists=False,
        dependency_state_matches=False,
        model_cached=True,
    )


@given("the local virtual environment matches the pinned dependency state")
def valid_bootstrap_state(ctx: dict) -> None:
    ctx["bootstrap_state"] = BootstrapState(
        venv_exists=True,
        dependency_state_matches=True,
        model_cached=True,
    )


@given("no supported Audex model is cached")
def no_model_cached(ctx: dict) -> None:
    previous = ctx.get(
        "bootstrap_state",
        BootstrapState(
            venv_exists=True,
            dependency_state_matches=True,
            model_cached=True,
        ),
    )
    ctx["bootstrap_state"] = BootstrapState(
        venv_exists=previous.venv_exists,
        dependency_state_matches=previous.dependency_state_matches,
        model_cached=False,
    )


@when("the user runs ./start.sh")
def user_runs_start(ctx: dict) -> None:
    ctx["bootstrap_plan"] = plan_bootstrap(ctx["bootstrap_state"])


@then("start.sh creates a local virtual environment")
def creates_venv(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].create_venv is True


@then("installs huggingface_hub")
def installs_hf(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].install_huggingface_hub is True


@then("installs pinned project dependencies")
def installs_deps(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].install_pinned_dependencies is True


@then("start.sh does not reinstall dependencies by default")
def no_reinstall(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].install_pinned_dependencies is False


@then("proceeds to model selection")
def proceeds_to_model_selection(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].proceed_to_model_selection is True


@then("start.sh explains the selected model size and NVIDIA license")
def explains_download(ctx: dict) -> None:
    notice = model_download_notice(AUDEX_2B_REPO, "about 10 GB")
    ctx["download_notice"] = notice
    assert "NVIDIA" in notice
    assert AUDEX_2B_REPO in notice


@then("asks for confirmation before downloading")
def asks_confirmation(ctx: dict) -> None:
    assert ctx["bootstrap_plan"].prompt_for_model_download is True


# Licensing


@given("the user reads README.md")
def read_readme(ctx: dict) -> None:
    ctx["readme"] = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )


@then("the README says Audex-Mac source code is MIT licensed")
def readme_mit(ctx: dict) -> None:
    assert "Audex-Mac source code is MIT licensed" in ctx["readme"]


@then("the README says NVIDIA model weights are governed by NVIDIA's license")
def readme_nvidia_license(ctx: dict) -> None:
    assert "model weights" in ctx["readme"]
    assert "NVIDIA" in ctx["readme"]


@then("the README links to the Audex model cards")
def readme_model_cards(ctx: dict) -> None:
    assert "Nemotron-Labs-Audex-2B" in ctx["readme"]
    assert "Nemotron-Labs-Audex-30B-A3B" in ctx["readme"]


@when("start.sh asks before downloading a model")
def start_asks_download(ctx: dict) -> None:
    ctx["download_notice"] = model_download_notice(AUDEX_2B_REPO, "about 10 GB")


@then("the prompt mentions NVIDIA's model license")
def prompt_mentions_nvidia(ctx: dict) -> None:
    assert "NVIDIA" in ctx["download_notice"]


@then("it does not imply the model weights are MIT licensed")
def prompt_not_mit_weights(ctx: dict) -> None:
    assert "MIT license applies only" in ctx["download_notice"]


# No extra semantic models


@given("the CLI is running in speech-to-speech mode")
def cli_sts_mode(ctx: dict) -> None:
    ctx["loaded_models"] = ["audex"]


@when("the user records an utterance with push-to-talk")
def record_ptt(ctx: dict) -> None:
    ctx["input_path"] = "audex_audio"


@then("the input audio is passed to Audex audio input processing")
def input_to_audex(ctx: dict) -> None:
    assert ctx["input_path"] == "audex_audio"


@then("no Whisper model is loaded")
def no_whisper(ctx: dict) -> None:
    validate_no_forbidden_models(ctx["loaded_models"])


@then("no Kokoro model is loaded")
def no_kokoro(ctx: dict) -> None:
    validate_no_forbidden_models(ctx["loaded_models"])


@then("no Silero VAD model is loaded")
def no_silero(ctx: dict) -> None:
    validate_no_forbidden_models(ctx["loaded_models"])


@then("the spoken response is decoded with the Audex causal speech decoder")
def audex_decoder(ctx: dict) -> None:
    ctx["decoder"] = "audex_causal_speech_decoder"
    assert ctx["decoder"] == "audex_causal_speech_decoder"


@given("the CLI captured audio from the microphone")
def captured_audio(ctx: dict) -> None:
    ctx["audio"] = "pcm"


@when("Audex-Mac prepares the audio for Audex")
def prepare_audio(ctx: dict) -> None:
    ctx["audio_ops"] = {"resample", "normalize", "codec_convert"}


@then("it may resample PCM")
def may_resample(ctx: dict) -> None:
    assert "resample" in ALLOWED_AUDIO_PLUMBING


@then("it may normalize audio samples")
def may_normalize(ctx: dict) -> None:
    assert "normalize" in ALLOWED_AUDIO_PLUMBING


@then("it may use codec tools for deterministic conversion")
def may_codec(ctx: dict) -> None:
    assert "codec_convert" in ALLOWED_AUDIO_PLUMBING


@then("it must not infer speech text with a separate model")
def no_separate_inference(ctx: dict) -> None:
    validate_no_forbidden_models(ctx.get("loaded_models", ["audex"]))


# Speech CLI non-thinking


@given("the CLI is started without a thinking flag")
def no_thinking_flag(ctx: dict) -> None:
    ctx["thinking_enabled"] = False


@when("Audex-Mac builds the assistant response prefix")
def build_prefix(ctx: dict) -> None:
    ctx["assistant_prefix"] = assistant_prefix(thinking_enabled=ctx["thinking_enabled"])


@then("it prepends <think></think>")
def prepends_non_thinking(ctx: dict) -> None:
    assert ctx["assistant_prefix"] == NON_THINKING_PREFIX


@then("it records thinking_enabled=false in the run log")
def records_thinking_false(ctx: dict) -> None:
    ctx["run_log"] = {"thinking_enabled": ctx["thinking_enabled"]}
    assert ctx["run_log"]["thinking_enabled"] is False


@given("one 16 kHz utterance shorter than 30 seconds")
def short_audio_utterance(ctx: dict) -> None:
    ctx["sample_count"] = 16_000
    ctx["loaded_models"] = ["audex"]


@when("Audex-Mac builds the native audio input prompt")
def build_native_audio_prompt(ctx: dict) -> None:
    plan = build_audio_prompt_plan(sample_count=ctx["sample_count"])
    ctx["audio_prompt"] = build_audio_chat_prompt(plan, thinking_enabled=False)


@then("the prompt contains exactly 750 <so_embedding> tokens")
def exact_sound_embedding_count(ctx: dict) -> None:
    assert ctx["audio_prompt"].count(SOUND_TOKEN) == 750


@then("the tokens are bracketed by <so_start> and <so_end>")
def sound_tokens_are_bracketed(ctx: dict) -> None:
    prompt = ctx["audio_prompt"]
    start = prompt.index(SOUND_START_TOKEN)
    end = prompt.index(SOUND_END_TOKEN)
    assert start < prompt.index(SOUND_TOKEN) < end
    assert prompt.count(SOUND_START_TOKEN) == 1
    assert prompt.count(SOUND_END_TOKEN) == 1


class FakeVllmStsTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.enable_thinking_values.append(enable_thinking)
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode(self, prompt: str) -> list[int]:
        return list(range(10 + prompt.count("<unk>") + prompt.count("Hello")))


class FakeBddSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBddVllmEngine:
    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def generate(self, prompts, sampling_params):
        self.calls.append(list(prompts))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="'ok'",
                        token_ids=(1,),
                        finish_reason="stop",
                    )
                ]
            )
            for _prompt in prompts
        ]


@given("a 16 kHz utterance for the vLLM speech-to-speech path")
def vllm_sts_utterance(ctx: dict) -> None:
    ctx["vllm_sts_audio"] = [0.0, 0.1]
    ctx["vllm_sts_tokenizer"] = FakeVllmStsTokenizer()


@when("Audex-Mac builds the vLLM speech-to-speech request plan")
def build_vllm_sts_request_plan(ctx: dict) -> None:
    tokenizer = ctx["vllm_sts_tokenizer"]
    ctx["vllm_asr_request"] = build_asr_request(tokenizer, ctx["vllm_sts_audio"])
    ctx["vllm_text_request"] = build_text_response_request(tokenizer, "Hello.")
    ctx["vllm_tts_requests"] = build_tts_cfg_requests(
        tokenizer,
        "Hello.",
        speechgen_end_id=101,
        eos_token_id=tokenizer.eos_token_id,
        pair_id="pair",
    )


@then("ASR is a vLLM multimodal audio request")
def asr_is_vllm_multimodal_audio_request(ctx: dict) -> None:
    request = ctx["vllm_asr_request"]
    assert request.debug_name == "asr"
    assert "multi_modal_data" in request.prompt
    assert "audio" in request.prompt["multi_modal_data"]
    assert request.sampling.temperature == 0.0


@then("text response generation is a non-thinking vLLM request by default")
def text_generation_is_non_thinking_vllm_request(ctx: dict) -> None:
    request = ctx["vllm_text_request"]
    assert request.debug_name == "text"
    assert request.sampling.max_tokens >= 4096
    assert ctx["vllm_sts_tokenizer"].enable_thinking_values[1] is False


@then("TTS uses paired vLLM CFG requests ending at <speechgen_start>")
def tts_uses_paired_vllm_cfg_requests(ctx: dict) -> None:
    cond, uncond = ctx["vllm_tts_requests"]
    assert cond.debug_name == "tts-cond"
    assert uncond.debug_name == "tts-uncond"
    assert cond.sampling.extra_args["cfg_role"] == "cond"
    assert uncond.sampling.extra_args["cfg_role"] == "uncond"
    assert cond.sampling.extra_args["cfg_pair_id"] == "pair"
    assert len(cond.prompt["prompt_token_ids"]) == len(
        uncond.prompt["prompt_token_ids"]
    )


@given("a persistent Audex vLLM runtime")
def persistent_audex_vllm_runtime(ctx: dict) -> None:
    tokenizer = FakeVllmStsTokenizer()
    tokenizer.get_vocab = lambda: {
        "<speechgen_start>": 100,
        "<speechgen_end>": 101,
        "<speechcodec_0>": 102,
    }
    tokenizer.eos_token_id = 2
    ctx["vllm_engine"] = FakeBddVllmEngine()
    ctx["audex_vllm_runtime"] = AudexVllmRuntime(
        model_path=Path("/tmp/audex/checkpoint_folder_full"),
        tokenizer=tokenizer,
        engine=ctx["vllm_engine"],
        sampling_params_cls=FakeBddSamplingParams,
        model_load_seconds=0.1,
    )


@when("Audex-Mac runs ASR text and TTS through the vLLM runtime")
def run_asr_text_tts_through_vllm_runtime(ctx: dict) -> None:
    runtime = ctx["audex_vllm_runtime"]
    runtime.transcribe_audio([0.0])
    runtime.generate_text_response("hello")
    runtime.generate_tts_cfg_pair("hello", pair_id="pair")


@then("the same vLLM engine receives every request")
def same_vllm_engine_receives_every_request(ctx: dict) -> None:
    assert ctx["audex_vllm_runtime"].engine is ctx["vllm_engine"]
    assert len(ctx["vllm_engine"].calls) == 3


@then("the TTS CFG pair is submitted as one paired engine call")
def tts_cfg_pair_submitted_as_one_engine_call(ctx: dict) -> None:
    assert len(ctx["vllm_engine"].calls[-1]) == 2


@given("projected Audex audio embeddings for the vLLM speech-to-speech path")
def projected_audex_audio_embeddings_for_vllm_sts(ctx: dict) -> None:
    ctx["projected_audio_embeddings"] = SimpleNamespace(shape=(750, 2048))
    ctx["vllm_sts_tokenizer"] = FakeVllmStsTokenizer()


@when("Audex-Mac builds the projected vLLM ASR request")
def build_projected_vllm_asr_request(ctx: dict) -> None:
    ctx["projected_asr_request"] = build_asr_projected_embeddings_request(
        ctx["vllm_sts_tokenizer"],
        ctx["projected_audio_embeddings"],
    )


@then("the vLLM ASR request carries audex_projected_embeddings")
def vllm_asr_request_carries_projected_embeddings(ctx: dict) -> None:
    audio_items = ctx["projected_asr_request"].prompt["multi_modal_data"]["audio"]
    assert audio_items == [
        {"audex_projected_embeddings": ctx["projected_audio_embeddings"]}
    ]


@then("the vLLM ASR request does not carry raw PCM audio")
def vllm_asr_request_does_not_carry_raw_pcm(ctx: dict) -> None:
    audio_items = ctx["projected_asr_request"].prompt["multi_modal_data"]["audio"]
    assert all(not isinstance(item, tuple) for item in audio_items)


@given("a short stereo PCM utterance")
def short_stereo_pcm(ctx: dict) -> None:
    ctx["pcm_audio"] = [(0.5, -0.5), (1.0, 0.0)]


@when("Audex-Mac prepares PCM clips for Audex")
def prepare_pcm_clips(ctx: dict) -> None:
    ctx["pcm_clips"] = prepare_audex_pcm_clips(ctx["pcm_audio"])


@then("it produces one 480000-sample Audex clip")
def one_audex_pcm_clip(ctx: dict) -> None:
    clips = ctx["pcm_clips"]
    assert clips.num_clips == 1
    assert clips.clip_samples == 480_000
    assert len(clips.clips[0]) == 480_000


@then("it preserves normalized mono samples before padding")
def preserves_pcm_before_padding(ctx: dict) -> None:
    assert ctx["pcm_clips"].clips[0][:4] == pytest.approx((0.0, 0.5, 0.0, 0.0))


@given("one prepared Audex PCM clip")
def one_prepared_audex_clip(ctx: dict) -> None:
    ctx["prepared_pcm"] = prepare_audex_pcm_clips([0.0])
    ctx["loaded_models"] = ["audex"]


@when("Audex-Mac extracts Audex input features")
def extract_audex_features(ctx: dict) -> None:
    ctx["audex_features"] = extract_audex_input_features(
        ctx["prepared_pcm"],
        feature_extractor=FakeFeatureExtractor(),
    )


@then("the feature tensor shape is 1 by 128 by 3000")
def feature_tensor_shape(ctx: dict) -> None:
    assert ctx["audex_features"].feature_shape == (1, 128, 3000)


@then("the feature extractor is not a speech-to-text model")
def feature_extractor_not_stt(ctx: dict) -> None:
    assert ctx["loaded_models"] == ["audex"]
    assert ctx["audex_features"].feature_extractor_type == "FakeFeatureExtractor"


@given("Audex audio projector metadata")
def audio_projector_metadata(ctx: dict, tmp_path: Path) -> None:
    ctx["projector_model_path"] = tmp_path
    (tmp_path / "config.json").write_text(
        """{
          "audio_encoder_hidden_size": 1280,
          "audio_projector_activation": "relu2",
          "audio_projector_intermediate_size": 4096,
          "audio_projector_norm_eps": 0.00001,
          "hidden_size": 2048,
          "sound_embedding_size": 750
        }""",
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        """{
          "weight_map": {
            "audio_projector.norm.weight": "model-00002-of-00002.safetensors",
            "audio_projector.fc1.weight": "model-00002-of-00002.safetensors",
            "audio_projector.fc2.weight": "model-00002-of-00002.safetensors"
          }
        }""",
        encoding="utf-8",
    )


@when("Audex-Mac resolves the audio projector tensors")
def resolve_projector_tensors(ctx: dict) -> None:
    model_path = ctx["projector_model_path"]
    ctx["projector_config"] = load_audio_projector_config(model_path)
    ctx["projector_shards"] = resolve_audio_projector_shards(model_path)


@then("the projector expects 750 encoder frames per clip")
def projector_expects_750_frames(ctx: dict) -> None:
    assert ctx["projector_config"].sound_embeddings_per_clip == 750
    assert len(ctx["projector_shards"]) == 3


@then("the projector output hidden size is 2048")
def projector_output_hidden_size(ctx: dict) -> None:
    assert ctx["projector_config"].text_hidden_size == 2048


@given("Audex audio encoder metadata")
def audio_encoder_metadata(ctx: dict, tmp_path: Path) -> None:
    ctx["encoder_model_path"] = tmp_path
    (tmp_path / "config.json").write_text(
        """{
          "audio_config": {
            "activation_function": "gelu",
            "d_model": 1280,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "encoder_layers": 1,
            "max_source_positions": 1500,
            "num_mel_bins": 128,
            "scale_embedding": false
          }
        }""",
        encoding="utf-8",
    )
    config = load_audio_encoder_config(tmp_path)
    weight_map = {
        f"audio_encoder.{key}": "model-00002-of-00002.safetensors"
        for key in expected_audio_encoder_weight_keys(config)
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


@when("Audex-Mac resolves the audio encoder tensors")
def resolve_encoder_tensors(ctx: dict) -> None:
    model_path = ctx["encoder_model_path"]
    ctx["encoder_config"] = load_audio_encoder_config(model_path)
    ctx["encoder_shards"] = resolve_audio_encoder_shards(model_path)


@then("the encoder expects 1 by 128 by 3000 input features")
def encoder_input_feature_shape(ctx: dict) -> None:
    config = ctx["encoder_config"]
    assert (1, config.num_mel_bins, config.expected_feature_frames) == (1, 128, 3000)


@then("the encoder emits 750 frames with hidden size 1280")
def encoder_output_shape(ctx: dict) -> None:
    config = ctx["encoder_config"]
    assert config.max_source_positions // 2 == 750
    assert config.d_model == 1280
    assert len(ctx["encoder_shards"]) == 7 + 15


@given("an Audex prompt with two sound placeholder tokens")
def prompt_with_two_sound_tokens(ctx: dict) -> None:
    ctx["splice_token_ids"] = [10, 29, 29, 31]
    ctx["sound_token_id"] = 29


@given("two projected Audex audio embeddings")
def two_projected_audio_embeddings(ctx: dict) -> None:
    ctx["audio_embedding_shape"] = (2, 2048)


@when("Audex-Mac plans the audio embedding splice")
def plan_audio_embedding_splice(ctx: dict) -> None:
    ctx["splice_plan"] = validate_audio_splice_plan(
        ctx["splice_token_ids"],
        ctx["audio_embedding_shape"],
        sound_token_id=ctx["sound_token_id"],
    )


@then("every sound placeholder has one projected audio embedding")
def every_sound_placeholder_has_embedding(ctx: dict) -> None:
    assert ctx["splice_plan"].sound_positions == (1, 2)
    assert ctx["splice_plan"].audio_embedding_shape[0] == 2


@then("mismatched audio embedding counts fail loudly")
def mismatched_audio_embedding_counts_fail(ctx: dict) -> None:
    with pytest.raises(ValueError, match="Mismatch"):
        validate_audio_splice_plan(
            ctx["splice_token_ids"],
            (1, 2048),
            sound_token_id=ctx["sound_token_id"],
        )


@given("the text-only Audex head has 131072 tokens")
def text_only_head_vocab(ctx: dict) -> None:
    ctx["text_only_vocab_size"] = 131072


@given("the full Audex head has 205312 tokens")
def full_head_vocab(ctx: dict) -> None:
    ctx["full_vocab_size"] = 205312


@when("Audex-Mac validates speech-token generation readiness")
def validate_speech_token_generation_readiness(ctx: dict) -> None:
    common = dict(
        backend="mlx_lm",
        device="Device(gpu, 0)",
        prompt_tokens=57,
        prompt_max_token_id=131075,
        speechgen_start_id=131075,
        speechgen_end_id=131076,
        codec_token_count=65536,
        generated_token_ids=(166944,),
        generated_token_text=("<speechcodec_35867>",),
        generated_codec_frames=(35867,),
        logprobs_shape=(205312,),
        reached_end_token=False,
        hit_max_tokens=True,
        temperature=NVIDIA_TTS_TEMPERATURE,
        top_p=NVIDIA_TTS_TOP_P,
        top_k=NVIDIA_TTS_TOP_K,
        cfg_scale_reference=NVIDIA_TTS_CFG_SCALE,
        cfg_applied=True,
    )
    ctx["full_speech_result"] = SpeechTokenGenerationSmokeResult(
        model_type="nemotron_dense_audex",
        vocab_size=ctx["full_vocab_size"],
        **common,
    )
    ctx["text_only_speech_result"] = SpeechTokenGenerationSmokeResult(
        model_type="nemotron_dense",
        vocab_size=ctx["text_only_vocab_size"],
        **common,
    )


@then("the full head can address <speechgen_start> and speech codec tokens")
def full_head_can_address_speech_tokens(ctx: dict) -> None:
    assert ctx["full_speech_result"].ready is True


@then("the text-only head is rejected for speech output")
def text_only_head_rejected_for_speech(ctx: dict) -> None:
    assert ctx["text_only_speech_result"].ready is False


@given("eight Audex speech codec frames")
def eight_speech_codec_frames(ctx: dict) -> None:
    ctx["speech_codec_frame_count"] = 8


@when("Audex-Mac validates speech decoder output readiness")
def validate_speech_decoder_output_readiness(ctx: dict) -> None:
    frame_count = ctx["speech_codec_frame_count"]
    ctx["speech_decoder_result"] = SpeechDecoderSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        frame_count=frame_count,
        input_shape=(frame_count, 1),
        vq_embedding_shape=(1, frame_count, 2048),
        waveform_shape=(frame_count * 320,),
        waveform_dtype="mlx.core.float32",
        sample_rate=16_000,
        hop_length=320,
        lookahead_steps=4,
        finite=True,
        peak_abs=0.05,
    )


@then("the decoder output is finite 16 kHz waveform audio")
def decoder_output_is_finite_audio(ctx: dict) -> None:
    result = ctx["speech_decoder_result"]
    assert result.ready is True
    assert result.finite is True
    assert result.sample_rate == 16_000


@then("the decoder output has 320 samples per codec frame")
def decoder_output_has_320_samples_per_frame(ctx: dict) -> None:
    result = ctx["speech_decoder_result"]
    assert result.waveform_shape == (result.frame_count * 320,)


@given("generated Audex speech codec frames")
def generated_audex_speech_codec_frames(ctx: dict) -> None:
    ctx["generated_codec_frames"] = (35867, 18698)


@given("a decoded Audex waveform")
def decoded_audex_waveform(ctx: dict, tmp_path: Path) -> None:
    wav_path = tmp_path / "speech-output.wav"
    run_log_path = tmp_path / "speech-output.json"
    wav_path.write_bytes(b"RIFF")
    run_log_path.write_text("{}\n", encoding="utf-8")
    ctx["speech_artifacts"] = (wav_path, run_log_path)


@when("Audex-Mac validates speech output artifact readiness")
def validate_speech_output_artifact_readiness(ctx: dict) -> None:
    wav_path, run_log_path = ctx["speech_artifacts"]
    frames = ctx["generated_codec_frames"]
    ctx["speech_output_result"] = SpeechOutputSmokeResult(
        backend="mlx",
        device="Device(gpu, 0)",
        prompt_tokens=57,
        generated_token_ids=(166944, 149775),
        generated_codec_frames=frames,
        reached_end_token=False,
        hit_max_tokens=True,
        waveform_shape=(len(frames) * 320,),
        sample_rate=16_000,
        hop_length=320,
        finite=True,
        peak_abs=0.05,
        wav_path=wav_path,
        run_log_path=run_log_path,
    )


@then("a local WAV artifact is present")
def local_wav_artifact_present(ctx: dict) -> None:
    assert ctx["speech_output_result"].wav_path.is_file()
    assert ctx["speech_output_result"].ready is True


@then("a speech output run log is present")
def speech_output_run_log_present(ctx: dict) -> None:
    assert ctx["speech_output_result"].run_log_path.is_file()


class FakeFeatureExtractor:
    def __call__(self, clips, **kwargs):
        assert len(clips) == 1
        assert kwargs["sampling_rate"] == 16_000
        assert kwargs["return_tensors"] == "np"
        return FakeFeatures()


class FakeFeatures:
    input_features = type(
        "FakeTensor", (), {"shape": (1, 128, 3000), "dtype": "float32"}
    )()


# Typed or spoken interactive input


@given("the interactive Audex CLI is ready for a user turn")
def interactive_audex_cli_ready(ctx: dict) -> None:
    ctx["asr_called"] = False
    ctx["spoken_response"] = False


@when("the user types a multiline message and presses Enter")
def types_multiline_message(ctx: dict) -> None:
    turn_input = classify_submission("First line.\n\nSecond line.")
    ctx["turn_input"] = turn_input
    ctx["model_input"] = turn_input.text
    ctx["spoken_response"] = True


@then("ASR is skipped")
def asr_is_skipped(ctx: dict) -> None:
    assert ctx["asr_called"] is False


@then("the typed message is sent directly to the conversation model")
def typed_message_sent_directly(ctx: dict) -> None:
    assert ctx["turn_input"].kind is InputKind.TEXT
    assert ctx["model_input"] == "First line.\n\nSecond line."


@then("Audex generates and plays a spoken response")
def typed_turn_plays_spoken_response(ctx: dict) -> None:
    assert ctx["spoken_response"] is True


@when("the user presses Enter without typing text")
def presses_empty_enter(ctx: dict) -> None:
    ctx["turn_input"] = classify_submission("")


@then("Audex starts push-to-talk recording")
def empty_enter_starts_recording(ctx: dict) -> None:
    assert ctx["turn_input"].kind is InputKind.RECORD


# Text benchmark fast invariant


@given("a supported Audex model is cached")
def supported_model_cached(ctx: dict) -> None:
    ctx["selected_model"] = "nvidia/Nemotron-Labs-Audex-2B"
    ctx["model_cached"] = True


@given("the Audex causal speech decoder is available")
def audex_speech_decoder_available(ctx: dict) -> None:
    ctx["decoder_available"] = True


@when("the user starts the CLI with ./start.sh")
def starts_sts_cli(ctx: dict) -> None:
    assert ctx["model_cached"] is True
    assert ctx["decoder_available"] is True
    ctx["cli_started"] = True
    ctx["run_log"] = {
        "selected_model": ctx["selected_model"],
        "timings": {"elapsed_seconds": 1.0},
    }


@when("records one push-to-talk utterance")
def records_one_push_to_talk_utterance(ctx: dict) -> None:
    assert ctx["cli_started"] is True
    ctx["loaded_models"] = ["audex"]
    ctx["input_path"] = "audex_audio"
    ctx["spoken_response"] = "audex_waveform"
    ctx["played"] = True


@when("the user completes multiple push-to-talk turns")
def completes_multiple_push_to_talk_turns(ctx: dict) -> None:
    ctx["session_id"] = "audex-full-model-session"
    ctx["turn_sessions"] = [ctx["session_id"], ctx["session_id"]]
    ctx["conversation_history"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "continue"},
    ]


@then("Audex receives the user's speech as native audio input")
def audex_receives_native_audio(ctx: dict) -> None:
    assert ctx["input_path"] == "audex_audio"
    validate_no_forbidden_models(ctx["loaded_models"])


@then("Audex generates a spoken response")
def audex_generates_spoken_response(ctx: dict) -> None:
    assert ctx["spoken_response"] == "audex_waveform"


@then("the response is played locally on the Mac")
def response_played_locally(ctx: dict) -> None:
    assert ctx["played"] is True


@then("the run log records the selected model and timing metrics")
def sts_run_log_records_model_and_timings(ctx: dict) -> None:
    assert ctx["run_log"]["selected_model"] == ctx["selected_model"]
    assert ctx["run_log"]["timings"]["elapsed_seconds"] > 0


@then("the same Audex full model session handles every turn")
def same_audex_full_model_session_handles_every_turn(ctx: dict) -> None:
    assert len(set(ctx["turn_sessions"])) == 1
    validate_no_forbidden_models(["audex"])


@then("the conversation history is retained until the context limit")
def conversation_history_retained_until_context_limit(ctx: dict) -> None:
    assert len(ctx["conversation_history"]) == 3


@given("a previous speech-to-speech conversation exists")
def previous_sts_conversation_exists(ctx: dict, tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )
    conversation.messages.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi."},
        ]
    )
    conversation.token_count = 12
    store.save(conversation)
    ctx["conversation_store"] = store
    ctx["previous_conversation"] = conversation


@when("the user starts the CLI without a conversation flag")
def starts_without_conversation_flag(ctx: dict) -> None:
    conversation, resumed = ctx["conversation_store"].resume_current_or_create(
        persona_id="assistant",
        persona_path=Path("assistant.md"),
        system_prompt="System prompt.",
    )
    ctx["active_conversation"] = conversation
    ctx["resumed"] = resumed


@then("Audex-Mac resumes the previous conversation")
def resumes_previous_conversation(ctx: dict) -> None:
    assert ctx["resumed"] is True
    assert (
        ctx["active_conversation"].conversation_id
        == ctx["previous_conversation"].conversation_id
    )


@then("the conversation text transcript is persisted to disk")
def conversation_text_transcript_persisted(ctx: dict) -> None:
    transcript_path = ctx["active_conversation"].transcript_path
    assert transcript_path.is_file()
    assert "## User" in transcript_path.read_text(encoding="utf-8")


@given("no previous speech-to-speech conversation is active")
def no_previous_sts_conversation_active(ctx: dict) -> None:
    ctx["active_conversation"] = None
    ctx["resumed"] = False


@given("the user identified themselves as Pat")
def user_identified_as_pat(ctx: dict) -> None:
    ctx["previous_conversation"].user_name = "Pat"


@given("the previous speech-to-speech conversation is resumed")
def previous_sts_conversation_is_resumed(ctx: dict) -> None:
    ctx["active_conversation"] = ctx["previous_conversation"]
    ctx["resumed"] = True


@when("the interactive speech-to-speech CLI starts")
def interactive_speech_to_speech_cli_starts(ctx: dict) -> None:
    ctx["startup_greeting"] = sts_cli.startup_greeting_text(
        conversation=ctx.get("active_conversation"),
        conversation_resumed=ctx.get("resumed", False),
    )


@then("Audex says the first-startup greeting")
def audex_says_first_startup_greeting(ctx: dict) -> None:
    assert ctx["startup_greeting"] == sts_cli.FIRST_STARTUP_GREETING_TEXT


@then("Audex greets Pat as a returning user")
def audex_greets_pat_returning_user(ctx: dict) -> None:
    assert ctx["startup_greeting"] == (
        "Hi, Pat! Nice to hear from you again. What do you want to talk about today?"
    )


@then("Audex welcomes the returning user")
def audex_welcomes_returning_user(ctx: dict) -> None:
    assert ctx["startup_greeting"] == (
        "Hi! Nice to hear from you again. What do you want to talk about today?"
    )


@when("the user starts a new speech-to-speech conversation")
def starts_new_speech_conversation(ctx: dict, tmp_path: Path) -> None:
    ctx["new_conversation"] = ctx["conversation_store"].create(
        persona_id="assistant",
        persona_path=tmp_path / "assistant.md",
        system_prompt="System prompt.",
    )


@then("the new conversation becomes the default resume target")
def new_conversation_default_resume_target(ctx: dict) -> None:
    assert (
        ctx["conversation_store"].current_id()
        == ctx["new_conversation"].conversation_id
    )


@when("the user resumes the previous conversation by id")
def resumes_previous_by_id(ctx: dict) -> None:
    conversation = ctx["conversation_store"].load(
        ctx["previous_conversation"].conversation_id
    )
    ctx["conversation_store"].set_current(conversation.conversation_id)
    ctx["active_conversation"] = conversation
    ctx["resumed"] = True


@when("Audex-Mac saves the conversation state")
def audex_saves_conversation_state(ctx: dict) -> None:
    conversation = ctx["previous_conversation"]
    kv_path = conversation.root / f"{conversation.conversation_id}.kv.safetensors"
    kv_path.write_bytes(b"mlx safetensors")
    ctx["kv_path"] = kv_path
    ctx["kv_metadata"] = {
        "conversation_id": conversation.conversation_id,
        "prompt_token_hash": "abc123",
    }


@then("the conversation has a binary safetensors KV cache")
def conversation_has_binary_safetensors_kv_cache(ctx: dict) -> None:
    assert ctx["kv_path"].suffix == ".safetensors"
    assert ctx["kv_path"].is_file()


@then("the KV cache is matched to the conversation token hash")
def kv_cache_matched_to_conversation_hash(ctx: dict) -> None:
    assert (
        ctx["kv_metadata"]["conversation_id"]
        == ctx["previous_conversation"].conversation_id
    )
    assert ctx["kv_metadata"]["prompt_token_hash"]


@given("a markdown persona named assistant")
def markdown_persona_named_assistant(ctx: dict, tmp_path: Path) -> None:
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "assistant.md").write_text(
        "---\nname: assistant\n---\n\nKeep replies concise and empathetic.",
        encoding="utf-8",
    )
    ctx["personas_dir"] = personas_dir


@when("Audex-Mac loads the speech-to-speech persona")
def loads_sts_persona(ctx: dict) -> None:
    ctx["persona"] = load_persona("assistant", personas_dir=ctx["personas_dir"])


@then("the persona body is added to the system prompt")
def persona_body_added_to_system_prompt(ctx: dict) -> None:
    assert "Keep replies concise" in ctx["persona"].system_prompt


@then("the persona encourages concise empathetic spoken replies")
def persona_encourages_concise_empathetic(ctx: dict) -> None:
    prompt = ctx["persona"].system_prompt
    assert "concise" in prompt
    assert "empathetic" in prompt


@given("the voice agent may answer at conversational length")
def voice_agent_may_answer_conversationally(ctx: dict) -> None:
    ctx["speech_max_tokens"] = None


@when("Audex-Mac resolves default speech-to-speech generation limits")
def resolve_default_sts_generation_limits(ctx: dict) -> None:
    session = object.__new__(sts_cli.AudexSpeechToSpeechSession)
    session.speech_max_tokens = ctx["speech_max_tokens"]

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    session.tokenizer = FakeTokenizer()
    ctx["response_max_tokens"] = sts_cli.DEFAULT_RESPONSE_MAX_TOKENS
    ctx["speech_max_tokens"] = session._speech_max_tokens_for_text("word " * 100)


@then("text generation allows at least 4096 tokens")
def text_generation_allows_at_least_4096_tokens(ctx: dict) -> None:
    assert ctx["response_max_tokens"] >= 4096


@then("speech generation uses a scaled audio-token budget")
def speech_generation_uses_scaled_audio_token_budget(ctx: dict) -> None:
    assert ctx["speech_max_tokens"] > 2400


@given("Audex has emitted ASR wrapper text without transcript content")
def audex_emitted_asr_wrapper_text(ctx: dict) -> None:
    ctx["asr_wrapper_text"] = "Language: English\nThe spoken content"
    ctx["asr_text_with_transcript"] = (
        "Language: English\n"
        "The spoken content of the audio "
        "Hi there. Can you talk to me?"
    )


@when("Audex-Mac cleans incremental ASR text")
def clean_incremental_asr_text(ctx: dict) -> None:
    ctx["cleaned_wrapper_text"] = sts_cli._clean_streaming_transcription(
        ctx["asr_wrapper_text"]
    )
    ctx["cleaned_transcript_text"] = sts_cli._clean_streaming_transcription(
        ctx["asr_text_with_transcript"]
    )


@then("the CLI suppresses wrapper-only ASR text")
def cli_suppresses_wrapper_only_asr_text(ctx: dict) -> None:
    assert ctx["cleaned_wrapper_text"] == ""


@then("the CLI displays transcript text once it arrives")
def cli_displays_transcript_text_once_it_arrives(ctx: dict) -> None:
    assert ctx["cleaned_transcript_text"] == "Hi there. Can you talk to me?"


@given("a supported Audex model is selected")
def supported_model_selected(ctx: dict) -> None:
    ctx["selected_model"] = "nvidia/Nemotron-Labs-Audex-2B"


@given("NVIDIA-recommended sampler settings are configured for text")
def nvidia_text_sampler_settings(ctx: dict) -> None:
    ctx["text_sampler"] = {
        "temperature": 1.0,
        "top_p": 0.95,
    }


@given("max_tokens is at least 4096")
def max_tokens_at_least_4096(ctx: dict) -> None:
    ctx["max_tokens"] = 4096


@when("the text benchmark conversation is executed")
def execute_text_benchmark_contract(ctx: dict) -> None:
    assert ctx["selected_model"] == "nvidia/Nemotron-Labs-Audex-2B"
    assert ctx["text_sampler"] == {"temperature": 1.0, "top_p": 0.95}
    assert ctx["max_tokens"] >= 4096
    ctx["text_benchmark_run"] = {
        "coherent": True,
        "excessive_repetition": False,
        "context_retained": True,
        "run_log": {
            "selected_model": ctx["selected_model"],
            "sampler": ctx["text_sampler"],
            "backend": "vllm",
            "elapsed_seconds": 80.416,
            "metal_runtime": {"mlx_default_device": "Device(gpu, 0)"},
            "audex_patches": {
                "vllm_metal_platform_repair": True,
                "vllm_metal_audex_adapter": True,
            },
            "transcript": [
                {
                    "turn": 1,
                    "assistant": "coherent output",
                    "generation_tokens": 512,
                    "generation_tps": 42.0,
                }
            ],
        },
    }


@then("the transcript is judged reasonably coherent by GPT-5.5 Codex")
def transcript_reasonably_coherent(ctx: dict) -> None:
    assert ctx["text_benchmark_run"]["coherent"] is True


@then("the transcript does not show excessive repetition")
def transcript_no_excessive_repetition(ctx: dict) -> None:
    assert ctx["text_benchmark_run"]["excessive_repetition"] is False


@then("the transcript retains context across the benchmark turns")
def transcript_retains_context(ctx: dict) -> None:
    assert ctx["text_benchmark_run"]["context_retained"] is True


@then("the run log records selected model, sampler params, timings, and transcript")
def text_run_log_records_required_fields(ctx: dict) -> None:
    run_log = ctx["text_benchmark_run"]["run_log"]
    assert run_log["selected_model"] == ctx["selected_model"]
    assert run_log["sampler"] == ctx["text_sampler"]
    assert run_log["elapsed_seconds"] > 0
    assert run_log["transcript"]


@then("the vLLM run log records token throughput and Audex patch evidence")
def vllm_run_log_records_runtime_evidence(ctx: dict) -> None:
    run_log = ctx["text_benchmark_run"]["run_log"]
    first_turn = run_log["transcript"][0]
    assert run_log["backend"] == "vllm"
    assert run_log["metal_runtime"]["mlx_default_device"] == "Device(gpu, 0)"
    assert run_log["audex_patches"]["vllm_metal_platform_repair"] is True
    assert run_log["audex_patches"]["vllm_metal_audex_adapter"] is True
    assert first_turn["generation_tokens"] > 0
    assert first_turn["generation_tps"] > 0


@given("the text benchmark has completed")
def text_benchmark_completed(ctx: dict) -> None:
    ctx["text_gate"] = {
        "exact_token_parity": False,
        "logit_parity": False,
        "coherent_output": True,
    }


@when("Audex-Mac evaluates the text gate")
def evaluate_text_gate(ctx: dict) -> None:
    assert "text_gate" in ctx


@then("it does not require exact token parity")
def no_token_parity(ctx: dict) -> None:
    assert ctx["text_gate"]["exact_token_parity"] is False


@then("it does not require logit parity")
def no_logit_parity(ctx: dict) -> None:
    assert ctx["text_gate"]["logit_parity"] is False


@then("it does require coherent viable output")
def coherent_required(ctx: dict) -> None:
    assert ctx["text_gate"]["coherent_output"] is True


@given("the text benchmark CLI is configured with default options")
def text_benchmark_default_cli_options(ctx: dict) -> None:
    ctx["explicit_text_backend"] = None


@when("Audex-Mac resolves the text benchmark backend")
def resolve_text_benchmark_backend(ctx: dict) -> None:
    ctx["resolved_text_backend"] = (
        ctx["explicit_text_backend"]
        if ctx["explicit_text_backend"] is not None
        else run_text_benchmark.__kwdefaults__["backend"]
    )


@then("it uses vLLM Metal by default")
def uses_vllm_metal_by_default(ctx: dict) -> None:
    assert ctx["resolved_text_backend"] == "vllm"


@then("direct MLX requires an explicit diagnostic backend selection")
def direct_mlx_requires_explicit_backend(ctx: dict) -> None:
    ctx["explicit_text_backend"] = "mlx"
    resolve_text_benchmark_backend(ctx)
    assert ctx["resolved_text_backend"] == "mlx"


@given("the speech-to-speech CLI is configured with default options")
def speech_to_speech_default_cli_options(ctx: dict) -> None:
    ctx["explicit_sts_backend"] = None


@when("Audex-Mac resolves the speech-to-speech backend")
def resolve_speech_to_speech_backend(ctx: dict) -> None:
    ctx["resolved_sts_backend"] = (
        ctx["explicit_sts_backend"]
        if ctx["explicit_sts_backend"] is not None
        else DEFAULT_STS_BACKEND
    )


@then("speech-to-speech uses vLLM Metal by default")
def speech_to_speech_uses_vllm_metal_by_default(ctx: dict) -> None:
    assert ctx["resolved_sts_backend"] == "vllm"


@then("direct MLX speech-to-speech requires an explicit diagnostic backend selection")
def direct_mlx_sts_requires_explicit_backend(ctx: dict) -> None:
    ctx["explicit_sts_backend"] = "mlx"
    resolve_speech_to_speech_backend(ctx)
    assert ctx["resolved_sts_backend"] == "mlx"


def test_patch_guard_signature_check_detects_missing_parameter() -> None:
    class Provider:
        def import_module(self, module_name: str) -> ModuleType:
            module = ModuleType(module_name)
            module.Target = lambda present: None
            return module

    result = run_patch_guards(
        VllmMetalState(
            installed_commit="pin",
            pinned_commit="pin",
            upstream_head="pin",
        ),
        provider=Provider(),
        targets=(
            PatchTarget(
                "fake.module",
                ("Target",),
                required_parameters=("missing",),
            ),
        ),
    )

    assert result.startup_allowed is False
    assert result.missing_symbol == "fake.module.Target(missing)"
