"""Supported Audex model metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace

AUDEX_2B_REPO = "nvidia/Nemotron-Labs-Audex-2B"
AUDEX_30B_REPO = "nvidia/Nemotron-Labs-Audex-30B-A3B"
AUDEX_30B_NVFP4_REPO = "txgsync/Nemotron-Labs-Audex-30B-A3B-NVFP4-mlx"


@dataclass(frozen=True, slots=True)
class AudexModel:
    repo_id: str
    label: str
    first_run_default: bool
    higher_reasoning: bool
    required_patterns: tuple[str, ...]
    speech_required_files: tuple[str, ...]
    speech_checkpoint_dirs: tuple[str, ...]
    text_required_files: tuple[str, ...]
    text_checkpoint_dirs: tuple[str, ...]


_AUDEX_30B = AudexModel(
    repo_id=AUDEX_30B_REPO,
    label="Audex-30B-A3B",
    first_run_default=False,
    higher_reasoning=True,
    required_patterns=(
        "checkpoint_folder_full/*",
        "audex_causal_speech_decoder/*",
        "nv-whisper/*",
    ),
    speech_required_files=(
        "checkpoint_folder_full/config.json",
        "checkpoint_folder_full/generation_config.json",
        "checkpoint_folder_full/model.safetensors.index.json",
        "checkpoint_folder_full/tokenizer.json",
        "checkpoint_folder_full/tokenizer_config.json",
        "checkpoint_folder_full/chat_template.jinja",
        "checkpoint_folder_full/modeling_nemotron_h_audio.py",
        "checkpoint_folder_full/audio_preprocessor/config.json",
        "checkpoint_folder_full/audio_preprocessor/preprocessor_config.json",
        "nv-whisper/config.json",
        "nv-whisper/model.safetensors.index.json",
        "nv-whisper/model.safetensors",
        "nv-whisper/preprocessor_config.json",
        "audex_causal_speech_decoder/config.json",
        "audex_causal_speech_decoder/model.safetensors",
        "audex_causal_speech_decoder/modeling_audex_causal_speech_decoder.py",
        "inference_scripts_vllm/audioqa_scripts/run_audioqa_vllm.py",
        "inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py",
    ),
    speech_checkpoint_dirs=("checkpoint_folder_full",),
    text_required_files=(
        "checkpoint_folder_textonly/config.json",
        "checkpoint_folder_textonly/generation_config.json",
        "checkpoint_folder_textonly/model.safetensors.index.json",
        "checkpoint_folder_textonly/tokenizer.json",
        "checkpoint_folder_textonly/tokenizer_config.json",
        "checkpoint_folder_textonly/modeling_nemotron_h.py",
        "inference_scripts_vllm/textonly_scripts/run_text_vllm_example.py",
    ),
    text_checkpoint_dirs=("checkpoint_folder_textonly",),
)


SUPPORTED_MODELS: tuple[AudexModel, ...] = (
    replace(
        _AUDEX_30B,
        repo_id=AUDEX_30B_NVFP4_REPO,
        label="Audex-30B-A3B NVFP4 (MLX)",
        text_required_files=_AUDEX_30B.speech_required_files,
        text_checkpoint_dirs=_AUDEX_30B.speech_checkpoint_dirs,
    ),
    _AUDEX_30B,
    AudexModel(
        repo_id=AUDEX_2B_REPO,
        label="Audex-2B",
        first_run_default=True,
        higher_reasoning=False,
        required_patterns=(
            "checkpoint_folder_full/*",
            "audex_causal_speech_decoder/*",
            "nv-whisper/*",
        ),
        speech_required_files=(
            "checkpoint_folder_full/config.json",
            "checkpoint_folder_full/generation_config.json",
            "checkpoint_folder_full/model.safetensors.index.json",
            "checkpoint_folder_full/tokenizer.json",
            "checkpoint_folder_full/tokenizer_config.json",
            "checkpoint_folder_full/chat_template.jinja",
            "checkpoint_folder_full/modeling_nemotron_h_audio.py",
            "checkpoint_folder_full/modeling_nemotron_dense.py",
            "checkpoint_folder_full/audio_preprocessor/config.json",
            "checkpoint_folder_full/audio_preprocessor/preprocessor_config.json",
            "nv-whisper/config.json",
            "nv-whisper/model.safetensors.index.json",
            "nv-whisper/model.safetensors",
            "nv-whisper/preprocessor_config.json",
            "audex_causal_speech_decoder/config.json",
            "audex_causal_speech_decoder/model.safetensors",
            "audex_causal_speech_decoder/modeling_audex_causal_speech_decoder.py",
            "inference_scripts_vllm/audioqa_scripts/run_audioqa_vllm.py",
            "inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/processing_audex_vllm.py",
            "inference_scripts_vllm/unified_s2s_scripts/cascaded_s2s_web_server.py",
        ),
        speech_checkpoint_dirs=("checkpoint_folder_full",),
        text_required_files=(
            "checkpoint_folder_textonly/config.json",
            "checkpoint_folder_textonly/generation_config.json",
            "checkpoint_folder_textonly/model.safetensors.index.json",
            "checkpoint_folder_textonly/tokenizer.json",
            "checkpoint_folder_textonly/tokenizer_config.json",
            "checkpoint_folder_textonly/modeling_nemotron_dense.py",
            "inference_scripts_vllm/textonly_scripts/run_text_vllm_example.py",
        ),
        text_checkpoint_dirs=("checkpoint_folder_textonly",),
    ),
)

DEFAULT_MODEL = next(model for model in SUPPORTED_MODELS if model.first_run_default)
HIGHER_REASONING_MODEL = next(
    model for model in SUPPORTED_MODELS if model.repo_id == AUDEX_30B_REPO
)
