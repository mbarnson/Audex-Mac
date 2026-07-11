"""Launch the complete local Audex browser application."""

from __future__ import annotations

import argparse
import os
import webbrowser
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from ..audio_evaluation_enhancement import (
    NvidiaEnhancementVae,
    enhancement_vae_artifact_identity,
)
from ..audio_evaluation_xcodec import XCodec1WavDecoder, xcodec1_artifact_identity
from ..audio_runtime import preflight_audio_runtime
from ..conversations import ConversationStore
from ..model_select import (
    HuggingFaceSnapshotProbe,
    download_model_snapshot,
    select_model,
)
from ..models import (
    AUDEX_2B_REPO,
    AUDEX_30B_NVFP4_REPO,
    AUDEX_30B_REPO,
    SUPPORTED_MODELS,
)
from ..personas import DEFAULT_PERSONA_NAME, load_persona
from ..sound_lab.adapters import (
    AudexSoundLabPlanner,
    AudexTtaSoundGenerator,
    AudexVariantDesigner,
)
from ..sound_lab.catalog import SoundLabCatalog
from ..sound_lab.cli import (
    _resolve_or_download_enhancement_vae,
    _resolve_or_download_xcodec1,
)
from ..sound_lab.session import SoundLabSession
from ..vllm_sts_cli import VllmSpeechToSpeechSession
from .chat import ChatCoordinator
from .runtime import SharedAudexRuntimeFactory, SoundLabWebBackend
from .server import AudexWebApplication, serve
from .store import WebChatStore

DEFAULT_WEB_ROOT = Path(".audex/web")
_MODEL_CHOICES = {
    "2b": AUDEX_2B_REPO,
    "30b": AUDEX_30B_REPO,
    "30b-nvfp4": AUDEX_30B_NVFP4_REPO,
}


def main(
    argv: list[str] | None = None,
    *,
    opener: Callable[[str], Any] = webbrowser.open,
    serve_fn: Callable[..., None] = serve,
) -> int:
    parser = argparse.ArgumentParser(
        description="Audex local browser chat and sound-generation interface"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--model",
        choices=("auto", *_MODEL_CHOICES),
        default="auto",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--yes-download", action="store_true")
    parser.add_argument("--persona", default=DEFAULT_PERSONA_NAME)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--response-max-tokens", type=int, default=4096)
    parser.add_argument("--speech-max-tokens", type=int, default=None)
    parser.add_argument("--xcodec1-path", type=Path, default=None)
    parser.add_argument("--enhancement-vae-path", type=Path, default=None)
    parser.add_argument(
        "--xcodec-device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error(
            "Audex web binds to loopback only; use an SSH tunnel for remote access"
        )
    if args.response_max_tokens <= 0:
        parser.error("--response-max-tokens must be positive")
    if args.speech_max_tokens is not None and args.speech_max_tokens <= 0:
        parser.error("--speech-max-tokens must be positive")

    _configure_web_environment(os.environ)
    selected_model, model_path = _resolve_model(
        model=args.model,
        model_path=args.model_path,
        yes_download=args.yes_download,
    )
    preflight = preflight_audio_runtime(selected_model)
    decoder_path = model_path / "audex_causal_speech_decoder"
    if args.model_path is None:
        if not preflight.ready or preflight.decoder_path is None:
            missing = ", ".join(preflight.missing_items)
            raise RuntimeError(f"Audex browser speech runtime is incomplete: {missing}")
        decoder_path = preflight.decoder_path

    persona = load_persona(args.persona)
    root = DEFAULT_WEB_ROOT.resolve()
    conversation_store = ConversationStore(root / "model-conversations")

    def load_session(conversation, store, loaded_persona):
        print(f"Audex web: loading {selected_model.repo_id}...", flush=True)
        return VllmSpeechToSpeechSession(
            full_model_path=model_path,
            decoder_path=decoder_path,
            selected_model_repo=selected_model.repo_id,
            output_dir=root / "runs",
            thinking_enabled=args.thinking,
            response_max_tokens=args.response_max_tokens,
            speech_max_tokens=args.speech_max_tokens,
            conversation=conversation,
            conversation_store=store,
            persona=loaded_persona,
        )

    def load_sound_backend(session: VllmSpeechToSpeechSession):
        return _load_sound_backend(
            session=session,
            root=root / "sound-lab",
            model_repo=selected_model.repo_id,
            model_name=_model_name(selected_model.repo_id),
            model_profile=_model_profile(selected_model.repo_id),
            xcodec1_path=args.xcodec1_path,
            enhancement_vae_path=args.enhancement_vae_path,
            xcodec_device=args.xcodec_device,
        )

    runtime_factory = SharedAudexRuntimeFactory(
        conversation_store=conversation_store,
        persona=persona,
        session_loader=load_session,
        sound_backend_loader=load_sound_backend,
    )
    coordinator = ChatCoordinator(
        store=WebChatStore(root / "chats"),
        runtime_factory=runtime_factory,
    )
    application = AudexWebApplication(
        coordinator=coordinator,
        upload_root=root / "uploads",
    )
    try:
        serve_fn(
            application,
            host=args.host,
            port=args.port,
            on_ready=(None if args.no_open else opener),
        )
    finally:
        runtime_factory.shutdown()
    return 0


def _resolve_model(
    *,
    model: str,
    model_path: Path | None,
    yes_download: bool,
):
    if model == "auto":
        selection = select_model(HuggingFaceSnapshotProbe(), readiness="speech")
        selected = selection.selected
        cached = selection.cached
    else:
        selected = next(
            item for item in SUPPORTED_MODELS if item.repo_id == _MODEL_CHOICES[model]
        )
        cached = HuggingFaceSnapshotProbe().is_cached(selected, readiness="speech")
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Audex model path does not exist: {resolved}")
        return selected, resolved
    if not cached:
        if not yes_download:
            raise RuntimeError(
                f"Audex model is not cached: {selected.repo_id}. "
                "Run again with --yes-download."
            )
        download_model_snapshot(selected, readiness="speech")
    preflight = preflight_audio_runtime(selected)
    if preflight.model_path is None:
        raise RuntimeError(
            f"Audex could not resolve model snapshot: {selected.repo_id}"
        )
    return selected, preflight.model_path


def _load_sound_backend(
    *,
    session: VllmSpeechToSpeechSession,
    root: Path,
    model_repo: str,
    model_name: str,
    model_profile: str,
    xcodec1_path: Path | None,
    enhancement_vae_path: Path | None,
    xcodec_device: str,
) -> SoundLabWebBackend:
    if session.async_runtime is None:
        raise RuntimeError("Audex browser Sound Lab requires the async vLLM runtime.")
    print("Audex web: loading Sound Lab decoders...", flush=True)
    xcodec_config = _resolve_or_download_xcodec1(
        xcodec1_path,
        device=xcodec_device,
    )
    enhancement_config = _resolve_or_download_enhancement_vae(
        enhancement_vae_path,
        model=model_name,
        device=xcodec_device,
    )
    decoder = XCodec1WavDecoder(
        xcodec_config,
        allow_nvidia_reference_output=True,
        target_seconds=10.0,
    )
    enhancer = NvidiaEnhancementVae(enhancement_config)
    catalog = SoundLabCatalog(root / "catalog.sqlite3")
    runner = session.run_model_awaitable
    sound_session = SoundLabSession(
        catalog=catalog,
        planner=AudexSoundLabPlanner(runtime=session.async_runtime, run_sync=runner),
        designer=AudexVariantDesigner(runtime=session.async_runtime, run_sync=runner),
        generator=AudexTtaSoundGenerator(
            runtime=session.async_runtime,
            decode_to_wav=decoder,
            enhance_wav=enhancer,
            run_sync=runner,
        ),
        asset_root=root / "assets",
        model_repo=model_repo,
        recipe=(
            f"nvidia-tta-cfg3-{model_profile}+xcodec1@"
            f"{xcodec1_artifact_identity(xcodec_config.path)}+enhancement-vae@"
            f"{enhancement_vae_artifact_identity(enhancement_config.root)}"
        ),
    )
    return SoundLabWebBackend(session=sound_session, catalog=catalog)


def _configure_web_environment(env: MutableMapping[str, str]) -> None:
    env.setdefault("AUDEX_VLLM_ENABLE_CFG_WIRING", "1")
    env.setdefault("AUDEX_VLLM_CFG_MAX_MODEL_LEN", "8192")
    if not env.get("AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"):
        env["AUDEX_VLLM_NONPAGED_KV_CAPACITY_SEQS"] = "4"


def _model_name(repo_id: str) -> str:
    return "2b" if repo_id == AUDEX_2B_REPO else "30b"


def _model_profile(repo_id: str) -> str:
    return "nvfp4" if repo_id == AUDEX_30B_NVFP4_REPO else "bf16"
