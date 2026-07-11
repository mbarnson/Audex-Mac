"""One-command terminal entrypoint for Audex Sound Lab."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..audio_evaluation_xcodec import (
    XCODEC1_REPO_ID,
    XCODEC1_REVISION,
    XCodec1WavDecoder,
    resolve_xcodec1_config,
    xcodec1_artifact_identity,
)
from ..audio_model_resolver import (
    audio_model_repo,
    load_audio_vllm_runtime,
    resolve_cached_audio_model,
)
from .adapters import (
    AudexSoundLabPlanner,
    AudexTtaSoundGenerator,
    AudexVariantDesigner,
)
from .board import SoundLabBoard
from .catalog import SoundLabCatalog
from .session import SoundLabSession

DEFAULT_SOUND_LAB_ROOT = Path(".audex/sound-lab")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audex Sound Lab: conversational non-speech audio exploration"
    )
    parser.add_argument("--model", choices=("30b", "2b"), default="30b")
    parser.add_argument("--profile", choices=("bf16", "nvfp4"), default="nvfp4")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--xcodec1-path", type=Path, default=None)
    parser.add_argument(
        "--xcodec-device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--board-port", type=int, default=0)
    parser.add_argument("--no-open-board", action="store_true")
    args = parser.parse_args(argv)

    if args.model == "2b" and args.profile == "nvfp4":
        parser.error("--profile nvfp4 is only defined for --model 30b")
    if not 0 <= args.board_port <= 65535:
        parser.error("--board-port must be between 0 and 65535")

    model_repo = audio_model_repo(args.model, args.profile)
    model_path = args.model_path
    if model_path is None:
        model_path, model_repo = resolve_cached_audio_model(args.model, args.profile)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Audex Sound Lab model path does not exist: {model_path}"
        )

    xcodec_config = _resolve_or_download_xcodec1(
        args.xcodec1_path,
        device=args.xcodec_device,
    )
    print(f"Audex Sound Lab: loading {model_repo}...", flush=True)
    runtime = load_audio_vllm_runtime(model_path, args.profile)
    decoder = XCodec1WavDecoder(xcodec_config)
    root = DEFAULT_SOUND_LAB_ROOT.resolve()
    catalog = SoundLabCatalog(root / "catalog.sqlite3")
    board = SoundLabBoard(
        catalog,
        port=args.board_port,
        opener=None if args.no_open_board else webbrowser.open,
    )

    session = SoundLabSession(
        catalog=catalog,
        planner=AudexSoundLabPlanner(runtime=runtime),
        designer=AudexVariantDesigner(runtime=runtime),
        generator=AudexTtaSoundGenerator(runtime=runtime, decode_to_wav=decoder),
        asset_root=root / "assets",
        model_repo=model_repo,
        recipe=(
            "nvidia-tta-cfg3+xcodec1@"
            f"{xcodec1_artifact_identity(xcodec_config.path)}"
        ),
    )
    with board:
        print(f"Audex Sound Lab board: {board.url}", flush=True)
        print(
            "Type a sound request, such as 'audition five different explosions'. "
            "Type q to quit.",
            flush=True,
        )
        return run_sound_lab_repl(session)


def run_sound_lab_repl(
    session: SoundLabSession,
    *,
    read_line: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    """Run the typed Phase 1 loop through the SoundLabSession interface."""

    while True:
        try:
            raw = read_line("You: ")
        except (EOFError, KeyboardInterrupt, StopIteration):
            print(file=output)
            return 0
        text = raw.strip()
        if text.lower() in {"q", "quit", "exit"}:
            return 0
        if not text:
            continue
        try:
            turn = session.handle(text)
        except Exception as exc:
            print(f"Sound Lab error: {type(exc).__name__}: {exc}", file=output)
            continue
        print(f"Audex: {turn.message}", file=output)
        if turn.job_id is not None:
            print(
                f"{turn.job_id}: {turn.ready_count} ready, "
                f"{turn.failed_count} failed. Audition them on the board.",
                file=output,
            )


def _resolve_or_download_xcodec1(
    explicit_path: Path | None,
    *,
    device: str,
) -> Any:
    if explicit_path is not None or os.environ.get("XCODEC1_PATH"):
        return resolve_xcodec1_config(explicit_path, device=device)
    from huggingface_hub import snapshot_download

    print(
        f"Audex Sound Lab: resolving {XCODEC1_REPO_ID}; downloading if absent...",
        flush=True,
    )
    path = snapshot_download(
        repo_id=XCODEC1_REPO_ID,
        revision=XCODEC1_REVISION,
        token=os.environ.get("HF_TOKEN"),
    )
    return resolve_xcodec1_config(path, device=device)
