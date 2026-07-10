"""Helper for regenerating Audex text-only checkpoint shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .checkpoints import HF_CACHE_ROOT, local_snapshot_path, verify_indexed_checkpoint
from .models import AUDEX_2B_REPO, AUDEX_30B_REPO, SUPPORTED_MODELS, AudexModel

MODEL_CHOICES = {
    "audex-2b": AUDEX_2B_REPO,
    "audex-30b-a3b": AUDEX_30B_REPO,
}
NVIDIA_CONVERSION_SCRIPT = "model_conversion_scripts/convert_full_HF_to_textonly_HF.py"
DEFAULT_SOURCE_FOLDERS = ("checkpoint_folder_audiogen", "checkpoint_folder_full")
DEFAULT_OUTPUT_FOLDER = "checkpoint_folder_textonly"


@dataclass(frozen=True, slots=True)
class TextOnlyConversionPlan:
    model: AudexModel
    snapshot_path: Path
    nvidia_script: Path
    input_folder: Path
    output_folder: Path
    missing_sidecars: tuple[str, ...]
    command: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_sidecars


def model_by_choice(choice: str) -> AudexModel:
    repo_id = MODEL_CHOICES[choice]
    return next(model for model in SUPPORTED_MODELS if model.repo_id == repo_id)


def build_conversion_plan(
    model: AudexModel,
    *,
    cache_root: Path = HF_CACHE_ROOT,
    input_folder_name: str | None = None,
    output_folder_name: str = DEFAULT_OUTPUT_FOLDER,
    overwrite: bool = False,
    python_executable: str = sys.executable,
) -> TextOnlyConversionPlan:
    snapshot_path = local_snapshot_path(model.repo_id, cache_root=cache_root)
    if snapshot_path is None:
        raise FileNotFoundError(f"No local Hugging Face snapshot for {model.repo_id}")

    nvidia_script = snapshot_path / NVIDIA_CONVERSION_SCRIPT
    if not nvidia_script.is_file():
        raise FileNotFoundError(f"Missing NVIDIA conversion script: {nvidia_script}")

    input_folder = _resolve_input_folder(snapshot_path, input_folder_name)
    output_folder = snapshot_path / output_folder_name
    missing_sidecars = _missing_textonly_sidecars(
        model,
        snapshot_path=snapshot_path,
        output_folder_name=output_folder_name,
    )

    command = [
        python_executable,
        str(nvidia_script),
        "--input-folder",
        str(input_folder),
        "--output-folder",
        str(output_folder),
    ]
    if overwrite:
        command.append("--overwrite")

    return TextOnlyConversionPlan(
        model=model,
        snapshot_path=snapshot_path,
        nvidia_script=nvidia_script,
        input_folder=input_folder,
        output_folder=output_folder,
        missing_sidecars=missing_sidecars,
        command=tuple(command),
    )


def run_conversion(plan: TextOnlyConversionPlan, *, dry_run: bool = False) -> int:
    print(f"Model: {plan.model.repo_id}")
    print(f"Snapshot: {plan.snapshot_path}")
    print(f"NVIDIA conversion script: {plan.nvidia_script}")
    print(f"Input checkpoint: {plan.input_folder}")
    print(f"Output checkpoint: {plan.output_folder}")
    print("Command:")
    print("  " + " ".join(plan.command))

    if plan.missing_sidecars:
        print("Text-only sidecar files are missing:")
        for item in plan.missing_sidecars:
            print(f"  - {item}")
        print(
            "Download the model's text-only config/tokenizer files before "
            "converting shards."
        )
        return 2

    if dry_run:
        return 0

    completed = subprocess.run(plan.command, check=False)
    return int(completed.returncode)


def _resolve_input_folder(
    snapshot_path: Path,
    input_folder_name: str | None,
) -> Path:
    if input_folder_name is not None:
        input_folder = snapshot_path / input_folder_name
        _require_complete_input_checkpoint(input_folder)
        return input_folder

    for candidate in DEFAULT_SOURCE_FOLDERS:
        input_folder = snapshot_path / candidate
        if not input_folder.is_dir():
            continue
        check = verify_indexed_checkpoint(input_folder)
        if check.complete:
            return input_folder

    candidates = ", ".join(DEFAULT_SOURCE_FOLDERS)
    raise FileNotFoundError(
        f"No complete source checkpoint found under {snapshot_path}; "
        f"checked {candidates}."
    )


def _require_complete_input_checkpoint(input_folder: Path) -> None:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    check = verify_indexed_checkpoint(input_folder)
    if not check.complete:
        missing = ", ".join(check.missing_shards)
        raise FileNotFoundError(f"Input checkpoint is incomplete: {missing}")


def _missing_textonly_sidecars(
    model: AudexModel,
    *,
    snapshot_path: Path,
    output_folder_name: str,
) -> tuple[str, ...]:
    generated_files = {
        f"{output_folder_name}/model.safetensors.index.json",
    }
    missing = []
    for rel_path in model.text_required_files:
        if rel_path in generated_files:
            continue
        if not (snapshot_path / rel_path).exists():
            missing.append(rel_path)
    return tuple(missing)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Audex text-only safetensors shards from a full local snapshot."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CHOICES),
        default="audex-2b",
        help="Audex model snapshot to convert; defaults to the smaller 2B model.",
    )
    parser.add_argument(
        "--input-folder",
        default=None,
        help=(
            "Source checkpoint folder inside the snapshot. Defaults to the first "
            "complete folder among checkpoint_folder_audiogen and checkpoint_folder_full."
        ),
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Output text-only checkpoint folder inside the snapshot.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing text-only safetensors shards and index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved conversion command without running it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_conversion_plan(
        model_by_choice(args.model),
        input_folder_name=args.input_folder,
        output_folder_name=args.output_folder,
        overwrite=args.overwrite,
    )
    return run_conversion(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
