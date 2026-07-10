from __future__ import annotations

import json
from pathlib import Path

import pytest

from audex_mac.models import AUDEX_2B_REPO, AUDEX_30B_REPO
from audex_mac.textonly_conversion import build_conversion_plan, model_by_choice

pytestmark = pytest.mark.fast


def make_repo_snapshot(
    cache_root: Path,
    repo_id: str,
    *,
    script: bool = True,
    audiogen_complete: bool = True,
    full_complete: bool = True,
) -> Path:
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "rev"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("rev", encoding="utf-8")

    if script:
        script_path = (
            snapshot / "model_conversion_scripts" / "convert_full_HF_to_textonly_HF.py"
        )
        script_path.parent.mkdir(parents=True)
        script_path.write_text("# fake nvidia conversion script\n", encoding="utf-8")

    for folder_name, complete in (
        ("checkpoint_folder_audiogen", audiogen_complete),
        ("checkpoint_folder_full", full_complete),
    ):
        folder = snapshot / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}}),
            encoding="utf-8",
        )
        if complete:
            (folder / "model.safetensors").write_bytes(b"fake")

    textonly = snapshot / "checkpoint_folder_textonly"
    textonly.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "generation_config.json",
        "modeling_nemotron_dense.py",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (textonly / filename).write_text("{}", encoding="utf-8")
    scripts = snapshot / "inference_scripts_vllm" / "textonly_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run_text_vllm_example.py").write_text("# example\n", encoding="utf-8")
    return snapshot


def test_conversion_plan_prefers_complete_audiogen_checkpoint(tmp_path: Path) -> None:
    snapshot = make_repo_snapshot(tmp_path, AUDEX_2B_REPO)

    plan = build_conversion_plan(
        model_by_choice("audex-2b"),
        cache_root=tmp_path,
        python_executable="/python",
    )

    assert plan.snapshot_path == snapshot
    assert plan.input_folder == snapshot / "checkpoint_folder_audiogen"
    assert plan.output_folder == snapshot / "checkpoint_folder_textonly"
    assert plan.ready is True
    assert plan.command[:2] == ("/python", str(plan.nvidia_script))


def test_conversion_plan_falls_back_to_full_checkpoint(tmp_path: Path) -> None:
    snapshot = make_repo_snapshot(
        tmp_path,
        AUDEX_30B_REPO,
        audiogen_complete=False,
        full_complete=True,
    )
    textonly = snapshot / "checkpoint_folder_textonly"
    (textonly / "modeling_nemotron_h.py").write_text("{}", encoding="utf-8")

    plan = build_conversion_plan(
        model_by_choice("audex-30b-a3b"),
        cache_root=tmp_path,
        python_executable="/python",
    )

    assert plan.input_folder == snapshot / "checkpoint_folder_full"
    assert plan.ready is True


def test_conversion_plan_reports_missing_textonly_sidecars(tmp_path: Path) -> None:
    make_repo_snapshot(tmp_path, AUDEX_2B_REPO)
    missing = (
        tmp_path
        / "models--nvidia--Nemotron-Labs-Audex-2B"
        / "snapshots"
        / "rev"
        / "checkpoint_folder_textonly"
        / "tokenizer.json"
    )
    missing.unlink()

    plan = build_conversion_plan(
        model_by_choice("audex-2b"),
        cache_root=tmp_path,
        python_executable="/python",
    )

    assert plan.ready is False
    assert plan.missing_sidecars == ("checkpoint_folder_textonly/tokenizer.json",)
