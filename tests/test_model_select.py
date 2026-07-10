from __future__ import annotations

import sys
import types
from fnmatch import fnmatchcase

import pytest

from audex_mac import model_select
from audex_mac.model_select import (
    MACOS_CASE_COLLISION_IGNORE_PATTERNS,
    HuggingFaceSnapshotProbe,
    download_model_snapshot,
)
from audex_mac.models import DEFAULT_MODEL

pytestmark = pytest.mark.fast


def test_download_model_snapshot_uses_readiness_allow_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/snapshot"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    download_model_snapshot(DEFAULT_MODEL, readiness="speech")

    assert calls == [
        {
            "repo_id": DEFAULT_MODEL.repo_id,
            "allow_patterns": list(DEFAULT_MODEL.required_patterns),
            "ignore_patterns": list(MACOS_CASE_COLLISION_IGNORE_PATTERNS),
            "local_files_only": False,
        }
    ]


def test_text_download_includes_the_full_checkpoint_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )

    download_model_snapshot(DEFAULT_MODEL, readiness="text")

    assert "checkpoint_folder_full/chat_template.jinja" in calls[0]["allow_patterns"]


@pytest.mark.parametrize("readiness", ["speech", "text"])
def test_model_download_excludes_case_colliding_license_paths(
    readiness: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )

    download_model_snapshot(DEFAULT_MODEL, readiness=readiness)

    patterns = calls[0]["allow_patterns"]
    assert calls[0]["ignore_patterns"] == ["license/*"]
    assert not any(fnmatchcase("LICENSE", pattern) for pattern in patterns)
    assert not any(
        fnmatchcase("license/NVIDIA-OneWay-Noncommercial-License.docx", pattern)
        for pattern in patterns
    )


def test_snapshot_probe_accepts_directly_verified_cache_before_hf_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_verify_snapshot(*_args, **_kwargs):
        calls.append("verify")
        return types.SimpleNamespace(complete=True)

    monkeypatch.setattr(model_select, "verify_snapshot", fake_verify_snapshot)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    assert HuggingFaceSnapshotProbe().is_cached(DEFAULT_MODEL, readiness="speech")
    assert calls == ["verify"]
