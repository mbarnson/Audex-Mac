from __future__ import annotations

import sys
import types

import pytest

from audex_mac import model_select
from audex_mac.model_select import HuggingFaceSnapshotProbe, download_model_snapshot
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
            "local_files_only": False,
        }
    ]


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
