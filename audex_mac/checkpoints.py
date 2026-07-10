"""Local Hugging Face snapshot and checkpoint verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import AudexModel

HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True, slots=True)
class IndexedCheckpointCheck:
    path: Path
    index_path: Path
    shard_names: tuple[str, ...]
    missing_shards: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_shards


@dataclass(frozen=True, slots=True)
class SnapshotCheck:
    model: AudexModel
    snapshot_path: Path | None
    required_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    checkpoint_checks: tuple[IndexedCheckpointCheck, ...]

    @property
    def complete(self) -> bool:
        return (
            self.snapshot_path is not None
            and not self.missing_files
            and all(check.complete for check in self.checkpoint_checks)
        )

    @property
    def missing_summary(self) -> tuple[str, ...]:
        missing = list(self.missing_files)
        for check in self.checkpoint_checks:
            missing.extend(f"{check.path.name}/{name}" for name in check.missing_shards)
        return tuple(missing)


def repo_cache_dir(repo_id: str, cache_root: Path = HF_CACHE_ROOT) -> Path:
    return cache_root / f"models--{repo_id.replace('/', '--')}"


def local_snapshot_path(repo_id: str, cache_root: Path = HF_CACHE_ROOT) -> Path | None:
    """Return a usable local snapshot, preferring refs/main when materialized."""

    candidates = _local_snapshot_candidates(repo_id, cache_root=cache_root)
    return candidates[0] if candidates else None


def _local_snapshot_candidates(
    repo_id: str,
    *,
    cache_root: Path,
) -> tuple[Path, ...]:
    """Return materialized snapshots with refs/main first, then newest first."""

    root = repo_cache_dir(repo_id, cache_root)
    preferred: Path | None = None
    ref = root / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision:
            snapshot = root / "snapshots" / revision
            if snapshot.is_dir():
                preferred = snapshot

    snapshots_root = root / "snapshots"
    if not snapshots_root.is_dir():
        return (preferred,) if preferred is not None else ()
    snapshots = sorted(
        (path for path in snapshots_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if preferred is not None:
        snapshots = [preferred, *(path for path in snapshots if path != preferred)]
    return tuple(snapshots)


def verify_indexed_checkpoint(path: Path) -> IndexedCheckpointCheck:
    """Verify every safetensors shard referenced by the checkpoint index."""

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return IndexedCheckpointCheck(
            path=path,
            index_path=index_path,
            shard_names=(),
            missing_shards=("model.safetensors.index.json",),
        )

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = tuple(sorted(set(raw.get("weight_map", {}).values())))
    missing = tuple(name for name in shard_names if not (path / name).exists())
    return IndexedCheckpointCheck(
        path=path,
        index_path=index_path,
        shard_names=shard_names,
        missing_shards=missing,
    )


def verify_snapshot(
    model: AudexModel,
    required_files: tuple[str, ...],
    checkpoint_dirs: tuple[str, ...],
    cache_root: Path = HF_CACHE_ROOT,
) -> SnapshotCheck:
    """Verify exact files and indexed shards for a local model snapshot."""

    candidates = _local_snapshot_candidates(model.repo_id, cache_root=cache_root)
    if not candidates:
        return SnapshotCheck(
            model=model,
            snapshot_path=None,
            required_files=required_files,
            missing_files=required_files,
            checkpoint_checks=(),
        )

    checks = tuple(
        _verify_snapshot_path(
            model,
            snapshot,
            required_files=required_files,
            checkpoint_dirs=checkpoint_dirs,
        )
        for snapshot in candidates
    )
    return next((check for check in checks if check.complete), checks[0])


def _verify_snapshot_path(
    model: AudexModel,
    snapshot: Path,
    *,
    required_files: tuple[str, ...],
    checkpoint_dirs: tuple[str, ...],
) -> SnapshotCheck:
    missing_files = tuple(
        path for path in required_files if not (snapshot / path).exists()
    )
    checkpoint_checks = tuple(
        verify_indexed_checkpoint(snapshot / dirname) for dirname in checkpoint_dirs
    )
    return SnapshotCheck(
        model=model,
        snapshot_path=snapshot,
        required_files=required_files,
        missing_files=missing_files,
        checkpoint_checks=checkpoint_checks,
    )
