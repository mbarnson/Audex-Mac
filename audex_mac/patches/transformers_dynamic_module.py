"""Transformers dynamic-module compatibility patches for local HF snapshots."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

PATCH_SENTINEL = "_audex_mac_local_dynamic_module_patch"
ORIGINAL_CACHED_FILE = "_audex_mac_original_cached_file"
ORIGINAL_LOCAL_HASH = "_audex_mac_original_local_source_hash"
LAST_ERROR: str | None = None


def patch_transformers_local_dynamic_modules() -> bool:
    """Keep local snapshot remote-code paths logical instead of blob-resolved."""

    global LAST_ERROR
    LAST_ERROR = None
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False

    if getattr(dynamic_module_utils, PATCH_SENTINEL, False):
        return True

    original_cached_file = getattr(dynamic_module_utils, "cached_file", None)
    original_local_hash = getattr(
        dynamic_module_utils,
        "_compute_local_source_files_hash",
        None,
    )
    get_relative_import_files = getattr(
        dynamic_module_utils,
        "get_relative_import_files",
        None,
    )
    if not callable(original_cached_file):
        LAST_ERROR = "transformers.dynamic_module_utils.cached_file unavailable"
        return False
    if not callable(original_local_hash):
        LAST_ERROR = (
            "transformers.dynamic_module_utils._compute_local_source_files_hash "
            "unavailable"
        )
        return False
    if not callable(get_relative_import_files):
        LAST_ERROR = (
            "transformers.dynamic_module_utils.get_relative_import_files " "unavailable"
        )
        return False

    def cached_file_preserving_local_snapshot(
        pretrained_model_name_or_path: str | os.PathLike[str],
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        model_path = Path(pretrained_model_name_or_path)
        local_file = model_path / filename
        if model_path.is_dir() and local_file.is_file():
            return str(local_file)
        return original_cached_file(
            pretrained_model_name_or_path,
            filename,
            *args,
            **kwargs,
        )

    def compute_local_source_files_hash_preserving_snapshot(
        pretrained_model_name_or_path: str | os.PathLike[str],
        resolved_module_file: str | os.PathLike[str],
    ) -> str:
        model_path = Path(pretrained_model_name_or_path).absolute()
        module_file = _absolute_without_resolving(resolved_module_file)
        if not _is_relative_to(module_file, model_path):
            return original_local_hash(
                pretrained_model_name_or_path,
                resolved_module_file,
            )

        files_to_hash: list[tuple[str, Path]] = [
            (_relative_source_path(module_file, model_path), module_file),
        ]
        for source_file in get_relative_import_files(module_file):
            source_path = _absolute_without_resolving(source_file)
            files_to_hash.append(
                (_relative_source_path(source_path, model_path), source_path)
            )

        source_files_hash = hashlib.sha256()
        for relative_path, file_path in sorted(files_to_hash, key=lambda item: item[0]):
            source_files_hash.update(relative_path.encode("utf-8"))
            source_files_hash.update(file_path.read_bytes())
        return source_files_hash.hexdigest()[:16]

    setattr(dynamic_module_utils, ORIGINAL_CACHED_FILE, original_cached_file)
    setattr(dynamic_module_utils, ORIGINAL_LOCAL_HASH, original_local_hash)
    dynamic_module_utils.cached_file = cached_file_preserving_local_snapshot
    dynamic_module_utils._compute_local_source_files_hash = (
        compute_local_source_files_hash_preserving_snapshot
    )
    setattr(dynamic_module_utils, PATCH_SENTINEL, True)
    return True


def _absolute_without_resolving(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().absolute()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _relative_source_path(path: Path, model_path: Path) -> str:
    try:
        return path.relative_to(model_path).as_posix()
    except ValueError:
        return path.as_posix()
