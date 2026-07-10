"""Small coercion helpers for JSON-like model configuration values."""

from __future__ import annotations

from typing import Any


def optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
