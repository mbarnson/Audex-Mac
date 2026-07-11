"""Strict parsing for the small Audex Sound Lab tool surface."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_RE = re.compile(
    r"(?P<prefix>.*?)"
    r"<tool_call>\s*"
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>"
    r"(?P<body>.*?)"
    r"</function>\s*"
    r"</tool_call>\s*\Z",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"\s*<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>"
    r"(?P<value>.*?)"
    r"</parameter>",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ALLOWED_RENDER_PARAMETERS = {
    "brief",
    "count",
    "constraints",
    "parent_asset_ids",
}


@dataclass(frozen=True, slots=True)
class RenderSoundsCall:
    """Validated arguments for one asynchronous sound-render request."""

    brief: str
    count: int
    constraints: dict[str, Any]
    parent_asset_ids: tuple[str, ...]
    preamble: str = ""


def parse_sound_lab_tool_call(raw: str) -> RenderSoundsCall:
    """Parse one Nemotron XML tool call and reject everything outside the contract."""

    match = _TOOL_CALL_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError("Sound Lab expected one complete tool call with no suffix")
    if match.group("name") != "render_sounds":
        raise ValueError(f"Sound Lab does not allow tool: {match.group('name')}")

    values: dict[str, str] = {}
    body = match.group("body")
    position = 0
    for parameter in _PARAMETER_RE.finditer(body):
        if body[position : parameter.start()].strip():
            raise ValueError("Sound Lab tool call contains malformed parameter XML")
        position = parameter.end()
        name = parameter.group("name")
        if name not in _ALLOWED_RENDER_PARAMETERS:
            raise ValueError(f"render_sounds does not accept parameter: {name}")
        if name in values:
            raise ValueError(f"render_sounds parameter is duplicated: {name}")
        values[name] = parameter.group("value").strip()
    if body[position:].strip():
        raise ValueError("Sound Lab tool call contains malformed parameter XML")

    brief = values.get("brief", "").strip()
    if not brief:
        raise ValueError("render_sounds brief must not be empty")
    try:
        count = int(values.get("count", ""))
    except ValueError as exc:
        raise ValueError("render_sounds count must be an integer") from exc
    if not 1 <= count <= 5:
        raise ValueError("render_sounds count must be between 1 and 5")

    constraints = _json_object(values.get("constraints", "{}"), "constraints")
    parent_asset_ids = _json_string_list(
        values.get("parent_asset_ids", "[]"),
        "parent_asset_ids",
    )
    preamble = _THINK_RE.sub("", match.group("prefix")).strip()
    return RenderSoundsCall(
        brief=brief,
        count=count,
        constraints=constraints,
        parent_asset_ids=parent_asset_ids,
        preamble=preamble,
    )


def _json_object(raw: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"render_sounds {name} must be valid JSON") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"render_sounds {name} must be a JSON object or null")
    return parsed


def _json_string_list(raw: str, name: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"render_sounds {name} must be valid JSON") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) and item.strip() for item in parsed
    ):
        raise ValueError(f"render_sounds {name} must be a JSON string array")
    return tuple(item.strip() for item in parsed)
