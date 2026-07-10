from __future__ import annotations

from pathlib import Path

import pytest

from audex_mac.personas import load_persona, parse_persona_markdown

pytestmark = pytest.mark.fast


def test_parse_persona_markdown_front_matter_and_prompt() -> None:
    metadata, prompt = parse_persona_markdown(
        "---\nname: marge\nvoice: af_heart\n---\n\nYou are concise.\n"
    )

    assert metadata == {"name": "marge", "voice": "af_heart"}
    assert prompt == "You are concise."


def test_load_persona_resolves_name_from_directory(tmp_path: Path) -> None:
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "assistant.md").write_text(
        "---\nname: assistant\n---\n\nSpeak briefly.",
        encoding="utf-8",
    )

    persona = load_persona("assistant", personas_dir=personas_dir)

    assert persona.persona_id == "assistant"
    assert persona.prompt == "Speak briefly."
    assert "helpful and harmless assistant" in persona.system_prompt
    assert "Speak briefly." in persona.system_prompt
