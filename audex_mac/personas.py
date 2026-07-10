"""Markdown persona loading for the Audex speech CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio_contract import DEFAULT_SYSTEM_PROMPT

DEFAULT_PERSONAS_DIR = Path(__file__).resolve().parents[1] / "personas"
DEFAULT_PERSONA_NAME = "assistant"


@dataclass(frozen=True, slots=True)
class Persona:
    persona_id: str
    path: Path
    metadata: dict[str, str]
    prompt: str

    @property
    def system_prompt(self) -> str:
        return f"{DEFAULT_SYSTEM_PROMPT}\n\n{self.prompt}".strip()


def load_persona(
    persona: str | Path = DEFAULT_PERSONA_NAME,
    *,
    personas_dir: Path = DEFAULT_PERSONAS_DIR,
) -> Persona:
    path = resolve_persona_path(persona, personas_dir=personas_dir)
    metadata, prompt = parse_persona_markdown(path.read_text(encoding="utf-8"))
    persona_id = metadata.get("name") or path.stem
    if not prompt:
        raise ValueError(f"Persona file has no prompt body: {path}")
    return Persona(
        persona_id=persona_id,
        path=path,
        metadata=metadata,
        prompt=prompt,
    )


def resolve_persona_path(
    persona: str | Path,
    *,
    personas_dir: Path = DEFAULT_PERSONAS_DIR,
) -> Path:
    candidate = Path(persona)
    if candidate.suffix == ".md" or candidate.is_absolute() or "/" in str(persona):
        path = candidate
    else:
        path = personas_dir / f"{candidate.name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Persona not found: {path}")
    return path


def parse_persona_markdown(markdown: str) -> tuple[dict[str, str], str]:
    lines = markdown.splitlines()
    metadata: dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = index + 1
                break
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")
        else:
            body_start = 0
            metadata = {}
    prompt = "\n".join(lines[body_start:]).strip()
    return metadata, prompt
