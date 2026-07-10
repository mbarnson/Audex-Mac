"""Audex text-chat template and assistant-history semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHAT_STOP_MARKERS = ("<|im_end|>", "<|end_of_text|>", "<|eot_id|>")
AUDEX_CHAT_TEMPLATE_RELATIVE_PATH = "checkpoint_folder_full/chat_template.jinja"
THINKING_GENERATION_PREFIX = "<think>\n"
NON_THINKING_GENERATION_PREFIX = "<think></think>"


@dataclass(frozen=True, slots=True)
class TextAssistantTurn:
    """One generated turn in display form and template-valid history form."""

    raw_content: str
    answer: str


def find_audex_chat_template(model_path: str | Path) -> Path | None:
    """Find the official template shipped beside an Audex checkpoint."""

    checkpoint = Path(model_path)
    candidates = (
        checkpoint / "chat_template.jinja",
        checkpoint.parent / "checkpoint_folder_full" / "chat_template.jinja",
        checkpoint.parent / "checkpoint_folder_audiogen" / "chat_template.jinja",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def ensure_audex_chat_template(tokenizer: Any, model_path: str | Path) -> None:
    """Install the model's shipped template when its tokenizer omitted it."""

    if getattr(tokenizer, "chat_template", None):
        return

    template_path = find_audex_chat_template(model_path)
    if template_path is None:
        raise RuntimeError(
            "Audex text generation requires the model's chat_template.jinja; "
            f"none was found beside {model_path}."
        )

    template = template_path.read_text(encoding="utf-8")
    tokenizer.chat_template = template
    if hasattr(tokenizer, "_chat_template"):
        tokenizer._tokenizer.chat_template = template
    if hasattr(tokenizer, "has_chat_template"):
        tokenizer.has_chat_template = True


def render_text_chat_prompt(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    model_path: str | Path,
    thinking_enabled: bool,
) -> str:
    """Render a benchmark turn with the selected Audex model's own template."""

    ensure_audex_chat_template(tokenizer, model_path)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_enabled,
    )


def complete_text_assistant_turn(
    generated: str,
    *,
    thinking_enabled: bool,
) -> TextAssistantTurn:
    """Restore the prompt-consumed prefix before saving assistant history."""

    completion = clean_text_completion(generated)
    prefix = (
        THINKING_GENERATION_PREFIX
        if thinking_enabled
        else NON_THINKING_GENERATION_PREFIX
    )
    raw_content = prefix + completion
    if "</think>" in raw_content:
        answer = raw_content.rsplit("</think>", 1)[-1].strip()
    elif "<think>" in raw_content:
        answer = ""
    else:
        answer = raw_content.strip()
    return TextAssistantTurn(raw_content=raw_content, answer=answer)


def clean_text_completion(text: str) -> str:
    """Trim the first model stop marker while preserving reasoning syntax."""

    cleaned = text
    for marker in CHAT_STOP_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index]
    return cleaned.strip()
