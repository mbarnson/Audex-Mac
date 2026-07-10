"""Terminal editor for typed-or-spoken Audex turns."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import IO, Any, Literal

SHIFT_ENTER_SENTINEL = "f24"
_KITTY_ENABLE = "\x1b[>1u"
_KITTY_DISABLE = "\x1b[<1u"
_MODIFY_OTHER_KEYS_ENABLE = "\x1b[>4;2m"
_MODIFY_OTHER_KEYS_DISABLE = "\x1b[>4;0m"


class InputKind(StrEnum):
    TEXT = "text"
    RECORD = "record"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class TurnInput:
    kind: InputKind
    text: str = ""


def classify_submission(text: str) -> TurnInput:
    stripped = text.strip()
    if not stripped:
        return TurnInput(InputKind.RECORD)
    if stripped.lower() in {"q", "quit", "exit"}:
        return TurnInput(InputKind.QUIT)
    return TurnInput(InputKind.TEXT, text)


def keyboard_protocol(
    *,
    term_program: str | None = None,
    term: str | None = None,
) -> Literal["kitty", "modify-other-keys"] | None:
    program = (term_program or "").lower()
    terminal = (term or "").lower()
    if any(name in program for name in ("ghostty", "kitty", "wezterm")) or any(
        name in terminal for name in ("ghostty", "kitty", "wezterm")
    ):
        return "kitty"
    if "iterm" in program:
        return "modify-other-keys"
    return None


def install_modified_enter_sequences(
    sequences: MutableMapping[str, Any],
    sentinel: Any,
) -> None:
    """Keep modified Enter distinct in prompt_toolkit's VT100 parser."""

    sequences["\x1b[13;2u"] = sentinel  # Kitty keyboard protocol: Shift+Enter.
    sequences["\x1b[27;2;13~"] = sentinel  # xterm modifyOtherKeys.


@contextmanager
def modified_key_reporting(
    protocol: Literal["kitty", "modify-other-keys"] | None,
    output: IO[str] = sys.stdout,
):
    if protocol == "kitty":
        enable, disable = _KITTY_ENABLE, _KITTY_DISABLE
    elif protocol == "modify-other-keys":
        enable, disable = _MODIFY_OTHER_KEYS_ENABLE, _MODIFY_OTHER_KEYS_DISABLE
    else:
        enable = disable = ""

    if enable and output.isatty():
        output.write(enable)
        output.flush()
    try:
        yield
    finally:
        if disable and output.isatty():
            output.write(disable)
            output.flush()


def _message_key_bindings():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    install_modified_enter_sequences(ANSI_SEQUENCES, Keys.F24)
    bindings = KeyBindings()

    @bindings.add("c-m")
    @bindings.add("c-j")
    def submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.F24)
    @bindings.add("escape", "c-m")
    @bindings.add("escape", "c-j")
    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def read_turn_input() -> TurnInput:
    """Read one editable message; an empty submission selects recording."""

    from prompt_toolkit import PromptSession

    protocol = keyboard_protocol(
        term_program=os.environ.get("TERM_PROGRAM"),
        term=os.environ.get("TERM"),
    )
    session: PromptSession[str] = PromptSession(
        multiline=True,
        key_bindings=_message_key_bindings(),
        prompt_continuation=lambda *_args: "… ",
        bottom_toolbar=(
            "Enter: send  •  Shift+Enter/Option+Enter: newline  •  "
            "empty Enter: talk  •  q: quit"
        ),
    )
    try:
        with modified_key_reporting(protocol):
            text = session.prompt("You: ")
    except (EOFError, KeyboardInterrupt):
        return TurnInput(InputKind.QUIT)
    return classify_submission(text)
