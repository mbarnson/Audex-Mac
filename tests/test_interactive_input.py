from __future__ import annotations

from io import StringIO

import pytest

from audex_mac.interactive_input import (
    SHIFT_ENTER_SENTINEL,
    InputKind,
    TurnInput,
    classify_submission,
    install_modified_enter_sequences,
    keyboard_protocol,
    modified_key_reporting,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("text", ["", " ", "\n\n"])
def test_empty_submission_selects_push_to_talk(text: str) -> None:
    assert classify_submission(text) == TurnInput(InputKind.RECORD)


@pytest.mark.parametrize("text", ["q", "Q", " quit ", "EXIT"])
def test_quit_submission_exits(text: str) -> None:
    assert classify_submission(text) == TurnInput(InputKind.QUIT)


def test_multiline_submission_preserves_internal_newlines() -> None:
    assert classify_submission("First line\n\nSecond line") == TurnInput(
        InputKind.TEXT,
        "First line\n\nSecond line",
    )


def test_modified_enter_sequences_keep_shift_enter_distinct() -> None:
    sequences: dict[str, object] = {}

    install_modified_enter_sequences(sequences, SHIFT_ENTER_SENTINEL)

    assert sequences["\x1b[13;2u"] == SHIFT_ENTER_SENTINEL
    assert sequences["\x1b[27;2;13~"] == SHIFT_ENTER_SENTINEL


@pytest.mark.parametrize(
    ("term_program", "term", "expected"),
    [
        ("ghostty", "xterm-ghostty", "kitty"),
        ("kitty", "xterm-kitty", "kitty"),
        ("WezTerm", "xterm-256color", "kitty"),
        ("iTerm.app", "xterm-256color", "modify-other-keys"),
        ("Apple_Terminal", "xterm-256color", None),
    ],
)
def test_keyboard_protocol_matches_terminal_capabilities(
    term_program: str,
    term: str,
    expected: str | None,
) -> None:
    assert keyboard_protocol(term_program=term_program, term=term) == expected


def test_keyboard_reporting_restores_kitty_protocol() -> None:
    class TerminalOutput(StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()

    with modified_key_reporting("kitty", output):
        assert output.getvalue() == "\x1b[>1u"

    assert output.getvalue() == "\x1b[>1u\x1b[<1u"
