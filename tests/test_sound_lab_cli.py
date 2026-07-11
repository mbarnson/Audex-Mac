from __future__ import annotations

from io import StringIO

import pytest

from audex_mac.sound_lab.cli import run_sound_lab_repl
from audex_mac.sound_lab.session import SoundLabTurn


class FakeSession:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def handle(self, text: str) -> SoundLabTurn:
        self.requests.append(text)
        return SoundLabTurn(
            message="I designed three distinct impacts.",
            job_id="job-1",
            ready_count=2,
            failed_count=1,
        )


@pytest.mark.fast
def test_sound_lab_repl_accepts_typed_request_and_reports_blind_job() -> None:
    session = FakeSession()
    lines = iter(("Make three impacts.", "q"))
    output = StringIO()

    result = run_sound_lab_repl(
        session,
        read_line=lambda _prompt: next(lines),
        output=output,
    )

    assert result == 0
    assert session.requests == ["Make three impacts."]
    rendered = output.getvalue()
    assert "I designed three distinct impacts." in rendered
    assert "job-1: 2 ready, 1 failed" in rendered


@pytest.mark.fast
def test_sound_lab_command_has_an_isolated_cli_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from audex_mac.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["sound-lab", "--help"])

    assert exit_info.value.code == 0
    assert "Audex Sound Lab" in capsys.readouterr().out
