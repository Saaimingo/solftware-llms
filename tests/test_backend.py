from __future__ import annotations

import subprocess
from collections.abc import Sequence

from vramopt.backend import run_command


def test_run_command_preserves_arguments_and_disables_shell() -> None:
    observed: dict[str, object] = {}

    def fake_run(
        args: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["args"] = list(args)
        observed.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    result = run_command(
        ["program.exe", "argument with spaces", "; not a command"],
        timeout_seconds=12,
        runner=fake_run,
    )

    assert result.returncode == 0
    assert result.stdout == "ok"
    assert observed["args"] == ["program.exe", "argument with spaces", "; not a command"]
    assert observed["shell"] is False
    assert observed["timeout"] == 12
    assert observed["capture_output"] is True
