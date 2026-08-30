from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence

from vramopt.monitor import query_gpu_memory_used_mib, run_monitored_command


def test_query_gpu_memory_used_mib_sums_visible_devices() -> None:
    def fake_run(
        args: Sequence[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="1024 MiB\n256 MiB\n", stderr="")

    assert query_gpu_memory_used_mib(runner=fake_run) == 1_280


def test_run_monitored_command_records_peak_gpu_memory() -> None:
    values = iter((100, 300, 700, 500, 200))
    last = 100

    def probe() -> int:
        nonlocal last
        last = next(values, last)
        return last

    def fake_run(
        args: Sequence[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        time.sleep(0.04)
        return subprocess.CompletedProcess(args, 0, stdout="done", stderr="")

    result = run_monitored_command(
        ["program.exe"],
        timeout_seconds=1,
        sample_interval_seconds=0.005,
        gpu_probe=probe,
        runner=fake_run,
    )

    assert result.command.returncode == 0
    assert result.gpu_baseline_mib == 100
    assert result.gpu_peak_mib == 700
    assert result.gpu_peak_delta_mib == 600
