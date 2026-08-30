"""Runtime GPU-memory telemetry."""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vramopt.backend import CommandResult, run_command


@dataclass(frozen=True, slots=True)
class MonitoredCommandResult:
    command: CommandResult
    gpu_baseline_mib: int | None
    gpu_peak_mib: int | None

    @property
    def gpu_peak_delta_mib(self) -> int | None:
        if self.gpu_baseline_mib is None or self.gpu_peak_mib is None:
            return None
        return max(0, self.gpu_peak_mib - self.gpu_baseline_mib)


def query_gpu_memory_used_mib(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Return total memory used across visible NVIDIA devices."""

    result = runner(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader",
        ],
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    values = [int(value) for value in re.findall(r"(\d+)\s*MiB", result.stdout)]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return sum(values)


def run_monitored_command(
    args: Sequence[str],
    *,
    timeout_seconds: float,
    sample_interval_seconds: float = 0.1,
    gpu_probe: Callable[[], int | None] = query_gpu_memory_used_mib,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> MonitoredCommandResult:
    """Run a command while sampling aggregate GPU memory in parallel."""

    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    samples: list[int] = []

    def sample_once() -> None:
        try:
            value = gpu_probe()
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            return
        if value is not None:
            samples.append(value)

    sample_once()
    baseline = samples[0] if samples else None
    stop = threading.Event()

    def sampler() -> None:
        while not stop.wait(sample_interval_seconds):
            sample_once()

    thread = threading.Thread(target=sampler, name="vramopt-gpu-monitor", daemon=True)
    thread.start()
    try:
        command = run_command(args, timeout_seconds=timeout_seconds, runner=runner)
    finally:
        stop.set()
        thread.join(timeout=max(1.0, sample_interval_seconds * 2))
        sample_once()
    return MonitoredCommandResult(
        command=command,
        gpu_baseline_mib=baseline,
        gpu_peak_mib=max(samples) if samples else None,
    )
