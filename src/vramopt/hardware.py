"""Hardware discovery for constrained-VRAM planning."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import psutil

NumberT = TypeVar("NumberT", int, float)


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    memory_total_mib: int
    memory_free_mib: int
    driver_version: str
    pcie_gen_current: int | None
    pcie_gen_max: int | None
    pcie_width_current: int | None
    pcie_width_max: int | None
    power_limit_w: float | None


@dataclass(frozen=True, slots=True)
class SystemInfo:
    os_name: str
    os_version: str
    cpu_name: str
    logical_cpus: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: tuple[GpuInfo, ...]


def _optional_number(value: str, cast: Callable[[str], NumberT]) -> NumberT | None:
    cleaned = value.strip().replace(" MiB", "").replace(" W", "")
    if cleaned.lower() in {"n/a", "[n/a]", "not supported", ""}:
        return None
    return cast(cleaned)


def parse_nvidia_smi_csv(output: str) -> list[GpuInfo]:
    """Parse the exact no-header CSV format requested by hardware discovery."""

    gpus: list[GpuInfo] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 9:
            raise ValueError(f"expected 9 NVIDIA fields, received {len(row)}")
        total = _optional_number(row[1], int)
        free = _optional_number(row[2], int)
        if total is None or free is None:
            raise ValueError("NVIDIA memory totals are unavailable")
        gpus.append(
            GpuInfo(
                name=row[0].strip(),
                memory_total_mib=total,
                memory_free_mib=free,
                driver_version=row[3].strip(),
                pcie_gen_current=_as_optional_int(row[4]),
                pcie_gen_max=_as_optional_int(row[5]),
                pcie_width_current=_as_optional_int(row[6]),
                pcie_width_max=_as_optional_int(row[7]),
                power_limit_w=_as_optional_float(row[8]),
            )
        )
    return gpus


def query_nvidia_gpus(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[GpuInfo]:
    """Query NVIDIA GPUs without parsing localized human-readable output."""

    fields = (
        "name,memory.total,memory.free,driver_version,"
        "pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max,power.limit"
    )
    result = runner(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return parse_nvidia_smi_csv(result.stdout)


def _as_optional_int(value: str) -> int | None:
    parsed = _optional_number(value, int)
    return parsed if isinstance(parsed, int) else None


def _as_optional_float(value: str) -> float | None:
    parsed = _optional_number(value, float)
    return parsed if isinstance(parsed, float) else None


def _memory_snapshot() -> tuple[int, int]:
    memory = psutil.virtual_memory()
    return int(memory.total), int(memory.available)


def _cpu_snapshot() -> tuple[str, int]:
    name = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return name.strip(), os.cpu_count() or 1


def collect_system_info(
    *,
    gpu_provider: Callable[[], list[GpuInfo]] = query_nvidia_gpus,
    memory_provider: Callable[[], tuple[int, int]] = _memory_snapshot,
    cpu_provider: Callable[[], tuple[str, int]] = _cpu_snapshot,
) -> SystemInfo:
    """Collect the hardware facts used to key tuning profiles."""

    ram_total, ram_available = memory_provider()
    cpu_name, logical_cpus = cpu_provider()
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_name=cpu_name,
        logical_cpus=logical_cpus,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        gpus=tuple(gpu_provider()),
    )
