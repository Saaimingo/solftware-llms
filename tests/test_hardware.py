from __future__ import annotations

import subprocess
from collections.abc import Sequence

from vramopt.hardware import (
    collect_system_info,
    parse_nvidia_smi_csv,
    query_nvidia_gpus,
)


def test_parse_nvidia_smi_csv_preserves_hardware_limits() -> None:
    raw = (
        "NVIDIA GeForce RTX 3060, 12288 MiB, 11003 MiB, 616.56, "
        "1, 3, 16, 16, 170.00 W\n"
    )

    gpu = parse_nvidia_smi_csv(raw)[0]

    assert gpu.name == "NVIDIA GeForce RTX 3060"
    assert gpu.memory_total_mib == 12288
    assert gpu.memory_free_mib == 11003
    assert gpu.driver_version == "616.56"
    assert gpu.pcie_gen_current == 1
    assert gpu.pcie_gen_max == 3
    assert gpu.pcie_width_current == 16
    assert gpu.pcie_width_max == 16
    assert gpu.power_limit_w == 170.0


def test_parse_nvidia_smi_csv_accepts_na_values() -> None:
    raw = "Example GPU, 4096 MiB, 2048 MiB, 1.0, [N/A], [N/A], [N/A], [N/A], [N/A]\n"

    gpu = parse_nvidia_smi_csv(raw)[0]

    assert gpu.pcie_gen_current is None
    assert gpu.power_limit_w is None


def test_query_nvidia_gpus_uses_machine_readable_fields() -> None:
    seen: list[str] = []

    def fake_run(
        args: Sequence[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        seen.extend(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                "NVIDIA GeForce RTX 3060, 12288 MiB, 11003 MiB, 616.56, "
                "1, 3, 16, 16, 170.00 W\n"
            ),
            stderr="",
        )

    gpu = query_nvidia_gpus(runner=fake_run)[0]

    assert gpu.memory_total_mib == 12288
    assert seen[0] == "nvidia-smi"
    assert "--format=csv,noheader" in seen
    assert any(argument.startswith("--query-gpu=") for argument in seen)


def test_collect_system_info_combines_ram_cpu_and_gpu() -> None:
    gpu = parse_nvidia_smi_csv(
        "NVIDIA GeForce RTX 3060, 12288 MiB, 11003 MiB, 616.56, "
        "1, 3, 16, 16, 170.00 W\n"
    )[0]

    info = collect_system_info(
        gpu_provider=lambda: [gpu],
        memory_provider=lambda: (40 * 2**30, 30 * 2**30),
        cpu_provider=lambda: ("AMD Ryzen 7 5700", 16),
    )

    assert info.cpu_name == "AMD Ryzen 7 5700"
    assert info.logical_cpus == 16
    assert info.ram_total_bytes == 40 * 2**30
    assert info.ram_available_bytes == 30 * 2**30
    assert info.gpus == (gpu,)
