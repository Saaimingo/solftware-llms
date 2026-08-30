"""Parse auditable llama.cpp benchmark evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlamaMetrics:
    load_time_ms: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    gpu_layers: int | None = None
    total_layers: int | None = None
    gpu_model_buffer_mib: float | None = None
    cpu_model_buffer_mib: float | None = None
    oom: bool = False
    fit_adjustment_disabled: bool = False


def _float_match(pattern: str, output: str) -> float | None:
    match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
    return float(match.group(1)) if match else None


def _sum_matches(pattern: str, output: str) -> float | None:
    values = [
        float(value)
        for value in re.findall(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
    ]
    return sum(values) if values else None


def parse_llama_metrics(output: str) -> LlamaMetrics:
    """Extract stable placement and speed fields from llama.cpp logs."""

    layers = re.search(
        r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
        output,
        flags=re.IGNORECASE,
    )
    oom_markers = (
        "out of memory",
        "failed to allocate cuda",
        "cuda error 2",
        "cuda_error_out_of_memory",
    )
    folded = output.casefold()
    prompt_tps = _float_match(
        r"prompt eval time[^\n]*?([0-9.]+)\s+tokens per second", output
    )
    if prompt_tps is None:
        prompt_tps = _float_match(r"\bPrompt:\s*([0-9.]+)\s*t/s", output)
    generation_tps = _float_match(
        r"llama_perf_context_print:\s*eval time[^\n]*?"
        r"([0-9.]+)\s+tokens per second",
        output,
    )
    if generation_tps is None:
        generation_tps = _float_match(r"\bGeneration:\s*([0-9.]+)\s*t/s", output)
    return LlamaMetrics(
        load_time_ms=_float_match(r"load time\s*=\s*([0-9.]+)\s*ms", output),
        prompt_tokens_per_second=prompt_tps,
        generation_tokens_per_second=generation_tps,
        gpu_layers=int(layers.group(1)) if layers else None,
        total_layers=int(layers.group(2)) if layers else None,
        gpu_model_buffer_mib=_sum_matches(
            r"CUDA\d+\s+model buffer size\s*=\s*([0-9.]+)\s*MiB", output
        ),
        cpu_model_buffer_mib=_sum_matches(
            r"CPU(?:_Mapped)?\s+model buffer size\s*=\s*([0-9.]+)\s*MiB", output
        ),
        oom=any(marker in folded for marker in oom_markers),
        fit_adjustment_disabled=(
            "fit will not adjust it" in folded
            or "fit will not adjust them" in folded
        ),
    )
