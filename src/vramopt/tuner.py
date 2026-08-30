"""Execute and rank hardware-specific tuning candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vramopt.benchmark import LlamaMetrics, parse_llama_metrics
from vramopt.block_predictor import build_hubs, estimate_prefetch, predict_blocks
from vramopt.monitor import MonitoredCommandResult, run_monitored_command
from vramopt.planner import TuningCandidate


@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    candidate_id: str
    strategy: str
    success: bool
    returncode: int
    duration_seconds: float
    gpu_baseline_mib: int | None
    gpu_peak_mib: int | None
    gpu_peak_delta_mib: int | None
    metrics: LlamaMetrics
    args: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["args"] = list(self.args)
        return data


def run_candidate(
    candidate: TuningCandidate,
    *,
    executable: Path,
    model: Path,
    predict: int,
    timeout_seconds: float,
    log_path: Path,
    monitor: Callable[..., MonitoredCommandResult] = run_monitored_command,
) -> CandidateRunResult:
    """Run one candidate and write its raw backend output for audit."""

    args = candidate.to_llama_args(executable, model, predict=predict)
    monitored = monitor(args, timeout_seconds=timeout_seconds)
    command = monitored.command
    output = command.stdout
    if command.stderr:
        output = f"{output}\n{command.stderr}" if output else command.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    metrics = parse_llama_metrics(output)
    speed = metrics.generation_tokens_per_second
    success = command.returncode == 0 and not metrics.oom and speed is not None and speed > 0
    return CandidateRunResult(
        candidate_id=candidate.id,
        strategy=candidate.strategy,
        success=success,
        returncode=command.returncode,
        duration_seconds=command.duration_seconds,
        gpu_baseline_mib=monitored.gpu_baseline_mib,
        gpu_peak_mib=monitored.gpu_peak_mib,
        gpu_peak_delta_mib=monitored.gpu_peak_delta_mib,
        metrics=metrics,
        args=command.args,
    )


def choose_best(
    results: Iterable[CandidateRunResult], *, max_gpu_peak_mib: int | None = None
) -> CandidateRunResult:
    """Choose the fastest successful result within an optional VRAM limit."""

    valid = [
        result
        for result in results
        if result.success
        and result.metrics.generation_tokens_per_second is not None
        and (
            max_gpu_peak_mib is None
            or (
                result.gpu_peak_mib is not None
                and result.gpu_peak_mib <= max_gpu_peak_mib
            )
        )
    ]
    if not valid:
        raise ValueError("no successful tuning candidates")
    return max(
        valid,
        key=lambda result: (
            result.metrics.generation_tokens_per_second or 0.0,
            result.metrics.prompt_tokens_per_second or 0.0,
        ),
    )


def choose_best_with_blocks(
    results: Iterable[CandidateRunResult],
    *,
    trace_analysis: dict[str, Any] | None,
    max_gpu_peak_mib: int | None = None,
) -> tuple[CandidateRunResult, dict[str, Any] | None]:
    """Escolhe best e anexa plano de prefetch em blocos se trace existir."""
    best = choose_best(results, max_gpu_peak_mib=max_gpu_peak_mib)
    plan = None
    if trace_analysis is not None:
        hubs = build_hubs(trace_analysis, top_k=8)
        blocks = predict_blocks(trace_analysis, layer=3)
        if blocks:
            prefetch = estimate_prefetch(blocks, hubs, hit_rate=0.85)
            plan = {
                "hubs": hubs[:6],
                "blocks_layer3": [{"layer": b.layer, "experts": b.experts, "count": b.count, "ratio": b.ratio} for b in blocks[:2]],
                "with_blocks": {
                    "layer": prefetch.layer,
                    "hub_experts": prefetch.hub_experts,
                    "estimated_miss_gb": prefetch.estimated_miss_gb,
                    "estimated_tps_ceiling": prefetch.estimated_tps_ceiling,
                },
                "note": "plano esteira: hubs fixos VRAM + double-buffer pinned",
            }
    return best, plan
