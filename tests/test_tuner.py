from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vramopt.backend import CommandResult
from vramopt.benchmark import LlamaMetrics
from vramopt.monitor import MonitoredCommandResult
from vramopt.planner import TuningCandidate, TuningRequest
from vramopt.tuner import CandidateRunResult, choose_best, run_candidate


def _candidate(identifier: str = "dense-layer-q8_0") -> TuningCandidate:
    return TuningCandidate(
        id=identifier,
        strategy="layer",
        request=TuningRequest(context_size=4_096, fit_margin_mib=1_024, threads=8),
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )


def _result(identifier: str, speed: float, success: bool) -> CandidateRunResult:
    return CandidateRunResult(
        candidate_id=identifier,
        strategy="layer",
        success=success,
        returncode=0 if success else 1,
        duration_seconds=1.0,
        gpu_baseline_mib=100,
        gpu_peak_mib=10_000,
        gpu_peak_delta_mib=9_900,
        metrics=LlamaMetrics(generation_tokens_per_second=speed),
        args=(),
    )


def test_run_candidate_combines_runtime_and_backend_metrics(tmp_path: Path) -> None:
    output = "[ Prompt: 12.2 t/s | Generation: 3.1 t/s ]"

    def fake_monitor(*_: object, **__: object) -> MonitoredCommandResult:
        return MonitoredCommandResult(
            command=CommandResult(
                args=("llama-cli.exe",),
                returncode=0,
                stdout=output,
                stderr="",
                duration_seconds=4.2,
            ),
            gpu_baseline_mib=500,
            gpu_peak_mib=11_400,
        )

    log_path = tmp_path / "candidate.log"
    result = run_candidate(
        _candidate(),
        executable=Path("llama-cli.exe"),
        model=Path("model.gguf"),
        predict=32,
        timeout_seconds=60,
        log_path=log_path,
        monitor=fake_monitor,
    )

    assert result.success is True
    assert result.metrics.generation_tokens_per_second == 3.1
    assert result.gpu_peak_mib == 11_400
    assert result.gpu_peak_delta_mib == 10_900
    assert log_path.read_text(encoding="utf-8") == output


def test_choose_best_rejects_oom_and_prefers_generation_speed() -> None:
    failed = CandidateRunResult(
        candidate_id="oom",
        strategy="layer",
        success=False,
        returncode=1,
        duration_seconds=1.0,
        gpu_baseline_mib=100,
        gpu_peak_mib=12_200,
        gpu_peak_delta_mib=12_100,
        metrics=LlamaMetrics(generation_tokens_per_second=99.0, oom=True),
        args=(),
    )
    slower = CandidateRunResult(
        candidate_id="slow",
        strategy="layer",
        success=True,
        returncode=0,
        duration_seconds=2.0,
        gpu_baseline_mib=100,
        gpu_peak_mib=10_000,
        gpu_peak_delta_mib=9_900,
        metrics=LlamaMetrics(generation_tokens_per_second=3.1),
        args=(),
    )
    faster = CandidateRunResult(
        candidate_id="fast",
        strategy="ffn",
        success=True,
        returncode=0,
        duration_seconds=2.5,
        gpu_baseline_mib=100,
        gpu_peak_mib=11_000,
        gpu_peak_delta_mib=10_900,
        metrics=LlamaMetrics(generation_tokens_per_second=4.0),
        args=(),
    )

    assert choose_best([failed, slower, faster]).candidate_id == "fast"


def test_choose_best_honors_measured_gpu_peak_limit() -> None:
    fast_but_tight = replace(
        _result("tight", 5.0, True), gpu_peak_mib=11_800
    )
    safe = replace(_result("safe", 4.0, True), gpu_peak_mib=11_100)

    assert choose_best(
        [fast_but_tight, safe], max_gpu_peak_mib=11_264
    ).candidate_id == "safe"
