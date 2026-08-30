from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from vramopt.gguf import GGUFInfo
from vramopt.planner import TuningRequest, build_candidates, candidate_from_payload


def _model(*, moe: bool) -> GGUFInfo:
    return GGUFInfo(
        path=Path("C:/models/model.gguf"),
        size_bytes=16_464_440_224,
        version=3,
        tensor_count=1_000,
        metadata_count=100,
        name="Qwen test",
        architecture="qwen35moe" if moe else "qwen35",
        file_type=15,
        quantization_version=2,
        context_length=262_144,
        block_count=64,
        expert_count=64 if moe else None,
        expert_used_count=16 if moe else None,
    )


def test_dense_candidates_compare_layer_and_ffn_offload() -> None:
    request = TuningRequest(context_size=8_192, fit_margin_mib=1_024, threads=8)
    candidates = build_candidates(_model(moe=False), request)

    assert candidates[0].strategy == "layer"
    assert candidates[0].n_cpu_ffn == 0
    assert any(candidate.n_cpu_ffn > 0 for candidate in candidates)
    assert all(candidate.n_cpu_moe == 0 for candidate in candidates)
    assert all(
        candidate.load_mode == ("none" if candidate.strategy == "ffn" else "mmap")
        for candidate in candidates
    )
    assert {candidate.cache_type_k for candidate in candidates} <= {"f16", "q8_0"}
    assert len({candidate.id for candidate in candidates}) == len(candidates)

    ffn = next(candidate for candidate in candidates if candidate.n_cpu_ffn > 0)
    args = ffn.to_llama_args(
        Path("llama-cli.exe"), _model(moe=False).path, predict=64
    )
    assert "--n-cpu-ffn" in args
    assert "--cpu-moe" not in args
    assert args[args.index("--ctx-size") + 1] == "8192"
    assert args[args.index("--fit-target") + 1] == "1024"
    assert args[args.index("--load-mode") + 1] == "none"
    assert "--fit-print" not in args
    assert "--ignore-eos" in args
    assert args[args.index("--log-verbosity") + 1] == "3"
    assert not any("Q2" in argument or "Q3" in argument for argument in args)


def test_moe_candidates_offload_experts_instead_of_dense_ffn() -> None:
    request = TuningRequest(context_size=4_096, fit_margin_mib=1_024, threads=8)
    candidates = build_candidates(_model(moe=True), request)

    assert any(candidate.n_cpu_moe > 0 for candidate in candidates)
    assert all(candidate.n_cpu_ffn == 0 for candidate in candidates)
    expert = next(candidate for candidate in candidates if candidate.n_cpu_moe > 0)
    args = expert.to_llama_args(
        Path("llama-cli.exe"), _model(moe=True).path, predict=32
    )
    assert "--n-cpu-moe" in args
    assert "--n-cpu-ffn" not in args


def test_candidate_profile_round_trip_and_runtime_commands() -> None:
    candidate = next(
        item
        for item in build_candidates(
            _model(moe=False),
            TuningRequest(context_size=4_096, fit_margin_mib=1_024, threads=8),
        )
        if item.n_cpu_ffn > 0
    )

    restored = candidate_from_payload(asdict(candidate))
    interactive = restored.to_interactive_args(
        Path("llama-cli.exe"), Path("model.gguf")
    )
    server = restored.to_server_args(
        Path("llama-server.exe"),
        Path("model.gguf"),
        host="127.0.0.1",
        port=8_080,
    )

    assert restored == candidate
    assert "--n-cpu-ffn" in interactive
    assert "--single-turn" not in interactive
    assert "--prompt" not in interactive
    assert server[-4:] == ["--host", "127.0.0.1", "--port", "8080"]


def test_candidate_profile_rejects_missing_request() -> None:
    with pytest.raises(TypeError, match="request"):
        candidate_from_payload({"id": "broken"})


def test_custom_offload_counts_are_validated_and_deduplicated() -> None:
    request = TuningRequest(context_size=4_096, fit_margin_mib=1_024, threads=8)

    candidates = build_candidates(
        _model(moe=False), request, offload_counts=[40, 36, 40]
    )
    counts = [item.n_cpu_ffn for item in candidates if item.strategy == "ffn"]

    assert counts == [36, 40]
    with pytest.raises(ValueError, match="between"):
        build_candidates(_model(moe=False), request, offload_counts=[0])
