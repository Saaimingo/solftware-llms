"""Generate reproducible llama.cpp tuning candidates.

The planner never changes weight quantization. It compares placement and cache
strategies only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vramopt.gguf import GGUFInfo


@dataclass(frozen=True, slots=True)
class TuningRequest:
    context_size: int = 4_096
    fit_margin_mib: int = 1_024
    threads: int = 8
    batch_size: int = 512
    ubatch_size: int = 128

    def __post_init__(self) -> None:
        for name, value in (
            ("context_size", self.context_size),
            ("fit_margin_mib", self.fit_margin_mib),
            ("threads", self.threads),
            ("batch_size", self.batch_size),
            ("ubatch_size", self.ubatch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TuningCandidate:
    id: str
    strategy: str
    request: TuningRequest
    cache_type_k: str
    cache_type_v: str
    load_mode: str = "mmap"
    n_cpu_ffn: int = 0
    n_cpu_moe: int = 0

    def _common_args(self, executable: Path, model: Path) -> list[str]:
        args = [
            str(executable),
            "--model",
            str(model),
            "--ctx-size",
            str(self.request.context_size),
            "--fit",
            "on",
            "--fit-target",
            str(self.request.fit_margin_mib),
            "--fit-ctx",
            str(self.request.context_size),
            "--gpu-layers",
            "auto",
            "--cache-type-k",
            self.cache_type_k,
            "--cache-type-v",
            self.cache_type_v,
            "--flash-attn",
            "auto",
            "--threads",
            str(self.request.threads),
            "--threads-batch",
            str(self.request.threads),
            "--batch-size",
            str(self.request.batch_size),
            "--ubatch-size",
            str(self.request.ubatch_size),
            "--load-mode",
            self.load_mode,
            "--offline",
            "--log-colors",
            "off",
            "--log-verbosity",
            "3",
        ]
        if self.n_cpu_ffn:
            args.extend(("--n-cpu-ffn", str(self.n_cpu_ffn)))
        if self.n_cpu_moe:
            args.extend(("--n-cpu-moe", str(self.n_cpu_moe)))
        return args

    def to_llama_args(
        self,
        executable: Path,
        model: Path,
        *,
        predict: int,
        prompt: str = "Reply with exactly: READY",
    ) -> list[str]:
        if predict <= 0:
            raise ValueError("predict must be positive")
        return self._common_args(executable, model) + [
            "--no-mmproj",
            "--simple-io",
            "--no-display-prompt",
            "--single-turn",
            "--ignore-eos",
            "--perf",
            "--show-timings",
            "--seed",
            "1",
            "--temp",
            "0",
            "--predict",
            str(predict),
            "--prompt",
            prompt,
        ]

    def to_interactive_args(self, executable: Path, model: Path) -> list[str]:
        return self._common_args(executable, model) + ["--no-mmproj"]

    def to_server_args(
        self,
        executable: Path,
        model: Path,
        *,
        host: str,
        port: int,
    ) -> list[str]:
        if not host:
            raise ValueError("host cannot be empty")
        if not 1 <= port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        return self._common_args(executable, model) + [
            "--host",
            host,
            "--port",
            str(port),
        ]


def _required_str(
    payload: Mapping[str, object], key: str, *, default: str | None = None
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate field {key!r} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"candidate field {key!r} must be an integer")
    return value


def candidate_from_payload(payload: Mapping[str, object]) -> TuningCandidate:
    """Validate and reconstruct a candidate stored in a JSON profile."""

    request_payload = payload.get("request")
    if not isinstance(request_payload, Mapping):
        raise TypeError("candidate request must be an object")
    request = TuningRequest(
        context_size=_required_int(request_payload, "context_size"),
        fit_margin_mib=_required_int(request_payload, "fit_margin_mib"),
        threads=_required_int(request_payload, "threads"),
        batch_size=_required_int(request_payload, "batch_size"),
        ubatch_size=_required_int(request_payload, "ubatch_size"),
    )
    return TuningCandidate(
        id=_required_str(payload, "id"),
        strategy=_required_str(payload, "strategy"),
        request=request,
        cache_type_k=_required_str(payload, "cache_type_k"),
        cache_type_v=_required_str(payload, "cache_type_v"),
        load_mode=_required_str(payload, "load_mode", default="mmap"),
        n_cpu_ffn=_required_int(payload, "n_cpu_ffn", default=0),
        n_cpu_moe=_required_int(payload, "n_cpu_moe", default=0),
    )


def _offload_counts(block_count: int | None) -> list[int]:
    if block_count is None or block_count <= 0:
        return []
    return sorted(
        {
            max(1, min(block_count, round(block_count * fraction)))
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }
    )


def _validated_offload_counts(
    counts: Sequence[int], block_count: int | None
) -> list[int]:
    maximum = block_count if block_count is not None else max(counts, default=0)
    if any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or count > maximum
        for count in counts
    ):
        raise ValueError(f"offload counts must be between 1 and {maximum}")
    return sorted(set(counts))


def build_candidates(
    model: GGUFInfo,
    request: TuningRequest,
    *,
    offload_counts: Sequence[int] | None = None,
) -> list[TuningCandidate]:
    """Build a small search space that preserves the model's existing weights."""

    prefix = "moe" if model.is_moe else "dense"
    candidates = [
        TuningCandidate(
            id=f"{prefix}-layer-f16",
            strategy="layer",
            request=request,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        TuningCandidate(
            id=f"{prefix}-layer-q8_0",
            strategy="layer",
            request=request,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
    ]
    counts = (
        _offload_counts(model.block_count)
        if offload_counts is None
        else _validated_offload_counts(offload_counts, model.block_count)
    )
    for count in counts:
        if model.is_moe:
            candidates.append(
                TuningCandidate(
                    id=f"{prefix}-experts-{count}-q8_0",
                    strategy="expert",
                    request=request,
                    cache_type_k="q8_0",
                    cache_type_v="q8_0",
                    load_mode="none",
                    n_cpu_moe=count,
                )
            )
        else:
            candidates.append(
                TuningCandidate(
                    id=f"{prefix}-ffn-{count}-q8_0",
                    strategy="ffn",
                    request=request,
                    cache_type_k="q8_0",
                    cache_type_v="q8_0",
                    load_mode="none",
                    n_cpu_ffn=count,
                )
            )
    return candidates
