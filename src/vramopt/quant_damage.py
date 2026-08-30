"""Scientific analysis of mathematical error introduced by quantization.

This module measures *compressibility of numeric residuals* only.  It makes no
claim that recovery of residual energy recovers model quality or intelligence.
Four mandatory controls for experiment 01 are implemented here:

1. functional error over activations (y_ref vs y_q vs y_corrected);
2. reproducible sampling across hub and non-hub experts (caller responsibility);
3. cost-benefit curve BPW -> functional recovery;
4. simple mixed-precision baselines.

All measurements are non-destructive and storage/compute aware.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
_DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
_DEFAULT_SPARSE_FRACTIONS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05)
_DEFAULT_HYBRID = ((8, 0.001), (16, 0.001), (16, 0.005), (32, 0.001), (32, 0.005), (64, 0.001), (64, 0.005))
_ENERGY_TARGETS = (0.5, 0.75, 0.9, 0.95, 0.99)
_GGUF_FIXED_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGML_TYPE_F16 = 1


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    """A GGUF tensor table entry; offsets are relative to the data section."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    data_offset: int


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated GGUF")
    return data


def _read_u32(stream: BinaryIO) -> int:
    return int(struct.unpack("<I", _read_exact(stream, 4))[0])


def _read_u64(stream: BinaryIO) -> int:
    return int(struct.unpack("<Q", _read_exact(stream, 8))[0])


def _read_string(stream: BinaryIO) -> str:
    size = _read_u64(stream)
    if size > 64 * 1024 * 1024:
        raise ValueError("GGUF string exceeds safety limit")
    return _read_exact(stream, size).decode("utf-8")


def _skip_gguf_value(stream: BinaryIO, value_type: int) -> int | None:
    if value_type in _GGUF_FIXED_TYPE_SIZES:
        stream.seek(_GGUF_FIXED_TYPE_SIZES[value_type], 1)
        return None
    if value_type == 8:
        _read_string(stream)
        return None
    if value_type == 9:
        item_type = _read_u32(stream)
        count = _read_u64(stream)
        if count > 10_000_000:
            raise ValueError("GGUF metadata array exceeds safety limit")
        for _ in range(count):
            _skip_gguf_value(stream, item_type)
        return None
    raise ValueError(f"unsupported GGUF metadata type: {value_type}")


def _align(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def list_gguf_tensors(path: str | Path) -> list[GGUFTensor]:
    """Read GGUF tensor table without allocating or dequantizing model weights."""
    model_path = Path(path)
    with model_path.open("rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError("invalid GGUF magic")
        version = _read_u32(stream)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version: {version}")
        tensor_count = _read_u64(stream)
        metadata_count = _read_u64(stream)
        alignment = 32
        for _ in range(metadata_count):
            key = _read_string(stream)
            value_type = _read_u32(stream)
            if key == "general.alignment" and value_type == 4:
                alignment = _read_u32(stream)
            else:
                _skip_gguf_value(stream, value_type)
        raw_entries: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = _read_string(stream)
            dimensions = _read_u32(stream)
            if dimensions == 0 or dimensions > 8:
                raise ValueError(f"unsupported GGUF tensor dimensions: {dimensions}")
            shape = tuple(_read_u64(stream) for _ in range(dimensions))
            ggml_type = _read_u32(stream)
            offset = _read_u64(stream)
            raw_entries.append((name, shape, ggml_type, offset))
        data_offset = _align(stream.tell(), alignment)
    return [
        GGUFTensor(name=name, shape=shape, ggml_type=ggml_type, offset=offset, data_offset=data_offset)
        for name, shape, ggml_type, offset in raw_entries
    ]


def extract_f16_tensor_slice(
    path: str | Path, tensor_name: str, *, slice_index: int | None
) -> NDArray[np.float32]:
    """Extract one F16 tensor or one final-axis slice from an experimental GGUF.

    This is intentionally limited to F16 entries emitted by a temporary
    `llama-quantize --tensor-type-file ...=f16` experiment. It never decodes
    arbitrary production GGUF tensor formats.
    """
    tensor = next((item for item in list_gguf_tensors(path) if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"tensor not found: {tensor_name}")
    if tensor.ggml_type != _GGML_TYPE_F16:
        raise ValueError(f"tensor {tensor_name} is type {tensor.ggml_type}, expected F16")
    if len(tensor.shape) not in {1, 2, 3}:
        raise ValueError(f"unsupported F16 tensor rank: {len(tensor.shape)}")
    if len(tensor.shape) == 3:
        if slice_index is None or not 0 <= slice_index < tensor.shape[2]:
            raise ValueError(f"slice_index must be in [0, {tensor.shape[2]})")
        elements = tensor.shape[0] * tensor.shape[1]
        byte_offset = tensor.data_offset + tensor.offset + slice_index * elements * 2
        output_shape = (tensor.shape[1], tensor.shape[0])
    else:
        if slice_index is not None:
            raise ValueError("slice_index is only valid for rank-3 tensors")
        elements = int(np.prod(tensor.shape))
        byte_offset = tensor.data_offset + tensor.offset
        output_shape = (tensor.shape[1], tensor.shape[0]) if len(tensor.shape) == 2 else (1, tensor.shape[0])
    with Path(path).open("rb") as stream:
        stream.seek(byte_offset)
        raw = _read_exact(stream, elements * 2)
    return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(output_shape)


def extract_gguf_tensor_as_f32(
    path: str | Path, tensor_name: str, *, slice_index: int | None = None
) -> NDArray[np.float32]:
    """Extract any GGUF tensor as float32, dequantizing via gguf library when needed.

    Uses `gguf.GGUFReader` + `gguf.quants.dequantize` to support Q2_K/Q3_K/Q4_K/Q6_K/Q8_0/F16/F32 etc.
    For MoE 3D tensors, slice_index selects a single expert slice on the last axis.
    Returns a 2D matrix (out, in) or (1, dim) for compatibility with analyze_quant_damage.
    """
    try:
        import gguf.quants
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("gguf package is required for generic tensor extraction") from exc

    reader = GGUFReader(str(path))
    candidate = next((t for t in reader.tensors if t.name == tensor_name), None)
    if candidate is None:
        raise KeyError(f"tensor not found: {tensor_name}")
    # Dequantize full tensor to f32 array
    # candidate.data is a memmap of quantized bytes; dequantize handles block layout
    dequantized = gguf.quants.dequantize(candidate.data, candidate.tensor_type)
    # gguf returns shape reversed vs GGUF logical shape: logical [2048,512,256] -> dequantized (256,512,2048)
    # Keep behavior compatible with F16 path which returns (cols, rows)
    if len(candidate.shape) == 3:
        if slice_index is None or not 0 <= slice_index < int(candidate.shape[2]):
            raise ValueError(f"slice_index must be in [0, {candidate.shape[2]}) for 3D tensor")
        # dequantized is (experts, dim1, dim0) = (256,512,2048)
        if dequantized.ndim == 3:
            # expert axis is 0 as shown by (256,512,2048)
            sliced = dequantized[int(slice_index), :, :]
        else:
            flat_shape = tuple(int(s) for s in candidate.shape)
            # reverse order as gguf does: (256,512,2048)
            rev_shape = tuple(reversed(flat_shape))
            reshaped = dequantized.reshape(rev_shape)
            sliced = reshaped[int(slice_index), :, :]
        # sliced is already (512,2048) which matches F16 convention (cols, rows)
        return np.asarray(sliced, dtype=np.float32)
    if len(candidate.shape) == 2:
        if slice_index is not None:
            raise ValueError("slice_index is only valid for rank-3 tensors")
        return np.asarray(dequantized.T if dequantized.shape == tuple(int(s) for s in candidate.shape) else dequantized.reshape(candidate.shape).T, dtype=np.float32)
    if len(candidate.shape) == 1:
        if slice_index is not None:
            raise ValueError("slice_index is only valid for rank-3 tensors")
        return np.asarray(dequantized.reshape(1, -1), dtype=np.float32)
    raise ValueError(f"unsupported tensor rank: {len(candidate.shape)}")


def select_non_hub_experts(
    *, n_experts: int, hubs: Iterable[int], n_non_hub: int = 2, seed: int = 0
) -> list[int]:
    """Reproducibly select non-hub expert indices.

    Deterministically picks the first n_non_hub from the sorted complement of hubs.
    This avoids bias from picking hubs as controls and ensures at least two non-hub
    samples as required by control 2.
    """
    hub_set = {int(h) for h in hubs}
    candidates = [i for i in range(int(n_experts)) if i not in hub_set]
    if len(candidates) < n_non_hub:
        raise ValueError(f"not enough non-hub experts: have {len(candidates)}, need {n_non_hub}")
    # Deterministic: sorted; seed reserved for future randomized but reproducible selection if needed
    _ = seed  # keep signature stable
    return candidates[:n_non_hub]


@dataclass(frozen=True, slots=True)
class ResidualSimilarity:
    cosine_similarity: float
    principal_left_similarity: float
    principal_right_similarity: float
    singular_spectrum_similarity: float


def compare_residuals(first: NDArray[Any], second: NDArray[Any]) -> ResidualSimilarity:
    """Compare two residual matrices; this does not build a shared codebook."""
    left = _as_float32(first)
    right = _as_float32(second)
    if left.shape != right.shape:
        raise ValueError(f"residual shapes differ: {left.shape} != {right.shape}")
    left_matrix = _matrix(left)
    right_matrix = _matrix(right)
    left_norm = _frobenius(left_matrix)
    right_norm = _frobenius(right_matrix)
    cosine = float(np.vdot(left_matrix.ravel(), right_matrix.ravel()) / (left_norm * right_norm)) if left_norm and right_norm else 0.0
    left_u, left_values, left_vt = _low_rank_components(left_matrix)
    right_u, right_values, right_vt = _low_rank_components(right_matrix)
    left_spectrum = left_values / (np.linalg.norm(left_values) or 1.0)
    right_spectrum = right_values / (np.linalg.norm(right_values) or 1.0)
    return ResidualSimilarity(
        cosine_similarity=cosine,
        principal_left_similarity=float(abs(np.vdot(left_u[:, 0], right_u[:, 0]))),
        principal_right_similarity=float(abs(np.vdot(left_vt[0, :], right_vt[0, :]))),
        singular_spectrum_similarity=float(np.vdot(left_spectrum, right_spectrum)),
    )


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    mse: float
    rmse: float
    mae: float
    max_error: float
    relative_error: float
    original_frobenius: float
    residual_frobenius: float
    residual_to_original_frobenius: float
    absolute_percentiles: dict[str, float]


@dataclass(frozen=True, slots=True)
class ActivationError:
    mse: float
    rmse: float
    mae: float
    max_error: float
    frobenius: float
    relative_frobenius: float


@dataclass(frozen=True, slots=True)
class LowRankResult:
    energy_by_rank: dict[int, float]
    rank_for_energy: dict[float, int | None]
    error_recovered_by_rank: dict[int, float]
    delta_bpw_by_rank: dict[int, float]
    activation_error_recovered_by_rank: dict[int, float | None] = field(default_factory=dict)
    compute_ops_by_rank: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SparseResult:
    fraction: float
    elements_kept: int
    energy_recovered: float
    mse_reduction: float
    activation_error_recovered: float | None
    memory_bytes: int
    delta_bpw: float
    value_bits: int
    index_bits: int
    metadata_bytes: int
    compute_ops: int = 0


@dataclass(frozen=True, slots=True)
class HybridResult:
    rank: int
    sparse_fraction: float
    energy_recovered: float
    mse_reduction: float
    activation_error_recovered: float | None
    memory_bytes: int
    delta_bpw: float
    compute_ops: int = 0


@dataclass(frozen=True, slots=True)
class CostBenefitRow:
    method: str
    config: str
    delta_bpw: float
    memory_bytes: int
    energy_recovered: float
    mse_reduction: float
    functional_recovered: float | None
    compute_ops: int


@dataclass(frozen=True, slots=True)
class MixedPrecisionBaseline:
    name: str
    delta_bpw: float
    memory_bytes: int
    assumption: str
    note: str


@dataclass(frozen=True, slots=True)
class QuantDamageReport:
    shape: tuple[int, ...]
    metrics: ErrorMetrics
    histogram_edges: list[float]
    histogram_counts: list[int]
    outliers: dict[float, int]
    sparsity_by_threshold: dict[float, float]
    activation_error: ActivationError | None
    low_rank: LowRankResult
    sparse: dict[float, SparseResult]
    hybrid: dict[tuple[int, float], HybridResult]
    base_bpw: float
    limitations: list[str]
    activation_provenance: dict[str, object] = field(default_factory=dict)
    cost_benefit: list[CostBenefitRow] = field(default_factory=list)
    mixed_precision_baselines: list[MixedPrecisionBaseline] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        sparse_data = data["sparse"]
        hybrid_data = data["hybrid"]
        if not isinstance(sparse_data, dict) or not isinstance(hybrid_data, dict):
            raise TypeError("serialized residual result must be a dictionary")
        data["sparse"] = {str(key): value for key, value in sparse_data.items()}
        data["hybrid"] = {
            f"rank_{rank}_top_{fraction}": value
            for (rank, fraction), value in hybrid_data.items()
        }
        return data


def _as_float32(array: NDArray[Any]) -> NDArray[np.float32]:
    return np.asarray(array, dtype=np.float32)


def _validate_pair(original: NDArray[Any], quantized: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    reference = _as_float32(original)
    candidate = _as_float32(quantized)
    if reference.shape != candidate.shape:
        raise ValueError(f"tensor shapes differ: {reference.shape} != {candidate.shape}")
    if reference.size == 0:
        raise ValueError("tensor must contain at least one element")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("tensors must be finite")
    return reference, candidate


def _matrix(array: NDArray[np.float32]) -> NDArray[np.float32]:
    if array.ndim == 0:
        raise ValueError("low-rank analysis needs at least one matrix dimension")
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _allowed_ranks(ranks: Iterable[int], matrix: NDArray[np.float32]) -> tuple[int, ...]:
    limit = min(matrix.shape)
    return tuple(sorted({rank for rank in ranks if 0 < rank <= limit}))


def _frobenius(array: NDArray[np.float32]) -> float:
    return float(np.linalg.norm(array.ravel()))


def _activation_error(
    original: NDArray[np.float32], quantized: NDArray[np.float32], activations: NDArray[Any] | None
) -> ActivationError | None:
    if activations is None:
        return None
    matrix_original = _matrix(original)
    matrix_quantized = _matrix(quantized)
    x = _as_float32(activations)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != matrix_original.shape[1]:
        raise ValueError(
            "activations must have shape (flattened_input_dimension, batch), "
            f"expected first dimension {matrix_original.shape[1]}, got {x.shape}"
        )
    y_original = matrix_original @ x
    error = y_original - (matrix_quantized @ x)
    reference_norm = _frobenius(y_original)
    residual_norm = _frobenius(error)
    abs_error = np.abs(error)
    return ActivationError(
        mse=float(np.mean(error**2)),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(abs_error)),
        max_error=float(np.max(abs_error)),
        frobenius=residual_norm,
        relative_frobenius=residual_norm / reference_norm if reference_norm else 0.0,
    )


def _make_proxy_activations(
    input_dim: int, *, batch: int = 32, seed: int = 0
) -> tuple[NDArray[np.float32], dict[str, object]]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(input_dim, batch)).astype(np.float32)
    provenance: dict[str, object] = {
        "kind": "proxy",
        "distribution": "standard_normal",
        "batch": batch,
        "seed": seed,
        "warning": "proxy activations are NOT real model activations; functional recovery is a synthetic control, not cognitive evidence",
    }
    return x, provenance


def _sparse_candidate(residual: NDArray[np.float32], fraction: float) -> NDArray[np.float32]:
    if not 0 < fraction <= 1:
        raise ValueError("sparse fractions must be in (0, 1]")
    flat = residual.ravel()
    count = max(1, ceil(flat.size * fraction))
    selected = np.argpartition(np.abs(flat), -count)[-count:]
    candidate = np.zeros_like(flat)
    candidate[selected] = flat[selected]
    return candidate.reshape(residual.shape)


def _energy_recovered(residual: NDArray[np.float32], approximation: NDArray[np.float32]) -> float:
    total = float(np.sum(residual**2))
    if total == 0:
        return 1.0
    remaining = residual - approximation
    return float(1.0 - np.sum(remaining**2) / total)


def _activation_recovered(
    residual: NDArray[np.float32], approximation: NDArray[np.float32], activations: NDArray[Any] | None
) -> float | None:
    if activations is None:
        return None
    base = _activation_error(residual, np.zeros_like(residual), activations)
    remaining = _activation_error(residual, approximation, activations)
    if base is None or remaining is None or base.frobenius == 0:
        return 1.0
    return float(1.0 - remaining.frobenius / base.frobenius)


def _low_rank_components(residual: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    return tuple(np.asarray(item, dtype=np.float32) for item in np.linalg.svd(_matrix(residual), full_matrices=False))  # type: ignore[return-value]


def _low_rank_approximation(
    left: NDArray[np.float32], values: NDArray[np.float32], right: NDArray[np.float32], rank: int, shape: tuple[int, ...]
) -> NDArray[np.float32]:
    approximation = (left[:, :rank] * values[:rank]) @ right[:rank, :]
    return np.asarray(approximation.reshape(shape), dtype=np.float32)


def _low_rank_bpw(shape: tuple[int, ...], rank: int, value_bits: int = 16, metadata_bits: int = 128) -> float:
    rows = shape[0] if len(shape) > 1 else 1
    columns = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    elements = rows * columns
    stored_values = rank * (rows + columns + 1)
    return (stored_values * value_bits + metadata_bits) / elements


def _sparse_storage(elements: int, kept: int, value_bits: int = 16, metadata_bytes: int = 16) -> tuple[int, int, int]:
    index_bits = 32 if elements <= 2**32 - 1 else 64
    memory_bytes = ceil((kept * (value_bits + index_bits)) / 8) + metadata_bytes
    return memory_bytes, index_bits, metadata_bytes


def _estimate_low_rank_ops(shape: tuple[int, ...], rank: int) -> int:
    rows = shape[0] if len(shape) > 1 else 1
    columns = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    # y_corrected = y_q + (U_r * diag(s) @ V_r) @ x  ->  O(rank*(rows+cols)*batch) simplified as rank*(rows+cols)
    return int(rank * (rows + columns))


def _estimate_sparse_ops(kept: int) -> int:
    return int(kept)


def _resolve_activations(
    reference: NDArray[np.float32], activations: NDArray[Any] | None, *, use_proxy: bool, proxy_batch: int
) -> tuple[NDArray[np.float32] | None, dict[str, object]]:
    if activations is not None:
        x = _as_float32(activations)
        provenance: dict[str, object] = {"kind": "real_or_external", "batch": int(x.shape[1]) if x.ndim == 2 else 1}
        # validate shape quickly
        mat = _matrix(reference)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != mat.shape[1]:
            raise ValueError(f"activations first dim {x.shape[0]} != expected {mat.shape[1]}")
        return x, provenance
    if use_proxy:
        mat = _matrix(reference)
        x_proxy, prov = _make_proxy_activations(int(mat.shape[1]), batch=proxy_batch, seed=0)
        return x_proxy, prov
    return None, {"kind": "none", "warning": "no activations provided; functional recovery will be null"}


def analyze_quant_damage(
    original: NDArray[Any],
    quantized: NDArray[Any],
    *,
    activations: NDArray[Any] | None = None,
    thresholds: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2),
    ranks: tuple[int, ...] = _DEFAULT_RANKS,
    sparse_fractions: tuple[float, ...] = _DEFAULT_SPARSE_FRACTIONS,
    hybrid_combinations: tuple[tuple[int, float], ...] = _DEFAULT_HYBRID,
    histogram_bins: int = 64,
    base_bpw: float = 2.0,
    use_proxy_activations: bool = True,
    proxy_batch: int = 32,
) -> QuantDamageReport:
    """Measure numeric error and candidate compact residual representations.

    `original` and `quantized` may be any finite tensor shape.  Low-rank analysis
    views it as ``(shape[0], prod(shape[1:]))``.  Callers should sample extremely
    large tensors before this function; full SVD is intentionally exact here.

    Functional recovery y_ref vs y_q vs y_corrected is measured whenever activations
    are available.  When no real activations are supplied, a deterministic proxy
    is used if use_proxy_activations is True; the provenance clearly marks it as
    synthetic and not cognitive evidence.
    """
    reference, candidate = _validate_pair(original, quantized)
    residual = reference - candidate
    abs_residual = np.abs(residual)
    original_norm = _frobenius(reference)
    residual_norm = _frobenius(residual)
    percentiles = np.percentile(abs_residual, [50, 90, 95, 99, 99.9])
    edges, counts = np.histogram(residual, bins=histogram_bins)
    stddev = float(np.std(residual))
    metrics = ErrorMetrics(
        mse=float(np.mean(residual**2)),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(abs_residual)),
        max_error=float(np.max(abs_residual)),
        relative_error=float(residual_norm / (original_norm + np.finfo(np.float32).eps)),
        original_frobenius=original_norm,
        residual_frobenius=residual_norm,
        residual_to_original_frobenius=float(residual_norm / (original_norm + np.finfo(np.float32).eps)),
        absolute_percentiles={name: float(value) for name, value in zip(("p50", "p90", "p95", "p99", "p99.9"), percentiles, strict=True)},
    )
    outliers = {sigma: int(np.count_nonzero(abs_residual > sigma * stddev)) for sigma in (2.0, 3.0, 4.0)}
    sparsity = {threshold: float(np.mean(abs_residual <= threshold)) for threshold in thresholds}
    # Resolve activations with provenance
    effective_activations, activation_provenance = _resolve_activations(
        reference, activations, use_proxy=use_proxy_activations, proxy_batch=proxy_batch
    )
    activation = _activation_error(reference, candidate, effective_activations)
    left, values, right = _low_rank_components(residual)
    allowed_ranks = _allowed_ranks(ranks, _matrix(residual))
    total_energy = float(np.sum(values**2))
    energy = {
        rank: float(np.sum(values[:rank] ** 2) / total_energy) if total_energy else 1.0
        for rank in allowed_ranks
    }
    rank_targets: dict[float, int | None] = {}
    for target in _ENERGY_TARGETS:
        rank_at_target = (
            next(
                (
                    rank
                    for rank in range(1, len(values) + 1)
                    if float(np.sum(values[:rank] ** 2) / total_energy) >= target
                ),
                None,
            )
            if total_energy
            else 1
        )
        rank_targets[target] = rank_at_target
    low_rank_recovery: dict[int, float] = {}
    low_rank_func_recovery: dict[int, float | None] = {}
    low_rank_ops: dict[int, int] = {}
    low_rank_approximations: dict[int, NDArray[np.float32]] = {}
    for rank in allowed_ranks:
        approximation = _low_rank_approximation(left, values, right, rank, residual.shape)
        low_rank_approximations[rank] = approximation
        low_rank_recovery[rank] = _energy_recovered(residual, approximation)
        low_rank_func_recovery[rank] = _activation_recovered(residual, approximation, effective_activations)
        low_rank_ops[rank] = _estimate_low_rank_ops(reference.shape, rank)
    low_rank = LowRankResult(
        energy_by_rank=energy,
        rank_for_energy=rank_targets,
        error_recovered_by_rank=low_rank_recovery,
        delta_bpw_by_rank={rank: _low_rank_bpw(reference.shape, rank) for rank in allowed_ranks},
        activation_error_recovered_by_rank=low_rank_func_recovery,
        compute_ops_by_rank=low_rank_ops,
    )
    sparse: dict[float, SparseResult] = {}
    for fraction in sparse_fractions:
        approximation = _sparse_candidate(residual, fraction)
        kept = max(1, ceil(residual.size * fraction))
        memory_bytes, index_bits, metadata_bytes = _sparse_storage(residual.size, kept)
        sparse[fraction] = SparseResult(
            fraction=fraction,
            elements_kept=kept,
            energy_recovered=_energy_recovered(residual, approximation),
            mse_reduction=_energy_recovered(residual, approximation),
            activation_error_recovered=_activation_recovered(residual, approximation, effective_activations),
            memory_bytes=memory_bytes,
            delta_bpw=memory_bytes * 8 / residual.size,
            value_bits=16,
            index_bits=index_bits,
            metadata_bytes=metadata_bytes,
            compute_ops=_estimate_sparse_ops(kept),
        )
    hybrid: dict[tuple[int, float], HybridResult] = {}
    for rank, fraction in hybrid_combinations:
        if rank not in low_rank_approximations or not 0 < fraction <= 1:
            continue
        low_rank_base = low_rank_approximations[rank]
        second = _sparse_candidate(residual - low_rank_base, fraction)
        approximation = low_rank_base + second
        kept = max(1, ceil(residual.size * fraction))
        sparse_bytes, _, _ = _sparse_storage(residual.size, kept)
        low_rank_bytes = ceil(_low_rank_bpw(reference.shape, rank) * residual.size / 8)
        memory_bytes = low_rank_bytes + sparse_bytes
        hybrid[(rank, fraction)] = HybridResult(
            rank=rank,
            sparse_fraction=fraction,
            energy_recovered=_energy_recovered(residual, approximation),
            mse_reduction=_energy_recovered(residual, approximation),
            activation_error_recovered=_activation_recovered(residual, approximation, effective_activations),
            memory_bytes=memory_bytes,
            delta_bpw=memory_bytes * 8 / residual.size,
            compute_ops=_estimate_low_rank_ops(reference.shape, rank) + _estimate_sparse_ops(kept),
        )
    # Build cost-benefit rows sorted by delta_bpw
    rows: list[CostBenefitRow] = []
    for rank in allowed_ranks:
        rows.append(
            CostBenefitRow(
                method="low_rank",
                config=f"rank={rank}",
                delta_bpw=float(_low_rank_bpw(reference.shape, rank)),
                memory_bytes=ceil(_low_rank_bpw(reference.shape, rank) * residual.size / 8),
                energy_recovered=float(low_rank_recovery[rank]),
                mse_reduction=float(low_rank_recovery[rank]),
                functional_recovered=low_rank_func_recovery.get(rank),
                compute_ops=int(low_rank_ops[rank]),
            )
        )
    for fraction, result in sparse.items():
        rows.append(
            CostBenefitRow(
                method="sparse",
                config=f"top_{fraction}",
                delta_bpw=float(result.delta_bpw),
                memory_bytes=int(result.memory_bytes),
                energy_recovered=float(result.energy_recovered),
                mse_reduction=float(result.mse_reduction),
                functional_recovered=result.activation_error_recovered,
                compute_ops=int(result.compute_ops),
            )
        )
    for (rank, fraction), hybrid_result in hybrid.items():
        rows.append(
            CostBenefitRow(
                method="hybrid",
                config=f"rank_{rank}_top_{fraction}",
                delta_bpw=float(hybrid_result.delta_bpw),
                memory_bytes=int(hybrid_result.memory_bytes),
                energy_recovered=float(hybrid_result.energy_recovered),
                mse_reduction=float(hybrid_result.mse_reduction),
                functional_recovered=hybrid_result.activation_error_recovered,
                compute_ops=int(hybrid_result.compute_ops),
            )
        )
    rows.sort(key=lambda r: r.delta_bpw)
    # Mixed precision baselines (single-tensor view)
    baselines: list[MixedPrecisionBaseline] = []
    # Assumptions documented explicitly; not claiming model-level BPW.
    higher_precisions = [
        ("Q2->Q3", 3.44, "Q3_K approximate"),
        ("Q2->Q4", 4.56, "Q4_K_M approximate"),
        ("Q2->Q5", 5.5, "Q5_K_M approximate"),
        ("Q2->F16", 16.0, "full F16 for tensor"),
    ]
    for name, target_bpw, assumption in higher_precisions:
        delta = float(target_bpw - base_bpw)
        if delta <= 0:
            continue
        baselines.append(
            MixedPrecisionBaseline(
                name=name,
                delta_bpw=delta,
                memory_bytes=ceil(delta * residual.size / 8),
                assumption=assumption,
                note="baseline keeps the entire tensor at higher precision; compare against R_approx that corrects only a fraction",
            )
        )
    # Outlier baseline: keep top 0.1% and 1% in F16 (similar to sparse 16-bit)
    for frac in (0.001, 0.01):
        kept = max(1, ceil(residual.size * frac))
        mem, _, _ = _sparse_storage(residual.size, kept, value_bits=16)
        baselines.append(
            MixedPrecisionBaseline(
                name=f"outliers_{frac}",
                delta_bpw=float(mem * 8 / residual.size),
                memory_bytes=int(mem),
                assumption=f"store top {frac*100:.1f}% values as FP16 + index",
                note="explicit outlier retention; comparable to sparse candidate with same fraction",
            )
        )
    # Provenance enrichment
    activation_provenance = {
        **activation_provenance,
        "effective_batch": int(effective_activations.shape[1]) if effective_activations is not None else 0,
        "input_dim": int(_matrix(reference).shape[1]),
    }
    limitations = [
        "Energy recovered is numeric residual energy, not intelligence or benchmark quality.",
        "Functional recovery y_ref vs y_q vs y_corrected is measured on supplied or proxy activations; proxy is synthetic and must not be read as cognitive evidence.",
        "Low-rank analysis uses an exact SVD of the supplied tensor/sample only.",
        "Sparse storage assumes FP16 values plus a real uint32/uint64 linear index and metadata.",
        "Mixed-precision baselines are single-tensor theoretical deltas (target_bpw - base_bpw); model-level gain requires weighting by tensor count.",
        "Origin of this measurement is MXFP4 -> Q2 requantization unless otherwise recorded; not equivalent to BF16 -> Q2.",
    ]
    return QuantDamageReport(
        shape=tuple(reference.shape),
        metrics=metrics,
        histogram_edges=[float(value) for value in edges],
        histogram_counts=[int(value) for value in counts],
        outliers=outliers,
        sparsity_by_threshold=sparsity,
        activation_error=activation,
        low_rank=low_rank,
        sparse=sparse,
        hybrid=hybrid,
        base_bpw=base_bpw,
        limitations=limitations,
        activation_provenance=activation_provenance,
        cost_benefit=rows,
        mixed_precision_baselines=baselines,
    )
