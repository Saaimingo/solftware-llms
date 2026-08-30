from __future__ import annotations

import numpy as np
import pytest

from vramopt.quant_damage import analyze_quant_damage


def test_rank_one_residual_has_all_energy_at_rank_one() -> None:
    left = np.array([[1.0], [2.0], [-1.0]], dtype=np.float32)
    right = np.array([[3.0, -2.0]], dtype=np.float32)
    residual = left @ right
    original = np.full_like(residual, 10.0) + residual
    quantized = original - residual

    report = analyze_quant_damage(original, quantized, ranks=(1, 2), sparse_fractions=(0.5,))

    assert report.low_rank.energy_by_rank[1] == pytest.approx(1.0)
    assert report.low_rank.rank_for_energy[0.99] == 1
    assert report.low_rank.error_recovered_by_rank[1] == pytest.approx(1.0)


def test_sparse_outliers_recover_residual_energy_and_activation_error() -> None:
    original = np.zeros((4, 4), dtype=np.float32)
    quantized = original.copy()
    quantized[1, 2] = -10.0
    activations = np.eye(4, dtype=np.float32)

    report = analyze_quant_damage(
        original,
        quantized,
        activations=activations,
        ranks=(1,),
        sparse_fractions=(0.0625, 0.25),
    )

    one_value = report.sparse[0.0625]
    assert one_value.elements_kept == 1
    assert one_value.energy_recovered == pytest.approx(1.0)
    assert one_value.mse_reduction == pytest.approx(1.0)
    assert one_value.activation_error_recovered == pytest.approx(1.0)


def test_random_residual_is_not_claimed_rank_one_compressible() -> None:
    rng = np.random.default_rng(7)
    original = rng.normal(size=(32, 32)).astype(np.float32)
    quantized = np.zeros_like(original)

    report = analyze_quant_damage(original, quantized, ranks=(1, 2, 4, 8), sparse_fractions=(0.01,))

    assert report.low_rank.energy_by_rank[1] < 0.2
    assert report.low_rank.rank_for_energy[0.5] > 1


def test_report_includes_metrics_histogram_outliers_and_hybrid() -> None:
    original = np.array([[2.0, -2.0], [4.0, -4.0]], dtype=np.float32)
    quantized = np.array([[1.0, -1.0], [4.0, -3.0]], dtype=np.float32)

    report = analyze_quant_damage(
        original,
        quantized,
        thresholds=(0.5, 1.5),
        ranks=(1, 2),
        sparse_fractions=(0.25, 0.5),
        hybrid_combinations=((1, 0.25),),
    )

    assert report.metrics.mse == pytest.approx(0.75)
    assert report.metrics.max_error == pytest.approx(1.0)
    assert report.metrics.residual_to_original_frobenius > 0
    assert set(report.outliers) == {2.0, 3.0, 4.0}
    assert set(report.sparsity_by_threshold) == {0.5, 1.5}
    assert (1, 0.25) in report.hybrid
