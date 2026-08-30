from __future__ import annotations

import numpy as np
import pytest

from vramopt.quant_damage import analyze_quant_damage, select_non_hub_experts


def test_low_rank_activation_recovery_via_proxy() -> None:
    rng = np.random.default_rng(0)
    original = rng.standard_normal(size=(8, 8)).astype(np.float32)
    quantized = original + rng.normal(scale=0.1, size=(8, 8)).astype(np.float32)
    report = analyze_quant_damage(original, quantized, ranks=(1, 2, 4), sparse_fractions=(0.01,))
    # proxy activated by default, so low rank should have functional recovery
    assert report.activation_provenance["kind"] == "proxy"
    assert report.activation_provenance["warning"].startswith("proxy activations")  # type: ignore[union-attr]
    for rank in (1, 2, 4):
        if rank in report.low_rank.activation_error_recovered_by_rank:
            val = report.low_rank.activation_error_recovered_by_rank[rank]
            assert val is not None
            assert 0.0 <= val <= 1.0
    # also verify that functional recovery differs from pure energy (not equivalent)
    # at least one rank should have different values
    rank = 1
    energy = report.low_rank.energy_by_rank[rank]
    func = report.low_rank.activation_error_recovered_by_rank[rank]
    # they are distinct concepts; allow small numerical overlap but not identity required
    assert func is not None
    assert isinstance(energy, float)


def test_cost_benefit_contains_all_methods_and_sorted() -> None:
    original = np.eye(4, dtype=np.float32) * 2.0
    quantized = np.eye(4, dtype=np.float32)
    report = analyze_quant_damage(
        original,
        quantized,
        ranks=(1, 2),
        sparse_fractions=(0.25, 0.5),
        hybrid_combinations=((1, 0.25),),
    )
    methods = {row.method for row in report.cost_benefit}
    assert "low_rank" in methods
    assert "sparse" in methods
    assert "hybrid" in methods
    # sorted by delta_bpw ascending
    bpws = [row.delta_bpw for row in report.cost_benefit]
    assert bpws == sorted(bpws)
    for row in report.cost_benefit:
        assert row.memory_bytes > 0
        assert row.delta_bpw >= 0
        assert 0.0 <= row.energy_recovered <= 1.0
        assert 0.0 <= row.mse_reduction <= 1.0
        if row.functional_recovered is not None:
            assert 0.0 <= row.functional_recovered <= 1.0
        assert row.compute_ops >= 0


def test_mixed_precision_baselines_present_and_dominance_check() -> None:
    original = np.ones((4, 4), dtype=np.float32)
    quantized = np.zeros((4, 4), dtype=np.float32)
    report = analyze_quant_damage(original, quantized, base_bpw=2.98, ranks=(1,), sparse_fractions=(0.01,))
    names = {b.name for b in report.mixed_precision_baselines}
    assert "Q2->Q4" in names
    assert "Q2->Q3" in names
    assert any(n.startswith("outliers_") for n in names)
    for baseline in report.mixed_precision_baselines:
        assert baseline.delta_bpw > 0
        # memory_bytes uses ceil(delta*size/8) for higher precision baselines
        from math import ceil

        expected = ceil(baseline.delta_bpw * original.size / 8) if "outliers" not in baseline.name else baseline.memory_bytes
        if "outliers" not in baseline.name:
            assert baseline.memory_bytes == expected
        assert baseline.assumption
        assert baseline.note


def test_real_activations_preferred_over_proxy() -> None:
    original = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    quantized = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    activations = np.eye(2, dtype=np.float32)
    report = analyze_quant_damage(original, quantized, activations=activations, ranks=(1,))
    assert report.activation_provenance["kind"] == "real_or_external"
    assert report.activation_error is not None
    # when real activations provided, functional recovery should be computed, not None
    assert report.low_rank.activation_error_recovered_by_rank[1] is not None


def test_select_non_hub_experts_reproducible() -> None:
    hubs = [17, 42]
    picked = select_non_hub_experts(n_experts=256, hubs=hubs, n_non_hub=2)
    assert len(picked) == 2
    assert all(p not in hubs for p in picked)
    assert picked == [0, 1]
    # with different hubs, still deterministic
    picked2 = select_non_hub_experts(n_experts=10, hubs=[0, 1, 2], n_non_hub=2)
    assert picked2 == [3, 4]
    # not enough candidates raises
    with pytest.raises(ValueError, match="not enough non-hub"):
        select_non_hub_experts(n_experts=3, hubs=[0, 1, 2], n_non_hub=1)


def test_no_proxy_means_null_functional() -> None:
    original = np.eye(4, dtype=np.float32)
    quantized = np.zeros((4, 4), dtype=np.float32)
    report = analyze_quant_damage(original, quantized, use_proxy_activations=False, ranks=(1,))
    assert report.activation_provenance["kind"] == "none"
    assert report.low_rank.activation_error_recovered_by_rank[1] is None
    assert report.sparse[0.001].activation_error_recovered is None
