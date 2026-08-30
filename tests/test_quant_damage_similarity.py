from __future__ import annotations

import numpy as np
import pytest

from vramopt.quant_damage import compare_residuals


def test_compare_residuals_detects_shared_rank_one_direction() -> None:
    first = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    second = first * 3.0

    comparison = compare_residuals(first, second)

    assert comparison.cosine_similarity == pytest.approx(1.0)
    assert comparison.principal_left_similarity == pytest.approx(1.0)
    assert comparison.principal_right_similarity == pytest.approx(1.0)


def test_compare_residuals_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compare_residuals(np.ones((2, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))
