from pathlib import Path

from vramopt.block_predictor import (
    build_hubs,
    predict_blocks,
    simulate_double_buffer,
)


def test_build_hubs():
    analysis = {"global_freq": {"17": 100, "42": 90, "5": 10}, "blocks": []}
    hubs = build_hubs(analysis, top_k=2)
    assert hubs == [17, 42]


def test_predict_blocks_filter():
    analysis = {"global_freq": {}, "blocks": [{"layer": 3, "experts": [1, 2], "count": 10, "ratio": 1.0}, {"layer": 7, "experts": [3, 4], "count": 5, "ratio": 0.5}]}
    assert len(predict_blocks(analysis)) == 2
    assert len(predict_blocks(analysis, layer=3)) == 1


def test_simulate_double_buffer(tmp_path=None):
    # usa trace real gerado
    p = sorted(Path("artifacts/traces").glob("Qwen3.6*.json"))[-1]
    res = simulate_double_buffer(p)
    assert "with_blocks" in res
    assert res["speedup"] > 1
    assert res["with_blocks"]["estimated_tps_ceiling"] > res["without_blocks_tps"]
