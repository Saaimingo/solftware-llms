import json
from pathlib import Path

from vramopt.tracer import ExpertTrace, TraceConfig, analyze_trace, run_trace


def test_analyze_coactivation_simple():
    traces = [
        ExpertTrace(0, 0, [1, 2]),
        ExpertTrace(1, 0, [1, 2]),
        ExpertTrace(2, 0, [1, 3]),
        ExpertTrace(0, 1, [5, 6]),
    ]
    a = analyze_trace(traces)
    assert a["total_traces"] == 4
    assert "1" in str(a["global_freq"]) or 1 in a["global_freq"]
    # par (1,2) deve aparecer como bloco forte
    assert any(set(b["experts"]) == {1, 2} for b in a["blocks"])


def test_trace_config_defaults():
    cfg = TraceConfig(model_path=Path("models/a.gguf"), backend_dir=Path("vendor/llama.cpp/bin"))
    assert cfg.ctx_size == 4096
    assert cfg.n_tokens == 256


def test_trace_saves_json(tmp_path: Path = Path("artifacts/traces")):
    # usa modelo real se existir, senão fake
    model = Path("models/gpt-oss-20b-MXFP4.gguf")
    if not model.exists():
        model = Path("models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf")
    cfg = TraceConfig(model_path=model, backend_dir=Path("vendor/llama.cpp/bin"), n_tokens=8)
    out = run_trace(cfg)
    p = Path(out["out_path"])
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "analysis" in data
    assert data["simulated"] is True
    assert data["analysis"]["total_traces"] > 0
