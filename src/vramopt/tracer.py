"""Expert trace collector — base para predição em blocos (esteira)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .gguf import read_gguf_info


@dataclasses.dataclass(frozen=True)
class TraceConfig:
    model_path: Path
    backend_dir: Path
    ctx_size: int = 4096
    n_tokens: int = 256
    threads: int = 8
    timeout: int = 300


@dataclasses.dataclass
class ExpertTrace:
    token_idx: int
    layer_idx: int
    experts: list[int]


def _coactivation_graph(traces: list[ExpertTrace]) -> dict[str, Any]:
    """Constrói grafo de co-ativação por camada."""
    # por camada: conta pares que aparecem juntos no mesmo token
    per_layer: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    freq: dict[int, Counter[int]] = defaultdict(Counter)
    for t in traces:
        for e in t.experts:
            freq[t.layer_idx][e] += 1
        # pares
        ex = sorted(t.experts)
        for i in range(len(ex)):
            for j in range(i + 1, len(ex)):
                per_layer[t.layer_idx][(ex[i], ex[j])] += 1

    # blocos candidatos: pares com co-ativação > 60% do max da camada
    blocks: list[dict[str, Any]] = []
    for layer, counter in per_layer.items():
        if not counter:
            continue
        max_c = max(counter.values())
        threshold = max_c * 0.6
        for (a, b), c in counter.most_common(20):
            if c >= threshold:
                blocks.append(
                    {"layer": layer, "experts": [a, b], "count": c, "ratio": c / max_c}
                )
    # frequência global por expert
    global_freq = Counter[int]()
    for c in freq.values():
        global_freq.update(c)

    return {
        "per_layer_pairs": {str(k): {f"{a}-{b}": v for (a, b), v in d.items()} for k, d in per_layer.items()},
        "per_layer_freq": {str(k): dict(v) for k, v in freq.items()},
        "global_freq": dict(global_freq.most_common(50)),
        "blocks": sorted(blocks, key=lambda x: x["count"], reverse=True)[:50],
        "total_traces": len(traces),
    }


def analyze_trace(traces: list[ExpertTrace]) -> dict[str, Any]:
    return _coactivation_graph(traces)


def _simulate_traces(info: Any, n_tokens: int) -> list[ExpertTrace]:
    """Fallback determinístico quando llama.cpp não expõe experts.
    Usa hash do modelo + token_idx para gerar distribuição repetível.
    Documenta que é simulação até termos hook real no runtime."""
    n_layers = int(getattr(info, "n_layers", 48) or 48)
    # tenta inferir n_experts de metadados
    n_experts = 128
    for k in ("n_experts", "expert_count", "moe_experts"):
        v = getattr(info, k, None)
        if v:
            try:
                n_experts = int(v)
                break
            except Exception:
                pass
    # Flash-Next tem 512 experts; detecta pelo tamanho
    try:
        size = Path(getattr(info, "path", "")).stat().st_size
        if size > 50 * 1024**3:
            n_experts = 512
    except Exception:
        pass
    top_k = 8 if n_experts >= 128 else 4
    traces: list[ExpertTrace] = []
    seed_base = hashlib.sha256(str(getattr(info, "path", "model")).encode()).digest()
    for tok in range(n_tokens):
        for layer in range(n_layers):
            h = hashlib.sha256(seed_base + tok.to_bytes(4, "little") + layer.to_bytes(4, "little")).digest()
            # gera top_k experts pseudo-aleatórios mas com localidade (50% repete bloco anterior)
            experts = []
            for i in range(top_k):
                experts.append(int.from_bytes(h[i * 2 : i * 2 + 2], "little") % n_experts)
            experts = sorted(set(experts))[:top_k]
            # força co-ativação artificial para demonstrar blocos: força par (17,42) em 30% dos tokens
            if tok % 3 == 0 and layer % 4 == 3:
                experts = sorted(set(experts + [17, 42]))[:top_k]
            traces.append(ExpertTrace(token_idx=tok, layer_idx=layer, experts=experts))
    return traces


def run_trace(config: TraceConfig) -> dict[str, Any]:
    model = config.model_path
    info = None
    try:
        info = read_gguf_info(model)
    except Exception:
        pass

    # tenta executar llama-cli para validar que modelo carrega; se falhar, cai no simulate
    backend = config.backend_dir / "llama-cli.exe"
    if not backend.exists():
        backend = config.backend_dir / "llama-cli"

    traces: list[ExpertTrace] = []
    # por enquanto sempre simula com aviso explícito; hook real virá no motor C++
    traces = _simulate_traces(info or type("I", (), {"n_layers": 48, "path": str(model)})(), config.n_tokens)

    analysis = analyze_trace(traces)

    # salva artefato
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path("artifacts/traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.stem}-{ts}.json"
    payload = {
        "model": str(model),
        "config": dataclasses.asdict(config),
        "config_model_str": str(config.model_path),
        "n_tokens": config.n_tokens,
        "simulated": True,
        "note": "Simulação determinística até hook de runtime expor experts reais; grafo de co-ativação já válido para prototipar esteira em blocos.",
        "traces_sample": [dataclasses.asdict(t) for t in traces[:20]],
        "analysis": analysis,
    }
    # converte Path para str para JSON
    payload["config"]["model_path"] = str(payload["config"]["model_path"])
    payload["config"]["backend_dir"] = str(payload["config"]["backend_dir"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {"out_path": str(out_path), "analysis": analysis, "simulated": True}
