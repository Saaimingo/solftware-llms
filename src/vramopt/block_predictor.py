"""Block predictor — esteira em blocos para superar COLIBRI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Block:
    layer: int
    experts: list[int]
    count: int
    ratio: float


@dataclass
class PrefetchPlan:
    layer: int
    predicted_blocks: list[Block]
    hub_experts: list[int]
    estimated_miss_gb: float
    estimated_tps_ceiling: float


PCIE_GBS = 12.0  # efetivo RTX 3060 PCIe 3.0 x16
BYTES_PER_EXPERT_Q4_GB = 0.02  # ~20MB por expert Q4 (6B/512 *0.5)


def load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_hubs(analysis: dict[str, Any], top_k: int = 8) -> list[int]:
    freq = analysis.get("global_freq", {})
    # keys são str
    sorted_experts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [int(k) for k, _ in sorted_experts[:top_k]]


def predict_blocks(analysis: dict[str, Any], layer: int | None = None) -> list[Block]:
    blocks = []
    for b in analysis.get("blocks", []):
        if layer is not None and b["layer"] != layer:
            continue
        blocks.append(Block(layer=b["layer"], experts=b["experts"], count=b["count"], ratio=b["ratio"]))
    return blocks


def estimate_prefetch(blocks: list[Block], hubs: list[int], hit_rate: float = 0.8) -> PrefetchPlan:
    # miss = (1 - hit_rate) * blocos * bytes por expert
    n_experts_miss = len(blocks) * 2 * (1 - hit_rate)  # cada bloco tem 2 experts
    miss_gb = n_experts_miss * BYTES_PER_EXPERT_Q4_GB
    # adiciona hubs que ficam residentes (não contam no miss)
    tps = PCIE_GBS / miss_gb if miss_gb > 0 else 999.0
    layer = blocks[0].layer if blocks else -1
    return PrefetchPlan(
        layer=layer,
        predicted_blocks=blocks[:4],
        hub_experts=hubs[:4],
        estimated_miss_gb=round(miss_gb, 4),
        estimated_tps_ceiling=round(tps, 1),
    )


def simulate_double_buffer(trace_path: Path) -> dict[str, Any]:
    data = load_trace(trace_path)
    analysis = data["analysis"]
    hubs = build_hubs(analysis)
    # pega camada 3 como exemplo (tem bloco forte [17,42])
    blocks = predict_blocks(analysis, layer=3)
    plan = estimate_prefetch(blocks, hubs, hit_rate=0.85)
    # compara sem bloco: cada token traz 8 experts soltos
    miss_no_block = 8 * BYTES_PER_EXPERT_Q4_GB * 0.5  # 50% hit rate sem predição
    tps_no_block = PCIE_GBS / miss_no_block
    return {
        "trace": str(trace_path),
        "hubs": hubs[:6],
        "blocks_layer3": [{"layer": b.layer, "experts": b.experts, "count": b.count, "ratio": b.ratio} for b in blocks[:3]],
        "with_blocks": {"layer": plan.layer, "hub_experts": plan.hub_experts, "estimated_miss_gb": plan.estimated_miss_gb, "estimated_tps_ceiling": plan.estimated_tps_ceiling, "predicted_blocks": [{"layer": b.layer, "experts": b.experts, "count": b.count, "ratio": b.ratio} for b in plan.predicted_blocks]},
        "without_blocks_tps": round(tps_no_block, 1),
        "speedup": round(plan.estimated_tps_ceiling / tps_no_block, 2) if tps_no_block else 0,
        "note": "Simulação Python do double-buffer; CUDA real usará pinned + cudaMemcpyAsync overlap",
    }
