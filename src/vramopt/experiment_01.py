"""
Experiment 01 runner — non-destructive sampling with 4-controls.

Implements the mandatory sampling expansion:
- hubs 17 and 42 (co-activated bloc [17,42] ratio 1.0)
- two non-hub experts reproducibly selected via select_non_hub_experts
- non-MoE control attn_q
All tensors use proxy activations (marked) and produce cost-benefit + baselines.
Does NOT touch runtime/CUDA kernels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from vramopt.quant_damage import (
    analyze_quant_damage,
    compare_residuals,
    extract_gguf_tensor_as_f32,
    select_non_hub_experts,
)


def wait_for_file_stable(path: Path, *, stable_secs: int = 5, poll: float = 2.0, timeout: int = 1200) -> None:
    start = time.time()
    last_size = -1
    stable_since = None
    while True:
        if not path.exists():
            if time.time() - start > timeout:
                raise TimeoutError(f"timeout waiting for {path}")
            time.sleep(poll)
            continue
        size = path.stat().st_size
        if size == last_size:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since >= stable_secs:
                return
        else:
            last_size = size
            stable_since = None
        if time.time() - start > timeout:
            raise TimeoutError(f"timeout waiting for {path} to stabilize")
        time.sleep(poll)


HUBS = [17, 42]
N_EXPERTS = 256  # blk.3.ffn_gate_exps has 256 experts per layer
BLOCK = 3


def _expert_sampling_plan() -> list[int]:
    non_hubs = select_non_hub_experts(n_experts=N_EXPERTS, hubs=HUBS, n_non_hub=2)
    # plan: hubs first, then non-hubs
    return HUBS + non_hubs


def run_once(
    *,
    reference_gguf: Path,
    quantized_gguf: Path,
    out_dir: Path,
    base_bpw: float = 2.98,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = _expert_sampling_plan()
    print(f"Sampling plan experts: hubs {HUBS} + non-hubs {plan[2:]}  block {BLOCK}")
    generated: list[Path] = []
    # Tensors: 4 MoE experts + 1 MoE average proxy? + 1 non-MoE control
    tensors = [(f"blk.{BLOCK}.ffn_gate_exps.weight", e, f"expert_{e}") for e in plan] + [
        (f"blk.{BLOCK}.attn_q.weight", None, "attn_q_control")
    ]
    # Ensure reference file stable
    print(f"Waiting for reference {reference_gguf} to stabilize...")
    wait_for_file_stable(reference_gguf)
    print(f"Reference ready: {reference_gguf.stat().st_size/1e9:.2f} GB")
    residuals = {}
    for tensor_name, slice_idx, label in tensors:
        print(f"\n== {label}  {tensor_name}[{slice_idx}] ==")
        try:
            ref = extract_gguf_tensor_as_f32(reference_gguf, tensor_name, slice_index=slice_idx)
            q = extract_gguf_tensor_as_f32(quantized_gguf, tensor_name, slice_index=slice_idx)
        except Exception as exc:  # noqa: BLE001 - non-destructive sampling must not crash on one tensor
            print(f"  SKIP {label}: {exc}")
            continue
        print(f"  shapes {ref.shape}  ref mean {ref.mean():.4f} std {ref.std():.4f}  q mean {q.mean():.4f}")
        report = analyze_quant_damage(ref, q, base_bpw=base_bpw, use_proxy_activations=True, proxy_batch=32)
        # collect residual for similarity matrix
        residuals[label] = ref - q
        # Cost-benefit top 5 rows
        print(f"  activation_provenance: {report.activation_provenance}")
        print(f"  mse {report.metrics.mse:.6f} rmse {report.metrics.rmse:.6f}  residual/orig {report.metrics.residual_to_original_frobenius:.4f}")
        # functional vs energy for rank 8 as example
        if 8 in report.low_rank.activation_error_recovered_by_rank:
            fr = report.low_rank.activation_error_recovered_by_rank[8]
            er = report.low_rank.energy_by_rank[8]
            print(f"  rank8 energy_recovered {er:.3f}  functional_recovered {fr:.3f}  delta_bpw {report.low_rank.delta_bpw_by_rank[8]:.3f}")
        # sparse 1% functional
        if 0.01 in report.sparse:
            sr = report.sparse[0.01]
            print(f"  sparse 1% energy {sr.energy_recovered:.3f} func {sr.activation_error_recovered:.3f} delta_bpw {sr.delta_bpw:.3f} bytes {sr.memory_bytes}")
        # cost-benefit best functional under 0.5 bpw
        cheap = [r for r in report.cost_benefit if r.delta_bpw <= 0.5]
        if cheap:
            best = max(cheap, key=lambda r: r.functional_recovered or 0)
            print(f"  best @ <=0.5bpw: {best.method} {best.config} func {best.functional_recovered:.3f} energy {best.energy_recovered:.3f}")
        # baseline dominance check
        q3_bpws = next(b.delta_bpw for b in report.mixed_precision_baselines if b.name == "Q2->Q3")
        q4_bpws = next(b.delta_bpw for b in report.mixed_precision_baselines if b.name == "Q2->Q4")
        print(f"  baselines Q3 delta {q3_bpws:.3f}  Q4 delta {q4_bpws:.3f}")
        # save per-tensor report
        out_path = out_dir / f"exp01_{label}.json"
        payload = {
            "label": label,
            "tensor": tensor_name,
            "slice_index": slice_idx,
            "plan": plan,
            "hubs": HUBS,
            "analysis": report.to_dict(),
            "cost_benefit": [r.__dict__ for r in report.cost_benefit],
            "baselines": [b.__dict__ for b in report.mixed_precision_baselines],
            "provenance": {
                "reference_gguf": str(reference_gguf),
                "quantized_gguf": str(quantized_gguf),
                "origin": "MXFP4->Q2 requantization (not BF16->Q2)",
                "n_experts": N_EXPERTS,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  saved {out_path}")
        generated.append(out_path)

    # pairwise similarity among residuals
    if len(residuals) >= 2:
        labels = list(residuals.keys())
        print("\n== Residual similarity (hub vs non-hub) ==")
        sim_path = out_dir / "exp01_similarity.json"
        sims = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                try:
                    s = compare_residuals(residuals[a], residuals[b])
                    sims[f"{a}__vs__{b}"] = s.__dict__
                    print(f"  {a} vs {b}: cosine {s.cosine_similarity:.3f} left {s.principal_left_similarity:.3f} spectrum {s.singular_spectrum_similarity:.3f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  similarity {a} vs {b} failed: {exc}")
        sim_path.write_text(json.dumps(sims, indent=2), encoding="utf-8")
        generated.append(sim_path)
    return generated


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run quantization damage experiment 01 with 4 controls")
    p.add_argument("--reference", type=Path, default=Path("D:/Models/Qwen3.6-35B-A3B-Q2_K-mxfp4-reference-f16.gguf"))
    p.add_argument("--quantized", type=Path, default=Path("D:/Models/Qwen3.6-35B-A3B-Q2_K.gguf"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/quant_damage"))
    args = p.parse_args()
    paths = run_once(reference_gguf=args.reference, quantized_gguf=args.quantized, out_dir=args.out_dir)
    print(f"\nDone. Generated {len(paths)} files:")
    for pp in paths:
        print(f"  {pp}")
