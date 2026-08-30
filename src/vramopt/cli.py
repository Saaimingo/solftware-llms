"""Command-line interface for hardware-aware GGUF tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vramopt import __version__
from vramopt.backend import run_command
from vramopt.gguf import GGUFInfo, read_gguf_info
from vramopt.hardware import SystemInfo, collect_system_info
from vramopt.planner import (
    TuningCandidate,
    TuningRequest,
    build_candidates,
    candidate_from_payload,
)
from vramopt.profiles import load_profile, save_profile
from vramopt.quant_damage import analyze_quant_damage
from vramopt.tracer import TraceConfig, run_trace
from vramopt.tuner import CandidateRunResult, choose_best, run_candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_payload(info: GGUFInfo, *, sha256: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(info.path),
        "size_bytes": info.size_bytes,
        "version": info.version,
        "tensor_count": info.tensor_count,
        "metadata_count": info.metadata_count,
        "name": info.name,
        "architecture": info.architecture,
        "file_type": info.file_type,
        "quantization_version": info.quantization_version,
        "context_length": info.context_length,
        "block_count": info.block_count,
        "expert_count": info.expert_count,
        "expert_used_count": info.expert_used_count,
        "is_moe": info.is_moe,
    }
    if sha256 is not None:
        payload["sha256"] = sha256
    return payload


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _backend_version(executable: Path) -> str:
    result = run_command([str(executable), "--version"], timeout_seconds=30)
    if result.returncode != 0:
        raise RuntimeError(f"backend version check failed: {result.stderr.strip()}")
    return result.stdout.strip() or result.stderr.strip()


def _candidate_payload(candidate: TuningCandidate) -> dict[str, object]:
    return asdict(candidate)


def _verified_profile_candidate(profile_path: Path, model_path: Path) -> TuningCandidate:
    profile = load_profile(profile_path)
    model_payload = profile.get("model")
    if not isinstance(model_payload, Mapping):
        raise TypeError("profile model must be an object")
    expected_hash = model_payload.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("profile model SHA-256 is missing or invalid")
    actual_hash = _sha256_file(model_path)
    if actual_hash != expected_hash:
        raise ValueError("profile does not match the selected model SHA-256")
    candidate_payload = profile.get("winner_candidate")
    if not isinstance(candidate_payload, Mapping):
        raise TypeError("profile winner_candidate must be an object")
    return candidate_from_payload(candidate_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vramopt",
        description="Tune GGUF placement for GPUs with constrained VRAM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect local hardware")
    doctor.add_argument("--json", action="store_true")

    inspect = commands.add_parser("inspect", help="inspect GGUF metadata")
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--sha256", action="store_true")
    inspect.add_argument("--json", action="store_true")

    tune = commands.add_parser("tune", help="benchmark and save the best profile")
    tune.add_argument("model", type=Path)
    tune.add_argument("--backend-dir", type=Path, default=Path("vendor/llama.cpp/bin"))
    tune.add_argument("--context", type=int, default=4_096)
    tune.add_argument("--margin-mib", type=int, default=1_024)
    tune.add_argument("--threads", type=int, default=8)
    tune.add_argument("--predict", type=int, default=32)
    tune.add_argument("--timeout", type=float, default=600.0)
    tune.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    tune.add_argument(
        "--offload-count",
        action="append",
        type=int,
        dest="offload_counts",
        help="FFN/expert count to test; repeat to refine the search",
    )

    run = commands.add_parser("run", help="start an interactive model session")
    run.add_argument("model", type=Path)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--backend-dir", type=Path, default=Path("vendor/llama.cpp/bin"))
    run.add_argument("--dry-run", action="store_true")

    serve = commands.add_parser("serve", help="start a local OpenAI-compatible server")
    serve.add_argument("model", type=Path)
    serve.add_argument("--profile", type=Path, required=True)
    serve.add_argument("--backend-dir", type=Path, default=Path("vendor/llama.cpp/bin"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8_080)
    serve.add_argument("--dry-run", action="store_true")

    trace = commands.add_parser("trace", help="collect expert co-activation trace")
    trace.add_argument("model", type=Path)
    trace.add_argument("--backend-dir", type=Path, default=Path("vendor/llama.cpp/bin"))
    trace.add_argument("--context", type=int, default=4096)
    trace.add_argument("--tokens", type=int, default=256)
    trace.add_argument("--threads", type=int, default=8)
    trace.add_argument("--output-dir", type=Path, default=Path("artifacts/traces"))

    damage = commands.add_parser(
        "quant-damage", help="scientific residual analysis for reference and quantized tensor samples"
    )
    damage.add_argument("--reference", type=Path, required=True, help="reference tensor .npy")
    damage.add_argument("--quantized", type=Path, required=True, help="quantized/dequantized tensor .npy")
    damage.add_argument("--activations", type=Path, help="optional input activations .npy")
    damage.add_argument("--model", required=True)
    damage.add_argument("--tensor", required=True)
    damage.add_argument("--origin", required=True)
    damage.add_argument("--reference-dtype", default="unknown")
    damage.add_argument("--quantization", default="Q2")
    damage.add_argument("--base-bpw", type=float, default=2.0)
    damage.add_argument("--output-dir", type=Path, default=Path("artifacts/quant_damage"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    system_provider: Callable[[], SystemInfo] = collect_system_info,
    candidate_runner: Callable[..., CandidateRunResult] = run_candidate,
    backend_version_provider: Callable[[Path], str] = _backend_version,
    runtime_runner: Callable[..., int] = subprocess.call,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        system_info = system_provider()
        payload = asdict(system_info)
        if args.json:
            _print_json(payload)
        else:
            print(f"CPU: {system_info.cpu_name} ({system_info.logical_cpus} threads)")
            print(f"RAM: {system_info.ram_total_bytes / 2**30:.2f} GiB")
            for gpu in system_info.gpus:
                print(f"GPU: {gpu.name} ({gpu.memory_total_mib} MiB)")
        return 0
    if args.command == "inspect":
        model_info = read_gguf_info(args.model)
        checksum = _sha256_file(model_info.path) if args.sha256 else None
        payload = _gguf_payload(model_info, sha256=checksum)
        if args.json:
            _print_json(payload)
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    if args.command == "tune":
        model_info = read_gguf_info(args.model)
        executable = args.backend_dir / "llama-cli.exe"
        if not executable.is_file():
            raise FileNotFoundError(f"llama-cli executable not found: {executable}")
        request = TuningRequest(
            context_size=args.context,
            fit_margin_mib=args.margin_mib,
            threads=args.threads,
        )
        candidates = build_candidates(
            model_info, request, offload_counts=args.offload_counts
        )
        created = datetime.now(UTC)
        created_at = created.replace(microsecond=0).isoformat()
        run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{model_info.path.stem}"
        log_dir = args.output_dir / "logs" / run_id
        results: list[CandidateRunResult] = []
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"[{index}/{len(candidates)}] {candidate.id}",
                file=sys.stderr,
                flush=True,
            )
            result = candidate_runner(
                candidate,
                executable=executable,
                model=model_info.path,
                predict=args.predict,
                timeout_seconds=args.timeout,
                log_path=log_dir / f"{candidate.id}.log",
            )
            results.append(result)
        system_info = system_provider()
        if not system_info.gpus:
            raise RuntimeError("tuning requires an NVIDIA GPU")
        gpu_total_mib = system_info.gpus[0].memory_total_mib
        max_gpu_peak_mib = gpu_total_mib - request.fit_margin_mib
        fastest = choose_best(results)
        winner = choose_best(results, max_gpu_peak_mib=max_gpu_peak_mib)
        if winner.gpu_peak_mib is None:
            raise RuntimeError("winner is missing GPU memory telemetry")
        measured_free_mib = gpu_total_mib - winner.gpu_peak_mib
        safety_payload: dict[str, object] = {
            "gpu_total_mib": gpu_total_mib,
            "maximum_peak_mib": max_gpu_peak_mib,
            "measured_peak_mib": winner.gpu_peak_mib,
            "measured_free_mib": measured_free_mib,
            "requested_margin_mib": request.fit_margin_mib,
            "meets_requested_margin": measured_free_mib >= request.fit_margin_mib,
        }
        winner_candidate = next(
            candidate for candidate in candidates if candidate.id == winner.candidate_id
        )
        model_payload = _gguf_payload(
            model_info,
            sha256=_sha256_file(model_info.path),
        )
        hardware_payload = asdict(system_info)
        backend_payload: dict[str, object] = {
            "executable": str(executable.resolve()),
            "version": backend_version_provider(executable),
        }
        report: dict[str, object] = {
            "schema_version": 1,
            "created_at": created_at,
            "model": model_payload,
            "hardware": hardware_payload,
            "backend": backend_payload,
            "request": asdict(request),
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
            "results": [result.to_dict() for result in results],
            "fastest_result": fastest.to_dict(),
            "winner_candidate": _candidate_payload(winner_candidate),
            "winner_result": winner.to_dict(),
            "safety": safety_payload,
        }
        report_path = args.output_dir / "benchmarks" / f"{run_id}.json"
        profile_path = (
            args.output_dir
            / "profiles"
            / f"{model_info.path.stem}-ctx{request.context_size}.json"
        )
        save_profile(report_path, report)
        profile: dict[str, object] = {
            "schema_version": 1,
            "created_at": created_at,
            "model": model_payload,
            "hardware": hardware_payload,
            "backend": backend_payload,
            "winner_candidate": _candidate_payload(winner_candidate),
            "winner_result": winner.to_dict(),
            "safety": safety_payload,
            "report": str(report_path.resolve()),
        }
        save_profile(profile_path, profile)
        _print_json(
            {
                "profile": str(profile_path.resolve()),
                "report": str(report_path.resolve()),
                "winner": winner.to_dict(),
                "safety": safety_payload,
            }
        )
        return 0
    if args.command in {"run", "serve"}:
        model_path = args.model.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        candidate = _verified_profile_candidate(args.profile, model_path)
        executable_name = "llama-cli.exe" if args.command == "run" else "llama-server.exe"
        executable = (args.backend_dir / executable_name).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"backend executable not found: {executable}")
        if args.command == "run":
            command = candidate.to_interactive_args(executable, model_path)
        else:
            command = candidate.to_server_args(
                executable,
                model_path,
                host=args.host,
                port=args.port,
            )
        if args.dry_run:
            _print_json({"command": command})
            return 0
        return runtime_runner(command, shell=False)
    if args.command == "quant-damage":
        reference = np.load(args.reference, allow_pickle=False)
        quantized = np.load(args.quantized, allow_pickle=False)
        activations = np.load(args.activations, allow_pickle=False) if args.activations else None
        analysis = analyze_quant_damage(
            reference,
            quantized,
            activations=activations,
            base_bpw=args.base_bpw,
        )
        created = datetime.now(UTC)
        report_path = args.output_dir / (
            f"{created.strftime('%Y%m%dT%H%M%SZ')}-{args.model}-{args.tensor}.json"
        )
        damage_report: dict[str, object] = {
            "schema_version": 1,
            "created_at": created.replace(microsecond=0).isoformat(),
            "provenance": {
                "model": args.model,
                "tensor": args.tensor,
                "reference_path": str(args.reference.resolve()),
                "quantized_path": str(args.quantized.resolve()),
                "activations_path": str(args.activations.resolve()) if args.activations else None,
                "reference_dtype": args.reference_dtype,
                "quantization": args.quantization,
                "origin": args.origin,
                "reference_sha256": _sha256_file(args.reference),
                "quantized_sha256": _sha256_file(args.quantized),
            },
            "analysis": analysis.to_dict(),
            "limitations": [
                "This is a non-destructive numeric experiment; no runtime or CUDA kernel is changed.",
                "Recovered residual energy is not a measure of intelligence or benchmark quality.",
            ],
        }
        save_profile(report_path, damage_report)
        _print_json({"report_path": str(report_path.resolve()), "analysis": analysis.to_dict()})
        return 0
    if args.command == "trace":
        cfg = TraceConfig(
            model_path=args.model,
            backend_dir=args.backend_dir,
            ctx_size=args.context,
            n_tokens=args.tokens,
            threads=args.threads,
        )
        trace_result = run_trace(cfg)
        _print_json(trace_result)
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
