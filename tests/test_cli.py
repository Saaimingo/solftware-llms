from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path

import pytest

from vramopt.benchmark import LlamaMetrics
from vramopt.cli import main
from vramopt.hardware import GpuInfo, SystemInfo
from vramopt.planner import TuningCandidate, TuningRequest
from vramopt.tuner import CandidateRunResult


def _tiny_gguf(path: Path) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = [
        string("general.name") + struct.pack("<I", 8) + string("Tiny"),
        string("general.architecture") + struct.pack("<I", 8) + string("test"),
        string("test.block_count") + struct.pack("<II", 4, 2),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)) + b"".join(metadata))


def _system() -> SystemInfo:
    return SystemInfo(
        os_name="Windows",
        os_version="11",
        cpu_name="AMD Ryzen 7 5700",
        logical_cpus=16,
        ram_total_bytes=40 * 2**30,
        ram_available_bytes=30 * 2**30,
        gpus=(
            GpuInfo(
                name="NVIDIA GeForce RTX 3060",
                memory_total_mib=12_288,
                memory_free_mib=11_000,
                driver_version="616.56",
                pcie_gen_current=1,
                pcie_gen_max=3,
                pcie_width_current=16,
                pcie_width_max=16,
                power_limit_w=170.0,
            ),
        ),
    )


def test_doctor_json_reports_hardware(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["doctor", "--json"], system_provider=_system)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["cpu_name"] == "AMD Ryzen 7 5700"
    assert payload["gpus"][0]["memory_total_mib"] == 12_288


def test_inspect_json_reads_gguf_and_hash(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    _tiny_gguf(model)

    exit_code = main(["inspect", str(model), "--sha256", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["name"] == "Tiny"
    assert payload["block_count"] == 2
    assert len(payload["sha256"]) == 64


def test_help_exposes_complete_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    output = capsys.readouterr().out
    assert raised.value.code == 0
    for command in ("doctor", "inspect", "tune", "run", "serve"):
        assert command in output


def test_tune_saves_report_and_reusable_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    model = tmp_path / "tiny.gguf"
    _tiny_gguf(model)
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "llama-cli.exe").write_bytes(b"")
    output_dir = tmp_path / "artifacts"

    def fake_candidate_runner(
        candidate: TuningCandidate, **_: object
    ) -> CandidateRunResult:
        strategy = candidate.strategy
        identifier = candidate.id
        speed = 4.0 if strategy == "ffn" else 3.0
        return CandidateRunResult(
            candidate_id=identifier,
            strategy=strategy,
            success=True,
            returncode=0,
            duration_seconds=1.0,
            gpu_baseline_mib=100,
            gpu_peak_mib=10_000,
            gpu_peak_delta_mib=9_900,
            metrics=LlamaMetrics(
                prompt_tokens_per_second=12.0,
                generation_tokens_per_second=speed,
            ),
            args=("llama-cli.exe",),
        )

    exit_code = main(
        [
            "tune",
            str(model),
            "--backend-dir",
            str(backend),
            "--output-dir",
            str(output_dir),
            "--predict",
            "4",
            "--offload-count",
            "1",
        ],
        system_provider=_system,
        candidate_runner=fake_candidate_runner,
        backend_version_provider=lambda _: "llama.cpp test-build",
    )

    summary = json.loads(capsys.readouterr().out)
    report = Path(summary["report"])
    profile = Path(summary["profile"])
    assert exit_code == 0
    assert summary["winner"]["strategy"] == "ffn"
    assert report.is_file()
    assert profile.is_file()
    assert profile.name.endswith("-ctx4096.json")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(report_payload["candidates"]) == 3
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    assert profile_payload["winner_candidate"]["strategy"] == "ffn"
    assert profile_payload["safety"]["meets_requested_margin"] is True


def test_run_and_serve_dry_run_reuse_winner_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    model = tmp_path / "model.gguf"
    _tiny_gguf(model)
    backend = tmp_path / "backend"
    backend.mkdir()
    for executable in ("llama-cli.exe", "llama-server.exe"):
        (backend / executable).write_bytes(b"")
    candidate = TuningCandidate(
        id="dense-ffn-32-q8_0",
        strategy="ffn",
        request=TuningRequest(context_size=4_096, fit_margin_mib=1_024, threads=8),
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        n_cpu_ffn=32,
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
                "winner_candidate": asdict(candidate),
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                str(model),
                "--profile",
                str(profile),
                "--backend-dir",
                str(backend),
                "--dry-run",
            ]
        )
        == 0
    )
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["command"][0].endswith("llama-cli.exe")
    assert "--n-cpu-ffn" in run_payload["command"]
    assert "--single-turn" not in run_payload["command"]

    assert (
        main(
            [
                "serve",
                str(model),
                "--profile",
                str(profile),
                "--backend-dir",
                str(backend),
                "--port",
                "9090",
                "--dry-run",
            ]
        )
        == 0
    )
    serve_payload = json.loads(capsys.readouterr().out)
    assert serve_payload["command"][0].endswith("llama-server.exe")
    assert serve_payload["command"][-4:] == ["--host", "127.0.0.1", "--port", "9090"]
