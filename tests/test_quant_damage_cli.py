from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vramopt.cli import main


def test_quant_damage_cli_saves_non_destructive_json(tmp_path: Path, capsys: object) -> None:
    reference = tmp_path / "reference.npy"
    quantized = tmp_path / "quantized.npy"
    np.save(reference, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.save(quantized, np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32))

    exit_code = main(
        [
            "quant-damage",
            "--reference",
            str(reference),
            "--quantized",
            str(quantized),
            "--model",
            "synthetic-model",
            "--tensor",
            "synthetic.tensor",
            "--origin",
            "synthetic FP32 -> Q2",
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    report = Path(payload["report_path"])
    assert report.is_file()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["provenance"]["tensor"] == "synthetic.tensor"
    assert saved["analysis"]["metrics"]["mse"] == 0.5
