from __future__ import annotations

from pathlib import Path

from vramopt.profiles import load_profile, save_profile


def test_save_profile_is_round_trippable_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "profiles" / "qwen.json"
    payload = {
        "schema_version": 1,
        "model_sha256": "abc123",
        "candidate": {"id": "dense-ffn-32-q8_0", "n_cpu_ffn": 32},
    }

    save_profile(path, payload)

    assert load_profile(path) == payload
    assert list(path.parent.glob("*.tmp")) == []
