from __future__ import annotations

import struct
from pathlib import Path

import pytest

from vramopt.gguf import GGUFFormatError, read_gguf_info


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _metadata_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<II", 4, value)


def _write_tiny_qwen_moe(path: Path) -> None:
    metadata = [
        _metadata_string("general.name", "Tiny Qwen MoE"),
        _metadata_string("general.architecture", "qwen35moe"),
        _metadata_u32("general.file_type", 15),
        _metadata_u32("general.quantization_version", 2),
        _metadata_u32("qwen35moe.context_length", 262_144),
        _metadata_u32("qwen35moe.block_count", 64),
        _metadata_u32("qwen35moe.expert_count", 64),
        _metadata_u32("qwen35moe.expert_used_count", 16),
    ]
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, len(metadata))
        + b"".join(metadata)
    )


def test_read_gguf_info_extracts_planning_metadata(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    _write_tiny_qwen_moe(model)

    info = read_gguf_info(model)

    assert info.version == 3
    assert info.tensor_count == 0
    assert info.metadata_count == 8
    assert info.name == "Tiny Qwen MoE"
    assert info.architecture == "qwen35moe"
    assert info.file_type == 15
    assert info.context_length == 262_144
    assert info.block_count == 64
    assert info.expert_count == 64
    assert info.expert_used_count == 16
    assert info.is_moe is True
    assert info.size_bytes == model.stat().st_size


def test_read_gguf_info_rejects_wrong_magic(tmp_path: Path) -> None:
    model = tmp_path / "bad.gguf"
    model.write_bytes(b"NOPE" + bytes(32))

    with pytest.raises(GGUFFormatError, match="magic"):
        read_gguf_info(model)
