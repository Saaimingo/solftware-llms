from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from vramopt.quant_damage import extract_f16_tensor_slice, list_gguf_tensors


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _metadata_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _tensor_entry(name: str, shape: tuple[int, ...], ggml_type: int, offset: int) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", len(shape))
        + b"".join(struct.pack("<Q", value) for value in shape)
        + struct.pack("<IQ", ggml_type, offset)
    )


def _tiny_f16_gguf(path: Path) -> None:
    metadata = [_metadata_u32("general.alignment", 32)]
    tensor = _tensor_entry("expert.weight", (2, 3), 1, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    padding = bytes((-len(header)) % 32)
    values = np.arange(6, dtype=np.float16).tobytes()
    path.write_bytes(header + padding + values)


def test_list_gguf_tensors_and_extract_f16_matrix_slice(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    _tiny_f16_gguf(model)

    tensors = list_gguf_tensors(model)
    extracted = extract_f16_tensor_slice(model, "expert.weight", slice_index=None)

    assert len(tensors) == 1
    assert tensors[0].name == "expert.weight"
    assert tensors[0].shape == (2, 3)
    assert tensors[0].ggml_type == 1
    np.testing.assert_array_equal(extracted, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32))
