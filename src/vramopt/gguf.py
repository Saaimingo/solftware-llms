"""Small, bounded GGUF metadata reader.

It inspects planning metadata without allocating model tensors or loading weights.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

_MAGIC = b"GGUF"
_MAX_METADATA_ITEMS = 1_000_000
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ITEMS = 10_000_000
_FIXED_TYPE_SIZES = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class GGUFFormatError(ValueError):
    """Raised when a file is not a supported, safely readable GGUF."""


@dataclass(frozen=True, slots=True)
class GGUFInfo:
    path: Path
    size_bytes: int
    version: int
    tensor_count: int
    metadata_count: int
    name: str | None
    architecture: str | None
    file_type: int | None
    quantization_version: int | None
    context_length: int | None
    block_count: int | None
    expert_count: int | None
    expert_used_count: int | None

    @property
    def is_moe(self) -> bool:
        return bool(
            (self.expert_count is not None and self.expert_count > 0)
            or (self.architecture and "moe" in self.architecture.casefold())
        )


class _Reader:
    def __init__(self, stream: BinaryIO, file_size: int) -> None:
        self.stream = stream
        self.file_size = file_size

    def read_exact(self, count: int) -> bytes:
        if count < 0 or count > self.file_size - self.stream.tell():
            raise GGUFFormatError("truncated GGUF metadata")
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFFormatError("truncated GGUF metadata")
        return data

    def unpack(self, fmt: str) -> int | float | bool:
        parser = struct.Struct(fmt)
        return cast(int | float | bool, parser.unpack(self.read_exact(parser.size))[0])

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        size = self.u64()
        if size > _MAX_STRING_BYTES:
            raise GGUFFormatError(f"GGUF string exceeds safety limit: {size} bytes")
        try:
            return self.read_exact(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFFormatError("GGUF metadata contains invalid UTF-8") from exc

    def skip(self, count: int) -> None:
        if count < 0 or count > self.file_size - self.stream.tell():
            raise GGUFFormatError("truncated GGUF metadata")
        self.stream.seek(count, os.SEEK_CUR)


def _read_value(reader: _Reader, value_type: int) -> object:
    if value_type in _SCALAR_FORMATS:
        return reader.unpack(_SCALAR_FORMATS[value_type])
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        element_type = reader.u32()
        count = reader.u64()
        if count > _MAX_ARRAY_ITEMS:
            raise GGUFFormatError(f"GGUF array exceeds safety limit: {count} items")
        fixed_size = _FIXED_TYPE_SIZES.get(element_type)
        if fixed_size is not None:
            reader.skip(count * fixed_size)
        else:
            for _ in range(count):
                _read_value(reader, element_type)
        return None
    raise GGUFFormatError(f"unsupported GGUF metadata type: {value_type}")


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def read_gguf_info(path: str | Path) -> GGUFInfo:
    """Read model-planning metadata from a GGUF file with bounded allocations."""

    model_path = Path(path)
    size_bytes = model_path.stat().st_size
    try:
        with model_path.open("rb") as stream:
            reader = _Reader(stream, size_bytes)
            if reader.read_exact(4) != _MAGIC:
                raise GGUFFormatError("invalid GGUF magic")
            version = reader.u32()
            if version not in {2, 3}:
                raise GGUFFormatError(f"unsupported GGUF version: {version}")
            tensor_count = reader.u64()
            metadata_count = reader.u64()
            if metadata_count > _MAX_METADATA_ITEMS:
                raise GGUFFormatError(
                    f"GGUF metadata count exceeds safety limit: {metadata_count}"
                )

            metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                key = reader.string()
                value_type = reader.u32()
                metadata[key] = _read_value(reader, value_type)
    except OSError as exc:
        raise GGUFFormatError(f"cannot read GGUF: {model_path}") from exc

    def suffix(name: str) -> object | None:
        return next((value for key, value in metadata.items() if key.endswith(name)), None)

    return GGUFInfo(
        path=model_path.resolve(),
        size_bytes=size_bytes,
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        name=_optional_str(metadata.get("general.name")),
        architecture=_optional_str(metadata.get("general.architecture")),
        file_type=_optional_int(metadata.get("general.file_type")),
        quantization_version=_optional_int(metadata.get("general.quantization_version")),
        context_length=_optional_int(suffix(".context_length")),
        block_count=_optional_int(suffix(".block_count")),
        expert_count=_optional_int(suffix(".expert_count")),
        expert_used_count=_optional_int(suffix(".expert_used_count")),
    )
