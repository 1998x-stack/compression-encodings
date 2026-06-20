#!/usr/bin/env python3
"""
Compression Encodings — Shared Abstract Base
=============================================
All encoding modules inherit from CompressionEncoder.
Unified Result dataclass and common utilities.
"""

import logging
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Logging ─────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).resolve().parent.parent / "compression.log"


def get_logger(name: str) -> logging.Logger:
    """Create a logger for an encoding module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


log = get_logger("base")


# ── Shared Result ───────────────────────────────────────────────────────────


@dataclass
class CompressionResult:
    """Unified result from any compression encoder."""

    algorithm: str
    original_size: int                      # bytes (UTF-8)
    compressed_data: bytes                  # raw compressed bytes
    header_size: int = 0                    # metadata overhead
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def payload_size(self) -> int:
        return len(self.compressed_data) - self.header_size

    @property
    def total_size(self) -> int:
        return len(self.compressed_data)

    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.total_size / self.original_size

    def summary(self) -> str:
        lines = [
            f"  Algorithm:             {self.algorithm}",
            f"  Original UTF-8 size:   {self.original_size:>10d} bytes",
            f"  Header overhead:       {self.header_size:>10d} bytes",
            f"  Payload:               {self.payload_size:>10d} bytes",
            f"  Total compressed:      {self.total_size:>10d} bytes",
            f"  Compression ratio:     {self.compression_ratio:>10.2%}",
            f"  Space saved:           {self.original_size - self.total_size:>10d} bytes",
        ]
        for k, v in self.extra_info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── Abstract Base ───────────────────────────────────────────────────────────


class CompressionEncoder(ABC):
    """
    Abstract base for all compression encoders.
    Subclasses implement compress(text) -> CompressionResult.
    """

    @abstractmethod
    def algorithm_name(self) -> str:
        ...

    @abstractmethod
    def compress(self, text: str) -> CompressionResult:
        ...

    def serialize_header(self, obj: Any) -> bytes:
        """Pickle any Python object as header bytes."""
        return pickle.dumps(obj)

    @staticmethod
    def bitstring_to_bytes(bitstring: str) -> tuple[bytes, int]:
        """Pack a bit-string into bytes. Returns (bytes, padding_bits)."""
        padding = (8 - len(bitstring) % 8) % 8
        padded = bitstring + "0" * padding
        result = bytearray()
        for i in range(0, len(padded), 8):
            result.append(int(padded[i:i + 8], 2))
        return bytes(result), padding


# ── Factory ─────────────────────────────────────────────────────────────────


class EncoderFactory:
    """
    Unified factory for all compression encoders.
    Each module calls register() from its own code.
    """

    _registry: dict[str, type[CompressionEncoder]] = {}

    @classmethod
    def register(cls, key: str, encoder_cls: type[CompressionEncoder]):
        if key in cls._registry:
            log.warning("Overwriting existing encoder: %s -> %s", key, encoder_cls.__name__)
        else:
            log.info("Registered encoder: %s -> %s", key, encoder_cls.__name__)
        cls._registry[key] = encoder_cls

    @classmethod
    def list_strategies(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def create(cls, key: str) -> CompressionEncoder:
        if key not in cls._registry:
            raise KeyError(f"Unknown encoder '{key}'. Available: {cls._registry.keys()}")
        return cls._registry[key]()
