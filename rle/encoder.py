#!/usr/bin/env python3
"""
RLE (Run-Length Encoding) Compression
=====================================
Scans text and replaces runs of the same character (length >= 3)
with a special escape byte + count + character.

Escape byte: \\x00 (null byte)
Format per run:   \\x00 + count_byte + char_byte
Literal chars:    emitted directly (no escaping)
Escape literal:   \\x00 in input → \\x00 + \\x01 + \\x00
Runs > 255:        split into multiple escape sequences
"""

import sys
from pathlib import Path

# Ensure the project base is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

logger = get_logger("rle")

# ── Constants ───────────────────────────────────────────────────────────────

ESCAPE_BYTE = 0x00          # Null byte used as escape marker
MIN_RUN_LENGTH = 3           # Minimum run length to trigger compression
MAX_COUNT = 255              # Max count in one escape sequence


class RLEEncoder(CompressionEncoder):
    """
    Run-Length Encoding compressor.

    Encodes runs of >= 3 identical consecutive characters as:
        ESCAPE_BYTE (0x00) + count (1 byte) + character (1 byte)

    A literal null byte in the input is encoded as:
        ESCAPE_BYTE + 0x01 + ESCAPE_BYTE
    """

    def algorithm_name(self) -> str:
        return "rle"

    def compress(self, text: str) -> CompressionResult:
        """
        Compress a string using RLE.

        Args:
            text: The input string to compress.

        Returns:
            CompressionResult with compressed data and metadata.
        """
        original_bytes = text.encode("utf-8")
        original_size = len(original_bytes)

        if original_size == 0:
            return CompressionResult(
                algorithm=self.algorithm_name(),
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        result = bytearray()
        i = 0
        n = len(original_bytes)

        while i < n:
            ch = original_bytes[i]

            # Count how many times this byte repeats consecutively
            run_end = i + 1
            while run_end < n and original_bytes[run_end] == ch:
                run_end += 1

            run_len = run_end - i

            if ch == ESCAPE_BYTE:
                # Escape byte in input → always encode with count=1
                # regardless of run length (each null gets its own escape)
                result.append(ESCAPE_BYTE)
                result.append(1)
                result.append(ESCAPE_BYTE)
                i += 1
                # If there's a run of nulls, we handle each individually
                # (the loop will pick up the next one)
            elif run_len >= MIN_RUN_LENGTH:
                # Compress the run, splitting if > MAX_COUNT
                remaining = run_len
                while remaining > 0:
                    count = min(remaining, MAX_COUNT)
                    result.append(ESCAPE_BYTE)
                    result.append(count)
                    result.append(ch)
                    remaining -= count
                i = run_end
            else:
                # Short run: emit literal bytes
                for _ in range(run_len):
                    result.append(ch)
                i = run_end

        compressed = bytes(result)
        logger.debug(
            "RLE: %d → %d bytes (ratio %.2f%%)",
            original_size,
            len(compressed),
            len(compressed) / original_size * 100 if original_size else 0,
        )

        return CompressionResult(
            algorithm=self.algorithm_name(),
            original_size=original_size,
            compressed_data=compressed,
            header_size=0,
            extra_info={
                "escape_byte": f"0x{ESCAPE_BYTE:02x}",
                "min_run_length": MIN_RUN_LENGTH,
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("rle", RLEEncoder)
