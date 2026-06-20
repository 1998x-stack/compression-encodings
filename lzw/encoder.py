#!/usr/bin/env python3
"""
LZW (Lempel-Ziv-Welch) Compression
===================================
Variable-width dictionary-based encoder with bit packing.

Registered as: 'lzw'

Algorithm:
  - Initialize dictionary with 256 single-byte strings (0..255).
  - Max dictionary size = 65536 (16-bit codes).
  - Start outputting 9-bit codes; when dict reaches 2^width entries,
    increment width (9→10→...→16 bits).
  - For each input byte: build string w. If w + c in dict, w = w + c;
    otherwise emit code for w, add w + c to dict, reset w = c.
  - After the loop, emit the remaining code for w.

Output format:
  +-- 4 bytes: number of codes (uint32 big-endian)
  +-- 2 bytes: initial dictionary size (uint16 big-endian)
  +-- packed bitstream of variable-width codes
"""

import struct
import sys
from pathlib import Path
from typing import Any

# Ensure the project base is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("lzw")

# ── Constants ───────────────────────────────────────────────────────────────

INITIAL_DICT_SIZE = 256        # 0..255 = single-byte entries
MAX_DICT_SIZE = 65536          # 2^16 maximum
INITIAL_CODE_WIDTH = 9         # Start with 9-bit codes
MAX_CODE_WIDTH = 16            # Max code width in bits


class LZWEncoder(CompressionEncoder):
    """
    LZW dictionary-based compression encoder.

    Uses variable-width bit packing: codes start at 9 bits and grow
    dynamically up to 16 bits as the dictionary expands.
    """

    def algorithm_name(self) -> str:
        return "lzw"

    # ── Core LZW encoding ──────────────────────────────────────────────

    def _build_initial_dict(self) -> dict[bytes, int]:
        """Initialize dictionary with all single-byte strings 0..255."""
        return {bytes([i]): i for i in range(256)}

    def compress(self, text: str) -> CompressionResult:
        """
        Compress a UTF-8 string using LZW encoding.

        Args:
            text: Input string to compress.

        Returns:
            CompressionResult with compressed data and metadata.
        """
        original_bytes = text.encode("utf-8")
        original_size = len(original_bytes)

        if original_size == 0:
            return CompressionResult(
                algorithm="lzw",
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        # Build initial dictionary
        dictionary = self._build_initial_dict()
        next_code = INITIAL_DICT_SIZE
        current_width = INITIAL_CODE_WIDTH

        codes: list[int] = []

        # LZW main loop
        w = bytes()  # current accumulated string

        for byte in original_bytes:
            c = bytes([byte])
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                # Emit code for w
                codes.append(dictionary[w])
                # Add w + c to dictionary (if there's room)
                if next_code < MAX_DICT_SIZE:
                    dictionary[wc] = next_code
                    next_code += 1
                    # Check if we need to grow the code width
                    if next_code >= (1 << current_width) and current_width < MAX_CODE_WIDTH:
                        current_width += 1
                        log.debug(
                            "Code width grew to %d bits at dict size %d",
                            current_width,
                            next_code,
                        )
                # Reset w to current byte
                w = c

        # Emit remaining code for w
        if w:
            codes.append(dictionary[w])

        log.info(
            "LZW: %d bytes → %d codes, dict_size=%d, final_code_width=%d bits",
            original_size,
            len(codes),
            next_code,
            current_width,
        )

        # ── Pack codes into bitstream ──────────────────────────────────

        # Recalculate widths per code since the width grows during encoding.
        # Width for code i depends on dict size at that point.
        def _width_for_code(code_index: int) -> int:
            """Determine the code width used at position `code_index`."""
            # Determine dict size at the point this code was emitted.
            # Dict starts at 256, grows by 1 for each new entry.
            # Code at position idx was emitted when dict had (256 + idx) entries
            # (since each code emission that adds a new dict entry increments).
            # But the width check happens *after* adding, so if the new dict_size
            # crossed a power-of-2 boundary, the *next* code gets the wider width.
            # So: width for code idx is based on dict_size = 256 + idx.
            # The width needed to represent codes up to (dict_size - 1).
            dict_size_at_emit = INITIAL_DICT_SIZE + code_index
            # Find the smallest width that can represent dict_size_at_emit - 1
            # Actually: codes emitted are guaranteed < dict_size_at_emit
            # Width is ceil(log2(dict_size_at_emit)), but clamped to [9, 16]
            if dict_size_at_emit <= (1 << INITIAL_CODE_WIDTH):
                return INITIAL_CODE_WIDTH
            w = INITIAL_CODE_WIDTH
            while (1 << w) < dict_size_at_emit and w < MAX_CODE_WIDTH:
                w += 1
            return w

        bitstring_parts: list[str] = []
        for idx, code in enumerate(codes):
            width = _width_for_code(idx)
            bitstring_parts.append(format(code, f"0{width}b"))

        bitstring = "".join(bitstring_parts)
        packed_bytes, padding = self.bitstring_to_bytes(bitstring)

        # ── Build header ───────────────────────────────────────────────

        # 4 bytes: number of codes (uint32 big-endian)
        # 2 bytes: initial dictionary size (uint16 big-endian)
        header = struct.pack(">IH", len(codes), INITIAL_DICT_SIZE)
        header_size = len(header)

        combined = header + packed_bytes

        num_codes = len(codes)

        log.info(
            "Header: %d B, Payload: %d B, Total: %d B, Padding: %d bits",
            header_size,
            len(packed_bytes),
            len(combined),
            padding,
        )

        # ── Identify code width range ──────────────────────────────────

        widths_used = sorted(set(_width_for_code(i) for i in range(num_codes)))

        return CompressionResult(
            algorithm="lzw",
            original_size=original_size,
            compressed_data=combined,
            header_size=header_size,
            extra_info={
                "num_codes": num_codes,
                "dict_size": next_code,
                "initial_dict_size": INITIAL_DICT_SIZE,
                "max_dict_size": MAX_DICT_SIZE,
                "code_width_progression": f"{widths_used[0]}→{widths_used[-1]}" if widths_used else "N/A",
                "code_widths_used": widths_used,
                "padding_bits": padding,
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("lzw", LZWEncoder)
