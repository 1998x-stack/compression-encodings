#!/usr/bin/env python3
"""
LZ77 (Lempel-Ziv 1977) Compression
===================================
Sliding-window dictionary encoder with bit-packed match/literal tokens.

Registered as: 'lz77'

Algorithm:
  - Sliding window of WINDOW_SIZE (4096) bytes behind the current position.
  - Lookahead buffer of LOOKAHEAD_SIZE (18) bytes ahead.
  - For each position, find the longest match in the sliding window.
  - Emit tokens:
      * Literal: 1-bit flag (0) + 8-bit literal byte
      * Match:   1-bit flag (1) + 12-bit offset + 4-bit match-length-minus-3
  - Matches encode lengths 3..18 (stored as 0..15).
  - Token bitstream is packed into bytes using base.bitstring_to_bytes.

Output format:
  +-- 4 bytes: original input size (uint32 big-endian)
  +-- packed bitstream
"""

import struct
import sys
from pathlib import Path

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

log = get_logger("lz77")

# ── Constants ───────────────────────────────────────────────────────────────

WINDOW_SIZE = 4096             # Max search-back distance in bytes
LOOKAHEAD_SIZE = 18            # Max lookahead length in bytes
MIN_MATCH_LENGTH = 3           # Shortest match worth encoding
OFFSET_BITS = 12               # Bits for offset field (0..4095)
LENGTH_BITS = 4                # Bits for match length minus 3 (0..15)
LITERAL_BITS = 8               # Bits for a literal byte


class LZ77Encoder(CompressionEncoder):
    """
    LZ77 sliding-window compressor with bit-packed output.

    Each token is emitted as:
      - Literal token:  '0' + 8-bit character
      - Match token:    '1' + 12-bit offset + 4-bit (length - 3)

    The resulting bitstream is packed into bytes.
    """

    def algorithm_name(self) -> str:
        return "lz77"

    # ── Match finding ──────────────────────────────────────────────────

    def _find_longest_match(
        self,
        data: bytes,
        pos: int,
        window_start: int,
    ) -> tuple[int, int]:
        """
        Find the longest match in the sliding window for data starting at pos.

        Args:
            data: The full input bytes.
            pos: Current position in data (start of lookahead).
            window_start: Start index of the sliding window.

        Returns:
            (offset, length) tuple.
            offset = 0 means no match found (or match_len < MIN_MATCH_LENGTH).
            length = number of bytes matched (same as lookahead when offset > 0).
        """
        max_lookahead = min(LOOKAHEAD_SIZE, len(data) - pos)
        if max_lookahead < MIN_MATCH_LENGTH:
            return 0, 0

        best_offset = 0
        best_length = 0

        # Search backwards through the window
        search_start = max(window_start, pos - WINDOW_SIZE)
        for candidate in range(search_start, pos):
            # Fast bail-out: first byte must match
            if data[candidate] != data[pos]:
                continue

            # Measure how far this candidate matches
            match_len = 0
            max_possible = min(max_lookahead, len(data) - candidate)
            while (
                match_len < max_possible
                and data[candidate + match_len] == data[pos + match_len]
            ):
                match_len += 1

            if match_len > best_length:
                best_length = match_len
                best_offset = pos - candidate
                # Early exit if we found the maximum possible match
                if best_length == max_lookahead:
                    break

        if best_length >= MIN_MATCH_LENGTH:
            return best_offset, best_length
        return 0, 0

    # ── Compression ────────────────────────────────────────────────────

    def compress(self, text: str) -> CompressionResult:
        """
        Compress a UTF-8 string using LZ77 sliding-window encoding.

        Args:
            text: Input string to compress.

        Returns:
            CompressionResult with compressed data and metadata.
        """
        original_bytes = text.encode("utf-8")
        original_size = len(original_bytes)

        if original_size == 0:
            return CompressionResult(
                algorithm="lz77",
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        bitstring_parts: list[str] = []
        pos = 0
        n = len(original_bytes)

        literal_count = 0
        match_count = 0
        total_match_bytes = 0

        while pos < n:
            window_start = max(0, pos - WINDOW_SIZE)
            offset, length = self._find_longest_match(
                original_bytes, pos, window_start
            )

            if length >= MIN_MATCH_LENGTH and 1 <= offset <= WINDOW_SIZE:
                # Emit match token: 1 + 12-bit offset + 4-bit (length - 3)
                stored_len = length - MIN_MATCH_LENGTH  # 0..15
                bitstring_parts.append("1")
                bitstring_parts.append(format(offset, f"0{OFFSET_BITS}b"))
                bitstring_parts.append(format(stored_len, f"0{LENGTH_BITS}b"))
                pos += length
                match_count += 1
                total_match_bytes += length
            else:
                # Emit literal token: 0 + 8-bit char
                bitstring_parts.append("0")
                bitstring_parts.append(format(original_bytes[pos], f"0{LITERAL_BITS}b"))
                pos += 1
                literal_count += 1

        bitstring = "".join(bitstring_parts)
        packed_bytes, padding = self.bitstring_to_bytes(bitstring)

        # ── Build header ───────────────────────────────────────────────

        # 4 bytes: original input size (uint32 big-endian)
        header = struct.pack(">I", original_size)
        header_size = len(header)

        combined = header + packed_bytes

        total_tokens = literal_count + match_count
        if total_tokens > 0:
            match_ratio = match_count / total_tokens
        else:
            match_ratio = 0.0

        log.info(
            "LZ77: %d bytes → %d B (header=%d, payload=%d, padding=%d bits) | "
            "%d literals + %d matches (%.1f%% match), %.1f avg match bytes",
            original_size,
            len(combined),
            header_size,
            len(packed_bytes),
            padding,
            literal_count,
            match_count,
            match_ratio * 100,
            total_match_bytes / match_count if match_count else 0,
        )

        return CompressionResult(
            algorithm="lz77",
            original_size=original_size,
            compressed_data=combined,
            header_size=header_size,
            extra_info={
                "window_size": WINDOW_SIZE,
                "lookahead_size": LOOKAHEAD_SIZE,
                "min_match_length": MIN_MATCH_LENGTH,
                "offset_bits": OFFSET_BITS,
                "length_bits": LENGTH_BITS,
                "literal_count": literal_count,
                "match_count": match_count,
                "total_tokens": total_tokens,
                "match_token_ratio": f"{match_ratio:.2%}",
                "total_match_bytes": total_match_bytes,
                "padding_bits": padding,
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("lz77", LZ77Encoder)
