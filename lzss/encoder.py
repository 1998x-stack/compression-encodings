#!/usr/bin/env python3
"""
LZSS (Lempel-Ziv-Storer-Szymanski) Compression
================================================
Improvement over LZ77. Uses a sliding window (4 KB) and lookahead buffer (18 bytes)
to find longest matches. Only encodes a match when length >= 3, otherwise emits
a literal byte.

Output format (bit-packed):
  - Flag bits: 1 byte per 8 tokens. 0 = literal, 1 = match.
  - Token data:
      • Literal: 8 bits (raw byte value)
      • Match:   12 bits offset (0..4095) + 4 bits stored-length (actual - 3)

Constants:
  WINDOW_SIZE     = 4096   (12-bit window offset)
  LOOKAHEAD_SIZE  = 18     (max match length)
  MIN_MATCH       = 3      (only encode as match if >= 3 chars)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

logger = get_logger("lzss")

# ── Constants ───────────────────────────────────────────────────────────────

WINDOW_SIZE = 4096           # Bytes (12 bits)
LOOKAHEAD_SIZE = 18          # Max match length
MIN_MATCH = 3                # Minimum match length to encode
OFFSET_BITS = 12
LENGTH_BITS = 4              # Stored length = actual - MIN_MATCH, max 15 (i.e. actual max 18)


class LZSSEncoder(CompressionEncoder):
    """
    LZSS compression encoder.

    Algorithm:
      1. Slide a window over the input. For each position, search the preceding
         WINDOW_SIZE bytes for the longest match (up to LOOKAHEAD_SIZE).
      2. If a match of length >= MIN_MATCH is found, emit a (offset, length) match token.
      3. Otherwise, emit the single byte as a literal token.
      4. After collecting all tokens, build the flag bitstring (1 bit per token)
         and the data bitstring, then concatenate and pack into bytes.
    """

    def algorithm_name(self) -> str:
        return "lzss"

    def compress(self, text: str) -> CompressionResult:
        """
        Compress a UTF-8 string with LZSS.

        Args:
            text: Input string.

        Returns:
            CompressionResult with bit-packed LZSS data.
        """
        data = text.encode("utf-8")
        original_size = len(data)

        if original_size == 0:
            return CompressionResult(
                algorithm=self.algorithm_name(),
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        # ── Phase 1: tokenize ───────────────────────────────────────────
        tokens: list[tuple[str, object]] = []  # ("literal", byte) or ("match", (offset, length))
        pos = 0
        n = len(data)

        while pos < n:
            best_len = 0
            best_offset = 0

            # Search window: max WINDOW_SIZE bytes before current position
            search_start = max(0, pos - WINDOW_SIZE)
            search_end = pos

            # Max match length is limited by lookahead buffer and remaining data
            max_len = min(LOOKAHEAD_SIZE, n - pos)

            # Find longest match in the sliding window
            # Naive search: try every possible start in the window
            for candidate_start in range(search_start, search_end):
                match_len = 0
                while (
                    match_len < max_len
                    and data[candidate_start + match_len] == data[pos + match_len]
                ):
                    match_len += 1

                if match_len > best_len:
                    best_len = match_len
                    best_offset = pos - candidate_start
                    # Early exit if we hit max possible length
                    if best_len == max_len:
                        break

            if best_len >= MIN_MATCH:
                tokens.append(("match", (best_offset, best_len)))
                pos += best_len
            else:
                tokens.append(("literal", data[pos]))
                pos += 1

        # ── Phase 2: build bit-strings ──────────────────────────────────
        flag_bits: list[str] = []   # 0 = literal, 1 = match
        data_bits: list[str] = []

        for token_type, payload in tokens:
            if token_type == "literal":
                flag_bits.append("0")
                # 8-bit literal byte
                data_bits.append(format(payload, "08b"))
            else:  # match
                flag_bits.append("1")
                offset, length = payload
                stored_length = length - MIN_MATCH  # 0..15
                # 12-bit offset + 4-bit stored-length
                data_bits.append(format(offset, "012b") + format(stored_length, "04b"))

        flag_bitstring = "".join(flag_bits)
        data_bitstring = "".join(data_bits)
        combined_bitstring = flag_bitstring + data_bitstring

        flag_bytes, flag_padding = self.bitstring_to_bytes(flag_bitstring)
        data_bytes, data_padding = self.bitstring_to_bytes(data_bitstring)
        combined_bytes, combined_padding = self.bitstring_to_bytes(combined_bitstring)

        # ── Logging ─────────────────────────────────────────────────────
        total_tokens = len(tokens)
        match_count = sum(1 for t, _ in tokens if t == "match")
        literal_count = sum(1 for t, _ in tokens if t == "literal")

        logger.info(
            "LZSS: %d → %d bytes (%.1f%%).  %d tokens (%d literal, %d match).  "
            "flags %d bits (+%d pad), data %d bits (+%d pad), total packed %d bytes",
            original_size,
            len(combined_bytes),
            len(combined_bytes) / original_size * 100 if original_size else 0,
            total_tokens,
            literal_count,
            match_count,
            len(flag_bitstring),
            flag_padding,
            len(data_bitstring),
            data_padding,
            len(combined_bytes),
        )

        return CompressionResult(
            algorithm=self.algorithm_name(),
            original_size=original_size,
            compressed_data=combined_bytes,
            header_size=0,
            extra_info={
                "window_size": WINDOW_SIZE,
                "lookahead_size": LOOKAHEAD_SIZE,
                "min_match": MIN_MATCH,
                "tokens": total_tokens,
                "literals": literal_count,
                "matches": match_count,
                "flags_bytes": len(flag_bytes),
                "data_bytes": len(data_bytes),
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("lzss", LZSSEncoder)
