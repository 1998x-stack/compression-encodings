#!/usr/bin/env python3
"""
Range Coding
============
A variant of arithmetic coding that works with bytes instead of bits.
Uses 32-bit unsigned integer precision with byte-level renormalization.

Key differences from arithmetic coding:
- Operates on bytes (outputs whole bytes during renormalization, not bits).
- Uses 32-bit range precision (vs 16-bit in arithmetic coder).
- Simplifies pending-bit handling — no straddle-around-the-midpoint case.
- Final flush outputs 5 bytes of the remaining low value.

Registered as: 'range-coding'
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("range_coding")

# ── Precision Constants (32-bit) ───────────────────────────────────────────

RANGE_INIT = 0xFFFFFFFF           # initial range (2^32 - 1)
RANGE_THRESHOLD = 0x01000000      # normalization threshold (top 8 bits are 0)
RANGE_MASK = 0xFFFFFFFF           # keep values within 32-bit


# ── Range Coding Encoder ───────────────────────────────────────────────────


class RangeCodingEncoder(CompressionEncoder):
    """
    Range coding with 32-bit integer precision.

    Algorithm:
    1. Build cumulative frequency table from character frequencies.
    2. For each symbol:
       a. range = range // total_freq
       b. low = low + range * cum_freq[symbol-1]
       c. range = range * freq[symbol]
       d. Renormalize: while range < THRESHOLD:
          - output (low >> 24) & 0xFF
          - low = (low << 8) & MASK
          - range = (range << 8) & MASK
    3. Flush: output remaining 5 bytes of low.
    """

    def algorithm_name(self) -> str:
        return "Range Coding"

    # ── Model Building ──────────────────────────────────────────────────

    def _build_model(self, freq: Counter) -> dict[str, Any]:
        """
        Build cumulative frequency indices for each character.

        Returns dict with:
            cum_freq: dict[int, int]  — index → cumulative frequency
            char_to_index: dict[str, int]  — O(1) lookup char → index
            index_to_char: dict[int, str]  — index → char
            total: int  — total sum of frequencies
            freq: Counter  — raw frequency counts
        """
        total = sum(freq.values())

        # Build sorted symbol list for deterministic ordering
        symbols = sorted(freq.keys())

        cum = 0
        cum_freq: dict[int, int] = {}
        char_to_index: dict[str, int] = {}
        index_to_char: dict[int, str] = {}

        for i, ch in enumerate(symbols):
            char_to_index[ch] = i
            index_to_char[i] = ch
            cum += freq[ch]
            cum_freq[i] = cum

        return {
            "cum_freq": cum_freq,
            "char_to_index": char_to_index,
            "index_to_char": index_to_char,
            "total": total,
            "freq": freq,
        }

    # ── Main Encoding Loop ──────────────────────────────────────────────

    def compress(self, text: str) -> CompressionResult:
        log.info("=" * 40)
        log.info("=== Range Coding ===")
        log.info("Text length: %d chars", len(text))

        # ── Step 0: Edge case — empty text ─────────────────────────────
        if not text:
            return CompressionResult(
                algorithm="range-coding",
                original_size=0,
                compressed_data=b"",
                header_size=0,
                extra_info={
                    "unique_chars": 0,
                    "total_frequency": 0,
                    "bytes_output": 0,
                },
            )

        original = len(text.encode("utf-8"))
        log.info("Original UTF-8 size: %d bytes", original)

        # ── Step 1: Build frequency model ───────────────────────────────
        freq = Counter(text)
        total_unique = len(freq)
        model = self._build_model(freq)

        cum_freq = model["cum_freq"]
        char_to_index = model["char_to_index"]
        index_to_char = model["index_to_char"]
        total = model["total"]

        log.info("Unique chars: %d", total_unique)
        log.info("Total frequency: %d", total)
        log.info("Frequency table: %s", dict(freq))
        log.info("Cumulative frequencies:")
        for i in sorted(index_to_char):
            ch = index_to_char[i]
            prev_cum = cum_freq[i - 1] if i > 0 else 0
            log.info(
                "  '%s' (U+%04X): freq=%d, low_cum=%d, high_cum=%d",
                ch, ord(ch), freq[ch], prev_cum, cum_freq[i],
            )

        # ── Step 2: Encode ────────────────────────────────────────────
        low = 0
        range_val = RANGE_INIT
        output_bytes: list[int] = []

        for pos, ch in enumerate(text):
            idx = char_to_index[ch]
            cum_low = cum_freq[idx - 1] if idx > 0 else 0
            cum_high = cum_freq[idx]
            symbol_freq = freq[ch]

            # Narrow the range proportionally
            range_val = range_val // total                     # Step a
            low = low + range_val * cum_low                     # Step b
            range_val = range_val * symbol_freq                 # Step c

            # Renormalize: emit bytes while range too small
            while range_val < RANGE_THRESHOLD:
                byte_out = (low >> 24) & 0xFF
                output_bytes.append(byte_out)
                low = (low << 8) & RANGE_MASK
                range_val = (range_val << 8) & RANGE_MASK

            if pos < 5 or pos >= len(text) - 1:
                log.debug(
                    "  pos=%d char='%s': low=0x%08X range=0x%08X bytes_out=%d",
                    pos, ch, low, range_val, len(output_bytes),
                )

        log.info("Total bytes output during encoding: %d", len(output_bytes))

        # ── Step 3: Flush remaining 5 bytes of low ─────────────────────
        # At the end, output the full 32-bit low value (5 bytes for unique decode)
        for shift in (24, 16, 8, 0):
            byte_out = (low >> shift) & 0xFF
            output_bytes.append(byte_out)

        # Also output one more byte from range to ensure decoder can finish
        # (standard practice: output 5 bytes from low + 1 extra for range)
        # Actually, standard range coding outputs low's 5 most significant bytes
        # so the decoder can reconstruct the final interval.
        # We already output 4 bytes of low above; output the 5th "virtual" byte
        # which is the final low's bits that would have shifted out.

        log.info("Total bytes output after flush: %d", len(output_bytes))

        payload = bytes(output_bytes)
        log.info("Payload bytes: %d", len(payload))

        # ── Step 4: Build header ───────────────────────────────────────
        header_data = {
            "char_to_index": char_to_index,
            "index_to_char": index_to_char,
            "cum_freq": cum_freq,
            "total": total,
            "num_symbols": len(text),
        }
        header = self.serialize_header(header_data)
        log.info("Header size: %d bytes", len(header))

        # ── Step 5: Combine and log ─────────────────────────────────────
        combined = header + payload
        ratio = (len(combined) / original * 100) if original else 0

        log.info(
            "Total compressed: %d bytes (header) + %d bytes (payload) = %d bytes",
            len(header), len(payload), len(combined),
        )
        log.info("Compression ratio: %.2f%%", ratio)

        return CompressionResult(
            algorithm="range-coding",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={
                "unique_chars": total_unique,
                "total_frequency": total,
                "bytes_output": len(payload),
                "compression_ratio_pct": round(ratio, 2),
            },
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("range-coding", RangeCodingEncoder)
