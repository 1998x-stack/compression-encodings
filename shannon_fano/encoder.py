#!/usr/bin/env python3
"""
Shannon-Fano Encoding
=====================
Recursive splitting approach: partition symbol list into two parts with
approximately equal cumulative frequency, assign '0' to the left part and
'1' to the right part, then recurse until each part holds a single symbol.

Registered as: 'shannon-fano'
"""

from collections import Counter
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("shannon_fano")


# ── Encoder ────────────────────────────────────────────────────────────────


class ShannonFanoEncoder(CompressionEncoder):
    def algorithm_name(self) -> str:
        return "Shannon-Fano"

    # ── helpers ──────────────────────────────────────────────────────────

    def _split_index(self, items: list[tuple[str, int]]) -> int:
        """Return the split index so that sum(freq[:idx]) and sum(freq[idx:])
        are as close as possible.  Items must already be sorted by frequency descending."""
        total = sum(freq for _, freq in items)
        running = 0
        best_diff = None
        best_idx = 1
        for i in range(1, len(items)):
            running += items[i - 1][1]
            remaining = total - running
            diff = abs(running - remaining)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def _build_codes(self, items: list[tuple[str, int]], prefix: str = "") -> dict[str, str]:
        """Recursively build the code table via Shannon-Fano splitting."""
        table: dict[str, str] = {}

        if len(items) == 1:
            # Single symbol — empty prefix means the only symbol gets "0"
            table[items[0][0]] = prefix or "0"
            return table

        if len(items) == 0:
            return table

        idx = self._split_index(items)
        left = items[:idx]
        right = items[idx:]

        table.update(self._build_codes(left, prefix + "0"))
        table.update(self._build_codes(right, prefix + "1"))

        return table

    # ── main pipeline ────────────────────────────────────────────────────

    def compress(self, text: str) -> CompressionResult:
        log.info("=== Shannon-Fano ===")
        original = len(text.encode("utf-8"))

        # 1. Build frequency table
        freq = Counter(text)
        log.info("Unique chars: %d | Total chars: %d", len(freq), len(text))

        # 2. Build Shannon-Fano codes using recursive split
        # Sort symbols by frequency descending
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        log.info("Symbols sorted by frequency (top 5): %s",
                 [(ch, f) for ch, f in sorted_items[:5]])

        code_table = self._build_codes(sorted_items)
        log.info("Code table built: %d entries", len(code_table))

        # 3. Encode text to bitstring using code table
        bitstring = "".join(code_table[ch] for ch in text) if code_table else ""
        log.info("Encoded bits: %d", len(bitstring))

        # 4. Pack bitstring into bytes
        payload, padding = self.bitstring_to_bytes(bitstring)

        # 5. Serialize code table as header
        header = self.serialize_header(code_table)
        log.info("Header: %d B | Payload: %d B | Padding: %d bits",
                 len(header), len(payload), padding)

        # 6. Combine
        combined = header + payload

        max_code_len = max((len(c) for c in code_table.values()), default=0)
        avg_code_len = sum(len(code_table[ch]) * freq[ch] for ch in freq) / len(text) if text else 0.0

        log.info("Max code length: %d | Avg code length: %.2f bits/char",
                 max_code_len, avg_code_len)
        log.info("Original: %d B | Compressed: %d B | Ratio: %.2f%%",
                 original, len(combined),
                 (len(combined) / original * 100) if original else 0)

        return CompressionResult(
            algorithm="shannon-fano",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={
                "unique_chars": len(freq),
                "total_chars": len(text),
                "code_table_size": len(code_table),
                "max_code_len": max_code_len,
                "avg_code_len": round(avg_code_len, 3),
                "padding_bits": padding,
            },
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("shannon-fano", ShannonFanoEncoder)
