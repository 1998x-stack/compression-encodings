#!/usr/bin/env python3
"""
Arithmetic Coding
=================
Integer arithmetic coding with fixed-precision range renormalization.
Registered as: 'arithmetic'
"""

import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("arithmetic")


# ── Precision Constants ────────────────────────────────────────────────────

CODE_VALUE_BITS = 16
MAX_CODE = (1 << CODE_VALUE_BITS) - 1       # 65535
MAX_FREQ = (1 << 14) - 1                     # 16383

TOP_VALUE = MAX_CODE                          # 65535
FIRST_QTR = TOP_VALUE // 4 + 1               # 16384
HALF = FIRST_QTR * 2                          # 32768
THIRD_QTR = FIRST_QTR * 3                     # 49152


# ── Arithmetic Coding Encoder ──────────────────────────────────────────────


class ArithmeticEncoder(CompressionEncoder):
    """
    Integer arithmetic coding with 16-bit range precision.

    Algorithm:
    1. Build cumulative frequency table from character frequencies.
    2. Encode each symbol by narrowing [low, high) proportionally.
    3. Renormalize: whenever the range crosses the half-point, shift bits out.
    4. Flush remaining bits at the end.
    """

    def algorithm_name(self) -> str:
        return "Arithmetic Coding"

    # ── Model Building ──────────────────────────────────────────────────

    def _build_model(self, freq: Counter) -> dict:
        """
        Build cumulative frequency indices for each character.
        Returns:
            cum_freq: dict mapping character index to cumulative frequency.
            char_to_index: dict mapping char -> index for O(1) lookup.
            index_to_char: dict mapping index -> char for reconstruction.
            total: total cumulative frequency.
        """
        total = sum(freq.values())
        if total > MAX_FREQ:
            # Scale down frequencies to fit
            scale = MAX_FREQ / total
            total = 0
            scaled = {}
            for ch, f in freq.items():
                sf = max(1, int(f * scale))
                scaled[ch] = sf
                total += sf
            freq = scaled

        # Build sorted list of symbols
        symbols = sorted(freq.keys())

        cum = 0
        cum_freq = {}
        char_to_index = {}
        index_to_char = {}
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

    # ── Range Renormalization ───────────────────────────────────────────

    @staticmethod
    def _renormalize(
        low: int, high: int, bits_out: list, pending_bits: int
    ) -> tuple[int, int, int]:
        """
        Repeatedly shift bits out of the range midpoint while possible.
        Returns (low, high, pending_bits).
        """
        while True:
            if high < HALF:
                # Both low and high are in lower half → output 0
                bits_out.append("0")
                for _ in range(pending_bits):
                    bits_out.append("1")
                pending_bits = 0
                low = 2 * low
                high = 2 * high + 1
            elif low >= HALF:
                # Both low and high are in upper half → output 1
                bits_out.append("1")
                for _ in range(pending_bits):
                    bits_out.append("0")
                pending_bits = 0
                low = 2 * (low - HALF)
                high = 2 * (high - HALF) + 1
            elif low >= FIRST_QTR and high < THIRD_QTR:
                # Range straddles midpoint → increment pending bits
                pending_bits += 1
                low = 2 * (low - FIRST_QTR)
                high = 2 * (high - FIRST_QTR) + 1
            else:
                break

        return low, high, pending_bits

    # ── Main Encoding Loop ──────────────────────────────────────────────

    def compress(self, text: str) -> CompressionResult:
        log.info("=" * 40)
        log.info("=== Arithmetic Coding ===")
        log.info("Text length: %d chars", len(text))

        if not text:
            # Edge case: empty input
            return CompressionResult(
                algorithm="arithmetic",
                original_size=0,
                compressed_data=b"",
                header_size=0,
                extra_info={"unique_chars": 0, "bits_output": 0, "padding_bits": 0},
            )

        original = len(text.encode("utf-8"))
        log.info("Original UTF-8 size: %d bytes", original)

        # ── Step 1: Build frequency model ───────────────────────────────
        freq = Counter(text)
        log.info("Unique chars: %d", len(freq))
        log.info("Frequency table: %s", dict(freq))

        model = self._build_model(freq)
        cum_freq = model["cum_freq"]
        char_to_index = model["char_to_index"]
        index_to_char = model["index_to_char"]
        total = model["total"]

        log.info("Total frequency: %d", total)
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
        high = TOP_VALUE
        pending_bits = 0
        bits_out = []

        for pos, ch in enumerate(text):
            idx = char_to_index[ch]
            cum_low = cum_freq[idx - 1] if idx > 0 else 0
            cum_high = cum_freq[idx]

            range_width = high - low + 1

            new_low = low + (range_width * cum_low) // total
            new_high = low + (range_width * cum_high) // total - 1

            low = new_low
            high = new_high

            # Renormalize after each symbol
            low, high, pending_bits = self._renormalize(
                low, high, bits_out, pending_bits
            )

            if pos < 5 or pos >= len(text) - 1:
                log.debug(
                    "  pos=%d char='%s': low=%d high=%d range=%d pending=%d",
                    pos, ch, low, high, high - low + 1, pending_bits,
                )

        # ── Step 3: Flush remaining bits ──────────────────────────────
        # Output pending bits then the final bits of low
        pending_bits += 1
        if low < FIRST_QTR:
            bits_out.append("0")
            for _ in range(pending_bits):
                bits_out.append("1")
        else:
            bits_out.append("1")
            for _ in range(pending_bits):
                bits_out.append("0")

        bitstring = "".join(bits_out)
        log.info("Total bits output: %d", len(bitstring))

        # ── Step 4: Pack bits to bytes ────────────────────────────────
        payload, padding = self.bitstring_to_bytes(bitstring)
        log.info("Payload bytes: %d (padding: %d bits)", len(payload), padding)

        # ── Step 5: Build header (frequency table for decoding) ───────
        # Store: {char: freq} mapping
        header_data = {
            "char_to_index": char_to_index,
            "index_to_char": index_to_char,
            "cum_freq": cum_freq,
            "total": total,
            "num_symbols": len(text),
        }
        header = self.serialize_header(header_data)
        log.info("Header size: %d bytes", len(header))

        # Combine header + payload
        combined = header + payload
        log.info("Total compressed: %d bytes (header) + %d bytes (payload) = %d bytes",
                 len(header), len(payload), len(combined))
        log.info("Compression ratio: %.2f%%", (len(combined) / original * 100) if original else 0)

        return CompressionResult(
            algorithm="arithmetic",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={
                "unique_chars": len(freq),
                "total_frequency": total,
                "bits_output": len(bitstring),
                "padding_bits": padding,
                "pending_bits_at_flush": pending_bits,
            },
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("arithmetic", ArithmeticEncoder)
