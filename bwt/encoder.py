#!/usr/bin/env python3
"""
BWT + MTF Encoding
==================
Burrows-Wheeler Transform followed by Move-to-Front encoding.

BWT reorders text so that similar characters cluster together,
creating long runs.  MTF then converts those runs into sequences
of small integers (0, 0, 0, 1, 0, …) which are ideal input for
run-length encoding or arithmetic coding.

This module stores the MTF output as the "compressed" payload.
Header: pickled (original_index, initial_alphabet_list).
"""

import sys
from pathlib import Path

# Ensure parent directory (compression-encodings/) is on sys.path for 'from base import ...'
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("bwt")

# ── Sentinel ────────────────────────────────────────────────────────────────

SENTINEL = "\x02"  # STX — chosen because it is unlikely in natural text and < all printable ASCII


# ── BWT ─────────────────────────────────────────────────────────────────────


def _build_suffix_array_cyclic(text: str) -> list[int]:
    """
    Build a suffix array for cyclic rotations using prefix-doubling.

    This is O(n log n) time and O(n) memory.  Perfect for BWT where we
    need to sort all cyclic rotations of *text* lexicographically.

    Returns a list of start indices sorted by rotation order.
    """
    n = len(text)
    if n == 0:
        return []

    # ── Step 1: initial ranking by single character ────────────────────
    # Map each index to its rank (the character value).
    # We use ord() on the character at that position.
    sa = list(range(n))
    rank = [ord(c) for c in text]
    tmp = [0] * n

    k = 1
    while k < n:
        # Sort by (rank[i], rank[(i + k) % n])
        sa.sort(key=lambda i: (rank[i], rank[(i + k) % n]))

        # Re-rank: assign new ranks based on the sorted order.
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            prev_key = (rank[prev], rank[(prev + k) % n])
            cur_key = (rank[cur], rank[(cur + k) % n])
            tmp[cur] = tmp[prev] + (1 if cur_key != prev_key else 0)

        rank, tmp = tmp, rank

        # If all ranks are unique (0..n-1), we are done.
        if rank[sa[-1]] == n - 1:
            break

        k <<= 1

    return sa


def _burrows_wheeler_transform(text: str) -> tuple[str, int]:
    """
    Perform the Burrows-Wheeler transform on *text*.

    Returns (last_column, original_index) where *original_index* is the
    row of the sorted rotation matrix that equals the original string
    (including sentinel).
    """
    n = len(text)

    sa = _build_suffix_array_cyclic(text)

    # --- Last column L ------------------------------------------------------
    last_column = "".join(text[(i - 1) % n] for i in sa)

    # --- Original index -----------------------------------------------------
    # The original text (with sentinel appended at end) corresponds to the
    # rotation starting at the sentinel.  Since SENTINEL < all other chars,
    # this is the first row (sa where text[i]==SENTINEL).
    original_index = -1
    for row, i in enumerate(sa):
        if text[i] == SENTINEL:
            original_index = row
            break

    if original_index == -1:
        raise RuntimeError("BWT: could not locate sentinel row — internal error")

    return last_column, original_index


# ── MTF ─────────────────────────────────────────────────────────────────────


def _mtf_encode(data: str, alphabet: list[str]) -> tuple[list[int], list[str]]:
    """
    Move-to-Front encoding.

    Args:
        data: The string to encode (typically BWT last column).
        alphabet: Initial ordered list of unique characters.

    Returns:
        (mtf_indices, final_alphabet) — the encoded indices and the
        alphabet after processing (for informational purposes).
    """
    mtf = []
    # Work on a mutable copy of the alphabet.
    symbols = list(alphabet)

    for ch in data:
        # Find the index of ch in the current list.
        # Since each lookup modifies the list, this is O(k) per character
        # where k is the alphabet size — acceptable for typical text alphabets.
        idx = symbols.index(ch)  # guaranteed to exist by construction
        mtf.append(idx)
        # Move to front: pop and insert at position 0.
        del symbols[idx]
        symbols.insert(0, ch)

    return mtf, symbols


# ── Encoder ─────────────────────────────────────────────────────────────────


class BWTEncoder(CompressionEncoder):
    """
    BWT + MTF transform encoder.

    Registered as "bwt".

    Pipeline:
        1. Append sentinel (\\x02 STX) to the original text.
        2. Build the BWT rotation matrix and extract the last column L.
        3. Apply Move-to-Front encoding to L.
        4. Pack: pickled (original_index, alphabet) header + bytes of MTF indices.
    """

    def algorithm_name(self) -> str:
        return "bwt"

    def compress(self, text: str) -> CompressionResult:
        original_bytes = text.encode("utf-8")
        original_size = len(original_bytes)

        if original_size == 0:
            return CompressionResult(
                algorithm=self.algorithm_name(),
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        log.info("BWT+MTF encoding — original: %d bytes", original_size)

        # ── Step 1: Append sentinel ─────────────────────────────────────
        text_with_sentinel = text + SENTINEL

        # ── Step 2: BWT ────────────────────────────────────────────────
        last_column, original_index = _burrows_wheeler_transform(text_with_sentinel)

        log.info("  BWT last-column length: %d", len(last_column))
        log.info("  Original row index: %d", original_index)

        # ── Step 3: Build alphabet & MTF ───────────────────────────────
        # The alphabet is the sorted set of unique characters in the last column.
        unique_chars = sorted(set(last_column))
        mtf_indices, final_alphabet = _mtf_encode(last_column, unique_chars)

        log.info("  Alphabet size: %d", len(unique_chars))
        log.info("  MTF encoded size: %d indices", len(mtf_indices))

        # ── Step 4: Pack ───────────────────────────────────────────────
        # Header: pickled tuple of (original_index, initial_alphabet).
        header = self.serialize_header((original_index, unique_chars))

        # Payload: MTF indices stored as unsigned bytes (0–255).
        # The alphabet is bounded by len(unique_chars) which is ≤ text length,
        # but for very large alphabets (> 256 unique bytes) we'd need multi-byte.
        # For UTF-8 text this is unusual (would need many distinct byte values).
        # We use a simple byte array; if an index ever exceeds 255 we raise.
        payload_array = bytearray()
        for idx in mtf_indices:
            if idx > 255:
                raise ValueError(
                    f"MTF index {idx} exceeds 255 — alphabet too large for single-byte encoding"
                )
            payload_array.append(idx)
        payload = bytes(payload_array)

        compressed = header + payload

        log.info(
            "  Header: %d B, Payload: %d B, Total: %d B",
            len(header),
            len(payload),
            len(compressed),
        )

        return CompressionResult(
            algorithm=self.algorithm_name(),
            original_size=original_size,
            compressed_data=compressed,
            header_size=len(header),
            extra_info={
                "bwt_last_column_length": len(last_column),
                "bwt_original_index": original_index,
                "mtf_encoded_size": len(mtf_indices),
                "mtf_alphabet_size": len(unique_chars),
                "sentinel": repr(SENTINEL),
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("bwt", BWTEncoder)
