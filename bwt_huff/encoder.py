#!/usr/bin/env python3
"""
BWT + RLE + Huffman Combo Encoder (bzip2-style pipeline)
=========================================================

Pipeline:
  1. BWT (Burrows-Wheeler Transform) — reorder text to cluster similar chars
  2. MTF (Move-to-Front) — convert runs into sequences of small integers
  3. Zero-Run-Length Encoding — compact runs of zeros in MTF output
  4. Huffman Encoding — statistical compression of the RLE'd byte stream

Registered as: 'bwt-huff'
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# Ensure parent directory (compression-encodings/) is on sys.path
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("bwt-huff")

# ── Sentinel ────────────────────────────────────────────────────────────────

SENTINEL = "\x02"  # STX — unlikely in natural text, < all printable ASCII


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: BWT
# ═════════════════════════════════════════════════════════════════════════════


def _burrows_wheeler_transform(text: str) -> tuple[str, int]:
    """
    Burrows-Wheeler Transform on *text* (which already includes sentinel).

    Uses prefix-doubling suffix array construction — O(n log n) time,
    O(n) memory.  Avoids materializing all n cyclic rotation strings
    (which would be O(n²) memory for large inputs).

    Returns (last_column, original_index).
    """
    n = len(text)
    t0 = time.time()

    # ── Prefix-doubling suffix array ──────────────────────────────
    sa = list(range(n))
    # Initial rank: ord() of each character
    rank = [ord(c) for c in text]
    tmp = [0] * n

    k = 1
    while k < n:
        # Sort by (rank[i], rank[(i + k) % n])
        sa.sort(key=lambda i: (rank[i], rank[(i + k) % n]))

        # Re-rank
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            prev_key = (rank[prev], rank[(prev + k) % n])
            cur_key = (rank[cur], rank[(cur + k) % n])
            tmp[cur] = tmp[prev] + (1 if cur_key != prev_key else 0)

        rank, tmp = tmp, rank

        # All ranks unique → done
        if rank[sa[-1]] == n - 1:
            break

        k <<= 1

    # Last column L: the character BEFORE each rotation's start position
    last_column = "".join(text[(i - 1) % n] for i in sa)

    # Original index: row where sentinel appears at rotation start
    original_index = -1
    for row, i in enumerate(sa):
        if text[i] == SENTINEL:
            original_index = row
            break

    if original_index == -1:
        raise RuntimeError("BWT: could not locate sentinel row — internal error")

    elapsed = time.time() - t0
    log.info("  BWT: %.2fs — L length=%d, original_index=%d", elapsed, n, original_index)

    return last_column, original_index


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: MTF (Move-to-Front)
# ═════════════════════════════════════════════════════════════════════════════


def _mtf_encode(data: str, alphabet: list[str]) -> list[int]:
    """
    Move-to-Front encoding of *data* using the given ordered alphabet.

    For each character, emit its current position in the alphabet,
    then move that character to the front (position 0).

    Returns the list of integer indices.
    """
    t0 = time.time()
    symbols = list(alphabet)  # mutable copy
    mtf = []

    for ch in data:
        idx = symbols.index(ch)
        mtf.append(idx)
        # Move to front
        del symbols[idx]
        symbols.insert(0, ch)

    elapsed = time.time() - t0
    log.info("  MTF: %.2fs — %d indices, alphabet size=%d", elapsed, len(mtf), len(alphabet))

    return mtf


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Zero-Run-Length Encoding
# ═════════════════════════════════════════════════════════════════════════════


def _encode_variable_length(value: int) -> list[int]:
    """
    Encode a non-negative integer as variable-length bytes (7 bits per byte,
    MSB indicates continuation).

    while value >= 128: emit (value & 0x7F) | 0x80, value >>= 7
    then emit value & 0x7F
    """
    result = []
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return result


def _zero_run_encode(mtf_indices: list[int]) -> bytearray:
    """
    Encode runs of zeros in the MTF output.

    - A run of N consecutive zeros → emit byte 0, then (N-1) as a
      variable-length integer.
    - All other values → emit as-is (they are always ≤ 255 since
      alphabet size <= 256 for byte-level MTF).

    Returns a bytearray of the RLE-encoded stream.
    """
    t0 = time.time()
    result = bytearray()
    i = 0
    n = len(mtf_indices)

    while i < n:
        if mtf_indices[i] == 0:
            # Count zeros
            run_start = i
            while i < n and mtf_indices[i] == 0:
                i += 1
            run_len = i - run_start  # N consecutive zeros
            result.append(0)  # escape marker
            result.extend(_encode_variable_length(run_len - 1))
        else:
            result.append(mtf_indices[i])
            i += 1

    elapsed = time.time() - t0
    log.info("  RLE: %.2fs — %d MTF indices → %d bytes", elapsed, n, len(result))

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Huffman Encoding
# ═════════════════════════════════════════════════════════════════════════════


class _HuffmanNode:
    __slots__ = ("char", "freq", "left", "right")

    def __init__(self, char: Optional[int], freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(freq: Counter) -> Optional[_HuffmanNode]:
    """Build Huffman tree from a Counter of byte→frequency."""
    import heapq

    heap = [_HuffmanNode(b, f) for b, f in freq.items()]
    if not heap:
        return None
    heapq.heapify(heap)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _HuffmanNode(None, a.freq + b.freq, a, b))

    return heap[0]


def _generate_huffman_codes(node: _HuffmanNode) -> dict[int, str]:
    """Walk the Huffman tree and produce a byte→bitstring code table."""
    table: dict[int, str] = {}

    def walk(n: _HuffmanNode, prefix: str):
        if n.is_leaf():
            table[n.char] = prefix or "0"
        else:
            walk(n.left, prefix + "0")
            walk(n.right, prefix + "1")

    walk(node, "")
    return table


def _huffman_encode(data: bytearray) -> tuple[str, dict[int, str], int, Counter]:
    """
    Huffman-encode a bytearray.

    Returns (bitstring, code_table, padding_bits, freq_counter).
    """
    t0 = time.time()
    freq = Counter(data)
    log.info("  Huffman freq: %d unique byte values", len(freq))

    tree = _build_huffman_tree(freq)
    code_table = _generate_huffman_codes(tree)

    bitstring = "".join(code_table[b] for b in data)

    elapsed = time.time() - t0
    log.info(
        "  Huffman: %.2fs — %d bytes → %d bits (%.1f bits/byte)",
        elapsed,
        len(data),
        len(bitstring),
        len(bitstring) / len(data) if data else 0,
    )

    return bitstring, code_table, 0, freq  # padding computed later


# ═════════════════════════════════════════════════════════════════════════════
# Encoder Class
# ═════════════════════════════════════════════════════════════════════════════


class BWTHuffEncoder(CompressionEncoder):
    """
    BWT + MTF + Zero-Run-Length + Huffman combo encoder.

    Registered as "bwt-huff".

    Pipeline:
        1. Append sentinel (\\x02) to the original UTF-8 text.
        2. BWT — reorder characters to cluster runs.
        3. MTF — convert L-column into small-integer sequence.
        4. Zero-run-length encode — compact runs of zeros from MTF.
        5. Huffman encode — statistical compression of the RLE byte stream.

    Output format:
        Header: pickled (original_index, alphabet, huffman_code_table, rle_size)
        Payload: Huffman-encoded bitstream packed into bytes.
    """

    def algorithm_name(self) -> str:
        return "bwt-huff"

    def compress(self, text: str) -> CompressionResult:
        t_total = time.time()
        original_bytes = text.encode("utf-8")
        original_size = len(original_bytes)

        if original_size == 0:
            return CompressionResult(
                algorithm=self.algorithm_name(),
                original_size=0,
                compressed_data=b"",
                header_size=0,
            )

        log.info("=" * 50)
        log.info("BWT-HUFF — original: %d bytes (%d chars)", original_size, len(text))

        # ── Step 1: BWT ────────────────────────────────────────────────
        text_with_sentinel = text + SENTINEL
        last_column, original_index = _burrows_wheeler_transform(text_with_sentinel)

        # ── Step 2: MTF ────────────────────────────────────────────────
        # Build sorted alphabet from the last column's unique characters.
        alphabet = sorted(set(last_column))
        mtf_indices = _mtf_encode(last_column, alphabet)

        # ── Step 3: Zero-Run-Length Encoding ───────────────────────────
        rle_bytes = _zero_run_encode(mtf_indices)

        # ── Step 4: Huffman Encoding ───────────────────────────────────
        bitstring, code_table, _, freq = _huffman_encode(rle_bytes)

        # Pack bitstring into bytes
        payload, padding = self.bitstring_to_bytes(bitstring)

        # ── Header ─────────────────────────────────────────────────────
        header = self.serialize_header((original_index, alphabet, code_table, len(rle_bytes)))

        compressed = header + payload

        elapsed_total = time.time() - t_total
        log.info("  Total time: %.2fs", elapsed_total)
        log.info(
            "  Header: %d B | Payload: %d B | Total: %d B",
            len(header),
            len(payload),
            len(compressed),
        )
        log.info(
            "  Ratio: %.2f%% (%.2f bits/byte)",
            len(compressed) / original_size * 100,
            len(compressed) * 8 / original_size,
        )

        return CompressionResult(
            algorithm=self.algorithm_name(),
            original_size=original_size,
            compressed_data=compressed,
            header_size=len(header),
            extra_info={
                "bwt_last_column_length": len(last_column),
                "bwt_original_index": original_index,
                "mtf_size": len(mtf_indices),
                "mtf_alphabet_size": len(alphabet),
                "rle_size": len(rle_bytes),
                "huffman_unique_symbols": len(freq),
                "huffman_bits": len(bitstring),
                "huffman_padding": padding,
                "payload_bytes": len(payload),
                "total_time_s": round(elapsed_total, 3),
            },
        )


# ── Register with factory ───────────────────────────────────────────────────

EncoderFactory.register("bwt-huff", BWTHuffEncoder)
