#!/usr/bin/env python3
"""
Simplified DEFLATE Encoder (LZSS + Huffman)
============================================
Core algorithm used in gzip/zlib/PNG.

Pipeline:
  1. LZSS step — sliding window 4096, lookahead 18, min_match 3
     → tokens: (type, data) where type ∈ {'literal', 'match'}
  2. Separate streams — literal bytes, match lengths, match offsets
  3. Huffman-encode literals + lengths on combined alphabet:
       0..255   → literal bytes
       256..270 → match length − 3  (representing lengths 3..17)
       271      → end-of-block
     Match offsets: raw 12-bit values (not Huffman-coded)
  4. Output: pickled Huffman table + 4-byte token count + packed bitstream

Registered as: 'deflate'
"""

import heapq
import struct
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("deflate")

# ── Constants ─────────────────────────────────────────────────────────────

WINDOW_SIZE = 4096       # sliding window for LZSS
LOOKAHEAD = 18           # max match lookahead
MIN_MATCH = 3            # minimum match length
MAX_MATCH = 258          # theoretical max (we cap at lookahead = 18)

# Combined alphabet for Huffman coding of literals + lengths
# 0..255   → literal byte values
# 256..270 → match length − 3  (representing lengths 3..17)
# 271      → end-of-block symbol
LITERAL_MAX = 255
LENGTH_SYMBOL_OFFSET = 256
END_OF_BLOCK = 271
ALPHABET_SIZE = 272

# Match offset is emitted as raw 12 bits (0..4095 for window size 4096)
OFFSET_BITS = 12


# ── Huffman Tree Node ────────────────────────────────────────────────────


class HuffmanNode:
    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(self, symbol: Optional[int], freq: int, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other):
        return self.freq < other.freq


# ── LZSS Step ─────────────────────────────────────────────────────────────


def lzss_tokenize(data: bytes) -> list:
    """
    LZSS tokenization with sliding window.

    Returns list of tokens:
      ('literal', byte_value)     — single byte
      ('match', (offset, length)) — back-reference into window

    offset = distance from current position backward into the window.
    """
    n = len(data)
    pos = 0
    tokens: list = []

    while pos < n:
        best_len = 0
        best_offset = 0

        # Search for longest match in the sliding window
        search_start = max(0, pos - WINDOW_SIZE)
        search_end = pos

        # Minimal naive search — OK for a simplified encoder
        # Build a quick lookup: for each possible match start, compare
        for candidate in range(search_start, search_end):
            # Fast check: first byte must match
            if data[candidate] != data[pos]:
                continue

            # Extend match as far as possible
            match_len = 0
            max_extend = min(LOOKAHEAD, n - pos)
            while (
                match_len < max_extend
                and candidate + match_len < n
                and data[candidate + match_len] == data[pos + match_len]
            ):
                match_len += 1

            if match_len >= MIN_MATCH and match_len > best_len:
                best_len = match_len
                best_offset = pos - candidate
                if best_len == max_extend:
                    break  # can't do better

        if best_len >= MIN_MATCH:
            tokens.append(("match", (best_offset, best_len)))
            pos += best_len
        else:
            tokens.append(("literal", data[pos]))
            pos += 1

    return tokens


# ── Huffman Encoder (symbol → code table) ──────────────────────────────────


def build_huffman_tree(freq: dict[int, int]) -> HuffmanNode:
    """Build Huffman tree from frequency dict. Returns root node."""
    if not freq:
        return None

    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        # Single symbol case: create a dummy parent so we get a nonzero code
        node = heap[0]
        dummy = HuffmanNode(None, node.freq, node)
        return dummy

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))

    return heap[0]


def generate_canonical_codes(freq: dict[int, int]) -> dict[int, str]:
    """Build Huffman tree, get code lengths, then produce canonical codes."""
    if not freq:
        return {}

    tree = build_huffman_tree(freq)

    # Get code lengths
    def walk(node, depth=0, lengths=None):
        if lengths is None:
            lengths = {}
        if node.is_leaf():
            lengths[node.symbol] = max(depth, 1)  # at least 1 bit
        else:
            if node.left:
                walk(node.left, depth + 1, lengths)
            if node.right:
                walk(node.right, depth + 1, lengths)
        return lengths

    code_lengths = walk(tree)

    # Sort by (length, symbol) for canonical encoding
    sorted_symbols = sorted(code_lengths.items(), key=lambda x: (x[1], x[0]))
    canonical: dict[int, str] = {}
    current = 0
    prev_len = 0
    for sym, length in sorted_symbols:
        if length > prev_len:
            current <<= (length - prev_len)
            prev_len = length
        canonical[sym] = format(current, f"0{length}b")
        current += 1

    return canonical


def canonical_table_to_serializable(codes: dict[int, str]) -> dict:
    """Convert {symbol: bitstring} to a JSON-serializable (pickle-friendly) dict.

    We store as {str(symbol): bitstring} since pickle handles string keys cleanly.
    """
    return {str(sym): code for sym, code in codes.items()}


def serializable_to_code_table(data: dict) -> dict[int, str]:
    """Deserialize back: {str(symbol): bitstring} → {symbol: bitstring}."""
    return {int(sym): code for sym, code in data.items()}


# ── DEFLATE Encoder ────────────────────────────────────────────────────────


class DeflateEncoder(CompressionEncoder):
    """Simplified DEFLATE: LZSS + Huffman on literals/lengths, raw offsets."""

    def algorithm_name(self) -> str:
        return "Simplified DEFLATE (LZSS + Huffman)"

    def compress(self, text: str) -> CompressionResult:
        log.info("=== Simplified DEFLATE ===")

        # ── Prepare ───────────────────────────────────────────────────
        data = text.encode("utf-8")
        original_size = len(data)
        log.info("Original UTF-8: %d bytes (%d chars)", original_size, len(text))

        # ── Step 1: LZSS tokenization ──────────────────────────────────
        tokens = lzss_tokenize(data)
        token_count = len(tokens)
        log.info("LZSS tokens: %d (literals + matches)", token_count)

        # Count for logging
        literal_count = sum(1 for t in tokens if t[0] == "literal")
        match_count = sum(1 for t in tokens if t[0] == "match")
        log.info("  Literals: %d, Matches: %d", literal_count, match_count)

        # ── Step 2: Build combined literal/length stream ────────────────
        # Symbols: 0-255 literal bytes; 256-270 length-3; 271 EOB
        lit_len_symbols: list[int] = []
        offsets: list[int] = []

        for ttype, data_val in tokens:
            if ttype == "literal":
                lit_len_symbols.append(data_val)  # 0..255
            else:  # match
                offset, length = data_val
                length_code = length - MIN_MATCH  # 0..14 for lengths 3..17
                lit_len_symbols.append(LENGTH_SYMBOL_OFFSET + length_code)  # 256..270
                offsets.append(offset)
                # Log oversized offsets
                if offset >= (1 << OFFSET_BITS):
                    log.warning(
                        "Match offset %d exceeds %d-bit limit (%d), clamping",
                        offset,
                        OFFSET_BITS,
                        (1 << OFFSET_BITS) - 1,
                    )

        # Append end-of-block
        lit_len_symbols.append(END_OF_BLOCK)

        log.info("Literal/length symbols: %d (incl. EOB)", len(lit_len_symbols))
        log.info("Match offsets: %d", len(offsets))

        # ── Step 3: Huffman-encode the literal/length stream ───────────
        freq = Counter(lit_len_symbols)
        log.info("Unique symbols in lit/len stream: %d", len(freq))

        code_table = generate_canonical_codes(dict(freq))
        max_code_len = max(len(c) for c in code_table.values()) if code_table else 0
        log.info("Huffman code table: %d entries, max code length: %d bits",
                 len(code_table), max_code_len)

        # ── Step 4: Build bitstream ────────────────────────────────────
        offset_idx = 0
        bitstring_parts: list[str] = []

        for sym in lit_len_symbols:
            code = code_table[sym]
            bitstring_parts.append(code)

        # Interleave offsets after match length codes
        # We need to reconstruct: for each token in the original stream,
        # - literal: just the Huffman code
        # - match: Huffman length code + 12-bit raw offset
        # Then EOB Huffman code
        # The trick is that the lit_len_symbols stream matches tokens + EOB exactly.
        # So we iterate tokens, emit Huffman code for each, and for matches emit offset bits.
        # lit_len_symbols[-1] is EOB → emit its Huffman code alone.

        bit_parts: list[str] = []
        off_idx = 0
        for i, (ttype, _) in enumerate(tokens):
            sym = lit_len_symbols[i]
            bit_parts.append(code_table[sym])
            if ttype == "match":
                # Emit offset as 12 raw bits
                raw_offset = offsets[off_idx]
                if raw_offset >= (1 << OFFSET_BITS):
                    raw_offset = (1 << OFFSET_BITS) - 1  # clamp to max
                bit_parts.append(format(raw_offset, f"0{OFFSET_BITS}b"))
                off_idx += 1

        # Append EOB Huffman code
        eob_sym = lit_len_symbols[-1]
        bit_parts.append(code_table[eob_sym])

        bitstring = "".join(bit_parts)
        log.info("Bitstream length: %d bits (%d bytes theoretical)",
                 len(bitstring), (len(bitstring) + 7) // 8)

        # ── Pack bitstring to bytes ────────────────────────────────────
        payload_bytes, padding = self.bitstring_to_bytes(bitstring)

        # ── Header ─────────────────────────────────────────────────────
        # Serialize: (code_table_serializable, token_count)
        header_data = {
            "code_table": canonical_table_to_serializable(code_table),
            "token_count": token_count,
        }
        header_bytes = self.serialize_header(header_data)

        # 4-byte token count (uint32 big-endian)
        token_count_bytes = struct.pack(">I", token_count)

        # ── Combine ────────────────────────────────────────────────────
        combined = header_bytes + token_count_bytes + payload_bytes
        total_header = len(header_bytes) + len(token_count_bytes)

        log.info("Pickled header: %d B, Token count: 4 B, Payload: %d B → Total: %d B",
                 len(header_bytes), len(payload_bytes), len(combined))
        log.info("Padding bits: %d", padding)

        return CompressionResult(
            algorithm="deflate",
            original_size=original_size,
            compressed_data=combined,
            header_size=total_header,
            extra_info={
                "tokens": token_count,
                "literals": literal_count,
                "matches": match_count,
                "huffman_table_entries": len(code_table),
                "max_code_len": max_code_len,
                "bitstream_bits": len(bitstring),
                "padding_bits": padding,
            },
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("deflate", DeflateEncoder)
