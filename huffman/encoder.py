#!/usr/bin/env python3
"""
Huffman Encoding
================
Standard Huffman + Canonical Huffman.
Registered as: 'huffman', 'huffman-canonical'
"""

import heapq
from collections import Counter
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("huffman")


# ── Huffman Tree Node ─────────────────────────────────────────────────────


class HuffmanNode:
    def __init__(self, char: Optional[str], freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other):
        return self.freq < other.freq


# ── Standard Huffman ──────────────────────────────────────────────────────


class StandardHuffmanEncoder(CompressionEncoder):
    def algorithm_name(self) -> str:
        return "Standard Huffman"

    def _build_tree(self, freq: Counter) -> HuffmanNode:
        heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))
        return heap[0] if heap else None

    def _generate_codes(self, node, prefix="", table=None):
        if table is None:
            table = {}
        if node.is_leaf():
            table[node.char] = prefix or "0"
        else:
            self._generate_codes(node.left, prefix + "0", table)
            self._generate_codes(node.right, prefix + "1", table)
        return table

    def compress(self, text: str) -> CompressionResult:
        log.info("=== Standard Huffman ===")
        original = len(text.encode("utf-8"))

        freq = Counter(text)
        log.info("Unique chars: %d", len(freq))

        tree = self._build_tree(freq)
        code_table = self._generate_codes(tree)

        bitstring = "".join(code_table[ch] for ch in text)
        log.info("Encoded bits: %d", len(bitstring))

        payload, padding = self.bitstring_to_bytes(bitstring)
        header = self.serialize_header(code_table)

        combined = header + payload
        log.info("Header: %d B, Payload: %d B, Total: %d B", len(header), len(payload), len(combined))

        return CompressionResult(
            algorithm="huffman",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={"unique_chars": len(freq), "max_code_len": max(len(c) for c in code_table.values()), "padding_bits": padding},
        )


# ── Canonical Huffman ─────────────────────────────────────────────────────


class CanonicalHuffmanEncoder(CompressionEncoder):
    def algorithm_name(self) -> str:
        return "Canonical Huffman"

    def compress(self, text: str) -> CompressionResult:
        # Reuse standard tree-building, then canonicalize
        log.info("=== Canonical Huffman ===")
        original = len(text.encode("utf-8"))

        freq = Counter(text)
        log.info("Unique chars: %d", len(freq))

        # Build tree (same as standard)
        heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))
        tree = heap[0]

        # Get code lengths from standard tree
        def walk(node, depth=0, lengths=None):
            if lengths is None:
                lengths = {}
            if node.is_leaf():
                lengths[node.char] = depth
            else:
                walk(node.left, depth + 1, lengths)
                walk(node.right, depth + 1, lengths)
            return lengths

        code_lengths = walk(tree)

        # Canonical encoding: sort by (length, char), assign sequential codes
        sorted_chars = sorted(code_lengths.items(), key=lambda x: (x[1], x[0]))
        canonical = {}
        current = 0
        prev_len = 0
        for ch, length in sorted_chars:
            if length > prev_len:
                current <<= (length - prev_len)
                prev_len = length
            canonical[ch] = format(current, f"0{length}b")
            current += 1

        bitstring = "".join(canonical[ch] for ch in text)
        payload, padding = self.bitstring_to_bytes(bitstring)
        header = self.serialize_header(canonical)
        combined = header + payload

        log.info("Header: %d B, Payload: %d B, Total: %d B", len(header), len(payload), len(combined))

        return CompressionResult(
            algorithm="huffman-canonical",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={"unique_chars": len(freq), "max_code_len": max(len(c) for c in canonical.values()), "padding_bits": padding},
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("huffman", StandardHuffmanEncoder)
EncoderFactory.register("huffman-canonical", CanonicalHuffmanEncoder)
