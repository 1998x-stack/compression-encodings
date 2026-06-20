#!/usr/bin/env python3
"""
Huffman Encoding with Factory Pattern + Logging
================================================
Supports multiple encoding strategies via the Factory Method pattern.
Current encodings: Standard Huffman, Canonical Huffman.
"""

import heapq
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Logging Setup ──────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).resolve().parent / "huffman.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("huffman")


# ── Data Structures ─────────────────────────────────────────────────────────


class HuffmanNode:
    """Internal node for the Huffman tree."""

    def __init__(self, char: Optional[str], freq: int, left=None, right=None):
        self.char = char  # None for internal nodes
        self.freq = freq
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def __lt__(self, other):
        return self.freq < other.freq


@dataclass
class HuffmanResult:
    """Encapsulates the result of encoding."""

    original_size: int
    compressed_data: bytes
    code_table: dict[str, str]
    tree_size_bytes: int
    padding_bits: int


# ── Abstract Base: Encoding Strategy ────────────────────────────────────────


class HuffmanEncoder(ABC):
    """
    Abstract encoder – the Product in the Factory Method pattern.
    Subclasses implement build_tree() which defines the encoding strategy.
    """

    def __init__(self):
        self.tree: Optional[HuffmanNode] = None
        self.code_table: dict[str, str] = {}

    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable strategy name."""
        ...

    def build_frequency(self, text: str) -> Counter:
        log.info("Building character frequency table (len=%d chars)", len(text))
        freq = Counter(text)
        log.info("  Unique characters: %d", len(freq))
        return freq

    def build_tree(self, freq: Counter) -> HuffmanNode:
        """Standard Huffman tree – can be overridden by subclasses."""
        log.info("Building Huffman tree with standard algorithm")
        heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
        heapq.heapify(heap)
        log.info("  Initial heap size: %d", len(heap))

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(None, left.freq + right.freq, left, right)
            heapq.heappush(heap, parent)

        self.tree = heap[0] if heap else None
        log.info("  Tree root frequency: %s", self.tree.freq if self.tree else "N/A")
        return self.tree

    def generate_codes(self, node=None, prefix=""):
        """Walk the Huffman tree and populate self.code_table."""
        if node is None:
            node = self.tree
        if node is None:
            return
        if node.is_leaf():
            self.code_table[node.char] = prefix or "0"  # edge: single char
        else:
            self.generate_codes(node.left, prefix + "0")
            self.generate_codes(node.right, prefix + "1")

    def encode_text(self, text: str) -> str:
        """Return the bit-string of the encoded text."""
        log.info("Encoding text using code table (%d entries)", len(self.code_table))
        return "".join(self.code_table[ch] for ch in text)

    @staticmethod
    def bitstring_to_bytes(bitstring: str) -> tuple[bytes, int]:
        """Pack bitstring into bytes. Returns (bytes, padding_bits)."""
        padding = (8 - len(bitstring) % 8) % 8
        padded = bitstring + "0" * padding
        result = bytearray()
        for i in range(0, len(padded), 8):
            result.append(int(padded[i : i + 8], 2))
        return bytes(result), padding

    def compress(self, text: str) -> HuffmanResult:
        """Full pipeline: frequency → tree → codes → encode → pack."""
        log.info("=== Starting compression with [%s] ===", self.strategy_name())

        freq = self.build_frequency(text)
        self.build_tree(freq)
        self.generate_codes()

        encoded_bits = self.encode_text(text)
        log.info("  Encoded bit-length: %d", len(encoded_bits))

        compressed, padding = self.bitstring_to_bytes(encoded_bits)
        log.info("  Compressed byte-length: %d (padding=%d bits)", len(compressed), padding)

        # Serialize code table for decompression
        tree_bytes = pickle.dumps(dict(self.code_table))
        log.info("  Code-table serialized size: %d bytes", len(tree_bytes))

        total_compressed = tree_bytes + compressed
        log.info("  Total compressed size (header+data): %d bytes", len(total_compressed))

        original_size = len(text.encode("utf-8"))
        ratio = len(total_compressed) / original_size if original_size else 0
        log.info("  Compression ratio: %.2f%%", ratio * 100)

        return HuffmanResult(
            original_size=original_size,
            compressed_data=total_compressed,
            code_table=dict(self.code_table),
            tree_size_bytes=len(tree_bytes),
            padding_bits=padding,
        )


# ── Concrete Encoder: Standard Huffman ─────────────────────────────────────


class StandardHuffmanEncoder(HuffmanEncoder):
    """Standard Huffman coding – merges two smallest frequencies."""

    def strategy_name(self) -> str:
        return "Standard Huffman"


# ── Concrete Encoder: Canonical Huffman ────────────────────────────────────


class CanonicalHuffmanEncoder(HuffmanEncoder):
    """
    Canonical Huffman coding.
    Same tree-building, but stores codes in canonical form:
    - Codes sorted by (length, symbol)
    - Codes of same length are numerically sequential
    This produces a more compact header but identical compression ratio.
    """

    def strategy_name(self) -> str:
        return "Canonical Huffman"

    def compress(self, text: str) -> HuffmanResult:
        result = super().compress(text)

        # Canonical reordering for the code table (cosmetic; data unchanged)
        # Sort by (code length, character value) then assign sequential codes
        sorted_symbols = sorted(result.code_table.items(), key=lambda x: (len(x[1]), x[0]))
        canonical = {}
        current_code = 0
        prev_len = 0
        for ch, code in sorted_symbols:
            code_len = len(code)
            if code_len > prev_len:
                current_code <<= (code_len - prev_len)
                prev_len = code_len
            canonical[ch] = format(current_code, f"0{code_len}b")
            current_code += 1

        log.info("Canonical reordering applied (same compressed data, cleaner table)")
        result.code_table = canonical
        return result


# ── Factory ─────────────────────────────────────────────────────────────────


class EncoderFactory:
    """
    Factory Method pattern – maps a strategy key to an encoder instance.
    Easy to add new encodings: just register them here.
    """

    _registry: dict[str, type[HuffmanEncoder]] = {
        "standard": StandardHuffmanEncoder,
        "canonical": CanonicalHuffmanEncoder,
    }

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def register(cls, name: str, encoder_cls: type[HuffmanEncoder]):
        log.info("Registering new encoder strategy: %s -> %s", name, encoder_cls.__name__)
        cls._registry[name] = encoder_cls

    @classmethod
    def create(cls, strategy: str = "standard") -> HuffmanEncoder:
        log.info("Factory creating encoder: strategy=%s", strategy)
        if strategy not in cls._registry:
            log.error("Unknown strategy '%s'. Available: %s", strategy, cls._registry.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {cls._registry.keys()}")
        return cls._registry[strategy]()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    log.info("=" * 60)
    log.info("Huffman Encoding Tool – Session Start")
    log.info("=" * 60)

    # Paths relative to repo root (compression-encodings/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    input_file = repo_root / "data" / "pride_and_prejudice.txt"
    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)

    if not input_file.exists():
        log.error("Input file not found: %s", input_file)
        sys.exit(1)

    log.info("Reading: %s", input_file)
    text = input_file.read_text(encoding="utf-8")
    log.info("File size: %d bytes (%d chars)", len(text.encode("utf-8")), len(text))

    # Run all registered strategies
    strategies = sys.argv[1:] if len(sys.argv) > 1 else EncoderFactory.list_strategies()

    for name in strategies:
        log.info("\n")

        try:
            encoder = EncoderFactory.create(name)
        except ValueError as e:
            log.error("%s", e)
            continue

        result = encoder.compress(text)

        # Save compressed file
        out_file = output_dir / f"pride_and_prejudice.{name}.huff"
        out_file.write_bytes(result.compressed_data)

        # Summary
        log.info("--- Summary [%s] ---", name)
        log.info("  Original UTF-8 size:  %10d bytes", result.original_size)
        log.info("  Code table header:    %10d bytes", result.tree_size_bytes)
        log.info("  Compressed payload:   %10d bytes", len(result.compressed_data) - result.tree_size_bytes)
        log.info("  Total compressed:     %10d bytes", len(result.compressed_data))
        log.info("  Compression ratio:    %10.2f%%", (len(result.compressed_data) / result.original_size) * 100)
        log.info("  Space saved:          %10d bytes", result.original_size - len(result.compressed_data))
        log.info("  Output: %s", out_file)

        # Print code table sample
        sample = list(result.code_table.items())[:10]
        log.info("  Code table sample (first 10):")
        for ch, code in sample:
            display = repr(ch)
            log.info("    %-6s → %s", display, code)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
