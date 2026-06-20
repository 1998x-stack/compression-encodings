#!/usr/bin/env python3
"""
Tunstall Coding
===============
The dual of Huffman coding:
- Huffman: variable-length codes → fixed-length symbols (characters)
- Tunstall: fixed-length codes → variable-length symbols (strings)

Algorithm:
  1. Compute character frequencies / probabilities.
  2. Choose k = bits per output code (default 12 → 4096 codewords).
  3. Initialize partition with single-character strings.
  4. While |partition| + U - 1 <= 2^k (where U = unique_chars):
     - Find highest-probability string s (prefer |s| >= 2 when available).
     - Replace s with {s + c for c in alphabet}.
     - Single-char strings are never removed; expanding one ADDITIONALLY adds
       its extensions (net +U instead of +U-1).
  5. Assign each string a k-bit code.
  6. Encode: greedy longest-prefix match from the partition.
  7. Pack k-bit codes into bytes.

Header: pickle({parse_table: {string: code_index}, k})
Payload: packed bitstream

Registered as: 'tunstall'
"""

import math
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    CompressionEncoder,
    CompressionResult,
    EncoderFactory,
    get_logger,
)

log = get_logger("tunstall")

# ── Tunstall Encoder ───────────────────────────────────────────────────────


class TunstallEncoder(CompressionEncoder):
    """Tunstall coding: fixed-length output codes → variable-length input strings."""

    DEFAULT_K = 12  # 4096 codewords

    def algorithm_name(self) -> str:
        return "Tunstall Coding"

    def compress(self, text: str) -> CompressionResult:
        log.info("=== Tunstall Coding ===")
        original = len(text.encode("utf-8"))

        # ── 1. Character frequencies ───────────────────────────────────────
        freq = Counter(text)
        unique_chars = len(freq)
        total = sum(freq.values())

        # Probability of each character
        char_probs = {ch: n / total for ch, n in freq.items()}

        log.info("Unique chars: %d", unique_chars)

        # ── 2. Choose k ────────────────────────────────────────────────────
        k = self.DEFAULT_K
        max_codewords = 1 << k
        if unique_chars >= max_codewords:
            k = math.ceil(math.log2(unique_chars)) + 2
            max_codewords = 1 << k
            log.info("Unique chars (%d) >= default 2^k (%d), using k=%d",
                     unique_chars, 1 << self.DEFAULT_K, k)
        log.info("k = %d bits per code → max %d codewords", k, max_codewords)

        # ── 3. Build the partition (set of strings → probability) ──────────
        # Initialize with single-character strings.
        # {string: probability}
        partition: dict[str, float] = {}
        for ch, prob in char_probs.items():
            partition[ch] = prob

        # Single-char strings are permanent — they remain in the partition
        # forever so the greedy parser can always match at least one char.
        permanent: set[str] = set(partition.keys())

        while len(partition) < max_codewords:
            # Find highest-probability string to expand.
            # Prefer multi-char when available; they give net +U-1 growth.
            best_s = None
            best_p = -1.0

            # First pass: look for multi-char candidates
            for s, prob in partition.items():
                if len(s) >= 2 and prob > best_p:
                    best_p = prob
                    best_s = s

            if best_s is None:
                # No multi-char candidates.  Try expanding a single-char
                # string — this ADDS its extensions without removing the
                # original (net +U growth).
                for s, prob in partition.items():
                    if prob > best_p:
                        best_p = prob
                        best_s = s

            if best_s is None:
                break

            # Compute net growth
            if best_s in permanent:
                # Single-char: keep original, add U children → net +U
                net_growth = unique_chars
            else:
                # Multi-char: remove original, add U children → net +U-1
                net_growth = unique_chars - 1

            if len(partition) + net_growth > max_codewords:
                break  # Would exceed codeword limit

            # Perform expansion
            base_p = partition.pop(best_s)
            if best_s in permanent:
                # Put the permanent single-char back
                partition[best_s] = base_p

            for ch, cp in char_probs.items():
                new_s = best_s + ch
                new_p = base_p * cp
                partition[new_s] = new_p

        # ── 4. Assign k-bit codes ──────────────────────────────────────────
        # Sort strings for deterministic code assignment
        sorted_strings = sorted(partition.keys())
        string_to_code: dict[str, int] = {s: i for i, s in enumerate(sorted_strings)}

        max_string_len = max(len(s) for s in partition)
        log.info("Codewords used: %d", len(partition))
        log.info("Max string length in partition: %d", max_string_len)

        # ── 5. Build a trie for fast greedy encoding ───────────────────────
        # Trie node: dict mapping char → (child_dict, is_leaf, code_or_None)
        trie: dict = {}
        for s, code in string_to_code.items():
            node = trie
            for idx, ch in enumerate(s):
                is_last = (idx == len(s) - 1)
                if ch not in node:
                    node[ch] = ({}, is_last, code if is_last else None)
                elif is_last:
                    child_dict, _, _ = node[ch]
                    node[ch] = (child_dict, True, code)
                node = node[ch][0]

        # Encode using the trie: walk greedily, emit code when hitting a leaf
        bit_chunks: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            node = trie
            j = i
            last_code = None
            last_j = i
            while j < n:
                ch = text[j]
                if ch not in node:
                    break
                _, is_leaf, code = node[ch]
                j += 1
                if is_leaf:
                    last_code = code
                    last_j = j
                node = node[ch][0]  # descend

            if last_code is None:
                log.error("Parse failure at position %d char %r", i, text[i])
                raise RuntimeError(
                    f"Tunstall parse failure at position {i}: {text[i]!r}"
                )

            bit_chunks.append(format(last_code, f"0{k}b"))
            i = last_j

        bitstring = "".join(bit_chunks)
        total_bits = len(bitstring)
        log.info("Encoded bits: %d", total_bits)

        # ── 6. Pack into bytes ─────────────────────────────────────────────
        payload, padding = self.bitstring_to_bytes(bitstring)
        log.info("Padding bits: %d", padding)

        # ── 7. Serialize header ────────────────────────────────────────────
        header_obj = {"parse_table": string_to_code, "k": k}
        header = self.serialize_header(header_obj)

        combined = header + payload
        log.info("Header: %d B, Payload: %d B, Total: %d B",
                 len(header), len(payload), len(combined))

        return CompressionResult(
            algorithm="tunstall",
            original_size=original,
            compressed_data=combined,
            header_size=len(header),
            extra_info={
                "k_bits_per_code": k,
                "max_codewords": max_codewords,
                "codewords_used": len(string_to_code),
                "unique_chars": unique_chars,
                "max_string_len": max_string_len,
                "padding_bits": padding,
            },
        )


# ── Register ───────────────────────────────────────────────────────────────

EncoderFactory.register("tunstall", TunstallEncoder)
