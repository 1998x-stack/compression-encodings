#!/usr/bin/env python3
"""
Compression Encodings — Unified CLI
====================================
Usage:
    python3 main.py                          # run all registered encoders
    python3 main.py --list                   # list available encoders
    python3 main.py huffman rle lzw          # run specific encoders
    python3 main.py --input myfile.txt       # custom input file
"""

import importlib
import logging
import sys
from pathlib import Path

# ── Ensure all encoder modules are loaded ──────────────────────────────────

ENCODER_MODULES = [
    "huffman.encoder",
    "rle.encoder",
    "shannon_fano.encoder",
    "arithmetic.encoder",
    "lzw.encoder",
    "bwt.encoder",
]

sys.path.insert(0, str(Path(__file__).resolve().parent))

for mod_name in ENCODER_MODULES:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        pass  # module may not exist yet; skip silently


from base import EncoderFactory, get_logger, CompressionResult

log = get_logger("main")

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_input_file() -> Path:
    """Find the first .txt file in data/."""
    candidates = list(DATA_DIR.glob("*.txt"))
    if not candidates:
        log.error("No .txt files found in %s", DATA_DIR)
        sys.exit(1)
    return candidates[0]


def run_encoder(key: str, text: str) -> CompressionResult:
    try:
        encoder = EncoderFactory.create(key)
    except KeyError as e:
        log.error("%s", e)
        return None

    log.info("Running: %s", key)
    result = encoder.compress(text)

    # Save output
    out_file = OUTPUT_DIR / f"pride_and_prejudice.{key}.bin"
    out_file.write_bytes(result.compressed_data)
    log.info("Saved: %s (%d bytes)", out_file, result.total_size)

    return result


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("\nAvailable encoders:")
        for s in EncoderFactory.list_strategies():
            print(f"  {s}")
        return

    # Determine which encoders to run
    if args and not args[0].startswith("--"):
        strategies = [a for a in args if not a.startswith("--")]
    else:
        strategies = EncoderFactory.list_strategies()

    if not strategies:
        log.error("No encoders registered!")
        sys.exit(1)

    # Find input file
    if "--input" in args:
        idx = args.index("--input")
        input_file = Path(args[idx + 1])
    else:
        input_file = find_input_file()

    log.info("=" * 60)
    log.info("Compression Encodings — Batch Run")
    log.info("Input: %s", input_file)
    log.info("Encoders: %s", strategies)
    log.info("=" * 60)

    text = input_file.read_text(encoding="utf-8")
    log.info("File: %d bytes (%d chars)", len(text.encode("utf-8")), len(text))

    results: dict[str, CompressionResult] = {}

    for key in strategies:
        log.info("")
        result = run_encoder(key, text)
        if result:
            results[key] = result

    # ── Comparison Table ───────────────────────────────────────────────────
    if results:
        log.info("")
        log.info("=" * 75)
        log.info("COMPARISON SUMMARY")
        log.info("=" * 75)
        header = f"{'Encoder':<24} {'Original':>10} {'Compressed':>10} {'Ratio':>8} {'Saved':>10}"
        log.info(header)
        log.info("-" * 65)
        # Sort by compressed size (best first)
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_size)
        for key, r in sorted_results:
            log.info(
                "%-24s %10d %10d %7.1f%% %10d",
                key,
                r.original_size,
                r.total_size,
                r.compression_ratio * 100,
                r.original_size - r.total_size,
            )
        log.info("=" * 75)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
