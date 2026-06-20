# Compression Encodings

A collection of **13 lossless compression algorithms** implemented in Python, using the **Factory Method** design pattern for clean extensibility.

## Project Structure

```
compression-encodings/
├── base.py               # Shared ABC: CompressionEncoder, CompressionResult, Factory
├── main.py               # Unified CLI: run all encoders, comparison table, --list, --input
├── huffman/              # Standard + Canonical Huffman
├── shannon_fano/         # Shannon-Fano coding
├── arithmetic/           # Arithmetic Coding (integer precision)
├── range_coding/         # Range Coding (byte-renormalized arithmetic)
├── lzw/                  # Lempel-Ziv-Welch dictionary coding
├── lz77/                 # LZ77 sliding-window
├── lzss/                 # LZSS (LZ77 + flag bits)
├── deflate/              # Simplified DEFLATE (LZSS + Huffman)
├── tunstall/             # Tunstall Coding (Huffman's dual)
├── rle/                  # Run-Length Encoding
├── bwt/                  # Burrows-Wheeler Transform + MTF
├── bwt_huff/             # BWT + MTF + RLE + Huffman (bzip2 core)
├── data/                 # Input text corpora (gitignored)
├── output/               # Compressed output files (gitignored)
└── README.md
```

## Implemented Encodings

| # | Encoding | Key | Type | Description |
|---|----------|-----|------|-------------|
| 1 | Standard Huffman | `huffman` | Entropy | Min-heap merging, prefix-free codes |
| 2 | Canonical Huffman | `huffman-canonical` | Entropy | Same ratio, canonical ordering |
| 3 | Shannon-Fano | `shannon-fano` | Entropy | Recursive frequency split |
| 4 | Arithmetic Coding | `arithmetic` | Entropy | 16-bit integer, near-entropy-limit |
| 5 | Range Coding | `range-coding` | Entropy | 32-bit, byte-renormalized arithmetic |
| 6 | LZW | `lzw` | Dictionary | Variable-width codes, GIF style |
| 7 | LZ77 | `lz77` | Dictionary | Sliding window 4096, (offset,length,next) |
| 8 | LZSS | `lzss` | Dictionary | LZ77 + flag bits, min_match=3 |
| 9 | DEFLATE | `deflate` | Hybrid | LZSS + Huffman, gzip/PNG core |
| 10 | Tunstall | `tunstall` | Entropy | Fixed→variable, Huffman's dual |
| 11 | RLE | `rle` | Run-length | Escape-byte, runs ≥ 3 |
| 12 | BWT+MTF | `bwt` | Transform | Suffix-array BWT + Move-to-Front |
| 13 | BWT+Huff | `bwt-huff` | Hybrid | BWT + MTF + RLE + Huffman (bzip2 core) |

## Usage

```bash
# List all available encoders
python3 main.py --list

# Run all registered encoders
python3 main.py

# Run specific encoders
python3 main.py huffman lzw deflate bwt-huff

# Custom input file
python3 main.py --input data/myfile.txt
```

## Adding a New Encoding

1. Create a new subdirectory (e.g. `lzma/`)
2. Create `encoder.py` with a class inheriting `CompressionEncoder` from `base.py`
3. Implement `algorithm_name()` and `compress(text) -> CompressionResult`
4. Register at bottom of file: `EncoderFactory.register("my-key", MyEncoder)`
5. Add `"mymodule.encoder"` to `ENCODER_MODULES` list in `main.py`

## Compression Results — *Pride and Prejudice* (738,046 bytes / 721 KB)

| Rank | Encoder | Compressed | Ratio | Saved | Type |
|:---:|---------|-----------:|:-----:|------:|------|
| 🥇 | **BWT+Huff** | 208,655 B | **28.3%** | 529,391 B | Hybrid |
| 🥈 | **LZW** | 273,292 B | **37.0%** | 464,754 B | Dictionary |
| 🥉 | **DEFLATE** | 326,987 B | **44.3%** | 411,059 B | Hybrid |
| 4 | LZSS | 375,105 B | 50.8% | 362,941 B | Dictionary |
| 5 | LZ77 | 375,109 B | 50.8% | 362,937 B | Dictionary |
| 6 | Arithmetic | 409,353 B | 55.5% | 328,693 B | Entropy |
| 7 | Range Coding | 409,952 B | 55.5% | 328,094 B | Entropy |
| 8 | Huffman | 412,598 B | 55.9% | 325,448 B | Entropy |
| 9 | Huffman-Can. | 412,598 B | 55.9% | 325,448 B | Entropy |
| 10 | Shannon-Fano | 413,477 B | 56.0% | 324,569 B | Entropy |
| 11 | BWT+MTF | 729,249 B | 98.8% | 8,797 B | Transform* |
| 12 | RLE | 731,896 B | 99.2% | 6,150 B | Run-length |
| 13 | Tunstall | 796,492 B | 107.9% | -58,446 B | Entropy† |

> \* BWT+MTF is a preprocessing transform — needs secondary coding (see bwt-huff)  
> † Tunstall expands small-alphabet text; designed for fixed-rate channels, not compression

### Key Insights

- **BWT+Huff dominates** at 28.3% — transforms expose redundancy that entropy coders exploit
- **LZW** (37%) and **DEFLATE** (44%) excel on English text via dictionary matching
- **LZ77 ≈ LZSS** — essentially identical performance, LZSS is cleaner format
- **Arithmetic > Huffman > Shannon-Fano** — as information theory predicts
- **Range Coding ≈ Arithmetic** — byte vs bit renormalization, same theoretical bound
- **RLE / BWT alone** are weak on natural language — they shine in pipelines

### Theoretical Alignment

| Theory | Observed |
|--------|----------|
| Shannon entropy ≈ 4.45 bits/char | Arithmetic: 4.49 bits/char ✅ |
| BWT pipelines → ~2.3 bits/char | BWT+Huff: 2.29 bits/char ✅ |
| LZ-based > entropy-only on text | LZW/DEFLATE beat all entropy coders ✅ |

## License

MIT
