# Compression Encodings

A collection of lossless compression algorithms implemented in Python, using the **Factory Method** design pattern for clean extensibility.

## Project Structure

```
compression-encodings/
├── base.py               # Shared ABC: CompressionEncoder, CompressionResult, Factory
├── main.py               # Unified CLI: run all encoders, comparison table, --list, --input
├── huffman/
│   └── encoder.py        # Standard + Canonical Huffman
├── rle/
│   └── encoder.py        # Run-Length Encoding
├── shannon_fano/
│   └── encoder.py        # Shannon-Fano coding
├── arithmetic/
│   └── encoder.py        # Arithmetic Coding (integer precision)
├── lzw/
│   └── encoder.py        # Lempel-Ziv-Welch
├── bwt/
│   └── encoder.py        # Burrows-Wheeler Transform + Move-to-Front
├── data/                 # Input text corpora (gitignored)
├── output/               # Compressed output files (gitignored)
└── README.md
```

## Implemented Encodings

| # | Encoding | Key | Type | Description |
|---|----------|-----|------|-------------|
| 1 | Standard Huffman | `huffman` | Entropy | Min-heap merging, prefix-free codes |
| 2 | Canonical Huffman | `huffman-canonical` | Entropy | Same ratio, canonical ordering |
| 3 | Shannon-Fano | `shannon-fano` | Entropy | Recursive frequency split, Huffman predecessor |
| 4 | Arithmetic Coding | `arithmetic` | Entropy | Integer arithmetic, near-entropy-limit |
| 5 | LZW | `lzw` | Dictionary | Variable-width codes, GIF/compress style |
| 6 | BWT + MTF | `bwt` | Transform | Burrows-Wheeler + Move-to-Front preprocessing |
| 7 | Run-Length Encoding | `rle` | Run-length | Escape-byte based, runs ≥ 3 |

## Usage

```bash
# List all available encoders
python3 main.py --list

# Run all registered encoders
python3 main.py

# Run specific encoders
python3 main.py huffman lzw arithmetic

# Custom input file
python3 main.py --input data/myfile.txt
```

## Adding a New Encoding

1. Create a new subdirectory (e.g. `lz77/`)
2. Create `encoder.py` with a class inheriting `CompressionEncoder` from `base.py`
3. Implement `algorithm_name()` and `compress(text) -> CompressionResult`
4. Register at bottom of file: `EncoderFactory.register("my-key", MyEncoder)`
5. Add `"mymodule.encoder"` to `ENCODER_MODULES` list in `main.py`

## Compression Results — *Pride and Prejudice* (738,046 bytes)

| Rank | Encoder | Compressed | Ratio | Saved | Type |
|------|---------|-----------|-------|-------|------|
| 🥇 | **LZW** | 273,292 B | **37.0%** | 464,754 B | Dictionary |
| 🥈 | **Arithmetic** | 409,353 B | **55.5%** | 328,693 B | Entropy |
| 🥉 | **Huffman** | 412,598 B | **55.9%** | 325,448 B | Entropy |
| 4 | Huffman-Canonical | 412,598 B | 55.9% | 325,448 B | Entropy |
| 5 | Shannon-Fano | 413,477 B | 56.0% | 324,569 B | Entropy |
| 6 | BWT+MTF | 729,249 B | 98.8% | 8,797 B | Transform* |
| 7 | RLE | 731,896 B | 99.2% | 6,150 B | Run-length |

> \* BWT+MTF is a preprocessing transform; it needs a secondary entropy coder to achieve strong compression. Combined with Huffman or RLE it forms the core of bzip2.

### Key Insights

- **LZW dominates** on English text — dictionary-based coding captures repeated words/patterns extremely well
- **Arithmetic > Huffman > Shannon-Fano** — as theory predicts: arithmetic is closest to entropy limit
- Canonical Huffman is identical to standard Huffman in compression (just cleaner header)
- RLE and BWT alone are weak on natural language; they shine when combined with entropy coders

## License

MIT
