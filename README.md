# Compression Encodings

A collection of compression/encoding algorithms implemented in Python, using the **Factory Method** design pattern for clean extensibility.

## Project Structure

```
compression-encodings/
├── huffman/          # Huffman encoding module
│   └── main.py       # Standard + Canonical Huffman
├── data/             # Input text corpora (gitignored)
├── output/           # Compressed output files (gitignored)
└── README.md
```

## Implemented Encodings

| Encoding | Strategy Key | Description |
|----------|-------------|-------------|
| Standard Huffman | `standard` | Classic Huffman tree via min-heap merging |
| Canonical Huffman | `canonical` | Same compression ratio, canonical code ordering |

## Usage

```bash
# Run all registered strategies
python3 huffman/main.py

# Run specific strategies
python3 huffman/main.py standard canonical

# Add input files: place .txt files in data/
```

## Adding a New Encoding

1. Create a new subdirectory (e.g. `lzw/`)
2. Subclass `HuffmanEncoder` (or a shared base) from `huffman/main.py`
3. Register it: `EncoderFactory.register("lzw", LZWEncoder)`

## Compression Results — *Pride and Prejudice*

| Metric | Value |
|--------|-------|
| Original (UTF-8) | 738,046 bytes (721 KB) |
| Standard Huffman | 412,598 bytes (403 KB) |
| Canonical Huffman | 412,598 bytes (403 KB) |
| **Compression Ratio** | **55.9%** |
| **Space Saved** | **325,448 bytes (318 KB)** |

## License

MIT
