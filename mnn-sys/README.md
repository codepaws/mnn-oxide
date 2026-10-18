# mnn-rs-sys

[![Crates.io](https://img.shields.io/crates/v/mnn-rs-sys.svg)](https://crates.io/crates/mnn-rs-sys)
[![Documentation](https://docs.rs/mnn-rs-sys/badge.svg)](https://docs.rs/mnn-rs-sys)

Raw FFI bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network) inference engine.

This crate provides low-level unsafe bindings to MNN's C API. For a safe, idiomatic Rust API, see the [`mnn-rs`](https://crates.io/crates/mnn-rs) crate.

## Features

- **Explicit native dependency**: Cargo does not download or clone the MNN engine
- **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, Metal
- **Static/Dynamic Linking**: Flexible linking options
- **Cross-Platform**: Windows, Linux, macOS, Android, iOS
- **SIMD Optimizations**: SSE, AVX2, AVX512 support for x86

## Usage

### Using an explicitly selected MNN installation

```bash
# Set environment variables
export MNN_LIB_DIR=/path/to/mnn/lib
export MNN_INCLUDE_DIR=/path/to/mnn/include

# Select static or dynamic linking explicitly
cargo build --no-default-features --features cpu,static
```

Source builds require `build-from-source`, `MNN_SOURCE_PATH` pointing to an existing clean checkout, and `MNN_SOURCE_VERSION` set to its full 40-character commit SHA. Modified and untracked files are rejected; Cargo will not clone the engine.

`use-prebuilt` is an explicit user-supplied acquisition mode. It is never enabled by default and requires all of `MNN_PREBUILT_URL` (HTTPS), `MNN_PREBUILT_SHA256` (64 hexadecimal digits), `MNN_PREBUILT_TARGET` (exactly the Cargo target triple), `MNN_PREBUILT_VERSION` (cache label), and `MNN_PREBUILT_CACHE_DIR`. The supplied gzip-compressed tar archive must contain `include/MNN/Interpreter.hpp` plus a root `lib/` containing a recognized library filename for the target and selected static/dynamic link mode. Its bytes are verified before extraction; traversal paths, links, and special archive entries are rejected. The crate does not publish or select artifact URLs or checksums.

```bash
export MNN_PREBUILT_URL=https://example.invalid/path/to/your-mnn.tar.gz
export MNN_PREBUILT_SHA256=REPLACE_WITH_64_HEX_CHARACTER_SHA256
export MNN_PREBUILT_TARGET=x86_64-unknown-linux-gnu
export MNN_PREBUILT_VERSION=3.6.1-your-build-label
export MNN_PREBUILT_CACHE_DIR="$HOME/.cache/mnn-rs"
cargo build --no-default-features --features cpu,static,use-prebuilt
```

The example URL and checksum are placeholders and must not be copied literally. Ordinary Cargo builds do not download MNN; the network request occurs only with the explicit `use-prebuilt` feature and complete configuration.

## Features

| Feature | Description |
|---------|-------------|
| `static` (default) | Static link MNN |
| `dynamic` | Dynamic link MNN. Requires `--no-default-features` (conflicts with the default `static`) |
| `use-prebuilt` | Opt-in user-supplied HTTPS archive, SHA-256 verified before safe extraction |
| `build-from-source` | Build from an existing checkout pinned to a full commit SHA |
| `llm` | Enable MNN LLM engine bindings |
| `cuda` | Enable CUDA backend |
| `opencl` | Enable OpenCL backend |
| `vulkan` | Enable Vulkan backend |
| `metal` | Enable Metal backend |
| `sse` | Enable SSE instructions |
| `avx2` | Enable AVX2 instructions |
| `avx512` | Enable AVX512 instructions |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MNN_SOURCE_PATH` | Path to MNN source directory |
| `MNN_SOURCE_VERSION` | Full 40-character commit SHA matching `MNN_SOURCE_PATH` |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_PREBUILT_URL` | User-supplied HTTPS archive URL; requires `use-prebuilt` |
| `MNN_PREBUILT_SHA256` | Expected SHA-256 for the exact archive bytes |
| `MNN_PREBUILT_TARGET` | Exact Cargo target triple associated with the artifact |
| `MNN_PREBUILT_VERSION` | User-defined artifact version/cache label |
| `MNN_PREBUILT_CACHE_DIR` | Required directory for downloaded and extracted files |
| `MNN_DEBUG_BUILD` | Print debug info during build |

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.
