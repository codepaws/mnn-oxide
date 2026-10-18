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

Source builds require `build-from-source`, `MNN_SOURCE_PATH` pointing to an existing clean checkout, and `MNN_SOURCE_VERSION` set to its full 40-character commit SHA. Modified and untracked files are rejected; Cargo will not clone the engine. `use-prebuilt` currently fails explicitly because checksum-verified artifact acquisition is not implemented.

## Features

| Feature | Description |
|---------|-------------|
| `static` (default) | Static link MNN |
| `dynamic` | Dynamic link MNN. Requires `--no-default-features` (conflicts with the default `static`) |
| `use-prebuilt` | Unsupported until checksum-verified artifact acquisition is implemented |
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
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_DEBUG_BUILD` | Print debug info during build |

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.
