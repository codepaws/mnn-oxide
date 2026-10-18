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

# Default features use dynamic linking
cargo build

# To request static linking instead
cargo build --no-default-features --features cpu,static
```

Source builds require `build-from-source`, `MNN_SOURCE_PATH` pointing to an existing clean checkout, and `MNN_SOURCE_VERSION` set to its full 40-character commit SHA. Modified and untracked files are rejected; Cargo will not clone the engine.

`use-prebuilt` is an explicit user-supplied acquisition mode. It is never enabled by default and requires all of `MNN_PREBUILT_URL` (HTTPS), `MNN_PREBUILT_SHA256` (64 hexadecimal digits), `MNN_PREBUILT_TARGET` (a target label exactly matching Cargo `TARGET`), `MNN_PREBUILT_VERSION` (cache label), and `MNN_PREBUILT_CACHE_DIR`. The gzip-compressed tar archive must contain `include/MNN/Interpreter.hpp`, a root `lib/` with a recognized library filename, and a root `mnn-rs-artifact.json` manifest. Archive traversal paths, links, and special entries are rejected. The manifest's target and link mode must match this build; its declared backends and optional modules must include the Cargo features selected by this build; every named license/notice file must exist in the archive. These are declarations supplied by the artifact publisher, not binary inspection: the crate does not verify architecture, ABI, MNN version, feature contents, publisher, or license validity. The archive SHA-256 verifies bytes only. Callers remain responsible for provenance, redistribution rights, and confirming the artifact is compatible with their toolchain and runtime environment.

Manifest format version 1:

```json
{
  "schema_version": 1,
  "mnn_version": "3.6.1",
  "target": "x86_64-unknown-linux-gnu",
  "link_mode": "static",
  "backends": ["cpu"],
  "optional_modules": [],
  "notice_files": ["LICENSE-MNN", "THIRD_PARTY_NOTICES.txt"]
}
```

`target`, `link_mode`, `backends`, and `optional_modules` are compatibility declarations. `notice_files` contains archive-relative paths and must name at least one existing file. The MNN version is recorded for users but is not compared with the headers or native library. The crate does not publish or select artifact URLs or checksums.

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
| `dynamic` (default) | Dynamically link MNN |
| `static` | Statically link MNN. Select with `--no-default-features --features cpu,static` |
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
| `MNN_PREBUILT_TARGET` | User assertion; must exactly match Cargo `TARGET`, but does not validate the binary architecture or ABI |
| `MNN_PREBUILT_VERSION` | User-defined artifact version/cache label |
| `MNN_PREBUILT_CACHE_DIR` | Required directory for downloaded and extracted files |
| `MNN_DEBUG_BUILD` | Print debug info during build |

## License

The `mnn-rs-sys` wrapper code is dual-licensed under either the Apache License, Version 2.0 or the MIT License, at your option. This does not change the license of the native MNN engine: [Alibaba MNN](https://github.com/alibaba/MNN), including MNN DLLs and other native MNN libraries, is Apache-2.0. If you use or redistribute an MNN library, comply with MNN's license terms and preserve/include applicable license and attribution notices. Particular MNN builds may include components with additional licenses; check the notices shipped with the library distribution.
