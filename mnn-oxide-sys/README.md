# mnn-oxide-sys

[![Crates.io](https://img.shields.io/crates/v/mnn-oxide-sys.svg)](https://crates.io/crates/mnn-oxide-sys)
[![Documentation](https://docs.rs/mnn-oxide-sys/badge.svg)](https://docs.rs/mnn-oxide-sys)

Raw FFI bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network) inference engine.

This crate provides low-level unsafe bindings to MNN's C API. For a safe, idiomatic Rust API, see the [`mnn-oxide`](https://crates.io/crates/mnn-oxide) crate.

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

`MNN_CMAKE_GENERATOR` can optionally override the platform-selected CMake generator when the build host needs a different installed generator (for example, `Visual Studio 18 2026`). The selected value participates in the source-build cache key.

`use-prebuilt` is an explicit user-supplied acquisition mode. It is never enabled by default and requires all of `MNN_PREBUILT_URL` (HTTPS), `MNN_PREBUILT_SHA256` (64 hexadecimal digits), `MNN_PREBUILT_TARGET` (a target label exactly matching Cargo `TARGET`), `MNN_PREBUILT_VERSION` (cache label), and `MNN_PREBUILT_CACHE_DIR`. The gzip-compressed tar archive must contain the wrapper-required public headers, a root `lib/` with a recognized library filename, and a root `mnn-oxide-artifact.json` manifest in schema version 2. Archive traversal paths, links, special entries, and mismatches between the manifest's file inventory and package contents are rejected. The manifest declares the build source URL/revision, target, link mode, enabled backends/modules, build options, toolchain, and notices. Its MNN release version must match the selected `MNNDefine.h` version macros. Those fields remain publisher claims: this crate does not inspect binary architecture, ABI, exported symbols, enabled features, publisher identity, or license validity. File hashes detect package-content mismatches; they do not prove source provenance or attest that binaries were built from the recorded source. Callers remain responsible for provenance, redistribution rights, and confirming compatibility with their toolchain and runtime environment.

Manifest format version 2:

```json
{
  "schema_version": 2,
  "mnn_version": "3.6.1",
  "source_url": "https://github.com/alibaba/MNN.git",
  "source_commit": "0123456789abcdef0123456789abcdef01234567",
  "target": "x86_64-unknown-linux-gnu",
  "link_mode": "static",
  "backends": ["cpu"],
  "optional_modules": [],
  "build_options": {"MNN_BUILD_LLM": false},
  "toolchain": {"cmake": "3.x", "compiler": "g++"},
  "notice_files": ["notices/LICENSE-MNN"],
  "files": {
    "include/MNN/MNNDefine.h": "64-lowercase-hex-sha256",
    "lib/libMNN.a": "64-lowercase-hex-sha256",
    "notices/LICENSE-MNN": "64-lowercase-hex-sha256"
  }
}
```

`target`, `link_mode`, `backends`, `optional_modules`, `source_url`, `source_commit`, `build_options`, and `toolchain` are declarations. `notice_files` contains archive-relative paths and must name existing files. `files` maps every package file except the manifest and `SHA256SUMS` to its SHA-256 digest; the loader verifies the inventory and digests. The declared MNN release version must match the version macros in the selected `MNNDefine.h`. These checks establish package consistency only, not binary ABI, feature contents, source provenance, publisher identity, or legal sufficiency. `scripts/package_mnn_artifact.py` can generate this manifest, copy notices from an MNN source checkout, and produce `SHA256SUMS`; inspect its inputs and output before distributing an artifact. The crate does not publish or select artifact URLs or archive checksums.

The manifest above is illustrative only: its file hashes are placeholders and it is not a valid package manifest. Use the packaging helper to generate hashes from the actual artifact files.

```bash
export MNN_PREBUILT_URL=https://example.invalid/path/to/your-mnn.tar.gz
export MNN_PREBUILT_SHA256=REPLACE_WITH_64_HEX_CHARACTER_SHA256
export MNN_PREBUILT_TARGET=x86_64-unknown-linux-gnu
export MNN_PREBUILT_VERSION=3.6.1-your-build-label
export MNN_PREBUILT_CACHE_DIR="$HOME/.cache/mnn-oxide"
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
| `MNN_CMAKE_GENERATOR` | Optional source-build CMake generator override |
| `MNN_MSVC_RUNTIME` | Select matching `/MT` or `/MD` CRT for source-built MNN and the C++ wrapper |
| `MNN_OPENCL_LIB_DIR` | Optional Windows directory containing `OpenCL.lib`; otherwise the build searches `CUDA_PATH` and `WindowsSdkDir` |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_PREBUILT_URL` | User-supplied HTTPS archive URL; requires `use-prebuilt` |
| `MNN_PREBUILT_SHA256` | Expected SHA-256 for the exact archive bytes |
| `MNN_PREBUILT_TARGET` | User assertion; must exactly match Cargo `TARGET`, but does not validate the binary architecture or ABI |
| `MNN_PREBUILT_VERSION` | User-defined artifact version/cache label |
| `MNN_PREBUILT_CACHE_DIR` | Required directory for downloaded and extracted files |
| `MNN_DEBUG_BUILD` | Print debug info during build |

## License

The `mnn-oxide-sys` wrapper code is dual-licensed under either the Apache License, Version 2.0 or the MIT License, at your option. This does not change the license of the native MNN engine: [Alibaba MNN](https://github.com/alibaba/MNN), including MNN DLLs and other native MNN libraries, is Apache-2.0. If you use or redistribute an MNN library, comply with MNN's license terms and preserve/include applicable license and attribution notices. Particular MNN builds may include components with additional licenses; check the notices shipped with the library distribution.
