# mnn-rs

[![Crates.io](https://img.shields.io/crates/v/mnn-rs.svg)](https://crates.io/crates/mnn-rs)
[![Documentation](https://docs.rs/mnn-rs/badge.svg)](https://docs.rs/mnn-rs)
[![License](https://img.shields.io/crates/l/mnn-rs.svg)](https://github.com/codepaws/mnn-rs#license)

Rust bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network), Alibaba's efficient and lightweight deep learning inference framework.

## Project lineage

This Rust wrapper is derived from [wapznw/mnn-rs](https://github.com/wapznw/mnn-rs). The wrapped inference engine is [Alibaba MNN](https://github.com/alibaba/MNN).

## Features

- **Safe Rust API**: All MNN operations are wrapped in safe Rust types with proper error handling
- **Cross-platform codebase**: Native runtime qualification remains pending for each target and backend
- **Explicit native dependency**: Cargo never downloads, clones, or builds the MNN engine in its default configuration.
- **Backend options**: CPU, CUDA, OpenCL, Vulkan, and Metal when present in the selected MNN library
- **Static/Dynamic Linking**: Choose between static or dynamic linking
- **Async Support**: Optional async API with Tokio integration
- **LLM Support**: Run large language models (Qwen, Llama, ChatGLM, etc.) locally
- **Build from Source**: Option to build MNN locally when needed

## Quick Start

### Configure an installed MNN

Supply matching headers and libraries explicitly. A clean `cargo build` without these paths fails with an actionable configuration error; it does not acquire the engine.

```bash
# Set MNN library paths
export MNN_LIB_DIR=/path/to/mnn/lib
export MNN_INCLUDE_DIR=/path/to/mnn/include

# Default features use dynamic linking
cargo build

# To request static linking instead
cargo build --no-default-features --features cpu,static
```

### Linking a Dynamic MNN (DLL)

To link against a system dynamic MNN (e.g. a Windows Dynamic/MD package):

```bash
# Windows (set MNN library paths)
set MNN_LIB_DIR=C:\path\to\mnn\lib
set MNN_INCLUDE_DIR=C:\path\to\mnn\include

# Dynamic linking
cargo build --no-default-features --features cpu,dynamic,system-mnn
```

Prebuilt acquisition is opt-in and accepts only an artifact you supply: Cargo downloads it only when `use-prebuilt` is enabled and all required `MNN_PREBUILT_*` settings are provided. The gzip-compressed tar must contain `include/MNN/Interpreter.hpp`, a root `lib/` directory with the selected link-mode library, and a root `mnn-rs-artifact.json` manifest. The manifest declares schema version, MNN version, target triple, link mode, backends, optional modules, and archive-relative license/notice files. The build rejects target or link-mode mismatches, missing required Cargo-feature declarations, and missing named notice files. These manifest values are publisher declarations, not verified binary facts: the binary architecture, ABI, MNN version, and feature contents are not inspected. SHA-256 verifies archive bytes, not publisher, provenance, license validity, or compatibility. You are responsible for confirming the artifact's provenance, license rights, and compatibility with your toolchain and runtime. This project does not publish or select a prebuilt MNN artifact. Ordinary Cargo commands never download MNN. Source builds require an existing clean MNN checkout, `MNN_SOURCE_PATH`, and `MNN_SOURCE_VERSION` set to that checkout's full 40-character commit SHA; modified or untracked files are rejected, and Cargo does not clone it. Dynamic linking is selected by default; static linking requires disabling defaults and selecting `static` explicitly.

To use a user-supplied prebuilt archive, set all five variables below. `MNN_PREBUILT_TARGET` must exactly match Cargo's target triple (for example, `x86_64-unknown-linux-gnu`), but it is not a binary architecture check; `MNN_PREBUILT_VERSION` is a cache label chosen by you. The cache stores the verified archive and re-extracts it safely on each build. Archive symlinks, hard links, special files, and paths escaping the archive root are rejected.

```bash
export MNN_PREBUILT_URL=https://example.invalid/path/to/your-mnn.tar.gz
export MNN_PREBUILT_SHA256=REPLACE_WITH_64_HEX_CHARACTER_SHA256
export MNN_PREBUILT_TARGET=x86_64-unknown-linux-gnu
export MNN_PREBUILT_VERSION=3.6.1-your-build-label
export MNN_PREBUILT_CACHE_DIR="$HOME/.cache/mnn-rs"
cargo build --no-default-features --features cpu,static,use-prebuilt
```

Replace the example URL and placeholder checksum with values from your own artifact source. The example is not a published release or a valid checksum; do not copy it literally.

On MSVC, the C++ wrapper is compiled with the CRT matching the link mode:
`/MT` for static MNN, `/MD` for dynamic MNN. If your MNN library was built
with a different CRT, override it with `MNN_MSVC_RUNTIME=MT|MD`.

## Usage

### Basic Inference

```rust
use mnn_rs::{prelude::*, BackendType, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Load model
    let interpreter = Interpreter::from_file("model.mnn")?;

    // Create session
    let config = ScheduleConfig::new()
        .backend(BackendType::CPU)
        .num_threads(4);
    let mut session = interpreter.create_session(config)?;

    // Get input tensor and write data
    let input = session.get_input(None)?;
    let input_data: Vec<f32> = vec![0.0; input.element_count()?];
    input.write(&input_data)?;

    // Run inference
    session.run()?;

    // Read output
    let output = session.get_output(None)?;
    let output_data: Vec<f32> = output.read()?;

    println!("Output: {:?}", &output_data[..10]);
    Ok(())
}
```

### Async Inference (with Tokio)

```rust
use mnn_rs::{prelude::*, BackendType, ScheduleConfig};

#[tokio::main]
async fn main() -> Result<(), MnnError> {
    let interpreter = Interpreter::from_file("model.mnn")?;
    let config = ScheduleConfig::new().backend(BackendType::CPU);
    let mut session = interpreter.create_session(config)?;

    // Async inference
    session.run_async().await?;

    Ok(())
}
```

### LLM Inference (requires `llm` feature)

```rust
use mnn_rs::{ChatMessage, Llm};

fn main() -> Result<(), mnn_rs::MnnError> {
    // config.json is produced by `transformers/llm/export/llmexport.py`
    let mut llm = Llm::create("model_dir/config.json")?;
    llm.load()?;

    // Streaming generation (closure receives each decoded chunk)
    llm.generate_stream("Hello!", 1, |chunk| print!("{chunk}"))?;
    println!();

    // Single-turn response
    let reply = llm.response("Hello!", Some(128))?;

    // Multi-turn chat with KV-cache reuse
    llm.set_config(r#"{"reuse_kv": true}"#)?;
    let history = vec![
        ChatMessage { role: "system".into(), content: "Be concise.".into() },
        ChatMessage { role: "user".into(), content: "What is Rust?".into() },
    ];
    let reply = llm.response_messages(&history, Some(128))?;

    Ok(())
}
```

## Features

### Linking Mode

| Feature | Description |
|---------|-------------|
| `dynamic` (default) | Dynamically link MNN library |
| `static` | Statically link MNN library. Select with `--no-default-features --features cpu,static` |

### Backend Support

| Feature | Description | Platform |
|---------|-------------|----------|
| `cpu` (default) | CPU backend | All |
| `cuda` | NVIDIA GPU backend | Windows, Linux |
| `opencl` | OpenCL GPU backend | Windows, Linux, macOS, Android |
| `vulkan` | Vulkan GPU backend | Windows, Linux, Android |
| `metal` | Metal GPU backend | macOS, iOS |

### Precision Support

| Feature | Description |
|---------|-------------|
| `fp16` | FP16 precision support |
| `int8` | INT8 precision support |
| `quantization` | Quantization support |

### x86 SIMD Optimizations

| Feature | Description | Default (x86_64) | Default (x86) |
|---------|-------------|------------------|---------------|
| `sse` | SSE instructions | ON | OFF |
| `avx2` | AVX2 instructions | OFF | OFF |
| `avx512` | AVX512 instructions | OFF | OFF |

### Build Options

| Feature | Default | Description |
|---------|---------|-------------|
| `use-prebuilt` | | Opt-in download of a user-supplied HTTPS tar.gz, verified by SHA-256; requires the five `MNN_PREBUILT_*` variables |
| `build-from-source` | | Build MNN from an existing checkout pinned to a full commit SHA |
| `system-mnn` | | Use system-installed MNN library |

### Async Support

| Feature | Description |
|---------|-------------|
| `async` | Enable async API with Tokio |

### LLM Support

| Feature | Description |
|---------|-------------|
| `llm` | Enable MNN LLM engine bindings (`Llm`, `Embedding`) |

The explicitly selected MNN library must include the LLM engine for `llm` to work:

```bash
cargo build --features llm
```

When building MNN from source, the `llm` feature adds
`MNN_BUILD_LLM=ON` (which turns on `MNN_LOW_MEMORY` and
`MNN_SUPPORT_TRANSFORMER_FUSE`), disables the MNN demo executables, and turns
off `LLM_SUPPORT_HTTP_RESOURCE` (the bundled cpp-httplib does not support
32-bit platforms, so models must be loaded from local files):

```bash
cargo build --features cpu,dynamic,llm,build-from-source --no-default-features
```

The LLM C++ engine (`MNN::Transformer::Llm` / `Embedding`) is merged into
`libMNN.a`/`mnn.lib` for static builds.

### Extended API Features

| Feature | Description |
|---------|-------------|
| `image-process` | Image processing support (requires MNN with the CV module) |
| `module` | Dynamic computation graph support (MNN Expr/Module) |
| `runtime` | Multi-session runtime sharing (MNN RuntimeManager) |

## Examples

See the `examples/` directory for more usage examples:

- `basic_inference.rs` - Basic inference workflow
- `async_inference.rs` - Async inference with Tokio
- `gpu_backend.rs` - Using GPU backends
- `image_process.rs` - Image processing with MNN's CV module
- `mnn_cv.rs` - OpenCV-style image operations
- `llm_chat.rs` - Interactive multi-turn LLM chat
- `llm_stream.rs` - Streaming LLM text generation
- `llm_embedding.rs` - Text embedding and cosine similarity

Note: `async_inference` requires the `async` feature, `gpu_backend` requires
the `cuda` feature, and the `llm_*` examples require the `llm` feature.

```bash
# Run basic inference example
cargo run --example basic_inference -- /path/to/model.mnn
```

## Cross-Compilation

Configure a target-compatible MNN installation explicitly. Source builds use an existing checkout at an exact commit SHA:

```bash
cargo build --target aarch64-linux-android --features cpu,dynamic,build-from-source --no-default-features
```

### Android

```bash
# Install Android target
rustup target add aarch64-linux-android

# Build for Android with explicit MNN paths
cargo build --target aarch64-linux-android --no-default-features --features cpu,static

# Or build from source (requires NDK and Ninja)
export ANDROID_NDK_HOME=/path/to/android-ndk
cargo build --target aarch64-linux-android --features cpu,dynamic,build-from-source --no-default-features
```

**Rust Android target identifiers (runtime qualification pending):**
- `aarch64-linux-android` (arm64-v8a)
- `armv7-linux-androideabi` (armeabi-v7a)
- `x86_64-linux-android`
- `i686-linux-android`

No first-party Android prebuilt is selected or qualified. The generic `use-prebuilt` route can consume an archive you have confirmed is Android-target compatible; its binary architecture is not inspected. Otherwise supply explicit MNN library and header paths.

**Building from source requirements:**
- Android NDK (set `ANDROID_NDK_HOME` or `NDK_HOME` environment variable)
- Ninja build system (`choco install ninja` on Windows, `brew install ninja` on macOS)

### iOS

```bash
# Install iOS target
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim

# Build for iOS device with explicit MNN paths
cargo build --target aarch64-apple-ios --no-default-features --features cpu,static

# Build for iOS simulator (Apple Silicon Macs)
cargo build --target aarch64-apple-ios-sim

# Build for iOS simulator (Intel Macs)
cargo build --target x86_64-apple-ios --no-default-features --features cpu,static
```

**Rust iOS target identifiers (runtime qualification pending):**
- `aarch64-apple-ios` (iOS device)
- `aarch64-apple-ios-sim` (iOS simulator on Apple Silicon)
- `x86_64-apple-ios` (iOS simulator on Intel Macs)

No first-party iOS prebuilt is selected or qualified. The generic `use-prebuilt` route can consume an archive you have confirmed is iOS-target compatible; its binary architecture is not inspected. Otherwise supply explicit MNN library and header paths.

### Linux

```bash
# Standard build
cargo build --target x86_64-unknown-linux-gnu

# Cross-compile for ARM
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu
```

### macOS

```bash
# Intel Macs
cargo build --target x86_64-apple-darwin

# Apple Silicon Macs
cargo build --target aarch64-apple-darwin
```

### Windows

```bash
# MSVC (recommended)
cargo build --target x86_64-pc-windows-msvc

# MinGW
rustup target add x86_64-pc-windows-gnu
cargo build --target x86_64-pc-windows-gnu --no-default-features --features cpu,static
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MNN_SOURCE_PATH` | Existing MNN source checkout for explicit source builds |
| `MNN_SOURCE_VERSION` | Full 40-character commit SHA matching `MNN_SOURCE_PATH` |
| `MNN_PREBUILT_URL` | HTTPS URL for a user-supplied MNN `.tar.gz` archive; used only with `use-prebuilt` |
| `MNN_PREBUILT_SHA256` | Expected 64-character hexadecimal SHA-256 of the exact archive bytes |
| `MNN_PREBUILT_TARGET` | User-asserted artifact target label; must exactly match Cargo `TARGET` (the binary architecture/ABI is not inspected) |
| `MNN_PREBUILT_VERSION` | User-defined artifact version/cache label, limited to ASCII letters, digits, `.`, `_`, and `-` |
| `MNN_PREBUILT_CACHE_DIR` | Required cache directory for the verified archive and safe extraction |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_DEBUG_BUILD` | Print debug information during build |
| `MNN_MSVC_RUNTIME` | Override MSVC C runtime for the C++ wrapper: `MT` (static CRT) or `MD` (dynamic CRT). Defaults to `/MT` for static linking and `/MD` for dynamic linking |
| `CUDA_PATH` | CUDA installation path |
| `ANDROID_NDK_HOME` | Android NDK installation path |
| `NDK_HOME` | Android NDK installation path (alternative to `ANDROID_NDK_HOME`) |

## API Documentation

See [https://docs.rs/mnn-rs](https://docs.rs/mnn-rs) for full API documentation.

## MNN Version

No prebuilt MNN artifact is selected, published, or qualified by this crate. The `use-prebuilt` route accepts a user-supplied archive and verifies its checksum; the caller remains responsible for provenance, license rights, architecture/ABI compatibility, and native runtime behavior. Source builds are pinned to the exact commit SHA in `MNN_SOURCE_VERSION`.

## License

The Rust wrapper code in this project is dual-licensed. You may use it under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. This dual license applies to the wrapper code; it does not change the license of the MNN engine.

[Alibaba MNN](https://github.com/alibaba/MNN), including MNN DLLs and other native MNN libraries, is licensed under the Apache License, Version 2.0. If you use or redistribute an MNN library with this wrapper, comply with MNN's license terms and preserve/include the applicable license and attribution notices. Components bundled with or linked into a particular MNN build may have additional licenses; check the notices shipped with that MNN distribution.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [MNN](https://github.com/alibaba/MNN) - Alibaba's Mobile Neural Network inference engine
- All contributors who helped with this project
