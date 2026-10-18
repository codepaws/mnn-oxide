# mnn-oxide

[![License](https://img.shields.io/crates/l/mnn-oxide.svg)](https://github.com/codepaws/mnn-oxide#license)

Rust bindings for [MNN](https://github.com/alibaba/MNN) (Mobile Neural Network), Alibaba's efficient and lightweight deep learning inference framework.

## Project lineage

This Rust wrapper is derived from [wapznw/mnn-rs](https://github.com/wapznw/mnn-rs). The wrapped inference engine is [Alibaba MNN](https://github.com/alibaba/MNN).

The workspace's minimum supported Rust version is 1.88.

## Features

- **Safe Rust API**: All MNN operations are wrapped in safe Rust types with proper error handling
- **Cross-platform codebase**: Native runtime qualification remains pending for each target and backend
- **Explicit native dependency**: Cargo never downloads, clones, or builds the MNN engine in its default configuration.
- **Backend options**: CPU, CUDA, OpenCL, Vulkan, and Metal when present in the selected MNN library
- **Static/Dynamic Linking**: Choose between static or dynamic linking
- **Local async facade**: Optional async-compatible calls that keep native handles thread-confined; MNN work still blocks the polling thread
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

Prebuilt acquisition is opt-in: Cargo downloads an artifact only when `use-prebuilt` is enabled and all required `MNN_PREBUILT_*` settings are provided. The gzip-compressed tar must contain the wrapper-required headers, a root `lib/` directory with the selected link-mode library, and an `mnn-oxide-artifact.json` manifest using schema version 2. The manifest declares source URL and commit, build options, toolchain, target, link mode, backends, optional modules, notices, and a per-file SHA-256 inventory. The build verifies the header release version, target/link mode, required feature declarations, notice presence, and that package files match the manifest inventory. These remain publisher claims: this crate does not inspect the native binary's architecture, ABI, symbols, or enabled feature contents, nor does it prove provenance, publisher identity, or legal sufficiency. File and archive hashes detect byte mismatches but do not attest how a binary was built. You are responsible for confirming artifact provenance, redistribution rights, and runtime compatibility. The local helper `scripts/package_mnn_artifact.py` prepares a manifest, copies upstream notices, and writes `SHA256SUMS`; inspect results before distribution. Tagged project releases currently publish bundles containing static MNN libraries for their listed targets and backends. Cargo does not automatically select or download those assets, and there is no dynamic-link release bundle yet. Ordinary Cargo commands never download MNN. Source builds require an existing clean MNN checkout, `MNN_SOURCE_PATH`, and `MNN_SOURCE_VERSION` set to that checkout's full 40-character commit SHA; modified or untracked files are rejected, and Cargo does not clone it. Dynamic linking is selected by default; static linking requires disabling defaults and selecting `static` explicitly.

To use a user-supplied prebuilt archive, set all five variables below. `MNN_PREBUILT_TARGET` must exactly match Cargo's target triple (for example, `x86_64-unknown-linux-gnu`), but it is not a binary architecture check; `MNN_PREBUILT_VERSION` is a cache label chosen by you. The cache stores the verified archive and re-extracts it safely on each build. Archive symlinks, hard links, special files, and paths escaping the archive root are rejected.

```bash
export MNN_PREBUILT_URL=https://example.invalid/path/to/your-mnn.tar.gz
export MNN_PREBUILT_SHA256=REPLACE_WITH_64_HEX_CHARACTER_SHA256
export MNN_PREBUILT_TARGET=x86_64-unknown-linux-gnu
export MNN_PREBUILT_VERSION=3.6.1-your-build-label
export MNN_PREBUILT_CACHE_DIR="$HOME/.cache/mnn-oxide"
cargo build --no-default-features --features cpu,static,use-prebuilt
```

Replace the example URL and placeholder checksum with values from your own artifact source. The example is not a published release or a valid checksum; do not copy it literally.

On MSVC source builds, MNN and the C++ wrapper use the same CRT: `/MT` for
static MNN by default and `/MD` for dynamic MNN. `MNN_MSVC_RUNTIME=MT|MD`
overrides both selections. Static MNN linking can still use `/MD`; this is
useful when the Rust application uses the dynamic CRT and avoids an MSVC
`LNK4098` conflict.

## Usage

### Basic Inference

```rust
use mnn_oxide::{prelude::*, BackendType, ScheduleConfig};

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

### Local Async-Compatible Inference

The optional `async` feature exposes async-compatible methods but executes MNN
calls synchronously on the polling thread. It does not make MNN work
non-blocking or move native handles across threads. For non-blocking inference,
create an application-owned worker thread and pass it owned input/output data.

```rust
use mnn_oxide::{prelude::*, BackendType, ScheduleConfig};

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
use mnn_oxide::{ChatMessage, Llm};

fn main() -> Result<(), mnn_oxide::MnnError> {
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

These Cargo features let the API represent and request a backend; they do not prove that the selected MNN library contains it or that the host has a working driver or device. The current safe API reports accelerator availability as unknown because native probing is not implemented. A successfully created session and correct inference are separate acceptance checks.

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
| `async` | Enable local async-compatible methods (native calls block the polling thread) |

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
| `runtime` | Multi-session runtime sharing through MNN `RuntimeInfo` |

## Examples

See the `examples/` directory for more usage examples:

- `basic_inference.rs` - Basic inference workflow
- `async_inference.rs` - Local async-compatible inference (native calls block the polling thread)
- `gpu_backend.rs` - Using GPU backends
- `image_process.rs` - Image processing with MNN's CV module
- `mnn_cv.rs` - OpenCV-style image operations
- `llm_chat.rs` - Interactive multi-turn LLM chat
- `llm_stream.rs` - Streaming LLM text generation
- `llm_embedding.rs` - Text embedding and cosine similarity
- `model_smoke.rs` - Deterministic CPU inference and output validation with the pinned PP-LCNet model

Note: `async_inference` requires the `async` feature, `gpu_backend` requires
the `cuda` feature, and the `llm_*` examples require the `llm` feature.

```bash
# Run basic inference example
cargo run --example basic_inference -- /path/to/model.mnn
```

To download the pinned Hugging Face MNN model, verify its SHA-256, and run the
model smoke example, configure MNN using one of the acquisition modes above and
run:

```bash
cargo run --example model_smoke --no-default-features --features cpu,static,build-from-source
```

On Windows/MSVC, set `MNN_MSVC_RUNTIME=MD` before this command when building
MNN statically so its CRT matches Rust's MSVC runtime. The example downloads
the pinned model on first use, verifies the SHA-256, and caches it under
`target/test-models/`. Pass a model path as an argument or set `MNN_TEST_MODEL`
to run a different local model; in that mode the example labels the result as
custom, while the pinned qualification test still requires the pinned SHA-256.

The model license, source revision, hash, and current platform/backend
qualification status are recorded in
[`docs/qualification/model-test-matrix.md`](docs/qualification/model-test-matrix.md).

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
| `MNN_CMAKE_GENERATOR` | Optional CMake generator override for source builds; defaults are selected for the target platform |
| `MNN_PREBUILT_URL` | HTTPS URL for a user-supplied MNN `.tar.gz` archive; used only with `use-prebuilt` |
| `MNN_PREBUILT_SHA256` | Expected 64-character hexadecimal SHA-256 of the exact archive bytes |
| `MNN_PREBUILT_TARGET` | User-asserted artifact target label; must exactly match Cargo `TARGET` (the binary architecture/ABI is not inspected) |
| `MNN_PREBUILT_VERSION` | User-defined artifact version/cache label, limited to ASCII letters, digits, `.`, `_`, and `-` |
| `MNN_PREBUILT_CACHE_DIR` | Required cache directory for the verified archive and safe extraction |
| `MNN_LIB_DIR` | Path to pre-built MNN library |
| `MNN_INCLUDE_DIR` | Path to MNN headers |
| `MNN_DEBUG_BUILD` | Print debug information during build |
| `MNN_MSVC_RUNTIME` | Select MSVC C runtime for the source-built MNN library and C++ wrapper: `MT` (static CRT) or `MD` (dynamic CRT). Defaults to `/MT` for static MNN linking and `/MD` for dynamic MNN linking |
| `CUDA_PATH` | CUDA installation path |
| `MNN_CUDA_ARCHS` | Optional CMake CUDA architecture list for source builds (for example `12.0` or `8.6;8.9`) |
| `MNN_CUDA_NATIVE_ARCH` | Optional MNN CUDA native-architecture mode (`ON`/`OFF`); useful with `MNN_CUDA_ARCHS` to avoid compiling MNN's default legacy architecture list |
| `MNN_OPENCL_LIB_DIR` | Optional Windows path containing `OpenCL.lib` when the OpenCL import library is outside the CUDA toolkit or Windows SDK paths |
| `ANDROID_NDK_HOME` | Android NDK installation path |
| `NDK_HOME` | Android NDK installation path (alternative to `ANDROID_NDK_HOME`) |

For source-built CUDA with a newer toolkit, MNN's default architecture list may
include targets that the installed `nvcc` no longer supports. Set both
`MNN_CUDA_ARCHS` and `MNN_CUDA_NATIVE_ARCH=ON` to build only a supported target
(for example, `12.0` with MNN revisions that include SM 120 support).

## API Documentation

See [https://docs.rs/mnn-oxide](https://docs.rs/mnn-oxide) for full API documentation.

## MNN Version

Tagged releases publish static MNN bundles; Cargo does not automatically select or download them. The `use-prebuilt` route requires an explicitly configured archive and verifies its package checksum; the caller remains responsible for provenance, license rights, architecture/ABI compatibility, and native runtime behavior. Source builds are pinned to the exact commit SHA in `MNN_SOURCE_VERSION`.

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
