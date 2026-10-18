# MNN Model Qualification Matrix

This matrix separates native crate/build checks from inference against a real
model. A passing library build does not imply model inference or accelerator
qualification. Update a row only after the listed command has completed on that
host; attach CI runs or native acceptance logs as evidence.

## Pinned model fixture

| Field | Value |
| --- | --- |
| Model | PP-LCNet text-line orientation classifier |
| Hugging Face repository | [`baicai1145/pp-ocr-mnn-models`](https://huggingface.co/baicai1145/pp-ocr-mnn-models) |
| Revision | `7746c6c454200777bb56fe0dadd350d58a6753e8` |
| File | `PP-LCNet_x1_0_textline_ori.mnn` (6,694,800 bytes) |
| SHA-256 | `8b63eed38352bd505e4790a4fcb4d213889c83d4b78bcc055a8ec74cbfe1f687` |
| License | Apache-2.0, as declared by the model repository; retain that attribution if redistributing the weights |
| Test input | Zero-filled float32 tensor with shape `[1, 3, 80, 160]` |
| Expected output | Two finite float32 class scores, shape `[1, 2]` |
| Local storage | `target/test-models/` (ignored; the model is not committed) |

The download and smoke run are reproducible with:

```powershell
cargo run --example model_smoke --no-default-features --features cpu,static,build-from-source
```

For a source build, set `MNN_SOURCE_PATH` and `MNN_SOURCE_VERSION` to a clean
checkout and its exact commit, as described in the README. On Windows/MSVC,
set `MNN_MSVC_RUNTIME=MD` when using static MNN linking. The Rust example
downloads the pinned revision, verifies its hash, and uses the cached fixture
on later runs. To use another MNN model file, pass its path as the first
argument or set `MNN_TEST_MODEL`; the example labels that inference as custom,
and the pinned integration qualification still requires the pinned SHA-256.
Downloads have a 120-second overall timeout and a 6,694,800-byte maximum.

## Native host and model coverage

| Host | Rust target | MNN build / backend | Native crate suite | Pinned model inference | Evidence / next action |
| --- | --- | --- | --- | --- | --- |
| Windows workstation | `x86_64-pc-windows-msvc` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, dynamic and static, CPU | Passed 2026-10-19 | Passed 2026-10-19 | Full dynamic suite passed; static model qualification test and example also passed |
| p3pro Linux host | `x86_64-unknown-linux-gnu` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, dynamic, CPU | Passed 2026-10-18 | Not run | Run the Rust model smoke example natively and record output |
| DGX Spark | `aarch64-unknown-linux-gnu` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, dynamic, CPU | Passed 2026-10-18 | Not run | Run the Rust model smoke example natively and record output |
| GitHub Actions Linux | `x86_64-unknown-linux-gnu` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, static, CPU | Pending first workflow run | Pending first workflow run | `model-qualification.yml` |
| GitHub Actions Windows | `x86_64-pc-windows-msvc` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, static, CPU | Pending first workflow run | Pending first workflow run | `model-qualification.yml` |
| GitHub Actions macOS ARM64 (`macos-15`) | `aarch64-apple-darwin` | MNN `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, static, CPU | Pending first workflow run | Pending first workflow run | Explicit ARM64 label and `uname -m` assertion |

The Windows dynamic result ran the full suite, including regular model
integration tests. The static source-build smoke command also passed both the
model qualification test and standalone example. The zero-filled input checks
model loading, session execution, output shape, and finite values; it does not
measure classification accuracy. The ordinary full test command intentionally
leaves model tests ignored unless the fixture path is supplied and the model
qualification test target is explicitly selected.

## Accelerator coverage

| Backend / device | Build or discovery check | Real-model inference | Status |
| --- | --- | --- | --- |
| CPU | Native builds on Windows x86-64, Linux x86-64, and Linux ARM64 | PP-LCNet inference verified on Windows x86-64 | CPU qualification is active; Linux model runs remain pending |
| OpenCL / AMD Radeon RX 9070 XT | Windows OpenCL driver/device enumeration passed; local MNN DLL requested OpenCL but reported no matching backend and selected CPU | Test correctly failed after detecting the CPU fallback | Not qualified; build/runtime backend registration still needs investigation |
| Vulkan / AMD Radeon RX 9070 XT | Windows Vulkan driver/device enumeration passed | PP-LCNet inference passed 2026-10-19; session reported Vulkan and returned two finite scores | Model-backed Vulkan path qualified on this workstation |
| CUDA | Not qualified on this fork | Not run | Pending a supported NVIDIA host and runtime |

CPU model inference is the portable baseline. GPU rows should be marked passed
only after the same model-backed smoke test runs successfully on the named
device/backend; backend compilation or device enumeration alone is not inference
evidence. The Vulkan pass used a device-to-host tensor transfer to seed the
model input and read the output, and verified the session backend reported by
MNN. The OpenCL run is deliberately left unqualified because MNN silently
selected CPU; the same backend assertion caught that fallback.

The GitHub matrix explicitly builds and links MNN statically so the test binary
does not depend on a separately deployed shared library at runtime.
`default_link_mode` now has separate feature-gated tests for the crate's normal
dynamic default and for explicitly selected static linking, so both modes are
checked without applying the default assertion to static workflows.

## Opt-in accelerator inference check

`tests/accelerator_qualification.rs` reuses the pinned PP-LCNet model and checks
that the selected backend can create a session, execute inference, and return
the expected finite outputs. It is ignored during ordinary `cargo test` runs so
CPU-only CI does not pretend to qualify an accelerator. GitHub Actions does
compile-check the Rust pathways for CUDA, OpenCL, and Vulkan on each target,
but does not claim native runtime qualification. Run it on a host with
the corresponding MNN backend and driver installed, selecting one accelerator
feature at a time. The same test is available for CUDA, OpenCL, Vulkan, and
Metal:

```powershell
cargo test --no-default-features --features cpu,dynamic,opencl --test accelerator_qualification -- --ignored --nocapture
cargo test --no-default-features --features cpu,dynamic,vulkan --test accelerator_qualification -- --ignored --nocapture
```

Use `cuda` or `metal` in place of `opencl` or `vulkan` on supported hosts. A
source-built MNN must also be configured with that backend enabled; merely
enabling the Rust feature does not build or install the native runtime. Set
`MNN_TEST_MODEL` to an already downloaded copy to skip model download. A pass
records a successful model-backed run on the selected backend; compilation and
device enumeration alone are not qualification. The test checks the backend
MNN assigned to the session, so a silent fallback to CPU fails the test.
