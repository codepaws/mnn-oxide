# Native MNN 3.6.1 Acceptance Record

Date: 2026-10-18
MNN source: `d407447ed56c4121a11ccbd266dc184ca1ead0c2`
Build mode: source build, dynamic MNN library, CPU backend; `image-process`, `runtime`, `module`, and `llm` wrapper surfaces enabled.

## Native builds

| Host | Target | Toolchain evidence | Built library |
| --- | --- | --- | --- |
| Windows workstation | `x86_64-pc-windows-msvc` | rustc 1.99.1; CMake generator `Visual Studio 18 2026` | `MNN.dll`; PE machine `IMAGE_FILE_MACHINE_AMD64` |
| p3pro | `x86_64-unknown-linux-gnu` | Ubuntu, Linux 7.0.0-38-generic; rustc 1.98.0; CMake 4.2.3; GCC 15.2.0 | ELF64 x86-64 `libMNN.so`; Build ID `a8a76dd59028f3c3f945b5434e9c3917180141f8` |
| DGX Spark | `aarch64-unknown-linux-gnu` | Ubuntu, Linux 7.0.0-1019-nvidia; rustc 1.99.1; CMake 3.28.3; GCC 13.3.0 | ELF64 AArch64 `libMNN.so`; Build ID `9901d04ae774e61e034f640bd2aadb80bd080335` |

Each host passed the full dynamic CPU suite with the listed optional wrapper features enabled: 45 unit tests, 1 dynamic-link test, 12 ordinary integration tests (3 model tests ignored), 2 callback tests, and 18 doctests (2 model doctests ignored). The Windows test process had the built MNN DLL directory on `PATH`; Linux acceptance used the build's `install/lib` on `LD_LIBRARY_PATH`.

The three ignored integration tests were manually enabled on Windows. Loading the checked-in `tests/models/mobilenetV3.mnn` and creating a session succeeded, while inference failed with `Create execution error: 12`; MNN reported that the benchmark graph has no weights. This fixture cannot qualify inference. Model-backed CPU or accelerator acceptance remains pending a valid, redistribution-compatible fixture.

## Link/package checks

- The package helper test suite passes 4/4 tests.
- The build configuration suite passes 12/12 tests on Windows, p3pro, and DGX Spark. It includes target-specific library naming, prebuilt filename checks, and OpenCL import-library directory validation; the case-sensitive lowercase-name rejection is exercised on Linux.
- Linux native libraries were checked with `file` as ELF64 for their respective architectures. The Windows MNN DLL PE header reports AMD64.
- Static Windows linking has previously passed unit and integration tests with `MNN_MSVC_RUNTIME=MD`. The default `/MT` setting still reports MSVC `LNK4098` against Rust's dynamic CRT.

## Accelerator status

Windows host enumeration finds the AMD Radeon RX 9070 XT and AMD Radeon integrated GPU through OpenCL 2.1 (AMD driver 3683.0) and Vulkan 1.4 (AMD driver 26.9.2). MNN 3.6.1 compiled with OpenCL enabled and passed the Rust suite on Windows (29 unit tests, 1 dynamic-link test, 12 ordinary integration tests with 3 model tests ignored, 2 callback tests, and 8 doctests with 2 model doctests ignored). This machine has the OpenCL runtime DLL and MinGW import archive but no Windows SDK OpenCL import library. For verification, an MSVC-compatible import library was generated under ignored `target/` from the system OpenCL DLL exports and passed via `MNN_OPENCL_LIB_DIR`; the build now documents that variable and fails with a direct message if the import library cannot be found. These checks establish backend compilation and host device enumeration only. No MNN OpenCL or Vulkan session was run, so GPU inference qualification remains open. The RX 9070 XT OpenCL regression check is especially pending until a valid model fixture is available.

The previously supplied `mnn-3.6.1-opencl-rx9070xt-crash.md` was not present in this checkout, and its earlier `D:\projects\detect\mnn-oxide\docs\` path no longer exists. The exact report needs to be restored before its reproduction details can be incorporated here.
