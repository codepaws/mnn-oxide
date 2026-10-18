//! Build script for mnn-oxide-sys. The engine is acquired only through explicit configuration.

use std::env;
use std::path::PathBuf;
use std::process::Command;

#[path = "build_config.rs"]
mod build_config;
#[cfg(feature = "use-prebuilt")]
#[path = "prebuilt.rs"]
mod prebuilt;

#[cfg(feature = "build-from-source")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "build-from-source")]
use std::hash::{Hash, Hasher};

fn selected_msvc_runtime(use_static_mnn: bool) -> String {
    env::var("MNN_MSVC_RUNTIME")
        .ok()
        .map(|value| value.to_uppercase())
        .filter(|value| value == "MT" || value == "MD")
        .unwrap_or_else(|| {
            if use_static_mnn {
                "MT".to_string()
            } else {
                "MD".to_string()
            }
        })
}

/// Get the NDK host tag based on the current host platform
fn get_ndk_host_tag() -> &'static str {
    // Check HOST environment variable first (set by Cargo)
    if let Ok(host) = env::var("HOST") {
        if host.contains("windows") {
            return "windows-x86_64";
        } else if host.contains("apple-darwin") {
            return "darwin-x86_64";
        } else if host.contains("linux") {
            return "linux-x86_64";
        }
    }

    // Fallback to checking the OS
    if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else {
        "linux-x86_64"
    }
}

fn main() {
    // Skip build process for docs.rs - it doesn't need to link native libraries
    // docs.rs sets DOCS_RS=1 environment variable
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-env=MNN_DOCS_RS=1");
        println!("cargo:rustc-env=MNN_SELECTED_HEADER_VERSION=unknown");
        println!("cargo:warning=mnn-oxide-sys: Building documentation on docs.rs, skipping native library linking");
        return;
    }

    // Check if we should print debug info
    let debug_build = env::var("MNN_DEBUG_BUILD").is_ok();

    if debug_build {
        println!("cargo:warning=mnn-oxide-sys: Starting build script");
    }

    // Get target information
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let target_os = get_target_os(&target);
    let target_env = get_target_env(&target);
    let target_arch = get_target_arch(&target);
    if debug_build {
        println!("cargo:warning=mnn-oxide-sys: Target: {}", target);
        println!("cargo:warning=mnn-oxide-sys: Target OS: {}", target_os);
        println!("cargo:warning=mnn-oxide-sys: Target Env: {}", target_env);
        println!("cargo:warning=mnn-oxide-sys: Target Arch: {}", target_arch);
    }

    // Read environment variables
    let mnn_source_path = env::var("MNN_SOURCE_PATH").ok();
    let mnn_version = env::var("MNN_SOURCE_VERSION").ok();
    let mnn_lib_dir = env::var("MNN_LIB_DIR").ok();
    let mnn_include_dir = env::var("MNN_INCLUDE_DIR").ok();
    #[cfg(feature = "use-prebuilt")]
    let prebuilt_url = env::var("MNN_PREBUILT_URL").ok();
    #[cfg(feature = "use-prebuilt")]
    let prebuilt_sha256 = env::var("MNN_PREBUILT_SHA256").ok();
    #[cfg(feature = "use-prebuilt")]
    let prebuilt_target = env::var("MNN_PREBUILT_TARGET").ok();
    #[cfg(feature = "use-prebuilt")]
    let prebuilt_version = env::var("MNN_PREBUILT_VERSION").ok();
    #[cfg(feature = "use-prebuilt")]
    let prebuilt_cache_dir = env::var("MNN_PREBUILT_CACHE_DIR").ok();
    let _mnn_use_system = env::var("MNN_USE_SYSTEM").is_ok() || cfg!(feature = "system-mnn");
    let build_from_source = cfg!(feature = "build-from-source");
    let use_prebuilt = cfg!(feature = "use-prebuilt");

    // Determine linking mode
    let static_link = cfg!(feature = "static");
    let dynamic_link = cfg!(feature = "dynamic");

    let mode = build_config::validate_acquisition(
        use_prebuilt,
        build_from_source,
        _mnn_use_system,
        mnn_lib_dir.is_some(),
        mnn_include_dir.is_some(),
    )
    .unwrap_or_else(|message| panic!("MNN acquisition configuration error: {message}"));
    if build_from_source {
        let source = mnn_source_path.as_deref().expect("build-from-source requires MNN_SOURCE_PATH to an existing checkout; Cargo will not clone MNN");
        let revision = mnn_version.as_deref().expect("build-from-source requires MNN_SOURCE_VERSION set to the exact 40-character source commit");
        if !build_config::is_full_commit_sha(revision) {
            panic!("MNN_SOURCE_VERSION must be a full 40-character commit SHA");
        }
        let output = Command::new("git")
            .args(["-C", source, "rev-parse", "HEAD"])
            .output()
            .expect("Could not run git to verify MNN_SOURCE_PATH revision");
        let actual = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !output.status.success() || !build_config::revision_matches(revision, &actual) {
            panic!("MNN_SOURCE_PATH revision mismatch: expected {revision}, found {actual}");
        }
        let status = Command::new("git")
            .args([
                "-C",
                source,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ])
            .output()
            .expect("Could not inspect MNN_SOURCE_PATH working tree");
        let dirty = String::from_utf8_lossy(&status.stdout).into_owned();
        let (expected_cutlass_revision, cutlass_version) = if cfg!(feature = "cuda") {
            if cfg!(feature = "llm") {
                (
                    Some("b995f933179c22d3fe0d871c3a53d11e4681950f"),
                    Some("v4.0.0"),
                )
            } else {
                (
                    Some("319a389f42b776fae5701afcb943fc03be5b5c25"),
                    Some("v2.9.0"),
                )
            }
        } else {
            (None, None)
        };
        let cutlass_path = cutlass_version.map(|version| {
            PathBuf::from(source)
                .join("3rd_party/cutlass")
                .join(version)
        });
        let actual_cutlass_revision = cutlass_path.as_ref().and_then(|path| {
            let revision = Command::new("git")
                .args(["-C", path.to_str()?, "rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())?;
            let status = Command::new("git")
                .args([
                    "-C",
                    path.to_str()?,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ])
                .output()
                .ok()?;
            (status.status.success() && status.stdout.is_empty()).then_some(revision)
        });
        if !status.status.success()
            || !build_config::checkout_status_is_clean_with_cutlass(
                &dirty,
                expected_cutlass_revision,
                actual_cutlass_revision.as_deref(),
            )
        {
            panic!("MNN_SOURCE_PATH must be at the pinned commit; modified files and unexpected untracked paths are rejected (only MNN's clean, pinned CUDA CUTLASS FetchContent checkout is permitted):\n{dirty}");
        }
        emit_source_rerun_paths(source);
    }

    let use_static = build_config::validate_link_mode(static_link, dynamic_link)
        .unwrap_or_else(|message| panic!("MNN link mode configuration error: {message}"));

    // Explicit paths only. Default Cargo builds never acquire the MNN engine.
    // This allows testing with local artifacts without downloading
    let (lib_dir, include_dir) = if mode == build_config::AcquisitionMode::BringYourOwn {
        // Use environment variables if both are set (highest priority for testing)
        if debug_build {
            println!(
                "cargo:warning=mnn-oxide-sys: Using MNN_LIB_DIR and MNN_INCLUDE_DIR from environment"
            );
        }
        (
            mnn_lib_dir.as_ref().map(|p| PathBuf::from(p)),
            mnn_include_dir.as_ref().map(|p| PathBuf::from(p)),
        )
    } else if mode == build_config::AcquisitionMode::Prebuilt {
        #[cfg(feature = "use-prebuilt")]
        {
            let config = build_config::validate_prebuilt_config(
                prebuilt_url.as_deref(),
                prebuilt_sha256.as_deref(),
                prebuilt_target.as_deref(),
                prebuilt_version.as_deref(),
                prebuilt_cache_dir.as_deref(),
                &target,
            )
            .unwrap_or_else(|message| panic!("MNN prebuilt configuration error: {message}"));
            let requirements = prebuilt::ArtifactRequirements {
                target: target.clone(),
                link_mode: if use_static { "static" } else { "dynamic" }.into(),
                backends: [
                    ("cpu", cfg!(feature = "cpu")),
                    ("cuda", cfg!(feature = "cuda")),
                    ("opencl", cfg!(feature = "opencl")),
                    ("vulkan", cfg!(feature = "vulkan")),
                    ("metal", cfg!(feature = "metal")),
                ]
                .into_iter()
                .filter_map(|(name, enabled)| enabled.then(|| name.to_string()))
                .collect(),
                optional_modules: [
                    ("image-process", cfg!(feature = "image-process")),
                    ("module", cfg!(feature = "module")),
                    ("runtime", cfg!(feature = "runtime")),
                    ("llm", cfg!(feature = "llm")),
                ]
                .into_iter()
                .filter_map(|(name, enabled)| enabled.then(|| name.to_string()))
                .collect(),
            };
            let (lib, include) = prebuilt::acquire(&config, &requirements)
                .unwrap_or_else(|message| panic!("MNN prebuilt acquisition failed: {message}"));
            (Some(lib), Some(include))
        }
        #[cfg(not(feature = "use-prebuilt"))]
        unreachable!("prebuilt mode cannot be selected without the use-prebuilt feature")
    } else if build_from_source {
        get_lib_and_include_dirs(
            true,
            &mnn_source_path,
            &mnn_version,
            &mnn_lib_dir,
            &mnn_include_dir,
            use_static,
            debug_build,
            &target,
            &target_os,
            &target_env,
            &target_arch,
        )
    } else {
        // Explicit bring-your-own paths, no host-wide search.
        (
            mnn_lib_dir.map(PathBuf::from),
            mnn_include_dir.map(PathBuf::from),
        )
    };

    // Compile the C++ wrapper
    let wrapper_dir = PathBuf::from("wrapper");
    let wrapper_cpp = wrapper_dir.join("mnn_wrapper.cpp");

    // The checked-in C++ shim needs headers from the explicitly selected install.
    if include_dir.is_none() {
        panic!(
            "MNN headers not found!\n\
            Set MNN_INCLUDE_DIR and MNN_LIB_DIR to an explicitly installed MNN.\n\
            For an explicit source build, use build-from-source with MNN_SOURCE_PATH and the exact MNN_SOURCE_VERSION commit SHA.\n\
            Cargo will not download or clone the MNN engine."
        );
    }

    if lib_dir.is_none() {
        panic!("MNN library directory is not configured. Set MNN_LIB_DIR alongside MNN_INCLUDE_DIR; no system-wide search or fallback is performed.");
    }

    let selected_header_version = read_mnn_header_version(
        include_dir
            .as_ref()
            .expect("MNN include directory was validated above"),
    )
    .unwrap_or_else(|message| panic!("MNN header version error: {message}"));
    println!("cargo:rustc-env=MNN_SELECTED_HEADER_VERSION={selected_header_version}");
    println!(
        "cargo:rerun-if-changed={}",
        include_dir
            .as_ref()
            .unwrap()
            .join("MNN/MNNDefine.h")
            .display()
    );

    // Build the wrapper library
    let mut build = cc::Build::new();
    build.cpp(true).file(&wrapper_cpp).include(&wrapper_dir);

    // Add MNN include directory if available
    if let Some(ref inc_dir) = include_dir {
        if !inc_dir.as_os_str().is_empty() {
            build.include(inc_dir);
        }
    }

    // Feature-specific compile definitions
    // The selected MNN library must contain the CV module.
    if cfg!(feature = "image-process") {
        build.define("MNN_IMAGE_PROCESS", None);
        // MNN_IMGCODECS and MNN_BUILD_OPENCV require building MNN from source
        // CV codecs are enabled only for source builds.
        if cfg!(feature = "build-from-source") {
            build.define("MNN_BUILD_OPENCV", "ON");
            build.define("MNN_IMGCODECS", "ON");
        }
        println!("cargo:rustc-cfg=feature=\"image-process\"");
    }
    // Note: module feature requires MNN with training module (Express/Module)
    // Optional modules are compiled only in explicit source builds.
    if cfg!(feature = "module") && cfg!(feature = "build-from-source") {
        build.define("MNN_MODULE_ENABLED", None);
        println!("cargo:rustc-cfg=feature=\"module\"");
    } else if cfg!(feature = "module") {
        panic!("'module' requires an MNN source build with the training module");
    }
    // Runtime sharing uses Interpreter::RuntimeInfo and is enabled only for
    // explicit source builds because that API is not required in BYO kits.
    if cfg!(feature = "runtime") && cfg!(feature = "build-from-source") {
        build.define("MNN_RUNTIME_ENABLED", None);
        println!("cargo:rustc-cfg=feature=\"runtime\"");
    } else if cfg!(feature = "runtime") {
        panic!("'runtime' requires an MNN source build with RuntimeInfo support");
    }

    // Compile module wrapper if feature enabled
    // Note: This requires MNN with training module, only available when building from source
    let module_wrapper_cpp = wrapper_dir.join("mnn_module_wrapper.cpp");
    if cfg!(feature = "module")
        && cfg!(feature = "build-from-source")
        && module_wrapper_cpp.exists()
    {
        build.file(&module_wrapper_cpp);
    }

    // Compile LLM wrapper if feature enabled
    // Note: LLM requires MNN built with MNN_BUILD_LLM=ON. Both build-from-source
    // and the selected MNN library must include its headers (installed to <include>/llm/llm.hpp). For source builds
    // also add the source-tree LLM include dir as a fallback in case the install
    // step is incomplete.
    let llm_wrapper_cpp = wrapper_dir.join("mnn_llm_wrapper.cpp");
    if cfg!(feature = "llm") && llm_wrapper_cpp.exists() {
        build.file(&llm_wrapper_cpp);
        if cfg!(feature = "build-from-source") {
            let src_llm_include = mnn_source_path
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| get_target_directory().join("mnn-source"))
                .join("transformers")
                .join("llm")
                .join("engine")
                .join("include");
            if src_llm_include.exists() {
                build.include(&src_llm_include);
            }
        }
        println!("cargo:rustc-cfg=feature=\"llm\"");
    } else if cfg!(feature = "llm") {
        panic!("mnn-oxide-sys: 'llm' feature enabled but mnn_llm_wrapper.cpp is missing (corrupted checkout)");
    }

    // Configure Android NDK compiler for cross-compilation
    if target_os == "android" {
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");

        let ndk_path = std::path::Path::new(&ndk_home);

        // Android API level - must match what MNN was built with
        let api_level = 24;

        // Determine toolchain prefix and target triple based on target
        let (_tool_prefix, target_triple) = match &target_arch[..] {
            "aarch64" => ("aarch64-linux-android", "aarch64-linux-android"),
            "armv7" | "arm" => ("arm-linux-androideabi", "armv7-linux-androideabi"),
            "x86_64" => ("x86_64-linux-android", "x86_64-linux-android"),
            "x86" => ("i686-linux-android", "i686-linux-android"),
            _ => panic!("Unsupported Android architecture: {}", target_arch),
        };

        // Find NDK toolchain directory
        let host_tag = get_ndk_host_tag();
        let toolchain_bin = ndk_path
            .join("toolchains")
            .join("llvm")
            .join("prebuilt")
            .join(host_tag)
            .join("bin");

        if !toolchain_bin.exists() {
            panic!(
                "Android NDK toolchain not found.\n\
                Expected at: {:?}\n\
                Make sure ANDROID_NDK_HOME is set correctly and NDK is installed.",
                toolchain_bin
            );
        }

        let exe_suffix = if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        };

        // Set AR tool path - use llvm-ar from NDK
        let ar_path = toolchain_bin.join(format!("llvm-ar{}", exe_suffix));
        if ar_path.exists() {
            // Set for cc-rs to find the archiver
            build.archiver(&ar_path);
            if debug_build {
                println!("cargo:warning=mnn-oxide-sys: Using archiver: {:?}", ar_path);
            }
        }

        // Use clang with proper target and API level
        let clang_path = toolchain_bin.join(format!("clang++{}", exe_suffix));
        if clang_path.exists() {
            build.compiler(&clang_path);
            // Set target with API level
            build.flag(format!("--target={}{}", target_triple, api_level));
            build.flag(format!(
                "--sysroot={}",
                ndk_path
                    .join("toolchains")
                    .join("llvm")
                    .join("prebuilt")
                    .join(host_tag)
                    .join("sysroot")
                    .display()
            ));
        } else {
            panic!(
                "Could not find clang++ in NDK.\n\
                Expected at: {:?}",
                clang_path
            );
        }

        if debug_build {
            println!("cargo:warning=mnn-oxide-sys: Android NDK: {:?}", ndk_path);
            println!(
                "cargo:warning=mnn-oxide-sys: Android target: {}{}",
                target_triple, api_level
            );
        }
    }

    // Set C++ standard and compiler flags based on TARGET (not host)
    if target_env == "msvc" {
        build.flag("/std:c++14");
        // The C ABI wrappers catch MNN's C++ exceptions and translate them to
        // sentinel returns, so MSVC must emit normal C++ unwind semantics.
        build.flag("/EHsc");
        // The wrapper's CRT must match the linked MNN library's CRT:
        //   - static / prebuilt MNN is built with /MT (static CRT)
        //   - dynamic MNN (DLL, e.g. a system Dynamic/MD package) uses /MD
        // Mixing them triggers LNK4098 (LIBCMT conflicts with other libs).
        // Override with MNN_MSVC_RUNTIME=MT|MD when the MNN library was built
        // with a different CRT than the default for the chosen link mode.
        let msvc_runtime = selected_msvc_runtime(use_static);
        build.flag(format!("/{}", msvc_runtime));
        if debug_build {
            println!(
                "cargo:warning=mnn-oxide-sys: MSVC runtime: /{} (use_static={})",
                msvc_runtime, use_static
            );
        }
    } else {
        build.flag_if_supported("-std=c++14");
    }

    // Print compiler info in debug mode
    if debug_build {
        if let Ok(tool) = build.try_get_compiler() {
            println!(
                "cargo:warning=mnn-oxide-sys: C++ compiler: {:?}",
                tool.path()
            );
        }
    }

    build.compile("mnn_wrapper");

    // Configure include path
    if let Some(ref inc_dir) = include_dir {
        println!("cargo:include={}", inc_dir.display());
    }

    // Configure linking
    if let Some(ref lib_path) = lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }

    // Link the library
    let link_type = if use_static { "static" } else { "dylib" };
    // MNN's CMake target is case-sensitive on Unix and emits libMNN.* there.
    println!(
        "cargo:rustc-link-lib={}={}",
        link_type,
        build_config::native_link_name()
    );

    // Configure platform-specific settings
    configure_platform(&target_os, &target_env);

    // Configure backend-specific linking
    configure_backends();

    // Tell Cargo to re-run if relevant env vars change
    println!("cargo:rerun-if-env-changed=MNN_SOURCE_PATH");
    println!("cargo:rerun-if-env-changed=MNN_SOURCE_VERSION");
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_URL");
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_SHA256");
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_TARGET");
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_VERSION");
    println!("cargo:rerun-if-env-changed=MNN_PREBUILT_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MNN_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MNN_USE_SYSTEM");
    println!("cargo:rerun-if-env-changed=MNN_MSVC_RUNTIME");
    println!("cargo:rerun-if-env-changed=MNN_CMAKE_GENERATOR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MNN_OPENCL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MNN_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=MNN_CUDA_NATIVE_ARCH");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=NDK_HOME");

    // Re-run if wrapper changes
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.h");
    println!("cargo:rerun-if-changed=wrapper/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=wrapper/mnn_llm_wrapper.h");
    println!("cargo:rerun-if-changed=wrapper/mnn_llm_wrapper.cpp");

    if debug_build {
        println!("cargo:warning=mnn-oxide-sys: Build script completed successfully");
    }
}

fn read_mnn_header_version(include_dir: &std::path::Path) -> Result<String, String> {
    let path = include_dir.join("MNN/MNNDefine.h");
    let contents = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let read_macro = |name: &str| -> Result<u32, String> {
        contents
            .lines()
            .filter_map(|line| {
                let rest = line.trim_start().strip_prefix("#define")?;
                let mut parts = rest.split_whitespace();
                (parts.next() == Some(name)).then(|| parts.next()).flatten()
            })
            .next()
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| {
                format!(
                    "missing or invalid numeric macro {name} in {}",
                    path.display()
                )
            })
    };
    let major = read_macro("MNN_VERSION_MAJOR")?;
    let minor = read_macro("MNN_VERSION_MINOR")?;
    let patch = read_macro("MNN_VERSION_PATCH")?;
    Ok(format!("{major}.{minor}.{patch}"))
}

/// Tell Cargo to rerun this build script when the source tree changes.
/// Watching the root directory is recursive and avoids emitting a directive
/// for every MNN file (some of which Git quotes on Windows).
fn emit_source_rerun_paths(source: &str) {
    let root = Command::new("git")
        .args(["-C", source, "rev-parse", "--show-toplevel"])
        .output()
        .expect("Could not find MNN source root");
    if !root.status.success() {
        panic!("Could not find MNN source root for Cargo change tracking");
    }
    let root = PathBuf::from(String::from_utf8_lossy(&root.stdout).trim());
    println!("cargo:rerun-if-changed={}", root.display());
}

/// Parse target OS from target triple
fn get_target_os(target: &str) -> String {
    // Check for Android first (e.g., aarch64-linux-android)
    // Android targets have "android" in the triple, but also contain "linux"
    if target.contains("android") {
        return "android".to_string();
    }
    if target.contains("ios") {
        return "ios".to_string();
    }

    let parts: Vec<&str> = target.split('-').collect();
    for part in &parts {
        match *part {
            "windows" => return "windows".to_string(),
            "linux" => return "linux".to_string(),
            "macos" | "darwin" => return "macos".to_string(),
            _ => {}
        }
    }
    "unknown".to_string()
}

/// Parse target environment from target triple
fn get_target_env(target: &str) -> String {
    let parts: Vec<&str> = target.split('-').collect();
    if let Some(env) = parts.last() {
        match *env {
            "gnu" | "gnueabi" | "gnueabihf" => return "gnu".to_string(),
            "msvc" => return "msvc".to_string(),
            "android" => return "android".to_string(),
            "androideabi" => return "android".to_string(),
            _ => {}
        }
    }
    if target.contains("windows") {
        "msvc".to_string()
    } else {
        "gnu".to_string()
    }
}

/// Parse target architecture from target triple
fn get_target_arch(target: &str) -> String {
    if let Some(first) = target.split('-').next() {
        match first {
            "x86_64" | "x86-64" | "amd64" => return "x86_64".to_string(),
            "i686" | "i586" | "i386" | "x86" => return "x86".to_string(),
            "aarch64" | "arm64" => return "aarch64".to_string(),
            "arm" | "armv7" | "thumbv7" => return "arm".to_string(),
            _ => return first.to_string(),
        }
    }
    "unknown".to_string()
}

/// Get the project's target directory
fn get_target_directory() -> PathBuf {
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let mut current = out_dir.as_path();
    for _ in 0..4 {
        if let Some(parent) = current.parent() {
            current = parent;
        }
    }

    if current.file_name().map(|n| n == "target").unwrap_or(false) {
        return current.to_path_buf();
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    manifest_dir.join("target")
}

/// Get library and include directories for an explicit acquisition selection.
fn get_lib_and_include_dirs(
    build_from_source: bool,
    mnn_source_path: &Option<String>,
    mnn_version: &Option<String>,
    mnn_lib_dir: &Option<String>,
    mnn_include_dir: &Option<String>,
    use_static: bool,
    debug: bool,
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
) -> (Option<PathBuf>, Option<PathBuf>) {
    if build_from_source {
        // Get the project's target directory (shared across all builds)
        let target_dir = get_target_directory();

        let source_path = PathBuf::from(
            mnn_source_path
                .as_deref()
                .expect("build-from-source requires MNN_SOURCE_PATH; Cargo never clones MNN"),
        );

        // Keep output isolated by target, verified source revision, and source
        // checkout path so a changed pin can never reuse stale native output.
        let revision = mnn_version.as_deref().expect("validated source revision");
        let cache_id = build_config::source_cache_identity(&source_path, revision);
        let source_tree_mode = if target_os == "android" || cfg!(feature = "llm") {
            "isolated"
        } else {
            "direct"
        };
        let build_dir = target_dir
            .join("mnn-build")
            .join(target)
            .join(format!("{cache_id}-{source_tree_mode}"));

        let (lib_dir, include_dir) = build_mnn_from_source(
            &source_path,
            &build_dir,
            use_static,
            debug,
            target,
            target_os,
            target_env,
            target_arch,
        );
        (Some(lib_dir), Some(include_dir))
    } else {
        // Use environment variables
        (
            mnn_lib_dir.as_ref().map(PathBuf::from),
            mnn_include_dir.as_ref().map(PathBuf::from),
        )
    }
}

/// Compute configuration hash for cache invalidation
#[cfg(feature = "build-from-source")]
fn compute_config_hash(
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
    use_static: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash target info
    target.hash(&mut hasher);
    target_os.hash(&mut hasher);
    target_env.hash(&mut hasher);
    target_arch.hash(&mut hasher);
    use_static.hash(&mut hasher);
    std::env::var("MNN_CMAKE_GENERATOR")
        .unwrap_or_default()
        .hash(&mut hasher);

    // The MSVC CRT choice maps to MNN_WIN_RUNTIME_MT in build-from-source.
    // Hash it so the cache invalidates when the runtime selection changes
    // (otherwise a stale dynamic build could keep a /MT-built MNN).
    if target_os == "windows" && target_env == "msvc" {
        let msvc_runtime = selected_msvc_runtime(use_static);
        msvc_runtime.hash(&mut hasher);
    }

    // Hash features
    cfg!(feature = "cuda").hash(&mut hasher);
    cfg!(feature = "opencl").hash(&mut hasher);
    cfg!(feature = "vulkan").hash(&mut hasher);
    cfg!(feature = "metal").hash(&mut hasher);
    cfg!(feature = "fp16").hash(&mut hasher);
    cfg!(feature = "int8").hash(&mut hasher);
    cfg!(feature = "quantization").hash(&mut hasher);
    cfg!(feature = "sse").hash(&mut hasher);
    cfg!(feature = "avx2").hash(&mut hasher);
    cfg!(feature = "avx512").hash(&mut hasher);
    cfg!(feature = "image-process").hash(&mut hasher);
    cfg!(feature = "module").hash(&mut hasher);
    cfg!(feature = "runtime").hash(&mut hasher);
    cfg!(feature = "llm").hash(&mut hasher);
    build_config::source_build_layout_args().hash(&mut hasher);

    hasher.finish()
}

#[cfg(feature = "build-from-source")]
/// Workaround for upstream MNN bug: when MNN_SEP_BUILD is OFF, the `llm` target
/// in transformers/llm/engine is an
/// OBJECT library, but its CMakeLists.txt attaches a POST_BUILD command to it.
/// Modern CMake rejects POST_BUILD on OBJECT libraries, which fails Android
/// builds. Affects MNN 3.6.1 and current main. Patch a disposable build copy
/// (idempotently) to copy headers at configure time instead. Mirrors
/// scripts/patch_mnn_llm_cmake.py.
fn patch_llm_cmake_for_object_library(source_path: &std::path::Path, debug: bool) {
    let cmake_file = source_path
        .join("transformers")
        .join("llm")
        .join("engine")
        .join("CMakeLists.txt");
    if !cmake_file.exists() {
        return; // MNN version without the LLM engine; nothing to patch
    }
    let original = match std::fs::read_to_string(&cmake_file) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=mnn-oxide-sys: could not read {}: {}",
                cmake_file.display(),
                e
            );
            return;
        }
    };
    // Already patched (idempotency check)
    if original.contains("file(COPY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION") {
        return;
    }
    const NEEDLE: &str = r#"IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
add_custom_command(
  TARGET llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ELSE()"#;
    const REPLACEMENT: &str = r#"IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
IF(NOT MNN_SEP_BUILD)
  # mnn-oxide workaround for upstream MNN bug: `llm` is an OBJECT library when
  # MNN_SEP_BUILD is OFF, and CMake forbids POST_BUILD on OBJECT libraries.
  # Copy headers at configure time instead.
  file(COPY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${NATIVE_INCLUDE_OUTPUT})
ELSE()
add_custom_command(
  TARGET llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ENDIF()
ELSE()"#;
    if !original.contains(NEEDLE) {
        println!(
            "cargo:warning=mnn-oxide-sys: skipping CMake workaround patch (pattern not found) in {}",
            cmake_file.display()
        );
        return;
    }
    let patched = original.replace(NEEDLE, REPLACEMENT);
    match std::fs::write(&cmake_file, patched) {
        Ok(()) => {
            if debug {
                println!(
                    "cargo:warning=mnn-oxide-sys: patched {} (llm OBJECT-library POST_BUILD workaround)",
                    cmake_file.display()
                );
            }
        }
        Err(e) => println!(
            "cargo:warning=mnn-oxide-sys: failed to patch {}: {}",
            cmake_file.display(),
            e
        ),
    }
}

/// Build MNN from source using CMake
#[cfg(feature = "build-from-source")]
fn build_mnn_from_source(
    source_path: &std::path::Path,
    build_dir: &std::path::Path,
    _static: bool,
    debug: bool,
    target: &str,
    target_os: &str,
    target_env: &str,
    target_arch: &str,
) -> (PathBuf, PathBuf) {
    // Install directory is inside build_dir
    let install_dir = build_dir.join("install");

    if debug {
        println!("cargo:warning=mnn-oxide-sys: Building MNN from source");
        println!("cargo:warning=mnn-oxide-sys: Source: {:?}", source_path);
        println!("cargo:warning=mnn-oxide-sys: Build dir: {:?}", build_dir);
        println!(
            "cargo:warning=mnn-oxide-sys: Install dir: {:?}",
            install_dir
        );
    }

    // Compute configuration hash
    let config_hash = compute_config_hash(target, target_os, target_env, target_arch, _static);
    let config_file = build_dir.join(".mnn-build-config");

    // Check if we need to rebuild based on configuration changes
    let need_rebuild = if config_file.exists() {
        if let Ok(stored_hash) = std::fs::read_to_string(&config_file) {
            let stored: u64 = stored_hash.trim().parse().unwrap_or(0);
            stored != config_hash
        } else {
            true
        }
    } else {
        true
    };

    // Check if already built with same configuration using MNN's exact CMake
    // output casing and the selected static/dynamic mode.
    let lib_path = install_dir
        .join("lib")
        .join(build_config::native_library_file_name(
            target_os, target_env, _static,
        ));

    if lib_path.exists() && !need_rebuild {
        if debug {
            println!(
                "cargo:warning=mnn-oxide-sys: Using cached MNN build at {:?}",
                install_dir
            );
        }
        return (install_dir.join("lib"), install_dir.join("include"));
    }

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // The LLM object-library workaround edits its input file. Configure from a
    // local copy so the validated MNN checkout remains unchanged. Android also
    // uses an isolated tree for its NDK-specific CMake workaround.
    let cmake_source = if target_os == "android" || cfg!(feature = "llm") {
        let source_copy = build_dir.join("source-tree-tracked");
        let staging_copy = build_dir.join("source-tree-tracked.staging");
        if staging_copy.exists() {
            std::fs::remove_dir_all(&staging_copy)
                .expect("Failed to remove an incomplete staged MNN source copy");
        }
        if source_copy.exists() {
            std::fs::remove_dir_all(&source_copy)
                .expect("Failed to refresh the isolated MNN source copy");
        }
        copy_tracked_source_tree(source_path, &staging_copy)
            .expect("Failed to create isolated MNN source staging copy");
        if cfg!(feature = "llm") {
            patch_llm_cmake_for_object_library(&staging_copy, debug);
        }
        std::fs::rename(&staging_copy, &source_copy)
            .expect("Failed to install the isolated MNN source copy");
        source_copy
    } else {
        source_path.to_path_buf()
    };

    // Store new configuration hash
    std::fs::write(&config_file, config_hash.to_string()).expect("Failed to write config file");

    // Determine generator and architecture based on target
    let (generator, extra_args) = if target_os == "android" {
        // Android requires Ninja build system
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if !ninja_available {
            panic!(
                "Ninja build system is required for Android builds.\n\
                Install it with:\n\
                - Windows: choco install ninja\n\
                - macOS: brew install ninja\n\
                - Linux: apt install ninja-build"
            );
        }
        (Some("Ninja"), vec![])
    } else if target_os == "ios" {
        // iOS requires Ninja or Xcode
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if ninja_available {
            (Some("Ninja"), vec![])
        } else {
            (Some("Xcode"), vec![])
        }
    } else if target_os == "windows" {
        if target_env == "msvc" {
            let arch = match &target_arch[..] {
                "x86_64" => "x64",
                "x86" => "Win32",
                "aarch64" => "ARM64",
                _ => "x64",
            };
            (Some("Visual Studio 17 2022"), vec!["-A", arch])
        } else {
            (Some("MinGW Makefiles"), vec![])
        }
    } else if target_os == "linux" {
        // Linux defaults to Unix Makefiles or Ninja
        (None, vec![])
    } else if target_os == "macos" {
        // macOS: prefer Ninja for consistency, fallback to default
        let ninja_available = Command::new("ninja").arg("--version").output().is_ok();
        if ninja_available {
            (Some("Ninja"), vec![])
        } else {
            // Let CMake choose (usually Unix Makefiles on macOS)
            (None, vec![])
        }
    } else {
        (None, vec![])
    };
    let generator_override = std::env::var("MNN_CMAKE_GENERATOR")
        .ok()
        .filter(|generator| !generator.trim().is_empty());
    let generator = generator_override.as_deref().or(generator);

    // Configure CMake options based on features
    let mut cmake_args = vec![format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display())];
    cmake_args.extend(
        build_config::source_build_layout_args()
            .into_iter()
            .map(str::to_string),
    );

    // On Windows MSVC, build MNN with a CRT matching the link mode so the
    // wrapper and the linked MNN library agree: static -> /MT, dynamic -> /MD
    if target_os == "windows" && target_env == "msvc" {
        let msvc_runtime = selected_msvc_runtime(_static);
        cmake_args.push(format!(
            "-DMNN_WIN_RUNTIME_MT={}",
            if msvc_runtime == "MT" { "ON" } else { "OFF" }
        ));
    }

    // Build static or shared library based on feature
    if _static {
        cmake_args.push("-DMNN_BUILD_SHARED_LIBS=OFF".to_string());
    } else {
        cmake_args.push("-DMNN_BUILD_SHARED_LIBS=ON".to_string());
    }

    // Backend options
    if cfg!(feature = "cuda") {
        cmake_args.push("-DMNN_CUDA=ON".to_string());
        let cuda_args = build_config::cuda_build_args(
            env::var("MNN_CUDA_ARCHS").ok().as_deref(),
            env::var("MNN_CUDA_NATIVE_ARCH").ok().as_deref(),
        )
        .unwrap_or_else(|error| panic!("MNN CUDA build configuration error: {error}"));
        cmake_args.extend(cuda_args);
    }
    if cfg!(feature = "opencl") {
        cmake_args.push("-DMNN_OPENCL=ON".to_string());
    }
    if cfg!(feature = "vulkan") {
        cmake_args.push("-DMNN_VULKAN=ON".to_string());
    }
    if cfg!(feature = "metal") {
        cmake_args.push("-DMNN_METAL=ON".to_string());
    }

    // ImageProcess and CV module options (enabled with image-process feature)
    if cfg!(feature = "image-process") {
        cmake_args.push("-DMNN_BUILD_OPENCV=ON".to_string());
        cmake_args.push("-DMNN_IMGCODECS=ON".to_string());
    }

    // Precision options
    if cfg!(feature = "fp16") {
        cmake_args.push("-DMNN_SUPPORT_FP16=ON".to_string());
    }
    if cfg!(feature = "int8") {
        cmake_args.push("-DMNN_SUPPORT_INT8=ON".to_string());
    }
    if cfg!(feature = "quantization") {
        cmake_args.push("-DMNN_BUILD_QUANT=ON".to_string());
    }

    // LLM engine options - applied only when building MNN from source.
    // Prebuilt binaries already ship the LLM engine (MNN_BUILD_LLM=ON is set
    // by the CI workflow that produces them).
    if cfg!(feature = "llm") {
        cmake_args.push("-DMNN_BUILD_LLM=ON".to_string());
        // MNN_LLM_BUILD_DEMO defaults to ON and would compile 6 demo executables;
        // only the MNN library itself is needed.
        cmake_args.push("-DMNN_LLM_BUILD_DEMO=OFF".to_string());
        // LLM_SUPPORT_HTTP_RESOURCE defaults to ON and pulls in httplib.
        cmake_args.push("-DLLM_SUPPORT_HTTP_RESOURCE=OFF".to_string());
    }

    // The CUDA backend fetches CUTLASS from GitHub via FetchContent. MNN's
    // cuda CMakeLists expects the source at 3rd_party/cutlass/<version> (e.g.
    // v4.0.0 when MNN_SUPPORT_TRANSFORMER_FUSE is ON, which MNN_BUILD_LLM
    // forces). If that pre-placed directory exists, tell FetchContent to trust
    // it and skip all network git operations (offline build support).
    if cfg!(feature = "cuda") {
        let cutlass_version = if cfg!(feature = "llm") {
            "v4.0.0"
        } else {
            "v2.9.0"
        };
        let cutlass_dir = source_path
            .join("3rd_party")
            .join("cutlass")
            .join(cutlass_version);
        if cutlass_dir.join("CMakeLists.txt").exists() {
            cmake_args.push("-DFETCHCONTENT_FULLY_DISCONNECTED=ON".to_string());
            if debug {
                println!(
                    "cargo:warning=mnn-oxide-sys: Found pre-placed CUTLASS at {}, using offline mode",
                    cutlass_dir.display()
                );
            }
        } else {
            println!(
                "cargo:warning=mnn-oxide-sys: CUDA build needs CUTLASS, expected at {} but not found (FetchContent will try to download it)",
                cutlass_dir.display()
            );
        }

        // nvcc needs an absolute --compiler-bindir on Windows. When the build
        // runs without a VS developer prompt, FindCUDA falls back to MSBuild
        // variables ($(VCInstallDir)...) which never expand, so nvcc cannot
        // find cl.exe. Resolve cl.exe via the cc crate and pin CUDA_HOST_COMPILER
        // to the directory that contains it (nvcc's -ccbin expects a directory).
        let cxx_compiler = cc::Build::new().cpp(true).try_get_compiler().ok();
        if let Some(compiler) = cxx_compiler {
            let cl_path = compiler.path();
            if cl_path.exists() {
                let bindir = cl_path.parent().unwrap_or(cl_path);
                cmake_args.push(format!("-DCUDA_HOST_COMPILER={}", bindir.display()));
                if debug {
                    println!(
                        "cargo:warning=mnn-oxide-sys: CUDA_HOST_COMPILER set to {}",
                        bindir.display()
                    );
                }
            }
        }
    }

    // x86 SIMD options (controlled by features)
    // Default: enabled for x86_64, disabled for x86 (32-bit)
    let use_sse = cfg!(feature = "sse")
        || (target_arch == "x86_64" && !cfg!(feature = "sse") && !cfg!(feature = "avx2"));
    let use_avx2 = cfg!(feature = "avx2");
    let use_avx512 = cfg!(feature = "avx512");

    // For 32-bit x86, disable all SIMD by default unless explicitly enabled
    if target_arch == "x86" {
        cmake_args.push(format!(
            "-DMNN_USE_SSE={}",
            if cfg!(feature = "sse") { "ON" } else { "OFF" }
        ));
        cmake_args.push(format!(
            "-DMNN_AVX2={}",
            if cfg!(feature = "avx2") { "ON" } else { "OFF" }
        ));
        cmake_args.push(format!(
            "-DMNN_AVX512={}",
            if cfg!(feature = "avx512") {
                "ON"
            } else {
                "OFF"
            }
        ));
    } else if target_arch == "x86_64" {
        cmake_args.push(format!(
            "-DMNN_USE_SSE={}",
            if use_sse || use_avx2 || use_avx512 {
                "ON"
            } else {
                "OFF"
            }
        ));
        cmake_args.push(format!(
            "-DMNN_AVX2={}",
            if use_avx2 || use_avx512 { "ON" } else { "OFF" }
        ));
        cmake_args.push(format!(
            "-DMNN_AVX512={}",
            if use_avx512 { "ON" } else { "OFF" }
        ));
    }

    // Build type
    cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string());

    // Cross-compilation support
    if target_os == "windows" && target_env == "gnu" {
        if target_arch == "x86" {
            cmake_args.push(String::from("-DCMAKE_C_COMPILER=i686-w64-mingw32-gcc.exe"));
            cmake_args.push(String::from(
                "-DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++.exe",
            ));
            cmake_args.push(String::from("-DCMAKE_SYSTEM_NAME=Windows"));
        } else if target_arch == "x86_64" {
            cmake_args.push(String::from(
                "-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc.exe",
            ));
            cmake_args.push(String::from(
                "-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++.exe",
            ));
            cmake_args.push(String::from("-DCMAKE_SYSTEM_NAME=Windows"));
        }
    }

    // Android NDK cross-compilation support
    if target_os == "android" {
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");

        let ndk_path = std::path::Path::new(&ndk_home);

        // Determine Android ABI and platform
        let (abi, platform, _toolchain_name) = match &target_arch[..] {
            "aarch64" => ("arm64-v8a", "android-24", "aarch64-linux-android"),
            "armv7" | "arm" => ("armeabi-v7a", "android-24", "arm-linux-androideabi"),
            "x86_64" => ("x86_64", "android-24", "x86_64-linux-android"),
            "x86" => ("x86", "android-24", "i686-linux-android"),
            _ => panic!("Unsupported Android architecture: {}", target_arch),
        };

        // Use NDK toolchain file
        let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");
        if !toolchain_file.exists() {
            panic!(
                "Android NDK toolchain file not found at {:?}",
                toolchain_file
            );
        }

        cmake_args.push(format!(
            "-DCMAKE_TOOLCHAIN_FILE={}",
            toolchain_file.display()
        ));
        cmake_args.push(format!("-DANDROID_ABI={}", abi));
        cmake_args.push(format!("-DANDROID_PLATFORM={}", platform));
        cmake_args.push("-DANDROID_STL=c++_static".to_string());
        cmake_args.push("-DCMAKE_SYSTEM_NAME=Android".to_string());

        if debug {
            println!("cargo:warning=mnn-oxide-sys: Android NDK: {:?}", ndk_path);
            println!("cargo:warning=mnn-oxide-sys: Android ABI: {}", abi);
            println!(
                "cargo:warning=mnn-oxide-sys: Android Platform: {}",
                platform
            );
        }

        // Disable KleidiAI on Android as it has linking issues with stdout/stderr
        cmake_args.push("-DMNN_KLEIDIAI=OFF".to_string());
    }

    // iOS cross-compilation support
    if target_os == "ios" {
        // Determine iOS platform and architecture
        let (platform, arch, sdk_name) = match target {
            t if t.starts_with("aarch64-apple-ios") => ("OS64", "arm64", "iphoneos"),
            t if t.starts_with("aarch64-apple-ios-sim") => {
                ("SIMULATORARM64", "arm64", "iphonesimulator")
            }
            t if t.starts_with("x86_64-apple-ios") => ("SIMULATOR64", "x86_64", "iphonesimulator"),
            t if t.starts_with("i386-apple-ios") || t.starts_with("i686-apple-ios") => {
                ("SIMULATOR", "i386", "iphonesimulator")
            }
            t if t.starts_with("armv7-apple-ios") => ("OS", "armv7", "iphoneos"),
            t if t.starts_with("armv7s-apple-ios") => ("OS", "armv7s", "iphoneos"),
            _ => panic!("Unsupported iOS target: {}", target),
        };

        // Set CMake system name
        cmake_args.push("-DCMAKE_SYSTEM_NAME=iOS".to_string());
        cmake_args.push(format!("-DCMAKE_OSX_ARCHITECTURES={}", arch));
        cmake_args.push(format!("-DPLATFORM={}", platform));

        // Enable Metal backend by default for iOS
        if !cfg!(feature = "metal") {
            cmake_args.push("-DMNN_METAL=ON".to_string());
        }

        if debug {
            println!("cargo:warning=mnn-oxide-sys: iOS Platform: {}", platform);
            println!("cargo:warning=mnn-oxide-sys: iOS Arch: {}", arch);
            println!("cargo:warning=mnn-oxide-sys: iOS SDK: {}", sdk_name);
        }
    }

    if debug {
        println!("cargo:warning=mnn-oxide-sys: CMake args: {:?}", cmake_args);
        println!("cargo:warning=mnn-oxide-sys: Generator: {:?}", generator);
    }

    // Run CMake configure
    let mut cmd = Command::new("cmake");

    // Pass generator BEFORE other args (important for Android/Ninja)
    if let Some(gen) = generator {
        cmd.args(["-G", gen]);
    }

    cmd.args(&cmake_args);
    cmd.args(&extra_args);

    cmd.args(["-B", build_dir.to_str().unwrap()]);
    cmd.args(["-S", cmake_source.to_str().unwrap()]);

    if debug {
        println!(
            "cargo:warning=mnn-oxide-sys: CMake configure command: {:?}",
            cmd
        );
    }

    let status = cmd
        .status()
        .expect("Failed to run CMake configure. Make sure cmake is installed.");

    if !status.success() {
        panic!("CMake configure failed");
    }

    // Run CMake build with parallel jobs
    let num_jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus::get);

    // First, build the library
    let mut build_cmd = Command::new("cmake");
    build_cmd
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["--config", "Release"]);

    if target_os == "windows" && target_env == "msvc" {
        build_cmd.args(["--", "/m", &format!("/p:CL_MPCount={}", num_jobs)]);
    } else {
        build_cmd.args(["--", "-j", &num_jobs.to_string()]);
    }

    if debug {
        println!(
            "cargo:warning=mnn-oxide-sys: Build command: {:?}",
            build_cmd
        );
    }

    let build_status = build_cmd.status().expect("Failed to run CMake build");

    if !build_status.success() {
        panic!("CMake build failed");
    }

    // Then, install the library
    let mut install_cmd = Command::new("cmake");
    install_cmd
        .args(["--install", build_dir.to_str().unwrap()])
        .args(["--config", "Release"]);

    if debug {
        println!(
            "cargo:warning=mnn-oxide-sys: Install command: {:?}",
            install_cmd
        );
    }

    let install_status = install_cmd.status().expect("Failed to run CMake install");

    if !install_status.success() {
        // cmake --install might fail on some platforms, try manual installation
        if debug {
            println!(
                "cargo:warning=mnn-oxide-sys: cmake --install failed, trying manual installation"
            );
        }
    }

    // Copy CV headers if image-process feature is enabled
    // MNN CMake doesn't install CV headers by default
    if cfg!(feature = "image-process") {
        let cv_include_dest = install_dir.join("include").join("MNN").join("cv");
        let cv_include_src = source_path
            .join("tools")
            .join("cv")
            .join("include")
            .join("cv");

        if cv_include_src.exists() {
            if debug {
                println!(
                    "cargo:warning=mnn-oxide-sys: Copying CV headers from {:?}",
                    cv_include_src
                );
            }

            // Create destination directory
            std::fs::create_dir_all(&cv_include_dest).ok();

            // Copy header files recursively
            if let Err(e) = copy_dir_all(&cv_include_src, &cv_include_dest) {
                if debug {
                    println!(
                        "cargo:warning=mnn-oxide-sys: Failed to copy CV headers: {}",
                        e
                    );
                }
            }
        }
    }

    // Check if install directory was created
    let installed_lib = install_dir
        .join("lib")
        .join(build_config::native_library_file_name(
            target_os, target_env, _static,
        ));

    let (lib_dir, include_dir) = if installed_lib.exists() {
        // cmake --install worked
        (install_dir.join("lib"), install_dir.join("include"))
    } else {
        // Manual fallback: use build directory and source include
        if debug {
            println!("cargo:warning=mnn-oxide-sys: Using fallback paths for lib and include");
        }

        // Create install directories
        std::fs::create_dir_all(install_dir.join("lib")).ok();
        std::fs::create_dir_all(install_dir.join("include")).ok();

        // Copy library if it exists in build dir
        let lib_in_build = build_dir.join(if target_os == "windows" {
            if target_env == "gnu" {
                "libMNN.a"
            } else {
                "Release/MNN.lib"
            }
        } else {
            build_config::native_library_file_name(target_os, target_env, _static)
        });

        if lib_in_build.exists() {
            let dest = install_dir
                .join("lib")
                .join(installed_lib.file_name().unwrap());
            if !dest.exists() {
                if let Err(e) = std::fs::copy(&lib_in_build, &dest) {
                    if debug {
                        println!("cargo:warning=mnn-oxide-sys: Failed to copy library: {}", e);
                    }
                }
            }
        }

        // Use source include directory directly (most reliable)
        let src_include = source_path.join("include");
        if src_include.exists() {
            (install_dir.join("lib"), src_include)
        } else {
            (install_dir.join("lib"), install_dir.join("include"))
        }
    };

    if debug {
        println!("cargo:warning=mnn-oxide-sys: MNN built successfully");
        println!("cargo:warning=mnn-oxide-sys: Library dir: {:?}", lib_dir);
        println!(
            "cargo:warning=mnn-oxide-sys: Include dir: {:?}",
            include_dir
        );
    }

    (lib_dir, include_dir)
}

/// Copy a directory recursively
#[cfg(feature = "build-from-source")]
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_name() == ".git" {
            continue;
        }
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Copy only Git-indexed source files, recursively including initialized
/// submodules. This excludes `.git` history, ignored build output, and other
/// non-source files from the Android workaround snapshot.
#[cfg(feature = "build-from-source")]
fn copy_tracked_source_tree(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    let root_output = Command::new("git")
        .args([
            "-C",
            src.to_str().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "non-UTF8 MNN source path")
            })?,
            "rev-parse",
            "--show-toplevel",
        ])
        .output()?;
    if !root_output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "MNN source path is not a Git checkout",
        ));
    }
    let root = PathBuf::from(String::from_utf8_lossy(&root_output.stdout).trim());
    let index_output = Command::new("git")
        .args([
            "-C",
            src.to_str().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "non-UTF8 MNN source path")
            })?,
            "ls-files",
            "--stage",
            "--full-name",
            "-z",
        ])
        .output()?;
    if !index_output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "failed to enumerate tracked MNN source files",
        ));
    }
    let entries = crate::build_config::parse_index_paths(&index_output.stdout)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    std::fs::create_dir_all(dst)?;
    for (relative_path, is_submodule) in entries {
        let source_file = root.join(&relative_path);
        let destination = dst.join(&relative_path);
        if is_submodule {
            copy_tracked_source_tree(&source_file, &destination)?;
        } else {
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(source_file, destination)?;
        }
    }
    Ok(())
}

/// Build from source stub when cmake feature is not enabled
#[cfg(not(feature = "build-from-source"))]
fn build_mnn_from_source(
    _source_path: &std::path::Path,
    _build_dir: &std::path::Path,
    _static: bool,
    _debug: bool,
    _target: &str,
    _target_os: &str,
    _target_env: &str,
    _target_arch: &str,
) -> (PathBuf, PathBuf) {
    panic!(
        "'build-from-source' feature is not enabled. Enable it with --features build-from-source"
    );
}

/// Configure platform-specific settings
fn configure_platform(target_os: &str, target_env: &str) {
    if target_os == "windows" {
        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=cudart");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
            // Windows ships the OpenCL loader DLL with many GPU drivers, but
            // MSVC still needs an import library to link against it. Accept an
            // explicit directory first, then search the CUDA toolkit and SDK.
            let mut found_import_library = false;
            if let Ok(opencl_lib_dir) = env::var("MNN_OPENCL_LIB_DIR") {
                let path = PathBuf::from(opencl_lib_dir);
                if !build_config::opencl_import_library_exists(&path) {
                    panic!(
                        "MNN_OPENCL_LIB_DIR must contain OpenCL.lib (got {})",
                        path.display()
                    );
                }
                println!("cargo:rustc-link-search=native={}", path.display());
                found_import_library = true;
            }
            if !found_import_library {
                if let Ok(cuda_path) = env::var("CUDA_PATH") {
                    let cuda_lib = PathBuf::from(cuda_path).join("lib").join("x64");
                    if build_config::opencl_import_library_exists(&cuda_lib) {
                        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
                        found_import_library = true;
                    }
                }
            }
            if !found_import_library {
                if let Ok(kit_root) = env::var("WindowsSdkDir") {
                    if let Ok(entries) = std::fs::read_dir(&kit_root) {
                        for entry in entries.flatten() {
                            let sdk_lib = entry.path().join("um").join("x64");
                            if build_config::opencl_import_library_exists(&sdk_lib) {
                                println!("cargo:rustc-link-search=native={}", sdk_lib.display());
                                found_import_library = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !found_import_library {
                panic!("OpenCL backend selected on Windows but OpenCL.lib was not found; install an OpenCL SDK/import library or set MNN_OPENCL_LIB_DIR to the directory containing OpenCL.lib");
            }
        }

        if target_env == "gnu" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }

    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=stdc++"); // C++ standard library

        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=cudart");
        }
        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if target_os == "macos" {
        // macOS prebuilt binaries always have Metal enabled, so we always need these frameworks
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        // Objective-C runtime is needed for Metal backend
        println!("cargo:rustc-link-lib=objc");

        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "android") || target_os == "android" {
        // Android requires explicit linking to C++ standard library
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=log"); // Android logging

        // Link libc for stdout/stderr symbols (needed by KleidiAI)
        println!("cargo:rustc-link-lib=c");

        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cfg!(feature = "vulkan") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "ios") || target_os == "ios" {
        // iOS prebuilt binaries have Metal enabled, need these frameworks
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        // Objective-C runtime is needed for Metal backend
        println!("cargo:rustc-link-lib=objc");

        if cfg!(feature = "opencl") {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        }
    }
}

/// Configure backend-specific linking
fn configure_backends() {
    if cfg!(feature = "cuda") {
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let cuda_lib = if cfg!(target_os = "windows") {
                format!("{}/lib/x64", cuda_path)
            } else {
                format!("{}/lib64", cuda_path)
            };
            println!("cargo:rustc-link-search=native={}", cuda_lib);
        }
    }
}
