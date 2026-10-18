//! Explicit user-supplied, checksum-verified MNN archive acquisition.

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::build_config::PrebuiltConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRequirements {
    pub target: String,
    pub link_mode: String,
    pub backends: Vec<String>,
    pub optional_modules: Vec<String>,
}

pub fn acquire(
    config: &PrebuiltConfig,
    requirements: &ArtifactRequirements,
) -> Result<(PathBuf, PathBuf), String> {
    fs::create_dir_all(&config.cache_dir)
        .map_err(|e| format!("cannot create MNN_PREBUILT_CACHE_DIR: {e}"))?;
    let entry = config.cache_dir.join(&config.target).join(&config.version);
    fs::create_dir_all(&entry).map_err(|e| format!("cannot create prebuilt cache entry: {e}"))?;
    let archive = entry.join(format!("{}.tar.gz", config.sha256));
    if archive.exists() {
        verify_sha256(&archive, &config.sha256)?;
    } else {
        download_verified(&config.url, &archive, &config.sha256)?;
    }

    // Always extract anew from the verified cached archive. This prevents a
    // modified extracted cache from becoming the artifact actually linked.
    let extracted = entry.join(format!("unpacked-{}-{}", config.sha256, std::process::id()));
    if extracted.exists() {
        fs::remove_dir_all(&extracted)
            .map_err(|e| format!("cannot refresh prebuilt extraction: {e}"))?;
    }
    fs::create_dir(&extracted).map_err(|e| format!("cannot create prebuilt extraction: {e}"))?;
    if let Err(error) = extract_archive(&archive, &extracted) {
        let _ = fs::remove_dir_all(&extracted);
        return Err(error);
    }
    if let Err(error) = validate_extracted_archive(&extracted, requirements) {
        let _ = fs::remove_dir_all(&extracted);
        return Err(error);
    }
    Ok((extracted.join("lib"), extracted.join("include")))
}

/// Validate the expected extracted archive layout and declared compatibility.
pub fn validate_extracted_archive(
    extracted: &Path,
    requirements: &ArtifactRequirements,
) -> Result<(), String> {
    let include = extracted.join("include");
    let library = extracted.join("lib");
    if !include.is_dir() || !library.is_dir() {
        return Err(
            "prebuilt archive must contain include/ and lib/ directories at its root".into(),
        );
    }
    let mut required_headers = vec![
        "MNN/Interpreter.hpp",
        "MNN/MNNDefine.h",
        "MNN/MNNForwardType.h",
        "MNN/Tensor.hpp",
        "MNN/ImageProcess.hpp",
        "MNN/Matrix.h",
    ];
    if requirements
        .optional_modules
        .iter()
        .any(|module| module == "llm")
    {
        required_headers.push("llm/llm.hpp");
    }
    if requirements
        .optional_modules
        .iter()
        .any(|module| module == "module")
    {
        required_headers.push("MNN/expr/Module.hpp");
        required_headers.push("MNN/expr/Expr.hpp");
        required_headers.push("MNN/expr/Executor.hpp");
    }
    if requirements
        .optional_modules
        .iter()
        .any(|module| module == "runtime")
    {
        required_headers.push("MNN/expr/Executor.hpp");
    }
    for header in required_headers {
        if !include.join(header).is_file() {
            return Err(format!("prebuilt include/ must contain {header}"));
        }
    }
    if !contains_link_library(
        &library,
        &requirements.target,
        requirements.link_mode == "static",
    ) {
        return Err(format!(
            "prebuilt lib/ does not contain a recognized {} MNN library for {}",
            requirements.link_mode, requirements.target
        ));
    }
    validate_artifact_manifest(extracted, requirements)
}

/// Validate declared artifact compatibility and the presence of named notices.
/// The manifest is publisher-supplied metadata: the MNN release claim is checked
/// against the selected public header, but no binary version, architecture, ABI,
/// feature set, or provenance is inspected or proved here.
pub fn validate_artifact_manifest(
    artifact_root: &Path,
    requirements: &ArtifactRequirements,
) -> Result<(), String> {
    let manifest_path = artifact_root.join("mnn-oxide-artifact.json");
    let bytes = fs::read(&manifest_path)
        .map_err(|e| format!("prebuilt archive must contain mnn-oxide-artifact.json: {e}"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid mnn-oxide-artifact.json: {e}"))?;
    let object = manifest
        .as_object()
        .ok_or_else(|| "mnn-oxide-artifact.json must contain a JSON object".to_string())?;

    let schema = object
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "artifact manifest schema_version must be an integer".to_string())?;
    if schema != 2 {
        return Err(format!(
            "unsupported artifact manifest schema_version {schema}; supported version is 2"
        ));
    }

    let string_field = |name: &str| -> Result<&str, String> {
        object
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("artifact manifest requires non-empty string field `{name}`"))
    };
    let list_field = |name: &str| -> Result<Vec<&str>, String> {
        let list = object
            .get(name)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("artifact manifest requires array field `{name}`"))?;
        list.iter()
            .map(|value| {
                value
                    .as_str()
                    .filter(|entry| !entry.trim().is_empty())
                    .ok_or_else(|| {
                        format!("artifact manifest `{name}` entries must be non-empty strings")
                    })
            })
            .collect()
    };

    let declared_mnn_version = string_field("mnn_version")?;
    let source_url = string_field("source_url")?;
    if !source_url.starts_with("https://") {
        return Err("artifact manifest `source_url` must use HTTPS".into());
    }
    let source_commit = string_field("source_commit")?;
    if source_commit.len() != 40 || !source_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(
            "artifact manifest `source_commit` must be a full 40-character commit SHA".into(),
        );
    }
    for field in ["build_options", "toolchain"] {
        if !object.get(field).is_some_and(serde_json::Value::is_object) {
            return Err(format!("artifact manifest `{field}` must be a JSON object"));
        }
    }
    let header_mnn_version = read_mnn_header_version(artifact_root)?;
    if declared_mnn_version != header_mnn_version {
        return Err(format!(
            "artifact manifest declares MNN version `{declared_mnn_version}`, but selected include/MNN/MNNDefine.h declares `{header_mnn_version}`"
        ));
    }
    let declared_target = string_field("target")?;
    if declared_target != requirements.target {
        return Err(format!(
            "artifact manifest declares target `{declared_target}`, but Cargo target is `{}`",
            requirements.target
        ));
    }
    let declared_link_mode = string_field("link_mode")?;
    if !matches!(declared_link_mode, "static" | "dynamic") {
        return Err("artifact manifest `link_mode` must be `static` or `dynamic`".into());
    }
    if declared_link_mode != requirements.link_mode {
        return Err(format!(
            "artifact manifest declares `{declared_link_mode}` linking, but this build selected `{}`",
            requirements.link_mode
        ));
    }

    let declared_backends = list_field("backends")?;
    let declared_modules = list_field("optional_modules")?;
    for (kind, required, declared) in [
        ("backend", &requirements.backends, &declared_backends),
        (
            "optional module",
            &requirements.optional_modules,
            &declared_modules,
        ),
    ] {
        for name in required {
            if !declared.contains(&name.as_str()) {
                return Err(format!(
                    "artifact manifest does not declare required {kind} `{name}`"
                ));
            }
        }
    }

    let notice_files = list_field("notice_files")?;
    if notice_files.is_empty() {
        return Err(
            "artifact manifest `notice_files` must name at least one license or notice file".into(),
        );
    }
    for notice in notice_files {
        let relative = Path::new(notice);
        if !safe_relative_path(relative) {
            return Err(format!(
                "unsafe path in artifact manifest notice_files: `{notice}`"
            ));
        }
        let full_path = artifact_root.join(relative);
        if !full_path.is_file() {
            return Err(format!(
                "artifact manifest names missing license/notice file `{notice}`"
            ));
        }
    }
    validate_file_inventory(artifact_root, object)?;
    Ok(())
}

fn validate_file_inventory(
    artifact_root: &Path,
    manifest: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
    let declared = manifest
        .get("files")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "artifact manifest `files` must be a path-to-SHA-256 object".to_string())?;
    if declared.is_empty() {
        return Err("artifact manifest `files` must not be empty".into());
    }

    let mut actual = std::collections::BTreeMap::new();
    collect_artifact_files(artifact_root, artifact_root, &mut actual)?;
    if actual.len() != declared.len() {
        return Err(format!(
            "artifact manifest file inventory has {} entries but package contains {} files",
            declared.len(),
            actual.len()
        ));
    }

    for (name, value) in declared {
        let path = Path::new(name);
        if !safe_relative_path(path) {
            return Err(format!("unsafe path in artifact manifest files: `{name}`"));
        }
        let expected = value
            .as_str()
            .ok_or_else(|| format!("artifact manifest file hash for `{name}` must be a string"))?;
        if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "artifact manifest file hash for `{name}` must be a 64-character SHA-256"
            ));
        }
        let digest = actual
            .get(name)
            .ok_or_else(|| format!("artifact manifest file `{name}` is missing from package"))?;
        if !digest.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "artifact manifest file hash does not match `{name}`"
            ));
        }
    }
    Ok(())
}

fn collect_artifact_files(
    root: &Path,
    directory: &Path,
    files: &mut std::collections::BTreeMap<String, String>,
) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot enumerate artifact files: {error}"))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read artifact entry: {error}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot inspect artifact entry: {error}"))?;
        if file_type.is_symlink() {
            return Err(format!(
                "artifact contains unsupported symbolic link: {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            collect_artifact_files(root, &path, files)?;
            continue;
        }
        if !file_type.is_file() {
            return Err(format!(
                "artifact contains unsupported special file: {}",
                path.display()
            ));
        }
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "artifact file escaped artifact root".to_string())?;
        let name = relative.to_string_lossy().replace('\\', "/");
        if name == "mnn-oxide-artifact.json" || name == "SHA256SUMS" {
            continue;
        }
        files.insert(name, sha256_file(&path)?);
    }
    Ok(())
}

/// Read MNN's release version from the numeric macros in its selected public header.
/// This checks agreement between publisher metadata and headers only; it does not
/// inspect or establish the version of the linked native library.
fn read_mnn_header_version(artifact_root: &Path) -> Result<String, String> {
    let path = artifact_root.join("include/MNN/MNNDefine.h");
    let contents = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read selected MNN version header: {error}"))?;
    let major = parse_numeric_define(&contents, "MNN_VERSION_MAJOR", &path)?;
    let minor = parse_numeric_define(&contents, "MNN_VERSION_MINOR", &path)?;
    let patch = parse_numeric_define(&contents, "MNN_VERSION_PATCH", &path)?;
    Ok(format!("{major}.{minor}.{patch}"))
}

fn parse_numeric_define(contents: &str, name: &str, path: &Path) -> Result<u32, String> {
    let mut found = None;
    for line in contents.lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        if parts.next() != Some(name) {
            continue;
        }
        let value = parts
            .next()
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| {
                format!(
                    "selected MNN header has an invalid {name} in {}",
                    path.display()
                )
            })?;
        if found.is_some_and(|previous| previous != value) {
            return Err(format!(
                "selected MNN header defines conflicting {name} values in {}",
                path.display()
            ));
        }
        found = Some(value);
    }
    found.ok_or_else(|| {
        format!(
            "selected MNN header is missing {name} in {}",
            path.display()
        )
    })
}

fn contains_link_library(lib_dir: &Path, target: &str, static_link: bool) -> bool {
    crate::build_config::native_library_present(lib_dir, target, static_link)
}

fn download_verified(url: &str, destination: &Path, expected: &str) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("MNN_PREBUILT_URL must use HTTPS".into());
    }
    let temp = destination.with_extension(format!("part-{}", std::process::id()));
    let _ = fs::remove_file(&temp);
    // Do not follow redirects: a redirect could otherwise silently change the
    // transport chosen by the user. The configured HTTPS endpoint is used as-is.
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| format!("failed to download configured MNN prebuilt URL: {e}"))?;
    {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| format!("cannot create temporary prebuilt archive: {e}"))?;
        let mut reader = response.into_reader();
        io::copy(&mut reader, &mut output)
            .map_err(|e| format!("failed to save prebuilt archive: {e}"))?;
        output
            .flush()
            .map_err(|e| format!("failed to flush prebuilt archive: {e}"))?;
    }
    if let Err(error) = verify_sha256(&temp, expected) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    match fs::rename(&temp, destination) {
        Ok(()) => Ok(()),
        Err(error) if destination.exists() => {
            let _ = fs::remove_file(&temp);
            verify_sha256(destination, expected).map_err(|_| {
                format!(
                    "another build populated the prebuilt cache with an invalid artifact: {error}"
                )
            })
        }
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(format!("cannot store verified prebuilt archive: {error}"))
        }
    }
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot read prebuilt archive: {e}"))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot hash prebuilt archive: {e}"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let actual = sha256_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!(
            "MNN prebuilt SHA-256 mismatch: expected {}, got {}",
            expected.to_ascii_lowercase(),
            actual
        ));
    }
    Ok(())
}

/// Extract regular files/directories only. No links or special files are
/// accepted, so an archive cannot redirect a later entry outside the root.
pub fn extract_archive(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let file = File::open(archive_path).map_err(|e| format!("cannot open MNN archive: {e}"))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|e| format!("invalid MNN tar.gz archive: {e}"))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("invalid MNN archive entry: {e}"))?;
        let relative = entry
            .path()
            .map_err(|e| format!("invalid MNN archive path: {e}"))?
            .into_owned();
        if !safe_relative_path(&relative) {
            return Err(format!(
                "unsafe path in MNN archive: {}",
                relative.display()
            ));
        }
        let output = destination.join(&relative);
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            fs::create_dir_all(&output)
                .map_err(|e| format!("cannot create extracted directory: {e}"))?;
        } else if kind.is_file() {
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create extracted parent: {e}"))?;
            }
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)
                .map_err(|e| format!("cannot create extracted file {}: {e}", output.display()))?;
            io::copy(&mut entry, &mut file)
                .map_err(|e| format!("cannot extract archive file: {e}"))?;
        } else {
            return Err(format!(
                "unsupported link or special file in MNN archive: {}",
                relative.display()
            ));
        }
    }
    Ok(())
}

fn safe_relative_path(path: &Path) -> bool {
    if path.as_os_str().is_empty() || path.to_string_lossy().contains('\\') {
        return false;
    }
    path.components().all(|component| match component {
        Component::Normal(part) => !part.to_string_lossy().contains(':'),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::time::{SystemTime, UNIX_EPOCH};

    const MNN_DEFINE_3_6_1: &[u8] = b"#define MNN_VERSION_MAJOR 3\n#define MNN_VERSION_MINOR 6\n#define MNN_VERSION_PATCH 1\n#define MNN_VERSION STR(MNN_VERSION_MAJOR) \".\" STR(MNN_VERSION_MINOR) \".\" STR(MNN_VERSION_PATCH)\n";

    fn test_requirements() -> ArtifactRequirements {
        ArtifactRequirements {
            target: "x86_64-unknown-linux-gnu".into(),
            link_mode: "static".into(),
            backends: vec!["cpu".into()],
            optional_modules: vec!["llm".into()],
        }
    }

    fn write_manifest(root: &Path, json: &str) {
        fs::write(root.join("LICENSE-MNN"), "Apache-2.0 notice").unwrap();
        fs::create_dir_all(root.join("include/MNN")).unwrap();
        fs::write(root.join("include/MNN/MNNDefine.h"), MNN_DEFINE_3_6_1).unwrap();

        let mut manifest: serde_json::Value = serde_json::from_str(json).unwrap();
        if manifest["schema_version"] == 1 {
            manifest["schema_version"] = serde_json::json!(2);
        }
        manifest["source_url"] = serde_json::json!("https://github.com/alibaba/MNN.git");
        manifest["source_commit"] = serde_json::json!("a".repeat(40));
        manifest["build_options"] = serde_json::json!({ "MNN_BUILD_LLM": true });
        manifest["toolchain"] = serde_json::json!({ "compiler": "fixture" });
        let mut files = serde_json::Map::new();
        for path in fixture_files(root) {
            let name = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            files.insert(name, serde_json::json!(sha256_file(&path).unwrap()));
        }
        manifest["files"] = serde_json::Value::Object(files);
        fs::write(
            root.join("mnn-oxide-artifact.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn fixture_files(root: &Path) -> Vec<PathBuf> {
        fn visit(path: &Path, output: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, output);
                } else if path.file_name().unwrap() != "mnn-oxide-artifact.json" {
                    output.push(path);
                }
            }
        }
        let mut files = Vec::new();
        visit(root, &mut files);
        files.sort();
        files
    }

    fn make_archive_from_dir(path: &Path, root: &Path) {
        let output = File::create(path).unwrap();
        let encoder = GzEncoder::new(output, Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for file in fixture_files_including_manifest(root) {
            let relative = file.strip_prefix(root).unwrap();
            builder.append_path_with_name(&file, relative).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
    }

    fn fixture_files_including_manifest(root: &Path) -> Vec<PathBuf> {
        fn visit(path: &Path, output: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, output);
                } else {
                    output.push(path);
                }
            }
        }
        let mut files = Vec::new();
        visit(root, &mut files);
        files.sort();
        files
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mnn-prebuilt-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    fn make_archive(path: &Path, entries: &[(&str, tar::EntryType, &[u8])]) {
        let output = File::create(path).unwrap();
        let encoder = GzEncoder::new(output, Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for (name, kind, bytes) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(*kind);
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *bytes).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
    }

    fn make_traversal_archive(path: &Path) {
        // Build a valid header first, then mutate its raw name field because
        // tar::Header::set_path intentionally refuses parent components.
        let mut header = tar::Header::new_gnu();
        header.set_size(3);
        header.set_mode(0o644);
        header.set_cksum();
        let raw = header.as_mut_bytes();
        raw[..100].fill(0);
        raw[..9].copy_from_slice(b"../escape");
        header.set_cksum();

        let output = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(output, Compression::default());
        encoder.write_all(header.as_bytes()).unwrap();
        encoder.write_all(b"bad").unwrap();
        encoder.write_all(&[0u8; 509]).unwrap();
        encoder.write_all(&[0u8; 1024]).unwrap();
        encoder.finish().unwrap();
    }

    #[test]
    fn sha256_verification_accepts_exact_digest_and_rejects_mismatch() {
        let path = temp_path("digest");
        fs::write(&path, b"artifact").unwrap();
        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            digest,
            "c7c5c1d70c5dec4416ab6158afd0b223ef40c29b1dc1f97ed9428b94d4cadb1c"
        );
        assert!(verify_sha256(&path, &digest).is_ok());
        assert!(verify_sha256(&path, &"0".repeat(64)).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn extraction_rejects_parent_traversal_and_symlinks() {
        let archive = temp_path("unsafe.tar.gz");
        let destination = temp_path("unsafe-out");
        fs::create_dir(&destination).unwrap();
        assert!(!safe_relative_path(Path::new("../escape")));
        assert!(!safe_relative_path(Path::new("include\\..\\escape")));
        make_traversal_archive(&archive);
        assert!(extract_archive(&archive, &destination).is_err());
        make_archive(
            &archive,
            &[("include/link", tar::EntryType::Symlink, b"../../escape")],
        );
        assert!(extract_archive(&archive, &destination)
            .unwrap_err()
            .contains("unsupported link"));
        assert!(!destination.parent().unwrap().join("escape").exists());
        let _ = fs::remove_file(archive);
        let _ = fs::remove_dir_all(destination);
    }

    #[test]
    fn extracts_root_include_and_lib_layout() {
        let archive = temp_path("valid.tar.gz");
        let destination = temp_path("valid-out");
        fs::create_dir(&destination).unwrap();
        make_archive(
            &archive,
            &[
                (
                    "include/MNN/Interpreter.hpp",
                    tar::EntryType::Regular,
                    b"header",
                ),
                ("lib/libMNN.a", tar::EntryType::Regular, b"library"),
            ],
        );
        extract_archive(&archive, &destination).unwrap();
        assert_eq!(
            fs::read(destination.join("include/MNN/Interpreter.hpp")).unwrap(),
            b"header"
        );
        assert_eq!(
            fs::read(destination.join("lib/libMNN.a")).unwrap(),
            b"library"
        );
        let _ = fs::remove_file(archive);
        let _ = fs::remove_dir_all(destination);
    }

    #[test]
    fn artifact_manifest_accepts_matching_claims_and_existing_notice() {
        let root = temp_path("manifest-valid");
        fs::create_dir(&root).unwrap();
        write_manifest(
            &root,
            r#"{
                "schema_version": 1,
                "mnn_version": "3.6.1",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu", "cuda"],
                "optional_modules": ["llm"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );
        assert!(validate_artifact_manifest(&root, &test_requirements()).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_manifest_rejects_tampered_files_and_invalid_source_revision() {
        let root = temp_path("manifest-file-integrity");
        fs::create_dir(&root).unwrap();
        write_manifest(
            &root,
            r#"{
                "schema_version": 2,
                "mnn_version": "3.6.1",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu"],
                "optional_modules": ["llm"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );

        fs::write(root.join("LICENSE-MNN"), "replaced notice").unwrap();
        assert!(validate_artifact_manifest(&root, &test_requirements())
            .unwrap_err()
            .contains("file hash does not match"));

        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("mnn-oxide-artifact.json")).unwrap())
                .unwrap();
        manifest["source_commit"] = serde_json::json!("short");
        fs::write(
            root.join("mnn-oxide-artifact.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(validate_artifact_manifest(&root, &test_requirements())
            .unwrap_err()
            .contains("full 40-character commit SHA"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_manifest_rejects_version_that_disagrees_with_selected_headers() {
        let root = temp_path("manifest-version-mismatch");
        fs::create_dir(&root).unwrap();
        write_manifest(
            &root,
            r#"{
                "schema_version": 1,
                "mnn_version": "3.5.0",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu"],
                "optional_modules": ["llm"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );

        let error = validate_artifact_manifest(&root, &test_requirements()).unwrap_err();
        assert!(error.contains("declares MNN version `3.5.0`"));
        assert!(error.contains("MNNDefine.h declares `3.6.1`"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_manifest_rejects_unknown_schema_and_incompatible_claims() {
        let root = temp_path("manifest-invalid");
        fs::create_dir(&root).unwrap();
        let requirements = test_requirements();
        let valid = r#"{
            "schema_version": 1,
            "mnn_version": "3.6.1",
            "target": "x86_64-unknown-linux-gnu",
            "link_mode": "static",
            "backends": ["cpu"],
            "optional_modules": ["llm"],
            "notice_files": ["LICENSE-MNN"]
        }"#;
        write_manifest(
            &root,
            &valid.replace("\"schema_version\": 1", "\"schema_version\": 3"),
        );
        assert!(validate_artifact_manifest(&root, &requirements)
            .unwrap_err()
            .contains("unsupported artifact manifest schema_version"));

        for (json, expected_error) in [
            (
                valid.replace("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"),
                "declares target",
            ),
            (
                valid.replace("\"static\"", "\"dynamic\""),
                "declares `dynamic` linking",
            ),
            (
                valid.replace("\"cpu\"", "\"opencl\""),
                "required backend `cpu`",
            ),
            (
                valid.replace("\"llm\"", "\"module\""),
                "required optional module `llm`",
            ),
        ] {
            write_manifest(&root, &json);
            assert!(validate_artifact_manifest(&root, &requirements)
                .unwrap_err()
                .contains(expected_error));
        }

        write_manifest(&root, &valid.replace("LICENSE-MNN", "missing/NOTICE.txt"));
        assert!(validate_artifact_manifest(&root, &requirements)
            .unwrap_err()
            .contains("names missing license/notice file"));
        write_manifest(&root, &valid.replace("LICENSE-MNN", "../NOTICE.txt"));
        assert!(validate_artifact_manifest(&root, &requirements)
            .unwrap_err()
            .contains("unsafe path"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_manifest_is_required_after_archive_extraction() {
        let archive = temp_path("manifest-archive.tar.gz");
        let destination = temp_path("manifest-archive-out");
        let package = temp_path("manifest-archive-package");
        fs::create_dir_all(package.join("include/MNN")).unwrap();
        fs::create_dir_all(package.join("include/llm")).unwrap();
        fs::create_dir_all(package.join("lib")).unwrap();
        for header in [
            "Interpreter.hpp",
            "MNNForwardType.h",
            "Tensor.hpp",
            "ImageProcess.hpp",
            "Matrix.h",
        ] {
            fs::write(package.join("include/MNN").join(header), b"header").unwrap();
        }
        fs::write(package.join("include/MNN/MNNDefine.h"), MNN_DEFINE_3_6_1).unwrap();
        fs::write(package.join("include/llm/llm.hpp"), b"header").unwrap();
        fs::write(package.join("lib/libMNN.a"), b"library").unwrap();
        write_manifest(
            &package,
            r#"{
                "schema_version": 2,
                "mnn_version": "3.6.1",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu"],
                "optional_modules": ["llm"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );
        // The helper writes LICENSE-MNN and the selected version header, and
        // inventories every package file other than the manifest itself.
        make_archive_from_dir(&archive, &package);
        fs::create_dir(&destination).unwrap();
        extract_archive(&archive, &destination).unwrap();
        assert!(validate_extracted_archive(&destination, &test_requirements()).is_ok());

        fs::remove_file(destination.join("mnn-oxide-artifact.json")).unwrap();
        assert!(
            validate_extracted_archive(&destination, &test_requirements())
                .unwrap_err()
                .contains("must contain mnn-oxide-artifact.json")
        );
        let _ = fs::remove_file(archive);
        let _ = fs::remove_dir_all(destination);
        let _ = fs::remove_dir_all(package);
    }

    #[test]
    fn prebuilt_requires_headers_used_by_enabled_wrappers() {
        let root = temp_path("required-headers");
        fs::create_dir_all(root.join("include/MNN")).unwrap();
        fs::create_dir_all(root.join("include/llm")).unwrap();
        fs::create_dir_all(root.join("lib")).unwrap();
        for header in [
            "MNN/Interpreter.hpp",
            "MNN/MNNDefine.h",
            "MNN/MNNForwardType.h",
            "MNN/Tensor.hpp",
            "MNN/ImageProcess.hpp",
            "MNN/Matrix.h",
            "llm/llm.hpp",
        ] {
            let contents = if header == "MNN/MNNDefine.h" {
                MNN_DEFINE_3_6_1
            } else {
                b"header"
            };
            fs::write(root.join("include").join(header), contents).unwrap();
        }
        fs::write(root.join("lib/libMNN.a"), b"library").unwrap();
        write_manifest(
            &root,
            r#"{
                "schema_version": 1,
                "mnn_version": "3.6.1",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu"],
                "optional_modules": ["llm", "module"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );
        assert!(validate_extracted_archive(&root, &test_requirements()).is_ok());

        fs::remove_file(root.join("include/llm/llm.hpp")).unwrap();
        assert!(validate_extracted_archive(&root, &test_requirements())
            .unwrap_err()
            .contains("include/ must contain llm/llm.hpp"));

        fs::remove_file(root.join("include/MNN/Tensor.hpp")).unwrap();
        let mut without_llm = test_requirements();
        without_llm.optional_modules.clear();
        assert!(validate_extracted_archive(&root, &without_llm)
            .unwrap_err()
            .contains("include/ must contain MNN/Tensor.hpp"));

        fs::write(root.join("include/MNN/Tensor.hpp"), b"header").unwrap();
        fs::create_dir_all(root.join("include/MNN/expr")).unwrap();
        for header in ["Module.hpp", "Expr.hpp", "Executor.hpp"] {
            fs::write(root.join("include/MNN/expr").join(header), b"header").unwrap();
        }
        write_manifest(
            &root,
            r#"{
                "schema_version": 2,
                "mnn_version": "3.6.1",
                "target": "x86_64-unknown-linux-gnu",
                "link_mode": "static",
                "backends": ["cpu"],
                "optional_modules": ["module"],
                "notice_files": ["LICENSE-MNN"]
            }"#,
        );
        let mut with_module = without_llm;
        with_module.optional_modules.push("module".into());
        assert!(validate_extracted_archive(&root, &with_module).is_ok());

        fs::remove_file(root.join("include/MNN/expr/Expr.hpp")).unwrap();
        assert!(validate_extracted_archive(&root, &with_module)
            .unwrap_err()
            .contains("include/ must contain MNN/expr/Expr.hpp"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn link_library_detection_matches_target_and_link_mode() {
        let lib_dir = temp_path("libraries");
        fs::create_dir(&lib_dir).unwrap();
        fs::write(lib_dir.join("libMNN.a"), b"static").unwrap();
        assert!(contains_link_library(
            &lib_dir,
            "x86_64-unknown-linux-gnu",
            true
        ));
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-unknown-linux-gnu",
            false
        ));
        fs::remove_file(lib_dir.join("libMNN.a")).unwrap();

        // GNU Windows static linking accepts an archive, not an import lib.
        fs::write(lib_dir.join("libMNN.a"), b"static").unwrap();
        assert!(contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-gnu",
            true
        ));
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-gnu",
            false
        ));
        fs::remove_file(lib_dir.join("libMNN.a")).unwrap();

        // GNU Windows dynamic linking requires both an import library and DLL.
        fs::write(lib_dir.join("libMNN.dll.a"), b"import").unwrap();
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-gnu",
            false
        ));
        fs::write(lib_dir.join("MNN.dll"), b"dll").unwrap();
        assert!(contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-gnu",
            false
        ));
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-gnu",
            true
        ));
        fs::remove_file(lib_dir.join("libMNN.dll.a")).unwrap();
        fs::remove_file(lib_dir.join("MNN.dll")).unwrap();

        // MSVC .lib names overlap, so dynamic mode also requires the DLL.
        fs::write(lib_dir.join("MNN.lib"), b"lib").unwrap();
        assert!(contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-msvc",
            true
        ));
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-msvc",
            false
        ));
        fs::write(lib_dir.join("MNN.dll"), b"dll").unwrap();
        assert!(contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-msvc",
            false
        ));
        fs::remove_file(lib_dir.join("MNN.lib")).unwrap();
        assert!(!contains_link_library(
            &lib_dir,
            "x86_64-pc-windows-msvc",
            false
        ));
        let _ = fs::remove_dir_all(lib_dir);
    }
}
