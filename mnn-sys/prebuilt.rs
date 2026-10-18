//! Explicit user-supplied, checksum-verified MNN archive acquisition.

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::build_config::PrebuiltConfig;

pub fn acquire(config: &PrebuiltConfig, static_link: bool) -> Result<(PathBuf, PathBuf), String> {
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
    let include = extracted.join("include");
    let library = extracted.join("lib");
    if !include.is_dir() || !library.is_dir() {
        let _ = fs::remove_dir_all(&extracted);
        return Err(
            "prebuilt archive must contain include/ and lib/ directories at its root".into(),
        );
    }
    if !include.join("MNN").join("Interpreter.hpp").is_file() {
        let _ = fs::remove_dir_all(&extracted);
        return Err("prebuilt include/ must contain MNN/Interpreter.hpp".into());
    }
    if !contains_link_library(&library, &config.target, static_link) {
        let _ = fs::remove_dir_all(&extracted);
        return Err(format!(
            "prebuilt lib/ does not contain a recognized {} MNN library for {}",
            if static_link { "static" } else { "dynamic" },
            config.target
        ));
    }
    Ok((library, include))
}

fn contains_link_library(lib_dir: &Path, target: &str, static_link: bool) -> bool {
    let windows = target.contains("windows");
    let darwin = target.contains("apple") || target.contains("darwin");
    if windows {
        let (import_names, dll_names) = if target.contains("gnu") {
            (
                &["libMNN.dll.a", "libmnn.dll.a"][..],
                &["MNN.dll", "mnn.dll"][..],
            )
        } else {
            (&["MNN.lib", "mnn.lib"][..], &["MNN.dll", "mnn.dll"][..])
        };
        if static_link {
            let static_names: &[&str] = if target.contains("gnu") {
                &["libMNN.a", "libmnn.a"]
            } else {
                &["MNN.lib", "mnn.lib"]
            };
            return static_names.iter().any(|name| lib_dir.join(name).is_file());
        }
        let has_import = import_names.iter().any(|name| lib_dir.join(name).is_file());
        let has_dll = dll_names.iter().any(|name| lib_dir.join(name).is_file());
        return has_import && has_dll;
    }
    let names: &[&str] = if darwin {
        if static_link {
            &["libMNN.a", "libmnn.a"]
        } else {
            &["libMNN.dylib", "libmnn.dylib"]
        }
    } else if static_link {
        &["libMNN.a", "libmnn.a"]
    } else {
        &["libMNN.so", "libmnn.so"]
    };
    names.iter().any(|name| lib_dir.join(name).is_file())
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
