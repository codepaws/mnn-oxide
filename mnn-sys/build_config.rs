//! Pure acquisition configuration checks shared with build-script tests.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionMode {
    BringYourOwn,
    Source,
    Prebuilt,
}

pub fn validate_acquisition(
    prebuilt: bool,
    source: bool,
    system: bool,
    has_lib_path: bool,
    has_include_path: bool,
) -> Result<AcquisitionMode, String> {
    if has_lib_path != has_include_path {
        return Err("MNN_LIB_DIR and MNN_INCLUDE_DIR must be supplied together".into());
    }
    if usize::from(prebuilt) + usize::from(source) > 1 {
        return Err("use-prebuilt and build-from-source cannot be combined".into());
    }
    if prebuilt {
        if system || has_lib_path {
            return Err(
                "use-prebuilt cannot be combined with system mode or MNN_LIB_DIR/MNN_INCLUDE_DIR"
                    .into(),
            );
        }
        return Ok(AcquisitionMode::Prebuilt);
    }
    if source {
        if system || has_lib_path {
            return Err("source mode cannot be combined with system mode or library paths".into());
        }
        return Ok(AcquisitionMode::Source);
    }
    if system && !has_lib_path {
        return Err("system-mnn requires explicit MNN_LIB_DIR and MNN_INCLUDE_DIR paths".into());
    }
    if has_lib_path {
        return Ok(AcquisitionMode::BringYourOwn);
    }
    Err("configure MNN_LIB_DIR and MNN_INCLUDE_DIR explicitly".into())
}

#[cfg(any(feature = "use-prebuilt", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrebuiltConfig {
    pub url: String,
    pub sha256: String,
    pub target: String,
    pub version: String,
    pub cache_dir: std::path::PathBuf,
}

#[cfg(any(feature = "use-prebuilt", test))]
pub fn validate_prebuilt_config(
    url: Option<&str>,
    sha256: Option<&str>,
    target: Option<&str>,
    version: Option<&str>,
    cache_dir: Option<&str>,
    cargo_target: &str,
) -> Result<PrebuiltConfig, String> {
    fn required<'a>(value: Option<&'a str>, name: &str) -> Result<&'a str, String> {
        value
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| format!("use-prebuilt requires {name}"))
    }
    let url = required(url, "MNN_PREBUILT_URL")?;
    if !url.starts_with("https://")
        || url
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
    {
        return Err("MNN_PREBUILT_URL must be an HTTPS URL without whitespace".into());
    }
    let sha256 = required(sha256, "MNN_PREBUILT_SHA256")?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("MNN_PREBUILT_SHA256 must contain exactly 64 hexadecimal characters".into());
    }
    let target = required(target, "MNN_PREBUILT_TARGET")?;
    if target != cargo_target {
        return Err(format!(
            "MNN_PREBUILT_TARGET ({target}) must match Cargo target ({cargo_target})"
        ));
    }
    let version = required(version, "MNN_PREBUILT_VERSION")?;
    for (name, value) in [
        ("MNN_PREBUILT_TARGET", target),
        ("MNN_PREBUILT_VERSION", version),
    ] {
        if !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        {
            return Err(format!(
                "{name} may contain only ASCII letters, digits, '.', '_' and '-'"
            ));
        }
    }
    let cache_dir = std::path::PathBuf::from(required(cache_dir, "MNN_PREBUILT_CACHE_DIR")?);
    Ok(PrebuiltConfig {
        url: url.to_owned(),
        sha256: sha256.to_ascii_lowercase(),
        target: target.to_owned(),
        version: version.to_owned(),
        cache_dir,
    })
}

pub fn is_full_commit_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn revision_matches(requested: &str, actual: &str) -> bool {
    is_full_commit_sha(requested) && actual.eq_ignore_ascii_case(requested)
}

pub fn checkout_status_is_clean(status: &str) -> bool {
    status.is_empty()
}

/// Parse `git ls-files --stage -z`, returning tracked paths and submodule flags.
#[cfg(any(feature = "build-from-source", test))]
pub fn parse_index_paths(output: &[u8]) -> Result<Vec<(std::path::PathBuf, bool)>, String> {
    let mut paths = Vec::new();
    for entry in output
        .split(|&byte| byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        let tab = entry
            .iter()
            .position(|&byte| byte == b'\t')
            .ok_or_else(|| "invalid git ls-files --stage record".to_string())?;
        let metadata = std::str::from_utf8(&entry[..tab])
            .map_err(|_| "invalid git index metadata".to_string())?;
        let mode = metadata
            .split_whitespace()
            .next()
            .ok_or_else(|| "missing git index mode".to_string())?;
        let path = std::str::from_utf8(&entry[tab + 1..])
            .map_err(|_| "non-UTF8 MNN source path is unsupported".to_string())?;
        paths.push((std::path::PathBuf::from(path), mode == "160000"));
    }
    Ok(paths)
}

/// Return a cache identity that changes when either the verified checkout or
/// its location changes. The latter avoids sharing build outputs across
/// checkouts whose untracked/submodule inputs may differ despite equal HEADs.
pub fn source_cache_identity(source_path: &std::path::Path, revision: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_path.hash(&mut hasher);
    revision.to_ascii_lowercase().hash(&mut hasher);
    format!(
        "{}-{:#016x}",
        revision.to_ascii_lowercase(),
        hasher.finish()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_conflicting_modes_and_partial_paths() {
        assert!(validate_acquisition(true, true, false, false, false).is_err());
        assert!(validate_acquisition(false, true, true, false, false).is_err());
        assert!(validate_acquisition(false, false, false, true, false).is_err());
    }

    #[test]
    fn accepts_only_explicit_byolibrary_or_pinned_source_selection() {
        assert_eq!(
            validate_acquisition(false, false, false, true, true),
            Ok(AcquisitionMode::BringYourOwn)
        );
        assert_eq!(
            validate_acquisition(false, true, false, false, false),
            Ok(AcquisitionMode::Source)
        );
        assert!(validate_acquisition(false, false, true, false, false).is_err());
        assert_eq!(
            validate_acquisition(true, false, false, false, false),
            Ok(AcquisitionMode::Prebuilt)
        );
        assert!(validate_acquisition(true, false, true, false, false).is_err());
        assert!(validate_acquisition(true, false, false, true, true).is_err());
    }

    #[test]
    fn prebuilt_config_requires_https_sha_target_version_and_cache() {
        let target = "x86_64-unknown-linux-gnu";
        let digest = "a".repeat(64);
        let valid = || {
            validate_prebuilt_config(
                Some("https://example.invalid/mnn.tar.gz"),
                Some(&digest),
                Some(target),
                Some("3.6.1"),
                Some("cache"),
                target,
            )
        };
        assert!(valid().is_ok());
        assert_eq!(valid().unwrap().sha256, digest);
        assert!(validate_prebuilt_config(
            None,
            Some(&digest),
            Some(target),
            Some("3.6.1"),
            Some("cache"),
            target
        )
        .unwrap_err()
        .contains("MNN_PREBUILT_URL"));
        assert!(validate_prebuilt_config(
            Some("http://example.invalid/mnn.tar.gz"),
            Some(&digest),
            Some(target),
            Some("3.6.1"),
            Some("cache"),
            target
        )
        .is_err());
        assert!(validate_prebuilt_config(
            Some("https://example.invalid/mnn.tar.gz"),
            Some("abc"),
            Some(target),
            Some("3.6.1"),
            Some("cache"),
            target
        )
        .is_err());
        assert!(validate_prebuilt_config(
            Some("https://example.invalid/mnn.tar.gz"),
            Some(&digest),
            Some("aarch64-unknown-linux-gnu"),
            Some("3.6.1"),
            Some("cache"),
            target
        )
        .is_err());
        assert!(validate_prebuilt_config(
            Some("https://example.invalid/mnn.tar.gz"),
            Some(&digest),
            Some(target),
            Some("../3.6.1"),
            Some("cache"),
            target
        )
        .is_err());
        assert!(validate_prebuilt_config(
            Some("https://example.invalid/mnn.tar.gz"),
            Some(&digest),
            Some(target),
            Some("3.6.1"),
            None,
            target
        )
        .is_err());
    }

    #[test]
    fn requires_full_commit_sha_format() {
        assert!(is_full_commit_sha(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!is_full_commit_sha("3.6.1"));
        assert!(!is_full_commit_sha(
            "0123456789abcdef0123456789abcdef0123456z"
        ));
        let pin = "0123456789abcdef0123456789abcdef01234567";
        assert!(revision_matches(pin, pin));
        assert!(!revision_matches(
            pin,
            "1123456789abcdef0123456789abcdef01234567"
        ));
    }

    #[test]
    fn source_cache_identity_tracks_checkout_path_and_revision() {
        let path = std::path::Path::new("mnn-source-a");
        let revision_a = "0123456789abcdef0123456789abcdef01234567";
        let revision_b = "1123456789abcdef0123456789abcdef01234567";
        assert_ne!(
            source_cache_identity(path, revision_a),
            source_cache_identity(path, revision_b)
        );
        assert_ne!(
            source_cache_identity(path, revision_a),
            source_cache_identity(std::path::Path::new("mnn-source-b"), revision_a)
        );
    }

    #[test]
    fn source_checkout_must_have_no_tracked_or_untracked_changes() {
        assert!(checkout_status_is_clean(""));
        assert!(!checkout_status_is_clean(" M src/file.cpp\n"));
        assert!(!checkout_status_is_clean("?? local-input.cmake\n"));
    }

    #[test]
    fn tracked_snapshot_entries_include_submodules_but_not_untracked_files() {
        let entries = b"100644 abc123 0\tCMakeLists.txt\0160000 def456 0\tthird_party/module\0";
        assert_eq!(
            parse_index_paths(entries).unwrap(),
            vec![
                (std::path::PathBuf::from("CMakeLists.txt"), false),
                (std::path::PathBuf::from("third_party/module"), true),
            ]
        );
        assert!(parse_index_paths(b"not-an-index-record").is_err());
    }
}
