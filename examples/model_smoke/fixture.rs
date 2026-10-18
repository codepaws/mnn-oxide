#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

pub const MODEL_REPO: &str = "baicai1145/pp-ocr-mnn-models";
pub const MODEL_REVISION: &str = "7746c6c454200777bb56fe0dadd350d58a6753e8";
pub const MODEL_FILENAME: &str = "PP-LCNet_x1_0_textline_ori.mnn";
pub const MODEL_SIZE_BYTES: u64 = 6_694_800;
pub const MODEL_SHA256: &str = "8b63eed38352bd505e4790a4fcb4d213889c83d4b78bcc055a8ec74cbfe1f687";
pub const MODEL_URL: &str = "https://huggingface.co/baicai1145/pp-ocr-mnn-models/resolve/7746c6c454200777bb56fe0dadd350d58a6753e8/PP-LCNet_x1_0_textline_ori.mnn";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn cached_model_matches(path: &Path, expected_sha256: &str) -> bool {
    sha256_file(path)
        .map(|actual| actual.eq_ignore_ascii_case(expected_sha256))
        .unwrap_or(false)
}

pub fn verify_model_file(path: &Path, expected_sha256: &str) -> Result<(), Box<dyn Error>> {
    let actual_sha256 = sha256_file(path)?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(format!(
            "SHA-256 mismatch for model file: expected {expected_sha256}, got {actual_sha256}"
        )
        .into());
    }
    Ok(())
}

pub fn model_label(path: &Path, is_pinned_model: bool) -> String {
    if is_pinned_model {
        format!("Pinned PP-LCNet ({MODEL_REVISION})")
    } else {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("custom model file");
        format!("Custom model: {file_name}")
    }
}

pub fn ensure_model_from_url(
    destination: &Path,
    url: &str,
    expected_sha256: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    ensure_model_with_limits(
        destination,
        url,
        expected_sha256,
        MODEL_SIZE_BYTES,
        DOWNLOAD_TIMEOUT,
    )
}

pub fn ensure_model_with_limits(
    destination: &Path,
    url: &str,
    expected_sha256: &str,
    max_bytes: u64,
    timeout: Duration,
) -> Result<PathBuf, Box<dyn Error>> {
    if cached_model_matches(destination, expected_sha256) {
        return Ok(destination.to_path_buf());
    }

    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let response = ureq::get(url)
        .timeout(timeout)
        .set("User-Agent", "mnn-oxide-model-qualification/1.0")
        .call()?;
    if let Some(content_length) = response
        .header("Content-Length")
        .and_then(|length| length.parse::<u64>().ok())
    {
        if content_length > max_bytes {
            return Err(
                format!("Downloaded model exceeds the maximum size of {max_bytes} bytes").into(),
            );
        }
    }

    let mut body = response.into_reader().take(max_bytes.saturating_add(1));
    let downloaded_bytes = io::copy(&mut body, &mut temporary)?;
    if downloaded_bytes > max_bytes {
        return Err(
            format!("Downloaded model exceeds the maximum size of {max_bytes} bytes").into(),
        );
    }
    temporary.flush()?;

    let actual_sha256 = sha256_file(temporary.path())?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(format!(
            "SHA-256 mismatch for downloaded model: expected {expected_sha256}, got {actual_sha256}"
        )
        .into());
    }

    temporary.persist(destination)?;
    Ok(destination.to_path_buf())
}

pub fn default_model_path() -> PathBuf {
    PathBuf::from("target")
        .join("test-models")
        .join(MODEL_FILENAME)
}
