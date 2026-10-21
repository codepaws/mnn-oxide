#[path = "../examples/model_smoke/fixture.rs"]
mod fixture;

use std::{
    io::{Read, Write},
    net::TcpListener,
    path::Path,
    thread,
    time::Duration,
};

fn serve_once(body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 1024];
        let _ = stream.read(&mut request);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(&body).unwrap();
    });
    format!("http://{address}/model.mnn")
}

fn serve_without_content_length(body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 1024];
        let _ = stream.read(&mut request);
        write!(stream, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n").unwrap();
        stream.write_all(&body).unwrap();
    });
    format!("http://{address}/model.mnn")
}

fn serve_delayed(body: Vec<u8>, delay: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 1024];
        let _ = stream.read(&mut request);
        thread::sleep(delay);
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(&body);
    });
    format!("http://{address}/model.mnn")
}

#[test]
fn downloads_model_when_digest_matches() {
    let bytes = b"known model bytes".to_vec();
    let url = serve_once(bytes.clone());
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");
    let digest = fixture::sha256_bytes(&bytes);

    let downloaded = fixture::ensure_model_with_limits(
        &destination,
        &url,
        &digest,
        bytes.len() as u64,
        Duration::from_secs(2),
    )
    .unwrap();

    assert_eq!(downloaded, destination);
    assert_eq!(std::fs::read(&destination).unwrap(), bytes);
}

#[test]
fn rejects_bad_download_without_leaving_partial_model() {
    let url = serve_once(b"wrong model bytes".to_vec());
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");

    let error = fixture::ensure_model_with_limits(
        &destination,
        &url,
        &"0".repeat(64),
        128,
        Duration::from_secs(2),
    )
    .unwrap_err();

    assert!(error.to_string().contains("SHA-256 mismatch"));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn reuses_cached_model_only_when_digest_matches() {
    let bytes = b"cached model bytes";
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");
    std::fs::write(&destination, bytes).unwrap();

    assert!(fixture::cached_model_matches(
        &destination,
        &fixture::sha256_bytes(bytes)
    ));
    assert!(!fixture::cached_model_matches(
        &destination,
        &"0".repeat(64)
    ));
    assert!(!fixture::cached_model_matches(
        Path::new("missing.mnn"),
        &"0".repeat(64)
    ));
}

#[test]
fn replaces_a_stale_cached_model_only_after_validating_new_bytes() {
    let replacement = b"replacement model bytes".to_vec();
    let url = serve_once(replacement.clone());
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");
    std::fs::write(&destination, b"stale cached bytes").unwrap();

    fixture::ensure_model_with_limits(
        &destination,
        &url,
        &fixture::sha256_bytes(&replacement),
        replacement.len() as u64,
        Duration::from_secs(2),
    )
    .unwrap();

    assert_eq!(std::fs::read(&destination).unwrap(), replacement);
}

#[test]
fn rejects_responses_larger_than_the_configured_limit() {
    let body = b"this response exceeds the limit".to_vec();
    let url = serve_without_content_length(body);
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");

    let error = fixture::ensure_model_with_limits(
        &destination,
        &url,
        &"0".repeat(64),
        8,
        Duration::from_secs(2),
    )
    .unwrap_err();

    assert!(error.to_string().contains("exceeds the maximum size"));
    assert!(!destination.exists());
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn times_out_when_the_server_stalls_before_sending_a_response() {
    let url = serve_delayed(b"late response".to_vec(), Duration::from_millis(200));
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("model.mnn");

    let error = fixture::ensure_model_with_limits(
        &destination,
        &url,
        &"0".repeat(64),
        128,
        Duration::from_millis(30),
    )
    .unwrap_err();

    assert!(
        format!("{error:?}").to_lowercase().contains("timedout"),
        "unexpected timeout error: {error:?}"
    );
    assert!(!destination.exists());
}

#[test]
fn rejects_custom_fixture_paths_that_do_not_match_the_pinned_digest() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("custom-model.mnn");
    std::fs::write(&destination, b"not the pinned model").unwrap();

    let error = fixture::verify_model_file(&destination, &fixture::MODEL_SHA256).unwrap_err();

    assert!(error.to_string().contains("SHA-256 mismatch"));
}

#[test]
fn custom_model_label_does_not_claim_the_pinned_model_identity() {
    let label = fixture::model_label(Path::new("alternate.mnn"), false);

    assert!(label.contains("alternate.mnn"));
    assert!(!label.contains("PP-LCNet"));
}
