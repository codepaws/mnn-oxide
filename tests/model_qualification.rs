//! Opt-in, model-specific CPU qualification for the pinned PP-LCNet fixture.

#[path = "../examples/model_smoke/fixture.rs"]
mod fixture;

use mnn_oxide::prelude::*;
use std::path::PathBuf;

#[test]
#[ignore = "downloads or requires the pinned PP-LCNet MNN model"]
fn pp_lcnet_cpu_inference_returns_two_finite_class_scores() {
    let path = match std::env::var_os("MNN_TEST_MODEL") {
        Some(path) => PathBuf::from(path),
        None => fixture::ensure_model_from_url(
            &fixture::default_model_path(),
            fixture::MODEL_URL,
            fixture::MODEL_SHA256,
        )
        .expect("download and verify pinned PP-LCNet model"),
    };
    fixture::verify_model_file(&path, fixture::MODEL_SHA256)
        .expect("verify pinned PP-LCNet model digest");
    assert!(
        path.is_file(),
        "model fixture not found at {}",
        path.display()
    );

    let interpreter = Interpreter::from_file(&path).expect("load PP-LCNet model");
    let mut session = interpreter
        .create_session(ScheduleConfig::new().backend(BackendType::CPU))
        .expect("create CPU session");

    let mut input = session.get_input(None).expect("get model input");
    assert_eq!(input.dtype().expect("input dtype"), DataType::Float32);
    let input_shape = input.shape();
    assert_eq!(input_shape, [1, 3, 80, 160]);
    interpreter.resize_tensor(&mut input, &input_shape);
    interpreter.resize_session(&mut session);
    input
        .write(&vec![0.0_f32; input.element_count().expect("input size")])
        .expect("write deterministic input");

    session.run().expect("run PP-LCNet inference");
    assert!(session.has_run());

    let output = session.get_output(None).expect("get model output");
    assert_eq!(output.shape(), [1, 2]);
    let scores = output.read::<f32>().expect("read class scores");
    assert_eq!(scores.len(), 2);
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "scores: {scores:?}"
    );
}
