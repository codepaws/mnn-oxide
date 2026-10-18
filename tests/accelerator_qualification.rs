//! Opt-in model-backed accelerator qualification for a real device.
//!
//! Run with exactly one accelerator feature enabled and `-- --ignored`. The
//! test intentionally attempts the selected backend instead of trusting feature
//! flags or device enumeration as proof of runtime support.

#![cfg(any(
    feature = "cuda",
    feature = "opencl",
    feature = "vulkan",
    feature = "metal"
))]

#[path = "../examples/model_smoke/fixture.rs"]
mod fixture;

use mnn_oxide::prelude::*;
use std::path::PathBuf;

#[cfg(feature = "cuda")]
const QUALIFICATION_BACKEND: BackendType = BackendType::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "opencl"))]
const QUALIFICATION_BACKEND: BackendType = BackendType::OpenCL;
#[cfg(all(not(feature = "cuda"), not(feature = "opencl"), feature = "vulkan"))]
const QUALIFICATION_BACKEND: BackendType = BackendType::Vulkan;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "opencl"),
    not(feature = "vulkan"),
    feature = "metal"
))]
const QUALIFICATION_BACKEND: BackendType = BackendType::Metal;

#[test]
#[ignore = "requires the pinned PP-LCNet model and a working selected accelerator runtime"]
fn pinned_pp_lcnet_runs_on_selected_accelerator() {
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

    let interpreter = Interpreter::from_file(&path).expect("load PP-LCNet model");
    let mut session = interpreter
        .create_session(ScheduleConfig::new().backend(QUALIFICATION_BACKEND))
        .unwrap_or_else(|error| {
            panic!(
                "create explicit {} session (no CPU fallback is requested): {error}",
                QUALIFICATION_BACKEND
            )
        });
    let actual_backend = session.backend().expect("MNN reports the session backend");
    assert_eq!(
        actual_backend, QUALIFICATION_BACKEND,
        "MNN silently selected {actual_backend} instead of the requested {QUALIFICATION_BACKEND}"
    );

    let mut input = session.get_input(None).expect("get model input");
    assert_eq!(input.dtype().expect("input dtype"), DataType::Float32);
    let input_shape = input.shape();
    assert_eq!(input_shape, [1, 3, 80, 160]);
    interpreter.resize_tensor(&mut input, &input_shape);
    interpreter.resize_session(&mut session);
    let host_input = input
        .create_host_tensor(false)
        .expect("create host input tensor");
    host_input
        .write(&vec![0.0_f32; input.element_count().expect("input size")])
        .expect("write deterministic input");
    input
        .copy_from_host(&host_input)
        .expect("copy deterministic input to accelerator");

    session
        .run()
        .unwrap_or_else(|error| panic!("run PP-LCNet on {}: {error}", QUALIFICATION_BACKEND));
    assert!(session.has_run());

    let output = session.get_output(None).expect("get model output");
    assert_eq!(output.shape(), [1, 2]);
    let host_output = output
        .create_host_tensor(true)
        .expect("copy output tensor to host");
    let scores = host_output.read::<f32>().expect("read class scores");
    assert_eq!(scores.len(), 2);
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "{} output scores: {scores:?}",
        QUALIFICATION_BACKEND
    );
}
