//! Run a deterministic CPU inference against the pinned PP-LCNet test model.
//!
//! Download the model, verify its SHA-256, and run inference with:
//! `cargo run --example model_smoke --no-default-features --features cpu,static,build-from-source`
//!
//! The model is cached under `target/test-models/`. MNN itself must already be
//! configured through one of the crate's documented acquisition modes.

#[path = "model_smoke/fixture.rs"]
mod fixture;

use mnn_oxide::prelude::*;
use std::{env, process, time::Instant};

fn run(model_path: &str, model_label: &str) -> MnnResult<()> {
    let interpreter = Interpreter::from_file(model_path)?;
    let mut session = interpreter.create_session(
        ScheduleConfig::new()
            .backend(BackendType::CPU)
            .num_threads(1),
    )?;

    let mut input = session.get_input(None)?;
    let expected_input_shape = [1, 3, 80, 160];
    let actual_input_shape = input.shape();
    if actual_input_shape != expected_input_shape {
        return Err(MnnError::shape_mismatch(
            &expected_input_shape,
            &actual_input_shape,
        ));
    }
    if input.dtype()? != DataType::Float32 {
        return Err(MnnError::invalid_input(format!(
            "Expected float32 model input, got {}",
            input.dtype()?
        )));
    }

    interpreter.resize_tensor(&mut input, &actual_input_shape);
    interpreter.resize_session(&mut session);
    let input_data = vec![0.0_f32; input.element_count()?];
    input.write(&input_data)?;

    let start = Instant::now();
    session.run()?;
    let elapsed = start.elapsed();

    let output = session.get_output(None)?;
    if output.shape() != [1, 2] {
        return Err(MnnError::shape_mismatch(&[1, 2], &output.shape()));
    }
    if output.dtype()? != DataType::Float32 {
        return Err(MnnError::invalid_input(format!(
            "Expected float32 class scores, got {}",
            output.dtype()?
        )));
    }
    let scores = output.read::<f32>()?;
    if scores.len() != 2 || scores.iter().any(|score| !score.is_finite()) {
        return Err(MnnError::invalid_input(format!(
            "Expected two finite class scores, got {scores:?}"
        )));
    }

    println!("Model: {model_label}");
    println!("Input: {actual_input_shape:?} float32 CPU tensor");
    println!("Output: {scores:?} (two finite class scores)");
    println!("Inference: {elapsed:?}");
    Ok(())
}

fn main() {
    let requested_path = env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .or_else(|| env::var_os("MNN_TEST_MODEL").map(std::path::PathBuf::from));
    let should_download = requested_path.is_none();

    let model_path = match requested_path {
        Some(path) => path,
        None => fixture::default_model_path(),
    };

    if should_download {
        match fixture::ensure_model_from_url(&model_path, fixture::MODEL_URL, fixture::MODEL_SHA256)
        {
            Ok(path) => println!(
                "Verified pinned model {} at {}",
                format!("{}@{}", fixture::MODEL_REPO, fixture::MODEL_REVISION),
                path.display()
            ),
            Err(error) => {
                eprintln!("Model download or verification failed: {error}");
                process::exit(1);
            }
        }
    }

    let model_label = fixture::model_label(&model_path, should_download);
    if let Err(error) = run(model_path.to_string_lossy().as_ref(), &model_label) {
        eprintln!("Model smoke failed: {error}");
        process::exit(1);
    }
}
