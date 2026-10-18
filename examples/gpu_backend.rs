//! GPU backend inference example using MNN.
//!
//! This example demonstrates using GPU backends (CUDA, OpenCL, Vulkan, Metal)
//! for inference.
//!
//! Usage:
//! ```bash
//! cargo run --example gpu_backend --no-default-features --features cpu,cuda,dynamic -- /path/to/model.mnn
//! cargo run --example gpu_backend --no-default-features --features cpu,opencl,dynamic -- /path/to/model.mnn
//! cargo run --example gpu_backend --no-default-features --features cpu,vulkan,dynamic -- /path/to/model.mnn
//! ```

use mnn_oxide::{prelude::*, BackendType, MemoryMode, PrecisionMode, ScheduleConfig};

fn main() -> Result<(), MnnError> {
    // Get model path from command line
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: gpu_backend <model.mnn>");

    println!("Loading model from: {}", model_path);

    // Select backend based on feature flags
    #[cfg(feature = "cuda")]
    let backend_type = BackendType::Cuda;

    #[cfg(all(not(feature = "cuda"), feature = "opencl"))]
    let backend_type = BackendType::OpenCL;

    #[cfg(all(not(feature = "cuda"), not(feature = "opencl"), feature = "vulkan"))]
    let backend_type = BackendType::Vulkan;

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "opencl"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    let backend_type = BackendType::Metal;

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "opencl"),
        not(feature = "vulkan"),
        not(feature = "metal")
    ))]
    let backend_type = BackendType::CPU;

    let availability = backend_type.availability();
    println!("\nSelected backend: {} ({availability:?})", backend_type);
    match availability {
        mnn_oxide::BackendAvailability::Unavailable => {
            return Err(MnnError::backend_not_available(backend_type.name()));
        }
        mnn_oxide::BackendAvailability::Unknown => {
            println!(
                "Native backend probing is not implemented; attempting the explicitly selected backend."
            );
        }
        mnn_oxide::BackendAvailability::Available => {}
    }

    // Create interpreter
    let interpreter = Interpreter::from_file(&model_path)?;

    // Create session with GPU configuration
    let config = ScheduleConfig::new()
        .backend(backend_type)
        .memory_mode(MemoryMode::Normal)
        .precision_mode(PrecisionMode::Normal);

    let mut session = interpreter.create_session(config)?;
    println!("Session created with {} backend", backend_type);
    let actual_backend = session.backend()?;
    println!("MNN assigned backend: {actual_backend}");
    if actual_backend != backend_type {
        return Err(MnnError::backend_error(format!(
            "MNN assigned {actual_backend} instead of requested {backend_type}"
        )));
    }

    // Get input info
    let input = session.get_input(None)?;
    println!("Input shape: {:?}", input.shape());

    // Use deterministic input so this example exercises inference rather than
    // merely creating a session. Resize first in case the backend requires it.
    let mut input = input;
    let input_shape = input.shape();
    interpreter.resize_tensor(&mut input, &input_shape);
    interpreter.resize_session(&mut session);
    let values = vec![0.0_f32; input.element_count()?];
    if backend_type.is_gpu() {
        let host_input = input.create_host_tensor(false)?;
        host_input.write(&values)?;
        input.copy_from_host(&host_input)?;
    } else {
        input.write(&values)?;
    }

    // Run inference
    println!("\nRunning inference...");
    let start = std::time::Instant::now();
    session.run()?;
    let elapsed = start.elapsed();
    println!("Inference completed in {:?}", elapsed);

    // Get output
    let output = session.get_output(None)?;
    println!("Output shape: {:?}", output.shape());

    // Print stats
    println!("\nSession stats:");
    println!("  Memory: {} bytes", session.memory_usage());
    println!("  FLOPS: {}", session.flops());

    Ok(())
}
