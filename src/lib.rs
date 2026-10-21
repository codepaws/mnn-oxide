//! # MNN Rust Bindings
//!
//! Safe Rust bindings for Alibaba's MNN (Mobile Neural Network) inference engine.
//!
//! MNN is a highly efficient and lightweight deep learning inference framework.
//! This crate provides idiomatic Rust bindings for running inference with MNN.
//!
//! ## Features
//!
//! - **Safe API**: All MNN operations are wrapped in safe Rust types
//! - **Multiple Backends**: CPU, CUDA, OpenCL, Vulkan, Metal support
//! - **Async Compatibility**: Optional thread-local async facade (native calls still block)
//! - **Cross-Platform**: Windows, Linux, macOS, Android, iOS support
//!
//! ## Quick Start
//!
//! ```no_run
//! use mnn_oxide::{Interpreter, ScheduleConfig, BackendType};
//!
//! // Load a model
//! let interpreter = Interpreter::from_file("model.mnn")?;
//!
//! // Create a session
//! let config = ScheduleConfig::new()
//!     .backend(BackendType::CPU)
//!     .num_threads(4);
//!
//! let mut session = interpreter.create_session(config)?;
//!
//! // Get input tensor
//! let input = session.get_input(None)?;
//!
//! // Fill input with data (example)
//! // input.write(&my_data)?;
//!
//! // Run inference
//! session.run()?;
//!
//! // Get output
//! let output = session.get_output(None)?;
//!
//! # Ok::<(), mnn_oxide::MnnError>(())
//! ```
//!
//! ## Backend Configuration
//!
//! ```no_run
//! use mnn_oxide::{ScheduleConfig, BackendType, MemoryMode, PrecisionMode};
//!
//! // CPU with custom settings
//! let cpu_config = ScheduleConfig::new()
//!     .backend(BackendType::CPU)
//!     .num_threads(8)
//!     .memory_mode(MemoryMode::Low);
//!
//! // GPU (auto-detect best backend)
//! let gpu_config = ScheduleConfig::new()
//!     .backend(BackendType::Auto)
//!     .precision_mode(PrecisionMode::Low);
//!
//! # Ok::<(), mnn_oxide::MnnError>(())
//! ```
//!
//! ## Local Async Facade (requires "async" feature)
//!
//! The facade keeps handles on the polling thread. Native calls run
//! synchronously and can block the executor; use a dedicated application
//! worker thread for non-blocking inference.
//!
//! ```ignore
//! use mnn_oxide::{AsyncInterpreter, ScheduleConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mnn_oxide::MnnError> {
//!     let interpreter = AsyncInterpreter::from_file("model.mnn").await?;
//!     let mut session = interpreter.create_session(ScheduleConfig::default()).await?;
//!
//!     session.run_async().await?;
//!
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub use mnn_oxide_sys;

// Core modules
mod error;
pub use error::{MnnError, MnnResult};

mod backend;
#[allow(deprecated)]
pub use backend::{
    available_backends, is_backend_available, native_version_info, version, BackendAvailability,
    BackendCapabilities, BackendCapability, BackendConfig, BackendType, DataType,
    NativeVersionInfo,
};

mod config;
pub use config::{
    DataFormat, MemoryMode, PowerMode, PrecisionMode, ScheduleConfig, ScheduleConfigBuilder,
    SessionMode,
};

mod tensor;
pub use tensor::{Tensor, TensorData, TensorInfo, TensorView};

#[cfg(feature = "image-process")]
mod image_process;
#[cfg(feature = "image-process")]
pub use image_process::{
    imread, imwrite, resize, Filter, ImageConfig, ImageFormat, ImageProcess, ImreadFlags, Matrix,
    ResizeFilter, Wrap,
};

mod session;
pub use session::{Session, SessionGuard};

mod interpreter;
#[cfg(feature = "async")]
pub use interpreter::AsyncInterpreter;
pub use interpreter::Interpreter;

mod utils;
pub use utils::{calculate_element_count, calculate_tensor_size, convert_format};

#[cfg(feature = "runtime")]
mod runtime;
#[cfg(feature = "runtime")]
pub use runtime::{InterpreterRuntimeExt, RuntimeInfo};

#[cfg(feature = "module")]
mod module;
#[cfg(feature = "module")]
pub use module::{Module, ModuleConfig, Variable, VariableInfo};

// LLM inference support (requires building MNN from source with MNN_BUILD_LLM=ON)
#[cfg(feature = "llm")]
mod llm;
#[cfg(feature = "llm")]
mod llm_callback;
#[cfg(feature = "llm")]
pub use llm::{ChatMessage, Embedding, Llm, LlmPerformance, LlmStatus};

// Async module
#[cfg(feature = "async")]
mod async_mod;
#[cfg(feature = "async")]
pub use async_mod::{run_session_async, AsyncBatchInference, SessionPool};

// Re-export commonly used types at crate root
pub use crate::error::MnnError as Error;

/// Prelude for common MNN types.
pub mod prelude {
    //! Common types for MNN operations.
    //!
    //! This module re-exports the most commonly used types for convenience.
    //!
    //! ```
    //! use mnn_oxide::prelude::*;
    //! ```

    pub use crate::backend::{BackendCapability, BackendType, DataType};
    pub use crate::config::{
        DataFormat, MemoryMode, PowerMode, PrecisionMode, ScheduleConfig, ScheduleConfigBuilder,
    };
    pub use crate::error::{MnnError, MnnResult};
    #[cfg(feature = "image-process")]
    pub use crate::image_process::{Filter, ImageConfig, ImageFormat};
    pub use crate::interpreter::Interpreter;
    #[cfg(feature = "llm")]
    pub use crate::llm::{ChatMessage, Embedding, Llm, LlmPerformance, LlmStatus};
    #[cfg(feature = "module")]
    pub use crate::module::{Module, ModuleConfig, Variable, VariableInfo};
    #[cfg(feature = "runtime")]
    pub use crate::runtime::RuntimeInfo;
    pub use crate::session::Session;
    pub use crate::tensor::{Tensor, TensorData, TensorInfo};
}

/// Testing utilities (only available in tests).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        // Version should be a non-empty string
        assert!(!v.is_empty() || v == "unknown");
    }

    #[test]
    fn test_available_backends() {
        assert_eq!(
            BackendType::CPU.availability(),
            BackendAvailability::Available
        );
    }
}

#[cfg(all(test, feature = "module"))]
mod module_api_tests {
    use super::*;

    #[test]
    fn input_variable_round_trips_float_data_with_shape_metadata() {
        let mut input = Variable::input(&[2], DataFormat::Nchw, DataType::Float32).unwrap();
        input.write_f32(&[1.25, -3.5]).unwrap();

        assert_eq!(input.info().unwrap().shape, vec![2]);
        assert_eq!(input.read_f32().unwrap(), vec![1.25, -3.5]);
    }

    #[test]
    fn expression_add_keeps_shape_and_returns_owned_result() {
        let mut input = Variable::input(&[2], DataFormat::Nchw, DataType::Float32).unwrap();
        input.write_f32(&[1.0, 4.0]).unwrap();
        let rhs = Variable::constant_f32(&[2], DataFormat::Nchw, &[2.0, 3.0]).unwrap();
        let output = input.add(&rhs).unwrap();
        assert_eq!(output.info().unwrap().shape, vec![2]);
        assert_eq!(output.read_f32().unwrap(), vec![3.0, 7.0]);
    }
}
