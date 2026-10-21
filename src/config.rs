//! Configuration types for MNN sessions and backends.
//!
//! This module provides configuration structures for creating interpreter
//! sessions with specific backend settings, thread counts, and data formats.

use crate::backend::{BackendConfig, BackendType};
use crate::error::{MnnError, MnnResult};

impl BackendConfig {
    /// Translate user-facing backend modes to the enum values declared by
    /// MNN 3.6.1. The native enum order differs from the Rust enum order for
    /// `High` and `Low`, so these mappings must stay explicit.
    pub(crate) fn native_values(&self) -> MnnResult<(i32, i32, i32)> {
        if self.backend_type.capability().availability
            == crate::backend::BackendAvailability::NotConfigured
        {
            return Err(MnnError::unsupported(format!(
                "{} backend adapter is not enabled in this mnn-oxide build",
                self.backend_type
            )));
        }
        if self.device_id.is_some() {
            return Err(MnnError::unsupported(
                "device_id is not supported by MNN 3.6.1 BackendConfig",
            ));
        }

        let memory = match self.memory_mode {
            crate::config::MemoryMode::Normal => 0,
            crate::config::MemoryMode::High => 1,
            crate::config::MemoryMode::Low => 2,
        };
        let power = match self.power_mode {
            crate::config::PowerMode::Normal => 0,
            crate::config::PowerMode::High => 1,
            crate::config::PowerMode::Low => 2,
        };
        let precision = match self.precision_mode {
            crate::config::PrecisionMode::Normal => 0,
            crate::config::PrecisionMode::High => 1,
            crate::config::PrecisionMode::Low => 2,
            crate::config::PrecisionMode::LowBf16 => 3,
        };
        Ok((memory, power, precision))
    }
}

/// Data format for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DataFormat {
    /// NHWC format: (batch, height, width, channels)
    /// Common in TensorFlow models
    #[default]
    Nhwc,

    /// NCHW format: (batch, channels, height, width)
    /// Common in PyTorch/ONNX models
    Nchw,

    /// NC4HW4 format: Optimized format for GPU backends
    Nc4hw4,
}

impl DataFormat {
    /// Get the name of this format
    pub fn name(&self) -> &'static str {
        match self {
            DataFormat::Nhwc => "NHWC",
            DataFormat::Nchw => "NCHW",
            DataFormat::Nc4hw4 => "NC4HW4",
        }
    }

    /// Convert to MNN dimension type constant.
    pub(crate) fn to_mnn(&self) -> i32 {
        match self {
            DataFormat::Nhwc => mnn_oxide_sys::MNN_DATA_FORMAT_NHWC,
            DataFormat::Nchw => mnn_oxide_sys::MNN_DATA_FORMAT_NCHW,
            DataFormat::Nc4hw4 => mnn_oxide_sys::MNN_DATA_FORMAT_NC4HW4,
        }
    }
}

/// Power usage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryMode {
    /// Normal memory usage (balanced)
    #[default]
    Normal,

    /// Low memory usage (may impact performance)
    Low,

    /// High memory usage for better performance
    High,
}

/// Session mode for controlling interpreter behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SessionMode {
    /// Debug mode - allows callback and internal op info
    #[default]
    Debug,
    /// Release mode - no callback, optimized
    Release,
    /// Input tensor allocated by session
    InputInside,
    /// Input tensor allocated by user
    InputUser,
    /// Output tensor depends on session
    OutputInside,
    /// Output tensor can be used separately
    OutputUser,
    /// Resize session directly
    ResizeDirect,
    /// Resize session deferred
    ResizeDefer,
    /// Backend fixed by user setting
    BackendFix,
    /// Backend auto determined by MNN
    BackendAuto,
}

impl SessionMode {
    /// Convert to MNN constant.
    pub(crate) fn to_mnn(self) -> i32 {
        match self {
            SessionMode::Debug => mnn_oxide_sys::MNN_SESSION_MODE_DEBUG,
            SessionMode::Release => mnn_oxide_sys::MNN_SESSION_MODE_RELEASE,
            SessionMode::InputInside => mnn_oxide_sys::MNN_SESSION_MODE_INPUT_INSIDE,
            SessionMode::InputUser => mnn_oxide_sys::MNN_SESSION_MODE_INPUT_USER,
            SessionMode::OutputInside => mnn_oxide_sys::MNN_SESSION_MODE_OUTPUT_INSIDE,
            SessionMode::OutputUser => mnn_oxide_sys::MNN_SESSION_MODE_OUTPUT_USER,
            SessionMode::ResizeDirect => mnn_oxide_sys::MNN_SESSION_MODE_RESIZE_DIRECT,
            SessionMode::ResizeDefer => mnn_oxide_sys::MNN_SESSION_MODE_RESIZE_DEFER,
            SessionMode::BackendFix => mnn_oxide_sys::MNN_SESSION_MODE_BACKEND_FIX,
            SessionMode::BackendAuto => mnn_oxide_sys::MNN_SESSION_MODE_BACKEND_AUTO,
        }
    }
}

/// Power usage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PowerMode {
    /// Normal power usage (balanced)
    #[default]
    Normal,

    /// Low power mode (may impact performance)
    Low,

    /// High power mode for maximum performance
    High,
}

/// Precision mode for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionMode {
    /// Normal precision (default)
    #[default]
    Normal,

    /// Low precision (faster, may reduce accuracy)
    Low,

    /// High precision
    High,

    /// Low precision with BF16
    LowBf16,
}

/// Schedule configuration for creating sessions.
///
/// This configuration determines how MNN will execute the model,
/// including backend selection, thread count, and optimization settings.
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    /// Backend configuration
    pub backend_config: BackendConfig,

    /// Number of threads for CPU backend (default: 4)
    pub num_threads: u32,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            backend_config: BackendConfig::default(),
            num_threads: 4,
        }
    }
}

impl ScheduleConfig {
    /// Create a new schedule config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a schedule config for CPU backend.
    pub fn cpu() -> Self {
        Self::default()
    }

    /// Create a schedule config for a specific backend type.
    pub fn with_backend(backend: BackendType) -> Self {
        Self {
            backend_config: BackendConfig::new(backend),
            ..Default::default()
        }
    }

    /// Set the backend type.
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.backend_config.backend_type = backend;
        self
    }

    /// Set the number of threads for CPU backend.
    pub fn num_threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the memory mode.
    pub fn memory_mode(mut self, mode: MemoryMode) -> Self {
        self.backend_config.memory_mode = mode;
        self
    }

    /// Set the power mode.
    pub fn power_mode(mut self, mode: PowerMode) -> Self {
        self.backend_config.power_mode = mode;
        self
    }

    /// Set the precision mode.
    pub fn precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.backend_config.precision_mode = mode;
        self
    }

    /// Set the device ID for GPU backends.
    #[deprecated(
        note = "MNN 3.6.1 BackendConfig does not support device_id; session creation returns Unsupported"
    )]
    pub fn device_id(mut self, id: i32) -> Self {
        self.backend_config.device_id = Some(id);
        self
    }
}

/// Builder for creating schedule configurations.
///
/// Provides a fluent interface for constructing [`ScheduleConfig`].
#[derive(Debug, Default)]
pub struct ScheduleConfigBuilder {
    config: ScheduleConfig,
}

impl ScheduleConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend type.
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.config.backend_config.backend_type = backend;
        self
    }

    /// Set the number of threads.
    pub fn num_threads(mut self, threads: u32) -> Self {
        self.config.num_threads = threads;
        self
    }

    /// Set memory mode.
    pub fn memory_mode(mut self, mode: MemoryMode) -> Self {
        self.config.backend_config.memory_mode = mode;
        self
    }

    /// Set power mode.
    pub fn power_mode(mut self, mode: PowerMode) -> Self {
        self.config.backend_config.power_mode = mode;
        self
    }

    /// Set precision mode.
    pub fn precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.config.backend_config.precision_mode = mode;
        self
    }

    /// Set device ID.
    #[deprecated(
        note = "MNN 3.6.1 BackendConfig does not support device_id; session creation returns Unsupported"
    )]
    pub fn device_id(mut self, id: i32) -> Self {
        self.config.backend_config.device_id = Some(id);
        self
    }

    /// Build the schedule config.
    pub fn build(self) -> ScheduleConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ScheduleConfig::default();
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_builder() {
        let config = ScheduleConfigBuilder::new()
            .backend(BackendType::CPU)
            .num_threads(8)
            .memory_mode(MemoryMode::Low)
            .precision_mode(PrecisionMode::High)
            .build();

        assert_eq!(config.num_threads, 8);
        assert_eq!(config.backend_config.memory_mode, MemoryMode::Low);
        assert_eq!(config.backend_config.precision_mode, PrecisionMode::High);
    }

    #[test]
    fn backend_modes_map_to_mnn_enum_values_without_swapping_high_and_low() {
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_memory_mode(MemoryMode::High)
                .native_values()
                .unwrap(),
            (1, 0, 0),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_memory_mode(MemoryMode::Low)
                .native_values()
                .unwrap(),
            (2, 0, 0),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_power_mode(PowerMode::High)
                .native_values()
                .unwrap(),
            (0, 1, 0),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_power_mode(PowerMode::Low)
                .native_values()
                .unwrap(),
            (0, 2, 0),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_precision_mode(PrecisionMode::High)
                .native_values()
                .unwrap(),
            (0, 0, 1),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_precision_mode(PrecisionMode::Low)
                .native_values()
                .unwrap(),
            (0, 0, 2),
        );
        assert_eq!(
            BackendConfig::new(BackendType::CPU)
                .with_precision_mode(PrecisionMode::LowBf16)
                .native_values()
                .unwrap(),
            (0, 0, 3),
        );
    }

    #[test]
    #[allow(deprecated)]
    fn mnn_backend_config_rejects_unimplemented_device_ids() {
        let error = BackendConfig::new(BackendType::CPU)
            .with_device_id(1)
            .native_values()
            .unwrap_err();

        assert!(error.to_string().contains("device_id"));
        assert!(error.to_string().contains("not supported"));
    }

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn mnn_backend_config_rejects_provider_without_adapter_feature() {
        let error = BackendConfig::new(BackendType::Vulkan)
            .native_values()
            .unwrap_err();

        assert!(error.to_string().contains("Vulkan"));
        assert!(error.to_string().contains("not enabled"));
    }
}
