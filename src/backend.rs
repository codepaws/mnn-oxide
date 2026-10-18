//! Backend configuration and management for MNN.
//!
//! This module provides types for configuring compute backends (CPU, GPU, etc.)
//! and querying backend capabilities.

/// Compute backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendType {
    /// CPU backend (always available)
    #[default]
    CPU,

    /// CUDA backend (NVIDIA GPUs)
    #[cfg(feature = "cuda")]
    Cuda,

    /// OpenCL backend (cross-platform GPU)
    #[cfg(feature = "opencl")]
    OpenCL,

    /// Vulkan backend (cross-platform GPU)
    #[cfg(feature = "vulkan")]
    Vulkan,

    /// Metal backend (Apple platforms)
    #[cfg(feature = "metal")]
    Metal,

    /// Auto-detect best available backend
    Auto,
}

impl BackendType {
    /// Convert to MNN forward type constant.
    pub(crate) fn to_mnn_type(&self) -> i32 {
        match self {
            BackendType::CPU => mnn_rs_sys::MNN_FORWARD_CPU,
            #[cfg(feature = "cuda")]
            BackendType::Cuda => mnn_rs_sys::MNN_FORWARD_CUDA,
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => mnn_rs_sys::MNN_FORWARD_OPENCL,
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => mnn_rs_sys::MNN_FORWARD_VULKAN,
            #[cfg(feature = "metal")]
            BackendType::Metal => mnn_rs_sys::MNN_FORWARD_METAL,
            BackendType::Auto => mnn_rs_sys::MNN_FORWARD_AUTO,
        }
    }

    /// Convert from MNN forward type constant.
    pub(crate) fn from_mnn_type(code: i32) -> Self {
        match code {
            #[cfg(feature = "cuda")]
            x if x == mnn_rs_sys::MNN_FORWARD_CUDA => BackendType::Cuda,
            #[cfg(feature = "opencl")]
            x if x == mnn_rs_sys::MNN_FORWARD_OPENCL => BackendType::OpenCL,
            #[cfg(feature = "vulkan")]
            x if x == mnn_rs_sys::MNN_FORWARD_VULKAN => BackendType::Vulkan,
            #[cfg(feature = "metal")]
            x if x == mnn_rs_sys::MNN_FORWARD_METAL => BackendType::Metal,
            _ => BackendType::CPU,
        }
    }

    /// Get the name of this backend.
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::CPU => "CPU",
            #[cfg(feature = "cuda")]
            BackendType::Cuda => "CUDA",
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => "OpenCL",
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => "Vulkan",
            #[cfg(feature = "metal")]
            BackendType::Metal => "Metal",
            BackendType::Auto => "Auto",
        }
    }

    /// Check if this backend is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            BackendType::Cuda => true,
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => true,
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => true,
            #[cfg(feature = "metal")]
            BackendType::Metal => true,
            _ => false,
        }
    }

    /// Report whether this backend is known to be available on the current system.
    ///
    /// CPU is reported as [`BackendAvailability::Available`] by the current API
    /// contract. Accelerator probing is not implemented yet, so CUDA, OpenCL,
    /// Vulkan, Metal, and automatic selection return [`BackendAvailability::Unknown`].
    /// Cargo features only control which
    /// backend variants can be configured; they do not prove native support,
    /// driver availability, device presence, or successful execution.
    pub fn availability(&self) -> BackendAvailability {
        match self {
            BackendType::CPU => BackendAvailability::Available,
            #[cfg(feature = "cuda")]
            BackendType::Cuda => BackendAvailability::Unknown,
            #[cfg(feature = "opencl")]
            BackendType::OpenCL => BackendAvailability::Unknown,
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => BackendAvailability::Unknown,
            #[cfg(feature = "metal")]
            BackendType::Metal => BackendAvailability::Unknown,
            BackendType::Auto => BackendAvailability::Unknown,
        }
    }

    /// Check if this backend is available on the current system.
    ///
    /// This legacy boolean API cannot distinguish an unavailable backend from one
    /// whose availability has not been checked. Prefer [`BackendType::availability`].
    /// Accelerator backends currently return `false` because probing is unsupported;
    /// that result must not be interpreted as proof that the backend is unavailable.
    #[deprecated(note = "use availability() to distinguish unknown from unavailable")]
    pub fn is_available(&self) -> bool {
        self.availability() == BackendAvailability::Available
    }

    /// Get all available backends on this system.
    ///
    /// This legacy API returns only backends confirmed available. Since accelerator
    /// probing is currently unsupported, it currently returns CPU only. It does not
    /// mean that accelerator backends are confirmed unavailable. Cargo features do
    /// not establish runtime availability. Prefer querying [`BackendType::availability`]
    /// when unknown status matters.
    #[deprecated(note = "accelerator probing is unsupported; use BackendType::availability")]
    pub fn available_backends() -> Vec<BackendType> {
        [BackendType::CPU]
            .into_iter()
            .filter(|backend| backend.availability() == BackendAvailability::Available)
            .collect()
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get list of available backends on this system.
///
/// This legacy API returns only backends confirmed available. Accelerator probing
/// is currently unsupported, so the result currently contains CPU only and does
/// not mean accelerators were confirmed unavailable. Cargo features do not prove
/// native runtime availability. Prefer [`BackendType::availability`] when unknown
/// status matters.
#[deprecated(note = "accelerator probing is unsupported; use BackendType::availability")]
#[allow(deprecated)]
pub fn available_backends() -> Vec<BackendType> {
    BackendType::available_backends()
}

/// Check if a specific backend is available.
///
/// This legacy boolean API cannot distinguish unavailable from unknown. Accelerator
/// probing is currently unsupported, so accelerator values return `false` without
/// establishing that those backends are absent. Prefer [`BackendType::availability`].
#[deprecated(note = "use BackendType::availability to distinguish unknown")]
pub fn is_backend_available(backend: BackendType) -> bool {
    backend.availability() == BackendAvailability::Available
}

/// Result of checking whether a backend is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendAvailability {
    /// The runtime explicitly confirmed that the backend is available.
    Available,
    /// The runtime explicitly confirmed that the backend is unavailable.
    Unavailable,
    /// The backend has not been probed, or no reliable probe is implemented.
    Unknown,
}

/// Backend capabilities.
///
/// This legacy snapshot cannot represent unknown values and is deprecated. Its
/// [`BackendCapabilities::query`] method returns conservative placeholders: zero
/// dimensions and false feature flags mean “not established”, not “unsupported”.
#[deprecated(
    note = "query() cannot represent unknown capabilities; do not use for runtime decisions"
)]
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    /// Maximum supported tensor dimensions, or zero when unknown.
    pub max_tensor_dimensions: i32,

    /// Whether FP16 is confirmed; false also represents unknown.
    pub supports_fp16: bool,

    /// Whether INT8 is confirmed; false also represents unknown.
    pub supports_int8: bool,

    /// Whether BF16 is confirmed; false also represents unknown.
    pub supports_bf16: bool,
}

#[allow(deprecated)]
impl BackendCapabilities {
    /// Return an unverified capability placeholder for compatibility.
    ///
    /// No capability probe is currently implemented. The returned zero and false
    /// values do not assert that the backend lacks those capabilities. This method
    /// no longer derives native capability claims from Cargo features.
    #[deprecated(
        note = "capability probing is unsupported; returned values are unknown placeholders"
    )]
    pub fn query(_backend: BackendType) -> Self {
        Self {
            max_tensor_dimensions: 0,
            supports_fp16: false,
            supports_int8: false,
            supports_bf16: false,
        }
    }
}

/// Configuration for a compute backend.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// The backend type to use
    pub backend_type: BackendType,

    /// Device ID for GPU backends (default: 0)
    pub device_id: Option<i32>,

    /// Memory usage mode
    pub memory_mode: crate::config::MemoryMode,

    /// Power usage mode
    pub power_mode: crate::config::PowerMode,

    /// Precision mode
    pub precision_mode: crate::config::PrecisionMode,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::CPU,
            device_id: None,
            memory_mode: crate::config::MemoryMode::Normal,
            power_mode: crate::config::PowerMode::Normal,
            precision_mode: crate::config::PrecisionMode::Normal,
        }
    }
}

impl BackendConfig {
    /// Create a new backend config with default settings.
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            backend_type,
            ..Default::default()
        }
    }

    /// Create a CPU backend config.
    pub fn cpu() -> Self {
        Self::new(BackendType::CPU)
    }

    /// Create a GPU backend config with auto-detection.
    pub fn gpu() -> Self {
        Self::new(BackendType::Auto)
    }

    /// Set the device ID.
    pub fn with_device_id(mut self, id: i32) -> Self {
        self.device_id = Some(id);
        self
    }

    /// Set the memory mode.
    pub fn with_memory_mode(mut self, mode: crate::config::MemoryMode) -> Self {
        self.memory_mode = mode;
        self
    }

    /// Set the power mode.
    pub fn with_power_mode(mut self, mode: crate::config::PowerMode) -> Self {
        self.power_mode = mode;
        self
    }

    /// Set the precision mode.
    pub fn with_precision_mode(mut self, mode: crate::config::PrecisionMode) -> Self {
        self.precision_mode = mode;
        self
    }
}

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DataType {
    /// 32-bit floating point
    #[default]
    Float32,

    /// 16-bit floating point (half precision)
    #[cfg(feature = "fp16")]
    Float16,

    /// Brain float 16
    #[cfg(feature = "bf16")]
    BFloat16,

    /// 32-bit signed integer
    Int32,

    /// 8-bit signed integer
    #[cfg(feature = "int8")]
    Int8,

    /// 8-bit unsigned integer
    UInt8,

    /// 16-bit signed integer
    Int16,

    /// 64-bit floating point
    Float64,
}

impl DataType {
    /// Get the size in bytes of this data type.
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            #[cfg(feature = "fp16")]
            DataType::Float16 => 2,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => 2,
            DataType::Int32 => 4,
            #[cfg(feature = "int8")]
            DataType::Int8 => 1,
            DataType::UInt8 => 1,
            DataType::Int16 => 2,
            DataType::Float64 => 8,
        }
    }

    /// Get the name of this data type.
    pub fn name(&self) -> &'static str {
        match self {
            DataType::Float32 => "float32",
            #[cfg(feature = "fp16")]
            DataType::Float16 => "float16",
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => "bfloat16",
            DataType::Int32 => "int32",
            #[cfg(feature = "int8")]
            DataType::Int8 => "int8",
            DataType::UInt8 => "uint8",
            DataType::Int16 => "int16",
            DataType::Float64 => "float64",
        }
    }

    /// Check if this is a floating point type.
    pub fn is_float(&self) -> bool {
        match self {
            DataType::Float32 | DataType::Float64 => true,
            #[cfg(feature = "fp16")]
            DataType::Float16 => true,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => true,
            _ => false,
        }
    }

    /// Check if this is an integer type.
    pub fn is_integer(&self) -> bool {
        match self {
            DataType::Int32 | DataType::Int16 | DataType::UInt8 => true,
            #[cfg(feature = "int8")]
            DataType::Int8 => true,
            _ => false,
        }
    }

    /// Check if this is a signed type.
    pub fn is_signed(&self) -> bool {
        !matches!(self, DataType::UInt8)
    }

    /// Convert to MNN type code.
    pub(crate) fn to_type_code(&self) -> i32 {
        // MNN uses halide_type_t codes: (code << 8) | bits
        match self {
            DataType::Float32 => (2 << 8) | 32, // halide_type_float = 2
            DataType::Float64 => (2 << 8) | 64,
            DataType::Int32 => 32, // halide_type_int = 0
            DataType::Int16 => 16,
            #[cfg(feature = "int8")]
            DataType::Int8 => 8,
            DataType::UInt8 => (1 << 8) | 8, // halide_type_uint = 1
            #[cfg(feature = "fp16")]
            DataType::Float16 => (2 << 8) | 16,
            #[cfg(feature = "bf16")]
            DataType::BFloat16 => (4 << 8) | 16, // halide_type_bfloat = 4
        }
    }

    /// Create from MNN type code.
    pub(crate) fn from_type_code(code: i32) -> Self {
        Self::try_from_type_code(code).unwrap_or(DataType::Float32)
    }

    /// Decode only supported native tensor types, without guessing on unknown codes.
    pub(crate) fn try_from_type_code(code: i32) -> Option<Self> {
        // MNN halide_type_t codes: (type << 8) | bits
        let type_code = (code >> 8) & 0xFF;
        let bits = code & 0xFF;

        match (type_code, bits) {
            (2, 32) => Some(DataType::Float32),
            (2, 64) => Some(DataType::Float64),
            (0, 32) => Some(DataType::Int32),
            (0, 16) => Some(DataType::Int16),
            #[cfg(feature = "int8")]
            (0, 8) => Some(DataType::Int8),
            (1, 8) => Some(DataType::UInt8),
            #[cfg(feature = "fp16")]
            (2, 16) => Some(DataType::Float16),
            #[cfg(feature = "bf16")]
            (4, 16) => Some(DataType::BFloat16),
            _ => None,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get the MNN version string.
pub fn version() -> String {
    unsafe {
        let ptr = mnn_rs_sys::mnn_get_version();
        if ptr.is_null() {
            return String::from("unknown");
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_name() {
        assert_eq!(BackendType::CPU.name(), "CPU");
        assert_eq!(BackendType::Auto.name(), "Auto");
    }

    #[test]
    fn cpu_availability_is_confirmed() {
        assert_eq!(
            BackendType::CPU.availability(),
            BackendAvailability::Available
        );
    }

    #[test]
    fn auto_backend_availability_is_unknown() {
        assert_eq!(
            BackendType::Auto.availability(),
            BackendAvailability::Unknown
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_availability_is_unknown_without_native_probe() {
        assert_eq!(
            BackendType::Cuda.availability(),
            BackendAvailability::Unknown
        );
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_availability_is_unknown_without_native_probe() {
        assert_eq!(
            BackendType::OpenCL.availability(),
            BackendAvailability::Unknown
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_availability_is_unknown_without_native_probe() {
        assert_eq!(
            BackendType::Vulkan.availability(),
            BackendAvailability::Unknown
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_availability_is_unknown_without_native_probe() {
        assert_eq!(
            BackendType::Metal.availability(),
            BackendAvailability::Unknown
        );
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_capability_query_does_not_claim_native_support() {
        let capabilities = BackendCapabilities::query(BackendType::CPU);
        assert_eq!(capabilities.max_tensor_dimensions, 0);
        assert!(!capabilities.supports_fp16);
        assert!(!capabilities.supports_int8);
        assert!(!capabilities.supports_bf16);
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
    }

    #[test]
    fn data_type_codes_match_mnn_halide_runtime() {
        assert_eq!(DataType::Float32.to_type_code(), (2 << 8) | 32);
        assert_eq!(DataType::Float64.to_type_code(), (2 << 8) | 64);
        assert_eq!(DataType::Int32.to_type_code(), 32);
        assert_eq!(DataType::Int16.to_type_code(), 16);
        assert_eq!(DataType::UInt8.to_type_code(), (1 << 8) | 8);
    }

    #[cfg(feature = "fp16")]
    #[test]
    fn float16_uses_halide_float_code() {
        let code = DataType::Float16.to_type_code();
        assert_eq!(code, (2 << 8) | 16);
        assert_eq!(DataType::try_from_type_code(code), Some(DataType::Float16));
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn bfloat16_uses_distinct_halide_bfloat_code() {
        let code = DataType::BFloat16.to_type_code();
        assert_eq!(code, (4 << 8) | 16);
        assert_eq!(DataType::try_from_type_code(code), Some(DataType::BFloat16));
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.backend_type, BackendType::CPU);
        assert!(config.device_id.is_none());
    }
}
