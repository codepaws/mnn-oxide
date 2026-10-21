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
    Cuda,

    /// OpenCL backend (cross-platform GPU)
    OpenCL,

    /// Vulkan backend (cross-platform GPU)
    Vulkan,

    /// Metal backend (Apple platforms)
    Metal,

    /// Auto-detect best available backend
    Auto,
}

impl BackendType {
    /// Convert to MNN forward type constant.
    pub(crate) fn to_mnn_type(&self) -> i32 {
        match self {
            BackendType::CPU => mnn_oxide_sys::MNN_FORWARD_CPU,
            BackendType::Cuda => mnn_oxide_sys::MNN_FORWARD_CUDA,
            BackendType::OpenCL => mnn_oxide_sys::MNN_FORWARD_OPENCL,
            BackendType::Vulkan => mnn_oxide_sys::MNN_FORWARD_VULKAN,
            BackendType::Metal => mnn_oxide_sys::MNN_FORWARD_METAL,
            BackendType::Auto => mnn_oxide_sys::MNN_FORWARD_AUTO,
        }
    }

    /// Convert from MNN forward type constant.
    pub(crate) fn from_mnn_type(code: i32) -> Self {
        match code {
            x if x == mnn_oxide_sys::MNN_FORWARD_CUDA => BackendType::Cuda,
            x if x == mnn_oxide_sys::MNN_FORWARD_OPENCL => BackendType::OpenCL,
            x if x == mnn_oxide_sys::MNN_FORWARD_VULKAN => BackendType::Vulkan,
            x if x == mnn_oxide_sys::MNN_FORWARD_METAL => BackendType::Metal,
            _ => BackendType::CPU,
        }
    }

    /// Get the name of this backend.
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::CPU => "CPU",
            BackendType::Cuda => "CUDA",
            BackendType::OpenCL => "OpenCL",
            BackendType::Vulkan => "Vulkan",
            BackendType::Metal => "Metal",
            BackendType::Auto => "Auto",
        }
    }

    /// Check if this backend is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        match self {
            BackendType::Cuda => true,
            BackendType::OpenCL => true,
            BackendType::Vulkan => true,
            BackendType::Metal => true,
            _ => false,
        }
    }

    /// Report whether this backend is known to be available on the current system.
    ///
    /// CPU is reported as [`BackendAvailability::Available`] when compiled.
    /// Accelerators return [`BackendAvailability::NotCompiled`] if excluded,
    /// or [`BackendAvailability::Unknown`] until a device/session probe exists.
    /// Cargo features do not prove native support, driver availability, device
    /// presence, or successful execution. Use [`BackendType::capability`] to
    /// distinguish a backend excluded from this build from an unprobed backend,
    /// and `probe_native_compiled` when the `module` feature is enabled.
    pub fn availability(&self) -> BackendAvailability {
        self.capability().availability
    }

    /// Describe this backend's adapter configuration and current runtime
    /// availability without initializing a provider. Use the explicit native
    /// probes for creator registration and runtime assignment.
    pub fn capability(&self) -> BackendCapability {
        let adapter_enabled = match self {
            BackendType::CPU => cfg!(feature = "cpu"),
            BackendType::Cuda => cfg!(feature = "cuda"),
            BackendType::OpenCL => cfg!(feature = "opencl"),
            BackendType::Vulkan => cfg!(feature = "vulkan"),
            BackendType::Metal => cfg!(feature = "metal"),
            BackendType::Auto => true,
        };
        let (availability, reason) = if !adapter_enabled {
            (
                BackendAvailability::NotConfigured,
                Some("backend adapter support is not enabled in this mnn-oxide build"),
            )
        } else if *self == BackendType::CPU {
            (BackendAvailability::Available, None)
        } else if *self == BackendType::Auto {
            (
                BackendAvailability::Unknown,
                Some("MNN chooses a backend at session creation; no single provider is established before a model is loaded"),
            )
        } else {
            (
                BackendAvailability::Unknown,
                Some(
                    "native creator registration and device/runtime state require explicit probes",
                ),
            )
        };
        BackendCapability {
            backend: *self,
            adapter_enabled,
            availability,
            reason,
        }
    }

    /// Query the active native artifact's registered backend creator.
    ///
    /// MNN's `RuntimeManager::isBackendSupport` reports whether a native
    /// runtime creator is registered. It does not establish that a device or
    /// compatible driver is present. Requires the `module` feature because
    /// this API belongs to MNN Express' public RuntimeManager surface.
    #[cfg(feature = "module")]
    pub fn probe_native_compiled(&self) -> crate::MnnResult<bool> {
        let result = unsafe { mnn_oxide_sys::mnn_backend_probe_compiled(self.to_mnn_type()) };
        match result {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(crate::error::native_error(
                "MNN native backend support probe failed",
                crate::MnnError::backend_error(format!("could not probe {} support", self.name())),
            )),
        }
    }

    /// Initialize this provider's MNN runtime and return the backend MNN actually assigns.
    ///
    /// MNN may substitute a different provider when the requested runtime is
    /// unavailable. A successful return therefore does not imply that the
    /// requested provider was selected; compare the returned value with `self`.
    /// Runtime initialization can load provider drivers and is best-effort.
    #[cfg(feature = "module")]
    pub fn probe_native_runtime(&self) -> crate::MnnResult<Self> {
        let actual = unsafe { mnn_oxide_sys::mnn_backend_probe_runtime(self.to_mnn_type()) };
        match actual {
            x if x == mnn_oxide_sys::MNN_FORWARD_CPU => Ok(Self::CPU),
            x if x == mnn_oxide_sys::MNN_FORWARD_CUDA => Ok(Self::Cuda),
            x if x == mnn_oxide_sys::MNN_FORWARD_OPENCL => Ok(Self::OpenCL),
            x if x == mnn_oxide_sys::MNN_FORWARD_VULKAN => Ok(Self::Vulkan),
            x if x == mnn_oxide_sys::MNN_FORWARD_METAL => Ok(Self::Metal),
            x if x == mnn_oxide_sys::MNN_FORWARD_AUTO => Ok(Self::Auto),
            -1 => Err(crate::error::native_error(
                "MNN provider runtime probe failed",
                crate::MnnError::backend_error(format!(
                    "could not initialize {} runtime",
                    self.name()
                )),
            )),
            x => Err(crate::MnnError::unsupported(format!(
                "MNN runtime probe returned unknown backend value {x}"
            ))),
        }
    }

    /// Query an MNN provider-specific device information key.
    ///
    /// MNN does not standardize keys across providers. `None` means that the
    /// native provider did not supply the requested key; it does not imply
    /// that the provider is unavailable. Requires `module`.
    #[cfg(feature = "module")]
    pub fn device_info(&self, key: &str) -> crate::MnnResult<Option<String>> {
        let key = std::ffi::CString::new(key)?;
        let mut value = std::ptr::null_mut();
        let status = unsafe {
            mnn_oxide_sys::mnn_backend_get_device_info(self.to_mnn_type(), key.as_ptr(), &mut value)
        };
        match status {
            0 => Ok(None),
            1 if !value.is_null() => {
                let result = unsafe { std::ffi::CStr::from_ptr(value) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { mnn_oxide_sys::mnn_backend_device_info_free(value) };
                Ok(Some(result))
            }
            _ => Err(crate::error::native_error(
                "MNN native device information query failed",
                crate::MnnError::backend_error(format!(
                    "could not query {} device info",
                    self.name()
                )),
            )),
        }
    }

    /// All concrete MNN providers, including providers absent from this build.
    pub const PROVIDERS: [BackendType; 5] = [
        BackendType::CPU,
        BackendType::Cuda,
        BackendType::OpenCL,
        BackendType::Vulkan,
        BackendType::Metal,
    ];

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
    /// The backend adapter is not configured in this `mnn-oxide` build.
    NotConfigured,
    /// The runtime explicitly confirmed that the backend is available.
    Available,
    /// The runtime explicitly confirmed that the backend is unavailable.
    Unavailable,
    /// The backend has not been probed, or no reliable probe is implemented.
    Unknown,
}

/// Build and runtime capability snapshot for one MNN provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendCapability {
    /// Concrete MNN provider.
    pub backend: BackendType,
    /// Whether this `mnn-oxide` build enables the backend adapter. This does
    /// not establish that the selected native MNN artifact includes it.
    pub adapter_enabled: bool,
    /// Runtime availability; accelerators remain unknown until `probe_native_runtime`
    /// or a successful model session provides evidence. Registered creator support
    /// is separately available through `probe_native_compiled`.
    pub availability: BackendAvailability,
    /// Why availability is unknown or unavailable, when applicable.
    pub reason: Option<&'static str>,
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
    #[deprecated(
        note = "MNN 3.6.1 BackendConfig does not support device_id; session creation returns Unsupported"
    )]
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
        let ptr = mnn_oxide_sys::mnn_get_version();
        if ptr.is_null() {
            return String::from("unknown");
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Version information for the MNN headers selected at build time and the
/// native library loaded at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeVersionInfo {
    /// Release version declared by the selected MNN headers.
    pub selected_header: String,
    /// Version string reported by the loaded MNN library.
    pub runtime_library: String,
    /// Whether both version strings match, or `None` when either is unknown.
    pub versions_match: Option<bool>,
}

/// Compare the selected header release against the loaded library's version.
///
/// A match is a useful consistency check but does not establish binary ABI,
/// build options, artifact provenance, or publisher identity.
pub fn native_version_info() -> NativeVersionInfo {
    let selected_header = mnn_oxide_sys::SELECTED_HEADER_VERSION.to_owned();
    let runtime_library = version();
    let versions_match = compare_native_versions(&selected_header, &runtime_library);
    NativeVersionInfo {
        selected_header,
        runtime_library,
        versions_match,
    }
}

fn compare_native_versions(selected_header: &str, runtime_library: &str) -> Option<bool> {
    let selected_header = selected_header.trim();
    let runtime_library = runtime_library.trim();
    if selected_header.is_empty()
        || runtime_library.is_empty()
        || selected_header.eq_ignore_ascii_case("unknown")
        || runtime_library.eq_ignore_ascii_case("unknown")
    {
        None
    } else {
        Some(selected_header == runtime_library)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_version_comparison_reports_match_mismatch_and_unknown() {
        assert_eq!(compare_native_versions("3.6.1", "3.6.1"), Some(true));
        assert_eq!(compare_native_versions("3.6.1", "3.5.0"), Some(false));
        assert_eq!(compare_native_versions("unknown", "3.6.1"), None);
    }

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

    #[test]
    fn backend_capability_distinguishes_build_state_from_runtime_probe() {
        let capability = BackendType::Vulkan.capability();

        assert_eq!(capability.adapter_enabled, cfg!(feature = "vulkan"));
        if cfg!(feature = "vulkan") {
            assert_eq!(capability.availability, BackendAvailability::Unknown);
            assert!(capability.reason.unwrap().contains("native MNN artifact"));
        } else {
            assert_eq!(capability.availability, BackendAvailability::NotConfigured);
            assert!(capability.reason.unwrap().contains("not enabled"));
        }
    }

    #[cfg(feature = "module")]
    #[test]
    fn public_mnn_runtime_manager_probe_confirms_cpu_creator() {
        assert!(BackendType::CPU.probe_native_compiled().unwrap());
        assert_eq!(
            BackendType::CPU.probe_native_runtime().unwrap(),
            BackendType::CPU
        );
    }

    #[cfg(feature = "module")]
    #[test]
    fn unknown_device_info_key_returns_absence_not_provider_failure() {
        assert_eq!(
            BackendType::CPU
                .device_info("not_a_real_mnn_device_key")
                .unwrap(),
            None
        );
    }
}
