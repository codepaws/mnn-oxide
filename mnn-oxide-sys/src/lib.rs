//! Raw FFI bindings for MNN (Mobile Neural Network) inference engine.
//!
//! This crate provides unsafe raw bindings to the MNN C wrapper API.
//! Users should prefer the safe `mnn-oxide` crate instead.
//!
//! # Safety
//!
//! All functions and types in this module are unsafe and directly map to
//! the MNN C API. Proper usage requires understanding of the MNN documentation
//! and careful memory management.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Exercise the pure acquisition/link configuration shared with build.rs.
#[cfg(test)]
#[path = "../build_config.rs"]
mod build_config_tests;

use std::os::raw::{c_char, c_int, c_void};

/// Release version declared by the selected MNN headers at build time.
///
/// This identifies the header set used to compile the wrapper. It does not
/// prove that the linked library was built from those headers.
pub const SELECTED_HEADER_VERSION: &str = env!("MNN_SELECTED_HEADER_VERSION");

// ============================================================================
// Forward declarations (opaque handles)
// ============================================================================

/// Opaque handle to an MNN Interpreter
/// The Interpreter holds the model and can create multiple sessions
#[repr(C)]
pub struct MNNInterpreter {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Session
/// A Session represents an inference context with allocated resources
#[repr(C)]
pub struct MNNSession {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Tensor
/// Tensors hold multi-dimensional array data for inference
#[repr(C)]
pub struct MNNTensor {
    _private: [u8; 0],
}

/// Opaque handle to an MNN ImageProcess
/// ImageProcess handles image preprocessing and conversion
#[repr(C)]
pub struct MNNImageProcess {
    _private: [u8; 0],
}

/// Opaque handle to an MNN Matrix
/// Matrix represents a 3x3 affine transformation matrix
#[repr(C)]
pub struct MNNMatrix {
    _private: [u8; 0],
}

// ============================================================================
// Enum Definitions
// ============================================================================

/// Backend type for computation (matches MNNForwardType)
pub const MNN_FORWARD_CPU: c_int = 0;
pub const MNN_FORWARD_AUTO: c_int = 4;
pub const MNN_FORWARD_OPENCL: c_int = 3;
pub const MNN_FORWARD_OPENGL: c_int = 6;
pub const MNN_FORWARD_VULKAN: c_int = 7;
pub const MNN_FORWARD_METAL: c_int = 1;
pub const MNN_FORWARD_CUDA: c_int = 2;
/// Compatibility alias for MNN's generic `MNN_FORWARD_USER_0` slot.
pub const MNN_FORWARD_NPU: c_int = 8;

/// Error codes
pub const MNN_ERROR_NONE: c_int = 0;
pub const MNN_ERROR_OUT_OF_MEMORY: c_int = 1;
pub const MNN_ERROR_NOT_SUPPORT: c_int = 2;
pub const MNN_ERROR_EXECUTION: c_int = 9;

/// Data format
pub const MNN_DATA_FORMAT_NHWC: c_int = 0;
pub const MNN_DATA_FORMAT_NC4HW4: c_int = 1;
pub const MNN_DATA_FORMAT_NCHW: c_int = 2;

// ============================================================================
// Image Format Constants (matches MNN::CV::ImageFormat)
// ============================================================================

pub const MNN_IMAGE_FORMAT_RGBA: c_int = 0;
pub const MNN_IMAGE_FORMAT_RGB: c_int = 1;
pub const MNN_IMAGE_FORMAT_BGR: c_int = 2;
pub const MNN_IMAGE_FORMAT_GRAY: c_int = 3;
pub const MNN_IMAGE_FORMAT_BGRA: c_int = 4;
pub const MNN_IMAGE_FORMAT_YCRCB: c_int = 5;
pub const MNN_IMAGE_FORMAT_YUV: c_int = 6;
pub const MNN_IMAGE_FORMAT_HSV: c_int = 7;
pub const MNN_IMAGE_FORMAT_XYZ: c_int = 8;
pub const MNN_IMAGE_FORMAT_BGR555: c_int = 9;
pub const MNN_IMAGE_FORMAT_BGR565: c_int = 10;
pub const MNN_IMAGE_FORMAT_YUV_NV21: c_int = 11;
pub const MNN_IMAGE_FORMAT_YUV_NV12: c_int = 12;
pub const MNN_IMAGE_FORMAT_YUV_I420: c_int = 13;
pub const MNN_IMAGE_FORMAT_HSV_FULL: c_int = 14;

// ============================================================================
// Filter Type Constants (matches MNN::CV::Filter)
// ============================================================================

pub const MNN_FILTER_NEAREST: c_int = 0;
pub const MNN_FILTER_BILINEAR: c_int = 1;
pub const MNN_FILTER_BICUBIC: c_int = 2;

// ============================================================================
// Wrap Type Constants (matches MNN::CV::Wrap)
// ============================================================================

pub const MNN_WRAP_CLAMP_TO_EDGE: c_int = 0;
pub const MNN_WRAP_ZERO: c_int = 1;
pub const MNN_WRAP_REPEAT: c_int = 2;

// ============================================================================
// ImageProcess Config Structure
// ============================================================================

/// ImageProcess config structure for image preprocessing
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNImageProcessConfig {
    pub filterType: c_int,
    pub sourceFormat: c_int,
    pub destFormat: c_int,
    pub mean: [f32; 4],
    pub normal: [f32; 4],
    pub wrap: c_int,
}

// ============================================================================
// FFI Function Declarations (C Wrapper Functions)
// ============================================================================

extern "C" {
    // ========================================================================
    // Version and Info
    // ========================================================================

    /// Get MNN version string
    pub fn mnn_get_version() -> *const c_char;

    /// Get the last C++ exception captured by a wrapper call on this thread.
    /// The pointer is valid until the next guarded wrapper call.
    pub fn mnn_last_error() -> *const c_char;

    /// Legacy availability stub: returns 1 for CPU and 0 for other backend
    /// types because native accelerator probing is not implemented. A zero
    /// result does not confirm that a backend is unavailable.
    #[deprecated(note = "non-CPU availability is unprobed; use mnn-oxide BackendAvailability")]
    pub fn mnn_is_backend_available(type_: c_int) -> c_int;

    // ========================================================================
    // Interpreter Functions
    // ========================================================================

    /// Create interpreter from file
    pub fn mnn_interpreter_create_from_file(file: *const c_char) -> *mut MNNInterpreter;

    /// Create interpreter from buffer
    pub fn mnn_interpreter_create_from_buffer(
        buffer: *const c_void,
        size: usize,
    ) -> *mut MNNInterpreter;

    /// Destroy interpreter
    pub fn mnn_interpreter_destroy(interpreter: *mut MNNInterpreter);

    /// Create session
    pub fn mnn_interpreter_create_session(
        interpreter: *mut MNNInterpreter,
        type_: c_int,
        num_thread: c_int,
    ) -> *mut MNNSession;

    /// Create a session with explicit memory, power, and precision modes.
    pub fn mnn_interpreter_create_session_v2(
        interpreter: *mut MNNInterpreter,
        type_: c_int,
        num_thread: c_int,
        memory_mode: c_int,
        power_mode: c_int,
        precision_mode: c_int,
    ) -> *mut MNNSession;

    /// Release session
    pub fn mnn_interpreter_release_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    );

    /// Run session
    pub fn mnn_interpreter_run_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> c_int;

    /// Get session input tensor
    pub fn mnn_interpreter_get_session_input(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
        name: *const c_char,
    ) -> *mut MNNTensor;

    /// Get session output tensor
    pub fn mnn_interpreter_get_session_output(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
        name: *const c_char,
    ) -> *mut MNNTensor;

    /// Resize session
    pub fn mnn_interpreter_resize_session(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    );

    /// Get session memory in MB
    pub fn mnn_interpreter_get_session_memory(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> f32;

    /// Get session FLOPS in M
    pub fn mnn_interpreter_get_session_flops(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> f32;

    /// Get the first backend MNN assigned to a session, or -1 if unavailable.
    pub fn mnn_interpreter_get_session_backend(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> c_int;

    /// Get business code
    pub fn mnn_interpreter_get_biz_code(interpreter: *mut MNNInterpreter) -> *const c_char;

    /// Get UUID
    pub fn mnn_interpreter_get_uuid(interpreter: *mut MNNInterpreter) -> *const c_char;

    // ========================================================================
    // Tensor Functions
    // ========================================================================

    /// Get tensor dimensions count
    pub fn mnn_tensor_get_dimensions(tensor: *const MNNTensor) -> c_int;

    /// Get tensor shape element at index
    pub fn mnn_tensor_get_dim(tensor: *const MNNTensor, index: c_int) -> c_int;

    /// Get tensor element count
    pub fn mnn_tensor_get_element_count(tensor: *const MNNTensor) -> c_int;

    /// Get tensor size in bytes
    pub fn mnn_tensor_get_size(tensor: *const MNNTensor) -> c_int;

    /// Get tensor host data pointer
    pub fn mnn_tensor_get_host_data(tensor: *mut MNNTensor) -> *mut c_void;

    /// Get tensor type code
    pub fn mnn_tensor_get_type_code(tensor: *const MNNTensor) -> c_int;

    /// Get tensor dimension type
    pub fn mnn_tensor_get_dimension_type(tensor: *const MNNTensor) -> c_int;

    // ========================================================================
    // ImageProcess Functions
    // ========================================================================

    /// Create image process with config
    pub fn mnn_image_process_create(config: *const MNNImageProcessConfig) -> *mut MNNImageProcess;

    /// Destroy image process
    pub fn mnn_image_process_destroy(process: *mut MNNImageProcess);

    /// Set transform matrix
    pub fn mnn_image_process_set_matrix(process: *mut MNNImageProcess, matrix: *const MNNMatrix);

    /// Convert image to tensor
    pub fn mnn_image_process_convert(
        process: *mut MNNImageProcess,
        source: *const u8,
        iw: c_int,
        ih: c_int,
        stride: c_int,
        tensor: *mut MNNTensor,
    ) -> c_int;

    /// Create image tensor
    pub fn mnn_image_tensor_create(
        w: c_int,
        h: c_int,
        bpp: c_int,
        data: *mut c_void,
    ) -> *mut MNNTensor;

    /// Destroy image tensor
    pub fn mnn_image_tensor_destroy(tensor: *mut MNNTensor);

    /// Read image from file using MNN CV (requires MNN_IMGCODECS)
    pub fn mnn_imread(path: *const c_char, flags: c_int) -> *mut MNNTensor;

    /// Write image to file using MNN CV (requires MNN_IMGCODECS)
    pub fn mnn_imwrite(
        path: *const c_char,
        tensor: *const MNNTensor,
        params: *const c_void,
    ) -> c_int;

    /// Resize image tensor (requires MNN_BUILD_OPENCV)
    pub fn mnn_resize(
        src: *const MNNTensor,
        dstWidth: c_int,
        dstHeight: c_int,
        filter: c_int,
    ) -> *mut MNNTensor;

    // ========================================================================
    // Matrix Functions
    // ========================================================================

    /// Create identity matrix
    pub fn mnn_matrix_create_identity() -> *mut MNNMatrix;

    /// Create scale matrix
    pub fn mnn_matrix_create_scale(sx: f32, sy: f32) -> *mut MNNMatrix;

    /// Create translate matrix
    pub fn mnn_matrix_create_translate(dx: f32, dy: f32) -> *mut MNNMatrix;

    /// Create rotate matrix (degrees)
    pub fn mnn_matrix_create_rotate(degrees: f32) -> *mut MNNMatrix;

    /// Create matrix from raw data (9 floats)
    pub fn mnn_matrix_create(data: *const f32) -> *mut MNNMatrix;

    /// Clone matrix
    pub fn mnn_matrix_clone(matrix: *const MNNMatrix) -> *mut MNNMatrix;

    /// Destroy matrix
    pub fn mnn_matrix_destroy(matrix: *mut MNNMatrix);

    /// Get matrix element at (row, col)
    pub fn mnn_matrix_get(matrix: *const MNNMatrix, row: c_int, col: c_int) -> f32;

    /// Set matrix element at (row, col)
    pub fn mnn_matrix_set(matrix: *mut MNNMatrix, row: c_int, col: c_int, value: f32);

    /// Multiply two matrices
    pub fn mnn_matrix_multiply(a: *const MNNMatrix, b: *const MNNMatrix) -> *mut MNNMatrix;

    /// Invert matrix
    pub fn mnn_matrix_invert(matrix: *const MNNMatrix) -> *mut MNNMatrix;

    // ========================================================================
    // Tensor Advanced Functions (GPU Memory Operations)
    // ========================================================================

    /// Copy data from host tensor to device tensor
    pub fn mnn_tensor_copy_from_host(dest: *mut MNNTensor, host_tensor: *const MNNTensor) -> c_int;

    /// Copy data from device tensor to host tensor
    pub fn mnn_tensor_copy_to_host(host_tensor: *mut MNNTensor, dest: *const MNNTensor) -> c_int;

    /// Create a host tensor with the same shape/type as a device tensor.
    pub fn mnn_tensor_create_host_from_device(
        device_tensor: *const MNNTensor,
        copy_data: c_int,
    ) -> *mut MNNTensor;

    /// Create a device tensor with given shape
    pub fn mnn_tensor_create_device(
        shape: *const c_int,
        dimensions: c_int,
        type_code: c_int,
        format: c_int,
    ) -> *mut MNNTensor;

    /// Clone a tensor
    pub fn mnn_tensor_clone(tensor: *const MNNTensor, deep_copy: c_int) -> *mut MNNTensor;

    /// Destroy a user-created tensor
    pub fn mnn_tensor_destroy(tensor: *mut MNNTensor);

    /// Get tensor device ID (for GPU tensors)
    pub fn mnn_tensor_device_id(tensor: *const MNNTensor) -> u64;

    /// Get tensor backend type
    pub fn mnn_tensor_get_backend(tensor: *const MNNTensor) -> c_int;
}

// ============================================================================
// Session Advanced Types
// ============================================================================

/// String array for returning names
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNStringArray {
    pub names: *mut *mut c_char,
    pub count: c_int,
}

// Session mode constants
pub const MNN_SESSION_MODE_DEBUG: c_int = 0;
pub const MNN_SESSION_MODE_RELEASE: c_int = 1;
pub const MNN_SESSION_MODE_INPUT_INSIDE: c_int = 2;
pub const MNN_SESSION_MODE_INPUT_USER: c_int = 3;
pub const MNN_SESSION_MODE_OUTPUT_INSIDE: c_int = 4;
pub const MNN_SESSION_MODE_OUTPUT_USER: c_int = 5;
pub const MNN_SESSION_MODE_RESIZE_DIRECT: c_int = 6;
pub const MNN_SESSION_MODE_RESIZE_DEFER: c_int = 7;
pub const MNN_SESSION_MODE_BACKEND_FIX: c_int = 8;
pub const MNN_SESSION_MODE_BACKEND_AUTO: c_int = 9;

extern "C" {
    // ========================================================================
    // Session Advanced Functions
    // ========================================================================

    /// Set session mode
    pub fn mnn_interpreter_set_session_mode(interpreter: *mut MNNInterpreter, mode: c_int);

    /// Set cache file for optimization
    pub fn mnn_interpreter_set_cache_file(
        interpreter: *mut MNNInterpreter,
        path: *const c_char,
        key_size: usize,
    );

    /// Update cache from session
    pub fn mnn_interpreter_update_cache(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> c_int;

    /// Set external file for model
    pub fn mnn_interpreter_set_external_file(
        interpreter: *mut MNNInterpreter,
        path: *const c_char,
        flag: usize,
    );

    /// Get input tensor names
    pub fn mnn_interpreter_get_input_names(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> MNNStringArray;

    /// Get output tensor names
    pub fn mnn_interpreter_get_output_names(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> MNNStringArray;

    /// Free string array
    pub fn mnn_string_array_free(array: *mut MNNStringArray);

    /// Resize tensor with new shape
    pub fn mnn_interpreter_resize_tensor(
        interpreter: *mut MNNInterpreter,
        tensor: *mut MNNTensor,
        shape: *const c_int,
        dims: c_int,
    );

    /// Get session operator count
    pub fn mnn_interpreter_get_session_op_count(
        interpreter: *mut MNNInterpreter,
        session: *mut MNNSession,
    ) -> c_int;
}

// ============================================================================
// Runtime Management Types
// ============================================================================

/// Opaque handle to MNN RuntimeManager
#[repr(C)]
pub struct MNNRuntimeManager {
    _private: [u8; 0],
}

extern "C" {
    // ========================================================================
    // Runtime Management
    // ========================================================================

    /// Create runtime manager from config
    pub fn mnn_runtime_manager_create(type_: c_int, num_threads: c_int) -> *mut MNNRuntimeManager;

    /// Create a shared runtime with explicit memory, power, and precision modes.
    pub fn mnn_runtime_manager_create_v2(
        type_: c_int,
        num_threads: c_int,
        memory_mode: c_int,
        power_mode: c_int,
        precision_mode: c_int,
    ) -> *mut MNNRuntimeManager;

    /// Destroy runtime manager
    pub fn mnn_runtime_manager_destroy(manager: *mut MNNRuntimeManager);

    /// Create session with shared runtime
    pub fn mnn_interpreter_create_session_with_runtime(
        interpreter: *mut MNNInterpreter,
        runtime: *mut MNNRuntimeManager,
        type_: c_int,
        num_threads: c_int,
    ) -> *mut MNNSession;

    /// Create a session with a shared runtime and explicit backend modes.
    pub fn mnn_interpreter_create_session_with_runtime_v2(
        interpreter: *mut MNNInterpreter,
        runtime: *mut MNNRuntimeManager,
        type_: c_int,
        num_threads: c_int,
        memory_mode: c_int,
        power_mode: c_int,
        precision_mode: c_int,
    ) -> *mut MNNSession;
}

// ============================================================================
// Dynamic Graph / Module API (requires feature = "module")
// ============================================================================

#[cfg(feature = "module")]
#[repr(C)]
pub struct MNNModule {
    _private: [u8; 0],
}

#[cfg(feature = "module")]
#[repr(C)]
pub struct MNNVar {
    _private: [u8; 0],
}

#[cfg(feature = "module")]
#[repr(C)]
pub struct MNNExecutor {
    _private: [u8; 0],
}

#[cfg(feature = "module")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MNNDimensionFormat {
    Nhwc = 0,
    Nc4hw4 = 1,
    Nchw = 2,
}

#[cfg(feature = "module")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNModuleConfig {
    pub dynamic: bool,
    pub shape_mutable: bool,
    pub rearrange: bool,
    pub backend_type: c_int,
}

#[cfg(feature = "module")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNVarInfo {
    pub shape: *mut c_int,
    pub dim_count: c_int,
    pub type_code: c_int,
    pub type_bits: c_int,
    pub format: MNNDimensionFormat,
}

#[cfg(feature = "module")]
extern "C" {
    pub fn mnn_module_load_from_file(
        inputs: *const *const c_char,
        input_count: c_int,
        outputs: *const *const c_char,
        output_count: c_int,
        file_path: *const c_char,
        config: *const MNNModuleConfig,
    ) -> *mut MNNModule;
    pub fn mnn_module_load_from_buffer(
        inputs: *const *const c_char,
        input_count: c_int,
        outputs: *const *const c_char,
        output_count: c_int,
        buffer: *const u8,
        buffer_size: usize,
        config: *const MNNModuleConfig,
    ) -> *mut MNNModule;
    pub fn mnn_module_destroy(module: *mut MNNModule);
    pub fn mnn_module_clone(module: *const MNNModule, share_params: bool) -> *mut MNNModule;
    pub fn mnn_module_forward(module: *mut MNNModule, input: *const MNNVar) -> *mut MNNVar;
    pub fn mnn_module_forward_multi(
        module: *mut MNNModule,
        inputs: *const *const MNNVar,
        input_count: c_int,
        output_count: *mut c_int,
    ) -> *mut *mut MNNVar;
    pub fn mnn_var_array_free(vars: *mut *mut MNNVar, count: c_int);
    pub fn mnn_var_array_take(vars: *mut *mut MNNVar, count: c_int, index: c_int) -> *mut MNNVar;
    pub fn mnn_module_get_parameter_count(module: *const MNNModule) -> c_int;
    pub fn mnn_module_get_parameters(
        module: *const MNNModule,
        count: *mut c_int,
    ) -> *mut *mut MNNVar;
    pub fn mnn_module_set_training(module: *mut MNNModule, is_training: bool);
    pub fn mnn_module_is_training(module: *const MNNModule) -> bool;
    pub fn mnn_module_get_name(module: *const MNNModule) -> *const c_char;
    pub fn mnn_module_set_name(module: *mut MNNModule, name: *const c_char);
    pub fn mnn_var_create_input(
        shape: *const c_int,
        dim_count: c_int,
        format: MNNDimensionFormat,
        type_code: c_int,
        type_bits: c_int,
    ) -> *mut MNNVar;
    pub fn mnn_var_create_constant_float(
        data: *const f32,
        shape: *const c_int,
        dim_count: c_int,
        format: MNNDimensionFormat,
    ) -> *mut MNNVar;
    pub fn mnn_var_destroy(var: *mut MNNVar);
    pub fn mnn_var_get_info(var: *const MNNVar) -> *mut MNNVarInfo;
    pub fn mnn_var_info_free(info: *mut MNNVarInfo);
    pub fn mnn_var_read_float(var: *const MNNVar, count: *mut c_int) -> *mut f32;
    pub fn mnn_float_array_free(values: *mut f32);
    pub fn mnn_var_write_float(var: *mut MNNVar, data: *const f32, count: c_int) -> bool;
    pub fn mnn_var_get_shape(var: *const MNNVar, count: *mut c_int) -> *mut c_int;
    pub fn mnn_int_array_free(values: *mut c_int);
    pub fn mnn_var_add(a: *const MNNVar, b: *const MNNVar) -> *mut MNNVar;
    pub fn mnn_var_sub(a: *const MNNVar, b: *const MNNVar) -> *mut MNNVar;
    pub fn mnn_var_mul(a: *const MNNVar, b: *const MNNVar) -> *mut MNNVar;
    pub fn mnn_var_div(a: *const MNNVar, b: *const MNNVar) -> *mut MNNVar;
    pub fn mnn_var_sum(
        var: *const MNNVar,
        axes: *const c_int,
        axis_count: c_int,
        keep_dims: bool,
    ) -> *mut MNNVar;
    pub fn mnn_var_mean(
        var: *const MNNVar,
        axes: *const c_int,
        axis_count: c_int,
        keep_dims: bool,
    ) -> *mut MNNVar;
    pub fn mnn_var_max(
        var: *const MNNVar,
        axes: *const c_int,
        axis_count: c_int,
        keep_dims: bool,
    ) -> *mut MNNVar;
    pub fn mnn_var_min(
        var: *const MNNVar,
        axes: *const c_int,
        axis_count: c_int,
        keep_dims: bool,
    ) -> *mut MNNVar;
    pub fn mnn_var_reshape(
        var: *const MNNVar,
        shape: *const c_int,
        dim_count: c_int,
    ) -> *mut MNNVar;
    pub fn mnn_var_transpose(
        var: *const MNNVar,
        perm: *const c_int,
        dim_count: c_int,
    ) -> *mut MNNVar;
    pub fn mnn_var_concat(vars: *const *const MNNVar, var_count: c_int, axis: c_int)
        -> *mut MNNVar;
    pub fn mnn_var_split(
        var: *const MNNVar,
        splits: c_int,
        axis: c_int,
        output_count: *mut c_int,
    ) -> *mut *mut MNNVar;
    pub fn mnn_backend_probe_compiled(backend_type: c_int) -> c_int;
    pub fn mnn_backend_probe_runtime(backend_type: c_int) -> c_int;
    pub fn mnn_backend_get_device_info(
        backend_type: c_int,
        key: *const c_char,
        value: *mut *mut c_char,
    ) -> c_int;
    pub fn mnn_backend_device_info_free(value: *mut c_char);
}

// ============================================================================
// LLM Support (requires feature = "llm")
// ============================================================================

/// Opaque handle to an MNN LLM instance (wraps MNN::Transformer::Llm)
#[cfg(feature = "llm")]
#[repr(C)]
pub struct MNNLlm {
    _private: [u8; 0],
}

/// Opaque handle to an MNN embedding model (wraps MNN::Transformer::Embedding)
#[cfg(feature = "llm")]
#[repr(C)]
pub struct MNNEmbedding {
    _private: [u8; 0],
}

/// Callback receiving incremental text chunks during streaming generation.
///
/// The `userdata` pointer is the value passed to `mnn_llm_generate_init`.
#[cfg(feature = "llm")]
pub type MnnLlmTextCb = unsafe extern "C" fn(text: *const c_char, userdata: *mut c_void);

/// Snapshot of the LLM context metrics (readable subset of MNN's LlmContext).
#[cfg(feature = "llm")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MNNLlmContext {
    /// Length of the prompt (prefill) in tokens
    pub prompt_len: c_int,
    /// Number of tokens generated this turn
    pub gen_seq_len: c_int,
    /// Total sequence length including history
    pub all_seq_len: c_int,
    /// Model load time in microseconds
    pub load_us: i64,
    /// Prefill time in microseconds
    pub prefill_us: i64,
    /// Decode time in microseconds
    pub decode_us: i64,
    /// Sampling time in microseconds
    pub sample_us: i64,
}

/// LLM status values (mirrors MNN::Transformer::LlmStatus).
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_NOT_LOADED: c_int = -1;
/// Generation is running.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_RUNNING: c_int = 0;
/// Generation finished normally.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_NORMAL_FINISHED: c_int = 1;
/// Generation stopped at the maximum token count.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_MAX_TOKENS_FINISHED: c_int = 2;
/// Generation was cancelled by the user.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_USER_CANCEL: c_int = 3;
/// An internal error occurred.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_INTERNAL_ERROR: c_int = 4;
/// Generation timed out.
#[cfg(feature = "llm")]
pub const MNN_LLM_STATUS_TIMEOUT: c_int = 5;

#[cfg(feature = "llm")]
extern "C" {
    // ========================================================================
    // LLM Lifecycle
    // ========================================================================

    /// Create an LLM instance from a model config.json path.
    pub fn mnn_llm_create(config_path: *const c_char) -> *mut MNNLlm;

    /// Destroy an LLM instance (may be NULL).
    pub fn mnn_llm_destroy(llm: *mut MNNLlm);

    /// Load model weights; returns true on success.
    pub fn mnn_llm_load(llm: *mut MNNLlm) -> bool;

    // ========================================================================
    // LLM Blocking Generation
    // ========================================================================

    /// Generate the full text response for a single prompt (blocking).
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_response_text(
        llm: *mut MNNLlm,
        text: *const c_char,
        max_new_tokens: c_int,
    ) -> *mut c_char;

    /// Generate the full text response for a chat message list (blocking).
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_response_messages(
        llm: *mut MNNLlm,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n: c_int,
        max_new_tokens: c_int,
    ) -> *mut c_char;

    // ========================================================================
    // LLM Token Generation (arrays)
    // ========================================================================

    /// Generate raw token ids for given input token ids (blocking).
    ///
    /// `out_n` is the capacity of `out` on input and the actual number of
    /// generated tokens on output. Returns the number of generated tokens
    /// (>= 0) or a negative value on error. If the result exceeds the output
    /// capacity it is truncated, but the full count is still reported through
    /// `out_n`.
    pub fn mnn_llm_generate_tokens(
        llm: *mut MNNLlm,
        input_ids: *const c_int,
        n: c_int,
        max_new_tokens: c_int,
        out: *mut c_int,
        out_n: *mut c_int,
    ) -> c_int;

    // ========================================================================
    // LLM Streaming Generation
    // ========================================================================

    /// Initialize streaming generation with a text callback.
    ///
    /// The callback is invoked synchronously with each chunk of generated
    /// text; `end_with` may be NULL.
    pub fn mnn_llm_generate_init(
        llm: *mut MNNLlm,
        text: *const c_char,
        cb: Option<MnnLlmTextCb>,
        userdata: *mut c_void,
        end_with: *const c_char,
    );

    /// Run one streaming generation step; returns true when stopped.
    pub fn mnn_llm_generate_step(llm: *mut MNNLlm, max_token: c_int) -> bool;

    /// Check whether streaming generation has stopped.
    pub fn mnn_llm_stoped(llm: *mut MNNLlm) -> bool;

    // ========================================================================
    // LLM Tokenizer and Chat Templates
    // ========================================================================

    /// Encode text into token ids.
    ///
    /// The returned array must be freed with `mnn_int_array_free`.
    pub fn mnn_llm_tokenizer_encode(
        llm: *mut MNNLlm,
        text: *const c_char,
        out_n: *mut c_int,
    ) -> *mut c_int;

    /// Decode a single token id into text.
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_tokenizer_decode(llm: *mut MNNLlm, token: c_int) -> *mut c_char;

    /// Check whether a token id is a stop token.
    pub fn mnn_llm_is_stop(llm: *mut MNNLlm, token: c_int) -> bool;

    /// Apply the chat template to a single user message.
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_apply_chat_template(llm: *mut MNNLlm, text: *const c_char) -> *mut c_char;

    /// Apply the chat template to a chat message list.
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_apply_chat_template_messages(
        llm: *mut MNNLlm,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n: c_int,
    ) -> *mut c_char;

    // ========================================================================
    // LLM Config and State
    // ========================================================================

    /// Set runtime configuration from a JSON string; returns true on success.
    pub fn mnn_llm_set_config(llm: *mut MNNLlm, json: *const c_char) -> bool;

    /// Dump the current runtime configuration as a JSON string.
    ///
    /// The returned string must be freed with `mnn_string_free`.
    pub fn mnn_llm_dump_config(llm: *mut MNNLlm) -> *mut c_char;

    /// Reset generation state (history, counters).
    pub fn mnn_llm_reset(llm: *mut MNNLlm);

    /// Check whether KV cache reuse is enabled.
    pub fn mnn_llm_reuse_kv(llm: *mut MNNLlm) -> bool;

    /// Get the current LLM status as an `MNN_LLM_STATUS_*` value, or -2 if
    /// the handle is NULL.
    pub fn mnn_llm_get_status(llm: *mut MNNLlm) -> c_int;

    /// Snapshot LLM context metrics into `out`; returns 0 on success and
    /// non-zero on failure.
    pub fn mnn_llm_get_context(llm: *mut MNNLlm, out: *mut MNNLlmContext) -> c_int;

    // ========================================================================
    // LLM Memory Helpers
    // ========================================================================

    /// Free a string previously returned by the LLM API (may be NULL).
    pub fn mnn_string_free(s: *mut c_char);

    /// Free an int array previously returned by the LLM API (may be NULL).
    pub fn mnn_int_array_free(p: *mut c_int);

    // ========================================================================
    // Embedding Model
    // ========================================================================

    /// Create an embedding model from a model config.json path.
    pub fn mnn_embedding_create(config_path: *const c_char, load: bool) -> *mut MNNEmbedding;

    /// Destroy an embedding model (may be NULL).
    pub fn mnn_embedding_destroy(embedding: *mut MNNEmbedding);

    /// Get the embedding vector dimension; returns a negative value on failure.
    pub fn mnn_embedding_dim(embedding: *mut MNNEmbedding) -> c_int;

    /// Embed a text string into a float vector.
    ///
    /// Returns the number of elements written (>= 0) or a negative value on
    /// error. If the embedding is larger than `out_cap` it is truncated.
    pub fn mnn_embedding_txt(
        embedding: *mut MNNEmbedding,
        text: *const c_char,
        out: *mut f32,
        out_cap: c_int,
    ) -> c_int;

    /// Embed token ids into a float vector.
    ///
    /// Returns the number of elements written (>= 0) or a negative value on
    /// error. If the embedding is larger than `out_cap` it is truncated.
    pub fn mnn_embedding_ids(
        embedding: *mut MNNEmbedding,
        ids: *const c_int,
        n: c_int,
        out: *mut f32,
        out_cap: c_int,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward_type_constants_match_mnn_3_6_1_header() {
        assert_eq!(super::MNN_FORWARD_CPU, 0);
        assert_eq!(super::MNN_FORWARD_METAL, 1);
        assert_eq!(super::MNN_FORWARD_CUDA, 2);
        assert_eq!(super::MNN_FORWARD_OPENCL, 3);
        assert_eq!(super::MNN_FORWARD_AUTO, 4);
        assert_eq!(super::MNN_FORWARD_OPENGL, 6);
        assert_eq!(super::MNN_FORWARD_VULKAN, 7);
        assert_eq!(super::MNN_FORWARD_NPU, 8);
    }
}

#[cfg(all(test, feature = "module"))]
mod module_abi_tests {
    use super::*;

    #[test]
    fn declared_expr_and_module_shim_surface_is_available() {
        let _: Option<
            unsafe extern "C" fn(
                *const c_int,
                c_int,
                MNNDimensionFormat,
                c_int,
                c_int,
            ) -> *mut MNNVar,
        > = Some(mnn_var_create_input);
        let _: Option<unsafe extern "C" fn(*const MNNModule, bool) -> *mut MNNModule> =
            Some(mnn_module_clone);
        let _: Option<unsafe extern "C" fn(*const MNNVar, *const MNNVar) -> *mut MNNVar> =
            Some(mnn_var_add);
        let _: Option<
            unsafe extern "C" fn(*const MNNVar, *const c_int, c_int, bool) -> *mut MNNVar,
        > = Some(mnn_var_sum);
        let _: Option<unsafe extern "C" fn(*const MNNVar, *const c_int, c_int) -> *mut MNNVar> =
            Some(mnn_var_reshape);
        let _: Option<
            unsafe extern "C" fn(*const MNNVar, c_int, c_int, *mut c_int) -> *mut *mut MNNVar,
        > = Some(mnn_var_split);
    }

    #[test]
    fn module_float_variable_write_and_read_preserve_values() {
        let shape = [2];
        let input = unsafe { mnn_var_create_input(&shape[0], 1, MNNDimensionFormat::Nchw, 2, 32) };
        assert!(!input.is_null());

        let expected = [1.25f32, -3.5f32];
        assert!(unsafe { mnn_var_write_float(input, expected.as_ptr(), expected.len() as c_int) });

        let mut count = 0;
        let values = unsafe { mnn_var_read_float(input, &mut count) };
        assert!(!values.is_null());
        assert_eq!(count, expected.len() as c_int);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(values, count as usize) },
            &expected
        );
        unsafe { mnn_float_array_free(values) };
        unsafe { mnn_var_destroy(input) };
    }

    #[test]
    fn packed_nc4hw4_raw_float_transfer_is_rejected_without_copying_logical_length() {
        let shape = [1, 3, 2, 2];
        let input = unsafe {
            mnn_var_create_input(
                shape.as_ptr(),
                shape.len() as c_int,
                MNNDimensionFormat::Nc4hw4,
                2,
                32,
            )
        };
        assert!(!input.is_null());
        let values = [0.0f32; 12];
        assert!(!unsafe { mnn_var_write_float(input, values.as_ptr(), values.len() as c_int) });
        let mut count = 0;
        assert!(unsafe { mnn_var_read_float(input, &mut count) }.is_null());
        unsafe { mnn_var_destroy(input) };
    }
}
