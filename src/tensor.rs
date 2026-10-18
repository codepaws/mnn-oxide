//! Tensor types for MNN inference.
//!
//! This module provides safe wrappers around MNN tensor operations,
//! including creating, reading, and writing tensor data.

use crate::backend::{BackendType, DataType};
use crate::config::DataFormat;
use crate::error::{native_error, MnnError, MnnResult};
use crate::session::SessionHandle;
use mnn_oxide_sys::MNNTensor;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

/// Information about a tensor's shape and type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name of the tensor (may be empty)
    pub name: String,

    /// Shape of the tensor (dimensions)
    pub shape: Vec<i32>,

    /// Data type of tensor elements
    pub dtype: DataType,

    /// Data format (layout)
    pub format: DataFormat,
}

impl TensorInfo {
    /// Calculate element count while rejecting negative dimensions and overflow.
    pub fn element_count(&self) -> MnnResult<usize> {
        checked_element_count(&self.shape)
    }

    /// Calculate byte size while rejecting invalid dimensions and overflow.
    pub fn byte_size(&self) -> MnnResult<usize> {
        self.element_count()?
            .checked_mul(self.dtype.size())
            .ok_or_else(|| MnnError::tensor_error("Tensor byte size overflow"))
    }
}

fn checked_element_count(shape: &[i32]) -> MnnResult<usize> {
    if shape.is_empty() {
        return Err(MnnError::tensor_error("Tensor shape is empty"));
    }
    shape.iter().try_fold(1usize, |count, &dim| {
        if dim < 0 {
            return Err(MnnError::tensor_error(
                "Tensor shape contains a negative dimension",
            ));
        }
        count
            .checked_mul(dim as usize)
            .ok_or_else(|| MnnError::tensor_error("Tensor element count overflow"))
    })
}

fn validate_device_shape(shape: &[i32]) -> MnnResult<i32> {
    // MNN 3.6.1 stores tensor dimensions in a fixed-size array of
    // MNN_MAX_TENSOR_DIM (9), so a longer shape is unsafe at the native edge.
    const MNN_MAX_TENSOR_DIMENSIONS: usize = 9;
    if shape.is_empty() {
        return Err(MnnError::invalid_input(
            "Device tensor shape cannot be empty",
        ));
    }
    if shape.len() > MNN_MAX_TENSOR_DIMENSIONS {
        return Err(MnnError::invalid_input(format!(
            "Device tensor rank cannot exceed {MNN_MAX_TENSOR_DIMENSIONS}"
        )));
    }
    let dimensions = i32::try_from(shape.len())
        .map_err(|_| MnnError::invalid_input("Device tensor has too many dimensions"))?;
    if shape.iter().any(|&dimension| dimension <= 0) {
        return Err(MnnError::invalid_input(
            "Device tensor dimensions must all be positive",
        ));
    }
    let elements = checked_element_count(shape)?;
    if elements > i32::MAX as usize {
        return Err(MnnError::invalid_input(
            "Device tensor element count exceeds MNN's supported limit",
        ));
    }
    Ok(dimensions)
}

fn decode_tensor_view_dtype(code: i32) -> MnnResult<DataType> {
    DataType::try_from_type_code(code).ok_or_else(|| {
        MnnError::tensor_error(format!("Unsupported native tensor type code: {code}"))
    })
}

/// A multi-dimensional array for neural network operations.
///
/// Tensors are the primary data structure for MNN inference,
/// holding input and output data for models.
///
/// Tensor handles are confined to their owning thread because they may alias
/// native session memory.
///
/// ```compile_fail,E0277
/// fn require_send<T: Send>() {}
/// require_send::<mnn_oxide::Tensor>();
/// ```
pub struct Tensor {
    inner: *mut MNNTensor,
    /// Name of the tensor (if any)
    name: Option<String>,
    /// Keeps session-owned tensor memory and its interpreter alive.
    _session_owner: Option<Arc<SessionHandle>>,
    destroy: Option<unsafe extern "C" fn(*mut MNNTensor)>,
}

impl Tensor {
    /// Create a tensor wrapper around an existing MNN tensor pointer.
    ///
    /// # Safety
    /// The pointer must be valid and remain valid for the lifetime of this tensor.
    pub(crate) unsafe fn from_session_ptr(
        ptr: *mut MNNTensor,
        name: Option<String>,
        owner: Arc<SessionHandle>,
    ) -> Self {
        Self {
            inner: ptr,
            name,
            _session_owner: Some(owner),
            destroy: None,
        }
    }

    /// Create a tensor wrapper around an existing MNN tensor pointer (public version).
    ///
    /// # Safety
    /// The pointer must be valid and remain valid for the lifetime of this tensor.
    pub unsafe fn from_ptr(ptr: *mut MNNTensor, name: Option<String>) -> Self {
        Self {
            inner: ptr,
            name,
            _session_owner: None,
            destroy: None,
        }
    }

    /// Wrap a native tensor allocated by a constructor that transfers ownership.
    pub(crate) unsafe fn from_owned_ptr(
        ptr: *mut MNNTensor,
        name: Option<String>,
        destroy: unsafe extern "C" fn(*mut MNNTensor),
        session_owner: Option<Arc<SessionHandle>>,
    ) -> Self {
        Self {
            inner: ptr,
            name,
            _session_owner: session_owner,
            destroy: Some(destroy),
        }
    }

    /// Get the mutable raw pointer to the underlying MNN tensor.
    pub fn inner_mut(&mut self) -> *mut MNNTensor {
        self.inner
    }

    /// Get the raw pointer to the underlying MNN tensor.
    pub fn as_ptr(&self) -> *const MNNTensor {
        self.inner
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dim_count = mnn_oxide_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_oxide_sys::mnn_tensor_get_dim(self.inner, i);
                shape.push(dim);
            }
            shape
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { mnn_oxide_sys::mnn_tensor_get_dimensions(self.inner) as usize }
    }

    /// Get the size of a specific dimension.
    ///
    /// # Arguments
    /// * `axis` - The dimension index (0-based)
    ///
    /// # Returns
    /// The size of the dimension, or an error if the axis is out of bounds.
    pub fn dim(&self, axis: usize) -> MnnResult<i32> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(MnnError::index_out_of_bounds(axis, 0, shape.len() as i32));
        }
        Ok(shape[axis])
    }

    /// Get the data type of the tensor.
    ///
    /// Returns an error when MNN reports a native type code this crate does
    /// not understand rather than guessing a fallback type.
    pub fn dtype(&self) -> MnnResult<DataType> {
        let code = unsafe { mnn_oxide_sys::mnn_tensor_get_type_code(self.inner) };
        decode_tensor_view_dtype(code)
    }

    fn checked_dtype(&self) -> MnnResult<DataType> {
        let code = unsafe { mnn_oxide_sys::mnn_tensor_get_type_code(self.inner) };
        DataType::try_from_type_code(code).ok_or_else(|| {
            MnnError::tensor_error(format!("Unsupported native tensor type code: {code}"))
        })
    }

    fn checked_native_byte_size(&self) -> MnnResult<usize> {
        let size = unsafe { mnn_oxide_sys::mnn_tensor_get_size(self.inner) };
        usize::try_from(size).map_err(|_| {
            MnnError::tensor_error(format!("Native tensor reported invalid byte size: {size}"))
        })
    }

    /// Get the data format of the tensor.
    pub fn format(&self) -> DataFormat {
        unsafe {
            let dim_type = mnn_oxide_sys::mnn_tensor_get_dimension_type(self.inner);
            match dim_type {
                0 => DataFormat::Nhwc,
                1 => DataFormat::Nc4hw4,
                2 => DataFormat::Nchw,
                _ => DataFormat::Nhwc,
            }
        }
    }

    /// Get the total number of elements in the tensor.
    pub fn element_count(&self) -> MnnResult<usize> {
        let count = unsafe { mnn_oxide_sys::mnn_tensor_get_element_count(self.inner) };
        usize::try_from(count).map_err(|_| {
            MnnError::tensor_error(format!(
                "Native tensor reported invalid element count: {count}"
            ))
        })
    }

    /// Get the size of the tensor data in bytes.
    pub fn byte_size(&self) -> MnnResult<usize> {
        self.checked_native_byte_size()
    }

    /// Get the name of the tensor.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Write data to the tensor.
    ///
    /// # Arguments
    /// * `data` - The data to write
    ///
    /// # Errors
    /// Returns an error if the data size doesn't match.
    pub fn write<T: TensorData>(&self, data: &[T]) -> MnnResult<()> {
        let byte_count = validate_transfer::<T>(
            self.checked_dtype()?,
            self.shape().as_slice(),
            self.checked_native_byte_size()?,
            data.len(),
        )?;
        if data.is_empty() {
            return Err(MnnError::EmptyData);
        }

        let expected_count = checked_element_count(&self.shape())?;
        if data.len() != expected_count {
            return Err(MnnError::shape_mismatch(
                &[expected_count as i32],
                &[data.len() as i32],
            ));
        }

        let host_data = unsafe { mnn_oxide_sys::mnn_tensor_get_host_data(self.inner) };
        if host_data.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, host_data, byte_count);
        }

        Ok(())
    }

    /// Read data from the tensor.
    ///
    /// # Returns
    /// A vector containing the tensor data.
    pub fn read<T: TensorData>(&self) -> MnnResult<Vec<T>> {
        let count = checked_element_count(&self.shape())?;
        let byte_count = validate_transfer::<T>(
            self.checked_dtype()?,
            self.shape().as_slice(),
            self.checked_native_byte_size()?,
            count,
        )?;
        let mut data = Vec::new();
        data.try_reserve_exact(count).map_err(|error| {
            MnnError::tensor_error(format!("Unable to allocate tensor output: {error}"))
        })?;
        data.resize(count, T::default());

        let host_data = unsafe { mnn_oxide_sys::mnn_tensor_get_host_data(self.inner) };
        if host_data.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(host_data, data.as_mut_ptr() as *mut c_void, byte_count);
        }

        Ok(data)
    }

    /// Get a mutable reference to the tensor's host data.
    ///
    /// # Safety
    /// The returned slice is valid only as long as no other operations
    /// are performed on the tensor.
    pub unsafe fn as_slice_mut<T: TensorData>(&mut self) -> MnnResult<&mut [T]> {
        let shape = self.shape();
        let count = checked_element_count(&shape)?;
        validate_transfer::<T>(
            self.checked_dtype()?,
            &shape,
            self.checked_native_byte_size()?,
            count,
        )?;
        let ptr = unsafe { mnn_oxide_sys::mnn_tensor_get_host_data(self.inner) };

        if ptr.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        Ok(unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) })
    }

    /// Get a reference to the tensor's host data.
    ///
    /// # Safety
    /// The returned slice is valid only as long as no other operations
    /// are performed on the tensor.
    pub unsafe fn as_slice<T: TensorData>(&self) -> MnnResult<&[T]> {
        let shape = self.shape();
        let count = checked_element_count(&shape)?;
        validate_transfer::<T>(
            self.checked_dtype()?,
            &shape,
            self.checked_native_byte_size()?,
            count,
        )?;
        let ptr = unsafe { mnn_oxide_sys::mnn_tensor_get_host_data(self.inner) };

        if ptr.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, count) })
    }

    /// Get the tensor info.
    pub fn info(&self) -> MnnResult<TensorInfo> {
        Ok(TensorInfo {
            name: self.name.clone().unwrap_or_default(),
            shape: self.shape(),
            dtype: self.dtype()?,
            format: self.format(),
        })
    }

    // ========================================================================
    // GPU Memory Operations
    // ========================================================================

    /// Copy data from a host tensor to this tensor (potentially on device).
    ///
    /// # Arguments
    /// * `host_tensor` - The host tensor to copy from
    ///
    /// # Returns
    /// Ok(()) on success, or an error on failure.
    pub fn copy_from_host(&mut self, host_tensor: &Tensor) -> MnnResult<()> {
        let result =
            unsafe { mnn_oxide_sys::mnn_tensor_copy_from_host(self.inner, host_tensor.inner) };

        if result != mnn_oxide_sys::MNN_ERROR_NONE as i32 {
            return Err(native_error(
                "Failed to copy from host tensor",
                MnnError::internal("Failed to copy from host tensor"),
            ));
        }

        Ok(())
    }

    /// Copy data from this tensor (potentially on device) to a host tensor.
    ///
    /// # Arguments
    /// * `host_tensor` - The host tensor to copy to
    ///
    /// # Returns
    /// Ok(()) on success, or an error on failure.
    pub fn copy_to_host(&self, host_tensor: &mut Tensor) -> MnnResult<()> {
        let result =
            unsafe { mnn_oxide_sys::mnn_tensor_copy_to_host(host_tensor.inner, self.inner) };

        if result != mnn_oxide_sys::MNN_ERROR_NONE as i32 {
            return Err(native_error(
                "Failed to copy to host tensor",
                MnnError::internal("Failed to copy to host tensor"),
            ));
        }

        Ok(())
    }

    /// Create a host tensor matching this tensor's shape and data type.
    ///
    /// For a device tensor, `copy_data` controls whether the current device
    /// values are copied into the new host tensor. Use `false` to prepare an
    /// input buffer, write it with [`Tensor::write`], then transfer it with
    /// [`Tensor::copy_from_host`].
    pub fn create_host_tensor(&self, copy_data: bool) -> MnnResult<Tensor> {
        let inner = unsafe {
            mnn_oxide_sys::mnn_tensor_create_host_from_device(self.inner, i32::from(copy_data))
        };
        if inner.is_null() {
            return Err(native_error(
                "Failed to create host tensor",
                MnnError::tensor_error("Failed to create host tensor from device tensor"),
            ));
        }

        Ok(unsafe {
            Tensor::from_owned_ptr(
                inner,
                self.name.clone(),
                mnn_oxide_sys::mnn_tensor_destroy,
                self._session_owner.clone(),
            )
        })
    }

    /// Create a device tensor with the given shape and format.
    ///
    /// # Arguments
    /// * `shape` - The tensor shape
    /// * `format` - The data format (NHWC, NCHW, etc.)
    /// * `dtype` - The data type
    ///
    /// # Returns
    /// A new device tensor on success, or an error on failure.
    pub fn create_device(shape: &[i32], format: DataFormat, dtype: DataType) -> MnnResult<Tensor> {
        let dimensions = validate_device_shape(shape)?;

        let type_code = dtype.to_type_code();
        let format_code = format.to_mnn();

        let inner = unsafe {
            mnn_oxide_sys::mnn_tensor_create_device(
                shape.as_ptr(),
                dimensions,
                type_code,
                format_code,
            )
        };

        if inner.is_null() {
            return Err(native_error(
                "Failed to create device tensor",
                MnnError::internal("Failed to create device tensor"),
            ));
        }

        Ok(unsafe { Tensor::from_owned_ptr(inner, None, mnn_oxide_sys::mnn_tensor_destroy, None) })
    }

    /// Clone this tensor.
    ///
    /// # Arguments
    /// * `deep_copy` - If true, copy data; if false, only copy metadata
    ///
    /// # Returns
    /// A cloned tensor on success, or an error on failure.
    pub fn clone(&self, deep_copy: bool) -> MnnResult<Tensor> {
        let inner =
            unsafe { mnn_oxide_sys::mnn_tensor_clone(self.inner, if deep_copy { 1 } else { 0 }) };

        if inner.is_null() {
            return Err(native_error(
                "Failed to clone tensor",
                MnnError::internal("Failed to clone tensor"),
            ));
        }

        Ok(unsafe {
            Tensor::from_owned_ptr(
                inner,
                None,
                mnn_oxide_sys::mnn_tensor_destroy,
                self._session_owner.clone(),
            )
        })
    }

    /// Get the device ID for this tensor (for GPU tensors).
    ///
    /// # Returns
    /// The device ID, or 0 if not a GPU tensor or unknown.
    pub fn device_id(&self) -> u64 {
        unsafe { mnn_oxide_sys::mnn_tensor_device_id(self.inner) }
    }

    /// Get the backend type for this tensor.
    ///
    /// # Returns
    /// The backend type (CPU, CUDA, OpenCL, etc.).
    pub fn backend(&self) -> BackendType {
        let backend_code = unsafe { mnn_oxide_sys::mnn_tensor_get_backend(self.inner) };
        BackendType::from_mnn_type(backend_code)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy {
            if !self.inner.is_null() {
                unsafe { destroy(self.inner) };
            }
        }
    }
}

fn validate_transfer<T: TensorData>(
    dtype: DataType,
    shape: &[i32],
    native_bytes: usize,
    count: usize,
) -> MnnResult<usize> {
    if T::dtype() != dtype {
        return Err(MnnError::TypeMismatch {
            expected: dtype.name().into(),
            actual: T::dtype().name().into(),
        });
    }
    let elements = checked_element_count(shape)?;
    if count != elements {
        return Err(MnnError::tensor_error(format!(
            "Tensor element count mismatch: expected {elements}, got {count}"
        )));
    }
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| MnnError::tensor_error("Tensor transfer byte size overflow"))?;
    let expected_bytes = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| MnnError::tensor_error("Tensor byte size overflow"))?;
    if bytes != expected_bytes || native_bytes != expected_bytes {
        return Err(MnnError::tensor_error(format!(
            "Tensor byte size mismatch: expected {expected_bytes}, native reports {native_bytes}"
        )));
    }
    Ok(expected_bytes)
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("format", &self.format())
            .field("name", &self.name)
            .finish()
    }
}

/// Trait for types that can be stored in a tensor.
///
/// This trait is implemented for primitive numeric types that MNN supports.
pub trait TensorData: Default + Clone + Copy + 'static {
    /// Get the MNN data type for this Rust type.
    fn dtype() -> DataType;
}

impl TensorData for f32 {
    fn dtype() -> DataType {
        DataType::Float32
    }
}

impl TensorData for f64 {
    fn dtype() -> DataType {
        DataType::Float64
    }
}

impl TensorData for i32 {
    fn dtype() -> DataType {
        DataType::Int32
    }
}

impl TensorData for i16 {
    fn dtype() -> DataType {
        DataType::Int16
    }
}

impl TensorData for u8 {
    fn dtype() -> DataType {
        DataType::UInt8
    }
}

#[cfg(feature = "fp16")]
impl TensorData for half::f16 {
    fn dtype() -> DataType {
        DataType::Float16
    }
}

#[cfg(feature = "int8")]
impl TensorData for i8 {
    fn dtype() -> DataType {
        DataType::Int8
    }
}

/// A view into a tensor's data without ownership.
///
/// This is useful for zero-copy access to tensor data.
pub struct TensorView<'a> {
    inner: *mut MNNTensor,
    _marker: PhantomData<&'a Tensor>,
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView").finish_non_exhaustive()
    }
}

impl<'a> TensorView<'a> {
    /// Create a view from a tensor reference.
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self {
            inner: tensor.inner,
            _marker: PhantomData,
        }
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dim_count = mnn_oxide_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_oxide_sys::mnn_tensor_get_dim(self.inner, i);
                shape.push(dim);
            }
            shape
        }
    }

    /// Get the data type.
    pub fn dtype(&self) -> MnnResult<DataType> {
        let code = unsafe { mnn_oxide_sys::mnn_tensor_get_type_code(self.inner) };
        decode_tensor_view_dtype(code)
    }
}

impl TensorInfo {
    /// Get the data type.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_types() {
        assert_eq!(f32::dtype(), DataType::Float32);
        assert_eq!(i32::dtype(), DataType::Int32);
        assert_eq!(u8::dtype(), DataType::UInt8);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![1, 3, 224, 224],
            dtype: DataType::Float32,
            format: DataFormat::Nchw,
        };

        assert_eq!(info.element_count().unwrap(), 1 * 3 * 224 * 224);
        assert_eq!(info.byte_size().unwrap(), 1 * 3 * 224 * 224 * 4);
    }

    #[test]
    fn checked_tensor_sizes_reject_invalid_dimensions_and_overflow() {
        let mut info = TensorInfo {
            name: String::new(),
            shape: vec![2, 3],
            dtype: DataType::Float32,
            format: DataFormat::Nchw,
        };
        assert_eq!(info.element_count().unwrap(), 6);
        assert_eq!(info.byte_size().unwrap(), 24);
        info.shape = vec![2, -1];
        assert!(info.element_count().is_err());
        info.shape = vec![i32::MAX, i32::MAX, i32::MAX];
        assert!(info.byte_size().is_err());
    }

    #[test]
    fn device_shape_validation_rejects_empty_negative_and_overflowing_shapes() {
        assert!(validate_device_shape(&[]).is_err());
        assert!(validate_device_shape(&[2, -1]).is_err());
        assert!(validate_device_shape(&[i32::MAX, i32::MAX, i32::MAX]).is_err());
        assert_eq!(validate_device_shape(&[1; 9]).unwrap(), 9);
        assert!(validate_device_shape(&[1; 10]).is_err());
        assert!(validate_device_shape(&[i32::MAX, 2]).is_err());
        assert_eq!(validate_device_shape(&[2, 3]).unwrap(), 2);
    }

    #[test]
    fn tensor_view_dtype_decoding_does_not_guess_float_for_unknown_codes() {
        assert_eq!(
            decode_tensor_view_dtype(DataType::Float32.to_type_code()).unwrap(),
            DataType::Float32
        );
        assert_eq!(
            decode_tensor_view_dtype(DataType::UInt8.to_type_code()).unwrap(),
            DataType::UInt8
        );
        assert!(decode_tensor_view_dtype(-1).is_err());
    }

    #[test]
    fn owned_tensor_destroys_native_object_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DESTROYED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn count_destroy(_: *mut MNNTensor) {
            DESTROYED.fetch_add(1, Ordering::SeqCst);
        }
        DESTROYED.store(0, Ordering::SeqCst);
        {
            let _owned = unsafe {
                Tensor::from_owned_ptr(1usize as *mut MNNTensor, None, count_destroy, None)
            };
        }
        assert_eq!(DESTROYED.load(Ordering::SeqCst), 1);
        let _borrowed = unsafe { Tensor::from_ptr(std::ptr::null_mut(), None) };
        assert_eq!(DESTROYED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn transfer_validation_checks_dtype_and_exact_bytes() {
        assert!(validate_transfer::<f32>(DataType::Int32, &[2], 8, 2).is_err());
        assert!(validate_transfer::<f32>(DataType::Float32, &[2], 4, 2).is_err());
        assert!(validate_transfer::<f32>(DataType::Float32, &[2], 8, 2).is_ok());
    }
}
