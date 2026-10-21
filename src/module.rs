//! Safe ownership wrappers for MNN's dynamic Module and Expr APIs.

use crate::{BackendType, DataFormat, DataType, MnnError, MnnResult};
use mnn_oxide_sys as sys;
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

#[derive(Debug)]
struct ModuleHandle(NonNull<sys::MNNModule>, PhantomData<Rc<()>>);
impl Drop for ModuleHandle {
    fn drop(&mut self) {
        unsafe { sys::mnn_module_destroy(self.0.as_ptr()) }
    }
}

/// Options used when loading an MNN dynamic module.
#[derive(Debug, Clone, Copy)]
pub struct ModuleConfig {
    /// Backend selected for module execution.
    pub backend: BackendType,
    /// Whether MNN may dynamically shape operators.
    pub dynamic: bool,
    /// Whether input shapes can change.
    pub shape_mutable: bool,
    /// Whether module weights should be rearranged at load time.
    pub rearrange: bool,
}
impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::CPU,
            dynamic: false,
            shape_mutable: false,
            rearrange: false,
        }
    }
}

/// Owned MNN dynamic module.
#[derive(Debug)]
pub struct Module {
    handle: std::sync::Arc<ModuleHandle>,
}
impl Module {
    /// Load a module from a model file with named inputs and outputs.
    pub fn from_file(
        path: impl AsRef<Path>,
        inputs: &[&str],
        outputs: &[&str],
        config: ModuleConfig,
    ) -> MnnResult<Self> {
        let p = CString::new(path.as_ref().to_string_lossy().as_bytes())?;
        let ins = cstrings(inputs)?;
        let outs = cstrings(outputs)?;
        let ip = ins.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let op = outs.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let cfg = raw_config(config)?;
        let ptr = unsafe {
            sys::mnn_module_load_from_file(
                ip.as_ptr(),
                ip.len() as i32,
                op.as_ptr(),
                op.len() as i32,
                p.as_ptr(),
                &cfg,
            )
        };
        Self::from_raw(ptr)
    }
    /// Run a module with one input variable.
    pub fn forward(&self, input: &Variable) -> MnnResult<Variable> {
        let p = unsafe { sys::mnn_module_forward(self.handle.0.as_ptr(), input.raw.as_ptr()) };
        Variable::from_raw(p, Some(self.handle.clone()))
    }
    /// Whether the module is in training mode.
    pub fn is_training(&self) -> bool {
        unsafe { sys::mnn_module_is_training(self.handle.0.as_ptr()) }
    }
    /// Set training mode.
    pub fn set_training(&mut self, enabled: bool) {
        unsafe { sys::mnn_module_set_training(self.handle.0.as_ptr(), enabled) }
    }
    /// Return module name, if present.
    pub fn name(&self) -> Option<String> {
        let p = unsafe { sys::mnn_module_get_name(self.handle.0.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        }
    }
    /// Set module name.
    pub fn set_name(&mut self, name: &str) -> MnnResult<()> {
        let s = CString::new(name)?;
        unsafe { sys::mnn_module_set_name(self.handle.0.as_ptr(), s.as_ptr()) };
        Ok(())
    }
    fn from_raw(ptr: *mut sys::MNNModule) -> MnnResult<Self> {
        Ok(Self {
            handle: std::sync::Arc::new(ModuleHandle(
                NonNull::new(ptr)
                    .ok_or_else(|| MnnError::invalid_model("MNN failed to load module"))?,
                PhantomData,
            )),
        })
    }
}

/// Tensor metadata exposed by an MNN Expr variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableInfo {
    /// Tensor dimensions.
    pub shape: Vec<i32>,
    /// MNN encoded type code.
    pub type_code: i32,
    /// Number of bits per element.
    pub type_bits: i32,
    /// Storage format.
    pub format: DataFormat,
}

/// Owned MNN Expr variable.
#[derive(Debug)]
pub struct Variable {
    raw: NonNull<sys::MNNVar>,
    owner: Vec<std::sync::Arc<ModuleHandle>>,
    _not_send: PhantomData<Rc<()>>,
}
impl Drop for Variable {
    fn drop(&mut self) {
        unsafe { sys::mnn_var_destroy(self.raw.as_ptr()) }
    }
}
impl Variable {
    /// Create a new model input variable.
    pub fn input(shape: &[i32], format: DataFormat, dtype: DataType) -> MnnResult<Self> {
        validate_shape(shape)?;
        validate_format(format)?;
        let code = dtype.to_type_code();
        let ptr = unsafe {
            sys::mnn_var_create_input(
                shape.as_ptr(),
                shape.len() as i32,
                raw_format(format),
                (code >> 8) & 255,
                code & 255,
            )
        };
        Self::from_raw(ptr, None)
    }
    /// Create a constant float variable.
    pub fn constant_f32(shape: &[i32], format: DataFormat, values: &[f32]) -> MnnResult<Self> {
        let n = validate_shape(shape)?;
        validate_format(format)?;
        if n != values.len() {
            return Err(MnnError::invalid_input(
                "constant value count does not match shape",
            ));
        }
        let ptr = unsafe {
            sys::mnn_var_create_constant_float(
                values.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                raw_format(format),
            )
        };
        Self::from_raw(ptr, None)
    }
    /// Read shape and type metadata.
    pub fn info(&self) -> MnnResult<VariableInfo> {
        let p = unsafe { sys::mnn_var_get_info(self.raw.as_ptr()) };
        let p = NonNull::new(p)
            .ok_or_else(|| MnnError::tensor_error("MNN did not return variable metadata"))?;
        let i = unsafe { p.as_ref() };
        let shape = if i.dim_count > 0 && !i.shape.is_null() {
            unsafe { std::slice::from_raw_parts(i.shape, i.dim_count as usize) }.to_vec()
        } else {
            Vec::new()
        };
        let format = match i.format {
            sys::MNNDimensionFormat::Nhwc => DataFormat::Nhwc,
            sys::MNNDimensionFormat::Nc4hw4 => DataFormat::Nc4hw4,
            sys::MNNDimensionFormat::Nchw => DataFormat::Nchw,
        };
        let out = VariableInfo {
            shape,
            type_code: i.type_code,
            type_bits: i.type_bits,
            format,
        };
        unsafe { sys::mnn_var_info_free(p.as_ptr()) };
        Ok(out)
    }
    /// Read f32 values from this variable.
    pub fn read_f32(&self) -> MnnResult<Vec<f32>> {
        validate_format(self.info()?.format)?;
        let mut count = 0;
        let p = unsafe { sys::mnn_var_read_float(self.raw.as_ptr(), &mut count) };
        if p.is_null() {
            return Err(MnnError::tensor_error("MNN failed to read variable data"));
        }
        let v = unsafe { std::slice::from_raw_parts(p, count.max(0) as usize) }.to_vec();
        unsafe { sys::mnn_float_array_free(p) };
        Ok(v)
    }
    /// Write f32 values to this variable.
    pub fn write_f32(&mut self, values: &[f32]) -> MnnResult<()> {
        let info = self.info()?;
        validate_format(info.format)?;
        let expected = validate_shape(&info.shape)?;
        if expected != values.len() {
            return Err(MnnError::invalid_input(
                "float data count does not match variable shape",
            ));
        }
        if !unsafe {
            sys::mnn_var_write_float(self.raw.as_ptr(), values.as_ptr(), values.len() as i32)
        } {
            return Err(MnnError::tensor_error("MNN failed to write variable data"));
        }
        Ok(())
    }
    /// Add another variable element-wise.
    pub fn add(&self, rhs: &Self) -> MnnResult<Self> {
        self.binary(rhs, sys::mnn_var_add)
    }
    /// Subtract another variable element-wise.
    pub fn sub(&self, rhs: &Self) -> MnnResult<Self> {
        self.binary(rhs, sys::mnn_var_sub)
    }
    /// Multiply by another variable element-wise.
    pub fn mul(&self, rhs: &Self) -> MnnResult<Self> {
        self.binary(rhs, sys::mnn_var_mul)
    }
    /// Divide by another variable element-wise.
    pub fn div(&self, rhs: &Self) -> MnnResult<Self> {
        self.binary(rhs, sys::mnn_var_div)
    }
    /// Reshape while preserving the number of elements.
    pub fn reshape(&self, shape: &[i32]) -> MnnResult<Self> {
        let old = validate_shape(&self.info()?.shape)?;
        let new = validate_shape(shape)?;
        if old != new {
            return Err(MnnError::invalid_input(
                "reshape must preserve the element count",
            ));
        }
        let ptr =
            unsafe { sys::mnn_var_reshape(self.raw.as_ptr(), shape.as_ptr(), shape.len() as i32) };
        Self::from_raw_with_owners(ptr, self.owner.clone())
    }
    /// Reduce by sum over the provided axes.
    pub fn sum(&self, axes: &[i32], keep_dims: bool) -> MnnResult<Self> {
        let rank = self.info()?.shape.len();
        if axes.iter().any(|&axis| axis < 0 || axis as usize >= rank) {
            return Err(MnnError::invalid_input("reduction axis is out of bounds"));
        }
        let ptr = unsafe {
            sys::mnn_var_sum(
                self.raw.as_ptr(),
                axes.as_ptr(),
                axes.len() as i32,
                keep_dims,
            )
        };
        Self::from_raw_with_owners(ptr, self.owner.clone())
    }
    fn binary(
        &self,
        rhs: &Self,
        op: unsafe extern "C" fn(*const sys::MNNVar, *const sys::MNNVar) -> *mut sys::MNNVar,
    ) -> MnnResult<Self> {
        if self.info()?.shape != rhs.info()?.shape {
            return Err(MnnError::invalid_input(
                "element-wise operands must have matching shapes",
            ));
        }
        let mut owners = self.owner.clone();
        owners.extend(rhs.owner.iter().cloned());
        let ptr = unsafe { op(self.raw.as_ptr(), rhs.raw.as_ptr()) };
        Self::from_raw_with_owners(ptr, owners)
    }
    fn from_raw(
        ptr: *mut sys::MNNVar,
        owner: Option<std::sync::Arc<ModuleHandle>>,
    ) -> MnnResult<Self> {
        Self::from_raw_with_owners(ptr, owner.into_iter().collect())
    }
    fn from_raw_with_owners(
        ptr: *mut sys::MNNVar,
        owners: Vec<std::sync::Arc<ModuleHandle>>,
    ) -> MnnResult<Self> {
        Ok(Self {
            raw: NonNull::new(ptr)
                .ok_or_else(|| MnnError::tensor_error("MNN operation returned a null variable"))?,
            owner: owners,
            _not_send: PhantomData,
        })
    }
}
fn cstrings(s: &[&str]) -> MnnResult<Vec<CString>> {
    s.iter()
        .map(|x| CString::new(*x).map_err(Into::into))
        .collect()
}
fn raw_config(c: ModuleConfig) -> MnnResult<sys::MNNModuleConfig> {
    if c.backend.capability().availability == crate::BackendAvailability::NotConfigured {
        return Err(MnnError::unsupported(format!(
            "{} adapter not enabled",
            c.backend
        )));
    }
    Ok(sys::MNNModuleConfig {
        dynamic: c.dynamic,
        shape_mutable: c.shape_mutable,
        rearrange: c.rearrange,
        backend_type: c.backend.to_mnn_type(),
    })
}
fn raw_format(f: DataFormat) -> sys::MNNDimensionFormat {
    match f {
        DataFormat::Nhwc => sys::MNNDimensionFormat::Nhwc,
        DataFormat::Nc4hw4 => sys::MNNDimensionFormat::Nc4hw4,
        DataFormat::Nchw => sys::MNNDimensionFormat::Nchw,
    }
}
fn validate_shape(shape: &[i32]) -> MnnResult<usize> {
    if shape.is_empty() || shape.len() > 9 || shape.iter().any(|&x| x <= 0) {
        return Err(MnnError::invalid_input(
            "shape must contain 1 to 9 positive dimensions",
        ));
    }
    let count = shape.iter().try_fold(1usize, |n, &x| {
        n.checked_mul(x as usize)
            .ok_or_else(|| MnnError::invalid_input("shape element count overflow"))
    })?;
    if count > i32::MAX as usize {
        return Err(MnnError::invalid_input(
            "shape element count exceeds MNN's supported limit",
        ));
    }
    Ok(count)
}

fn validate_format(format: DataFormat) -> MnnResult<()> {
    if format == DataFormat::Nc4hw4 {
        return Err(MnnError::unsupported("NC4HW4 Expr buffers require padded channel storage, which the safe f32 API does not currently expose"));
    }
    Ok(())
}

#[cfg(test)]
mod expr_boundary_tests {
    use super::*;

    #[test]
    fn expr_shape_rank_is_limited_to_mnns_fixed_dimension_capacity() {
        assert!(validate_shape(&[1; 10]).is_err());
    }

    #[test]
    fn packed_nc4hw4_is_rejected_until_padded_storage_is_supported() {
        assert!(validate_format(DataFormat::Nc4hw4).is_err());
    }
}
