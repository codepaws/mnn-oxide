//! Image processing and preprocessing for MNN.
//!
//! This module provides image processing capabilities for converting raw image data
//! to tensors suitable for neural network inference.

use crate::error::{native_error, MnnError, MnnResult};
use crate::tensor::Tensor;
use mnn_oxide_sys::*;
use std::ffi::c_void;

/// Image format for preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ImageFormat {
    /// RGBA: 4 channels, red-green-blue-alpha order
    #[default]
    Rgba,
    /// RGB: 3 channels, red-green-blue order
    Rgb,
    /// BGR: 3 channels, blue-green-red order (OpenCV default)
    Bgr,
    /// GRAY: 1 channel grayscale
    Gray,
    /// BGRA: 4 channels, blue-green-red-alpha order
    Bgra,
    /// YCrCb: 3 channels YCrCb color space
    YCrCb,
    /// YUV: 3 channels YUV color space
    Yuv,
    /// HSV: 3 channels HSV color space
    Hsv,
    /// XYZ: 3 channels XYZ color space
    Xyz,
    /// BGR555: 2 bytes per pixel, 5 bits per channel
    Bgr555,
    /// BGR565: 2 bytes per pixel, 5-6-5 bits per channel
    Bgr565,
    /// YUV_NV21: YUV 4:2:0 NV21 format (Android camera)
    YuvNv21,
    /// YUV_NV12: YUV 4:2:0 NV12 format
    YuvNv12,
    /// YUV_I420: YUV 4:2:0 I420 format
    YuvI420,
    /// HSV_FULL: HSV with full range values
    HsvFull,
}

impl ImageFormat {
    /// Convert to MNN constant.
    pub(crate) fn to_mnn(self) -> i32 {
        match self {
            ImageFormat::Rgba => MNN_IMAGE_FORMAT_RGBA,
            ImageFormat::Rgb => MNN_IMAGE_FORMAT_RGB,
            ImageFormat::Bgr => MNN_IMAGE_FORMAT_BGR,
            ImageFormat::Gray => MNN_IMAGE_FORMAT_GRAY,
            ImageFormat::Bgra => MNN_IMAGE_FORMAT_BGRA,
            ImageFormat::YCrCb => MNN_IMAGE_FORMAT_YCRCB,
            ImageFormat::Yuv => MNN_IMAGE_FORMAT_YUV,
            ImageFormat::Hsv => MNN_IMAGE_FORMAT_HSV,
            ImageFormat::Xyz => MNN_IMAGE_FORMAT_XYZ,
            ImageFormat::Bgr555 => MNN_IMAGE_FORMAT_BGR555,
            ImageFormat::Bgr565 => MNN_IMAGE_FORMAT_BGR565,
            ImageFormat::YuvNv21 => MNN_IMAGE_FORMAT_YUV_NV21,
            ImageFormat::YuvNv12 => MNN_IMAGE_FORMAT_YUV_NV12,
            ImageFormat::YuvI420 => MNN_IMAGE_FORMAT_YUV_I420,
            ImageFormat::HsvFull => MNN_IMAGE_FORMAT_HSV_FULL,
        }
    }

    /// Get the number of channels for this format.
    pub fn channels(&self) -> i32 {
        match self {
            ImageFormat::Rgba | ImageFormat::Bgra => 4,
            ImageFormat::Rgb
            | ImageFormat::Bgr
            | ImageFormat::YCrCb
            | ImageFormat::Yuv
            | ImageFormat::Hsv
            | ImageFormat::HsvFull
            | ImageFormat::Xyz => 3,
            ImageFormat::Gray => 1,
            ImageFormat::Bgr555 | ImageFormat::Bgr565 => 2, // stored as 2 bytes
            ImageFormat::YuvNv21 | ImageFormat::YuvNv12 | ImageFormat::YuvI420 => 1, // planar, 1 byte per pixel for Y
        }
    }
}

fn checked_image_buffer_len(
    format: ImageFormat,
    width: i32,
    height: i32,
    stride: i32,
) -> MnnResult<usize> {
    if width <= 0 || height <= 0 || stride < 0 {
        return Err(MnnError::InvalidInput(
            "Image width and height must be positive and stride cannot be negative".into(),
        ));
    }

    let width = width as usize;
    let height = height as usize;
    let requested_stride = stride as usize;
    let ceil_half = |value: usize| value / 2 + value % 2;

    let total = match format {
        ImageFormat::YuvNv21 | ImageFormat::YuvNv12 => {
            let row_stride = if stride == 0 { width } else { requested_stride };
            if row_stride < width {
                return Err(MnnError::InvalidInput(
                    "YUV row stride must be at least the image width".into(),
                ));
            }
            // MNN reads interleaved UV pairs. Odd widths therefore require an
            // even UV row width; an explicit stride must include that padding.
            let uv_row_stride = if stride == 0 {
                ceil_half(width).checked_mul(2)
            } else if row_stride < ceil_half(width).saturating_mul(2) {
                return Err(MnnError::InvalidInput(
                    "NV12/NV21 stride must fit the interleaved chroma row".into(),
                ));
            } else {
                Some(row_stride)
            }
            .ok_or_else(|| MnnError::InvalidInput("Image row size overflow".into()))?;
            row_stride.checked_mul(height).and_then(|luma| {
                uv_row_stride
                    .checked_mul(ceil_half(height))
                    .and_then(|chroma| luma.checked_add(chroma))
            })
        }
        ImageFormat::YuvI420 => {
            // MNN's copy path ignores the supplied stride while its nearest
            // path treats it as both luma and chroma stride. Reject the
            // ambiguous layout instead of guessing at the backing-buffer size.
            if stride != 0 {
                return Err(MnnError::InvalidInput(
                    "Explicit stride is not supported for I420 input".into(),
                ));
            }
            let row_stride = if stride == 0 { width } else { requested_stride };
            if row_stride < width {
                return Err(MnnError::InvalidInput(
                    "I420 row stride must be at least the image width".into(),
                ));
            }
            let chroma_stride = ceil_half(row_stride);
            row_stride.checked_mul(height).and_then(|luma| {
                chroma_stride
                    .checked_mul(ceil_half(height))
                    .and_then(|plane| plane.checked_mul(2))
                    .and_then(|chroma| luma.checked_add(chroma))
            })
        }
        _ => {
            let bytes_per_pixel = match format {
                ImageFormat::Rgba | ImageFormat::Bgra => 4,
                ImageFormat::Rgb
                | ImageFormat::Bgr
                | ImageFormat::YCrCb
                | ImageFormat::Yuv
                | ImageFormat::Hsv
                | ImageFormat::HsvFull
                | ImageFormat::Xyz => 3,
                ImageFormat::Bgr555 | ImageFormat::Bgr565 => 2,
                ImageFormat::Gray => 1,
                ImageFormat::YuvNv21 | ImageFormat::YuvNv12 | ImageFormat::YuvI420 => {
                    unreachable!()
                }
            };
            let minimum_stride = width
                .checked_mul(bytes_per_pixel)
                .ok_or_else(|| MnnError::InvalidInput("Image row size overflow".into()))?;
            let row_stride = if stride == 0 {
                minimum_stride
            } else {
                requested_stride
            };
            if row_stride < minimum_stride {
                return Err(MnnError::InvalidInput(
                    "Image row stride is smaller than the pixel row".into(),
                ));
            }
            row_stride.checked_mul(height)
        }
    }
    .ok_or_else(|| MnnError::InvalidInput("Image buffer size overflow".into()))?;

    if total > i32::MAX as usize {
        return Err(MnnError::InvalidInput(
            "Image buffer exceeds MNN's supported byte-size limit".into(),
        ));
    }
    Ok(total)
}

fn checked_image_tensor_len(width: i32, height: i32, bytes_per_pixel: i32) -> MnnResult<usize> {
    if width <= 0 || height <= 0 || bytes_per_pixel <= 0 {
        return Err(MnnError::InvalidInput(
            "Image tensor dimensions and bytes per pixel must be positive".into(),
        ));
    }
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel as usize))
        .ok_or_else(|| MnnError::InvalidInput("Image tensor size overflow".into()))?;
    if bytes > i32::MAX as usize {
        return Err(MnnError::InvalidInput(
            "Image tensor exceeds MNN's supported byte-size limit".into(),
        ));
    }
    Ok(bytes)
}

/// Filter type for image scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    /// Nearest neighbor (fastest, lowest quality)
    #[default]
    Nearest,
    /// Bilinear interpolation (good balance)
    Bilinear,
    /// Bicubic interpolation (slowest, best quality)
    Bicubic,
}

impl Filter {
    /// Convert to MNN constant.
    pub(crate) fn to_mnn(self) -> i32 {
        match self {
            Filter::Nearest => MNN_FILTER_NEAREST,
            Filter::Bilinear => MNN_FILTER_BILINEAR,
            Filter::Bicubic => MNN_FILTER_BICUBIC,
        }
    }
}

/// Wrap mode for edge handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    /// Clamp to edge pixels
    #[default]
    ClampToEdge,
    /// Zero padding
    Zero,
    /// Repeat/tile the image
    Repeat,
}

impl Wrap {
    /// Convert to MNN constant.
    pub(crate) fn to_mnn(self) -> i32 {
        match self {
            Wrap::ClampToEdge => MNN_WRAP_CLAMP_TO_EDGE,
            Wrap::Zero => MNN_WRAP_ZERO,
            Wrap::Repeat => MNN_WRAP_REPEAT,
        }
    }
}

/// Configuration for image processing.
#[derive(Debug, Clone)]
pub struct ImageConfig {
    /// Source image format
    pub source_format: ImageFormat,
    /// Destination format
    pub dest_format: ImageFormat,
    /// Filter type for scaling
    pub filter: Filter,
    /// Mean values for normalization (per channel)
    pub mean: [f32; 4],
    /// Normalization scales (per channel)
    pub normal: [f32; 4],
    /// Edge wrap mode
    pub wrap: Wrap,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            source_format: ImageFormat::Rgba,
            dest_format: ImageFormat::Rgba,
            filter: Filter::Bilinear,
            mean: [0.0f32; 4],
            normal: [1.0f32; 4],
            wrap: Wrap::ClampToEdge,
        }
    }
}

impl ImageConfig {
    /// Create a new image config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source format.
    pub fn with_source_format(mut self, format: ImageFormat) -> Self {
        self.source_format = format;
        self
    }

    /// Set the destination format.
    pub fn with_dest_format(mut self, format: ImageFormat) -> Self {
        self.dest_format = format;
        self
    }

    /// Set the filter type.
    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Set mean values for normalization.
    pub fn with_mean(mut self, mean: [f32; 4]) -> Self {
        self.mean = mean;
        self
    }

    /// Set normalization scales.
    pub fn with_normal(mut self, normal: [f32; 4]) -> Self {
        self.normal = normal;
        self
    }

    /// Set wrap mode.
    pub fn with_wrap(mut self, wrap: Wrap) -> Self {
        self.wrap = wrap;
        self
    }

    /// Common preset: RGB input with ImageNet normalization.
    pub fn imagenet_rgb() -> Self {
        Self {
            source_format: ImageFormat::Rgb,
            dest_format: ImageFormat::Rgb,
            filter: Filter::Bilinear,
            mean: [123.675, 116.28, 103.53, 0.0],
            normal: [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375, 1.0],
            wrap: Wrap::ClampToEdge,
        }
    }

    /// Common preset: BGR input (OpenCV) with ImageNet normalization.
    pub fn imagenet_bgr() -> Self {
        Self {
            source_format: ImageFormat::Bgr,
            dest_format: ImageFormat::Bgr,
            filter: Filter::Bilinear,
            mean: [103.53, 116.28, 123.675, 0.0],
            normal: [1.0 / 57.375, 1.0 / 57.12, 1.0 / 58.395, 1.0],
            wrap: Wrap::ClampToEdge,
        }
    }

    /// Convert to MNN config struct.
    pub(crate) fn to_mnn(&self) -> MNNImageProcessConfig {
        MNNImageProcessConfig {
            filterType: self.filter.to_mnn(),
            sourceFormat: self.source_format.to_mnn(),
            destFormat: self.dest_format.to_mnn(),
            mean: self.mean,
            normal: self.normal,
            wrap: self.wrap.to_mnn(),
        }
    }
}

/// Affine transformation matrix.
///
/// Represents a 3x3 matrix for 2D affine transformations including
/// translation, scale, rotation, and skew.
///
/// `Matrix` owns its native allocation and therefore does not implement
/// [`Clone`]. Use [`Matrix::try_clone`] to request an independent native copy.
///
/// ```compile_fail
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<mnn_oxide::Matrix>();
/// ```
///
/// Native matrix handles remain on their owning thread.
///
/// ```compile_fail,E0277
/// fn require_send<T: Send>() {}
/// require_send::<mnn_oxide::Matrix>();
/// ```
#[derive(Debug)]
pub struct Matrix {
    inner: *mut MNNMatrix,
}

impl Matrix {
    /// Create an identity matrix.
    pub fn identity() -> Self {
        let inner = unsafe { mnn_matrix_create_identity() };
        Self { inner }
    }

    /// Create a scale matrix.
    pub fn scale(sx: f32, sy: f32) -> Self {
        let inner = unsafe { mnn_matrix_create_scale(sx, sy) };
        Self { inner }
    }

    /// Create a translation matrix.
    pub fn translate(dx: f32, dy: f32) -> Self {
        let inner = unsafe { mnn_matrix_create_translate(dx, dy) };
        Self { inner }
    }

    /// Create a rotation matrix.
    pub fn rotate(degrees: f32) -> Self {
        let inner = unsafe { mnn_matrix_create_rotate(degrees) };
        Self { inner }
    }

    /// Create a matrix from raw data (9 floats, row-major order).
    pub fn from_array(data: &[f32; 9]) -> Self {
        let inner = unsafe { mnn_matrix_create(data.as_ptr()) };
        Self { inner }
    }

    /// Create a matrix from a pointer to 9 floats.
    ///
    /// # Safety
    /// The pointer must be valid and point to at least 9 floats.
    pub unsafe fn from_ptr(ptr: *const f32) -> Self {
        let inner = unsafe { mnn_matrix_create(ptr) };
        Self { inner }
    }

    /// Get a matrix element.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        if row > 2 || col > 2 {
            return 0.0;
        }
        unsafe { mnn_matrix_get(self.inner, row as i32, col as i32) }
    }

    /// Set a matrix element.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        if row > 2 || col > 2 {
            return;
        }
        unsafe { mnn_matrix_set(self.inner, row as i32, col as i32, value) }
    }

    /// Get all matrix elements as an array (row-major order).
    pub fn to_array(&self) -> [f32; 9] {
        let mut data = [0.0f32; 9];
        for row in 0..3 {
            for col in 0..3 {
                data[row * 3 + col] = self.get(row, col);
            }
        }
        data
    }

    /// Multiply two matrices.
    pub fn multiply(&self, other: &Matrix) -> Option<Matrix> {
        let inner = unsafe { mnn_matrix_multiply(self.inner, other.inner) };
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }

    /// Invert the matrix.
    pub fn invert(&self) -> Option<Matrix> {
        let inner = unsafe { mnn_matrix_invert(self.inner) };
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }

    /// Create an independent native copy of this matrix.
    ///
    /// Returns an error if MNN cannot allocate the copy.
    pub fn try_clone(&self) -> MnnResult<Matrix> {
        let inner = unsafe { mnn_matrix_clone(self.inner) };
        unsafe { Self::from_owned_ptr(inner) }
    }

    /// Convert a returned native pointer into an owned Matrix.
    ///
    /// # Safety
    /// A non-null pointer must be a valid MNNMatrix allocation owned by caller.
    unsafe fn from_owned_ptr(inner: *mut MNNMatrix) -> MnnResult<Self> {
        if inner.is_null() {
            Err(native_error(
                "Failed to clone matrix",
                MnnError::internal("Failed to clone matrix (MNN matrix clone returned null)"),
            ))
        } else {
            Ok(Self { inner })
        }
    }

    /// Get the raw pointer to the underlying MNN matrix.
    ///
    /// # Safety
    /// The returned pointer is owned by this Matrix and must not be freed.
    pub unsafe fn as_ptr(&self) -> *const MNNMatrix {
        self.inner
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::identity()
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                mnn_matrix_destroy(self.inner);
            }
        }
    }
}

/// Image processor for converting raw image data to tensors.
///
/// ImageProcess handles:
/// - Format conversion (RGB, BGR, RGBA, YUV, etc.)
/// - Image scaling with various filters
/// - Mean subtraction and normalization
/// - Affine transformations
///
/// # Example
/// ```no_run
/// use mnn_oxide::{ImageProcess, ImageConfig, ImageFormat, Filter};
///
/// // Create config for RGB image with normalization
/// let config = ImageConfig::new()
///     .with_source_format(ImageFormat::Rgb)
///     .with_filter(Filter::Bilinear)
///     .with_mean([123.675, 116.28, 103.53, 0.0])
///     .with_normal([1.0/58.395, 1.0/57.12, 1.0/57.375, 1.0]);
///
/// // Create the image processor
/// let processor = ImageProcess::new(&config)?;
///
/// // Convert image data to tensor
/// // let image_data: &[u8] = ...;
/// // processor.convert(image_data, width, height, &mut tensor)?;
/// # Ok::<(), mnn_oxide::MnnError>(())
/// ```
///
/// Image processors remain on their owning thread.
///
/// ```compile_fail,E0277
/// fn require_send<T: Send>() {}
/// require_send::<mnn_oxide::ImageProcess>();
/// ```
pub struct ImageProcess {
    inner: *mut MNNImageProcess,
    source_format: ImageFormat,
}

impl ImageProcess {
    /// Create a new image processor with the given configuration.
    pub fn new(config: &ImageConfig) -> MnnResult<Self> {
        let mnn_config = config.to_mnn();
        let inner = unsafe { mnn_image_process_create(&mnn_config) };

        if inner.is_null() {
            return Err(native_error(
                "Failed to create ImageProcess",
                MnnError::internal("Failed to create ImageProcess"),
            ));
        }

        Ok(Self {
            inner,
            source_format: config.source_format,
        })
    }

    /// Set the transformation matrix.
    pub fn set_matrix(&mut self, matrix: &Matrix) {
        unsafe {
            mnn_image_process_set_matrix(self.inner, matrix.as_ptr());
        }
    }

    /// Get the current transformation matrix.
    pub fn matrix(&self) -> Matrix {
        // We need to get the matrix from the internal MNN structure
        // For now, return identity (this would need MNN API support)
        Matrix::identity()
    }

    /// Convert image data to a tensor.
    ///
    /// # Arguments
    /// * `source` - Raw image data bytes
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `stride` - Number of bytes per row (0 = width * channels)
    /// * `tensor` - Destination tensor
    ///
    /// # Example
    /// ```no_run
    /// # use mnn_oxide::{ImageProcess, ImageConfig};
    /// # let processor = ImageProcess::new(&ImageConfig::default())?;
    /// # let mut tensor: mnn_oxide::Tensor = unsafe { std::mem::zeroed() };
    /// // Assuming a 224x224 RGB image
    /// let image_data: &[u8] = &[0u8; 224 * 224 * 3];
    /// processor.convert(image_data, 224, 224, 0, &mut tensor)?;
    /// # Ok::<(), mnn_oxide::MnnError>(())
    /// ```
    pub fn convert(
        &self,
        source: &[u8],
        width: i32,
        height: i32,
        stride: i32,
        tensor: &mut Tensor,
    ) -> MnnResult<()> {
        let required = checked_image_buffer_len(self.source_format, width, height, stride)?;
        if source.len() < required {
            return Err(MnnError::InvalidInput(format!(
                "Image source has {} bytes but the dimensions, format, and stride require at least {}",
                source.len(),
                required
            )));
        }

        let result = unsafe {
            mnn_image_process_convert(
                self.inner,
                source.as_ptr(),
                width,
                height,
                stride,
                tensor.inner_mut(),
            )
        };

        if result != MNN_ERROR_NONE as i32 {
            return Err(MnnError::internal(format!(
                "ImageProcess convert failed with error code: {}",
                result
            )));
        }

        Ok(())
    }

    /// Set padding value for ZERO wrap mode.
    pub fn set_padding(&mut self, value: u8) {
        // This would need a C wrapper function to be implemented
        // For now, it's a placeholder
        let _ = value;
    }

    /// Create an image tensor with the specified dimensions.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `bpp` - Bytes per pixel
    /// * `data` - Optional initial pixel data
    ///
    /// # Returns
    /// A new tensor suitable for image operations.
    pub fn create_image_tensor(
        width: i32,
        height: i32,
        bpp: i32,
        data: Option<&[u8]>,
    ) -> MnnResult<Tensor> {
        let required = checked_image_tensor_len(width, height, bpp)?;
        if let Some(data) = data {
            if data.len() < required {
                return Err(MnnError::InvalidInput(format!(
                    "Image tensor data has {} bytes but {} are required",
                    data.len(),
                    required
                )));
            }
        }

        let data_ptr = data.map_or(std::ptr::null_mut(), |bytes| bytes.as_ptr() as *mut c_void);

        let inner = unsafe { mnn_image_tensor_create(width, height, bpp, data_ptr) };

        if inner.is_null() {
            return Err(MnnError::internal("Failed to create image tensor"));
        }

        Ok(unsafe { Tensor::from_owned_ptr(inner, None, mnn_image_tensor_destroy, None) })
    }

    /// Get the raw pointer to the underlying MNN ImageProcess.
    ///
    /// # Safety
    /// The returned pointer is owned by this ImageProcess and must not be freed.
    pub unsafe fn as_ptr(&self) -> *const MNNImageProcess {
        self.inner
    }
}

/// Image read flags (same as OpenCV imread flags)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImreadFlags {
    /// Read image as-is (preserve original format)
    Unchanged = -1,
    /// Convert to grayscale
    Grayscale = 0,
    /// Convert to color (BGR)
    Color = 1,
}

/// Resize filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeFilter {
    /// Nearest neighbor interpolation
    Nearest,
    /// Bilinear interpolation
    #[default]
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

/// Read an image from file using MNN CV.
///
/// Requires MNN built with `-DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON`.
///
/// # Arguments
/// * `path` - Path to the image file (JPG, PNG, etc.)
/// * `flags` - Read flags (grayscale, color, unchanged)
///
/// # Returns
/// A tensor containing the image data (uint8 type) on success, or an error on failure.
///
/// # Example
/// ```no_run
/// use mnn_oxide::imread;
///
/// // Read as color image (BGR)
/// let tensor = imread("image.jpg", mnn_oxide::ImreadFlags::Color)?;
/// # Ok::<(), mnn_oxide::MnnError>(())
/// ```
pub fn imread<P: AsRef<std::path::Path>>(path: P, flags: ImreadFlags) -> MnnResult<Tensor> {
    let path_cstr = std::ffi::CString::new(path.as_ref().to_str().ok_or(MnnError::InvalidPath)?)
        .map_err(|_| MnnError::InvalidPath)?;

    let inner = unsafe { mnn_imread(path_cstr.as_ptr(), flags as i32) };

    if inner.is_null() {
        return Err(native_error(
            "Failed to read image",
            MnnError::internal("Failed to read image (MNN imread returned null)"),
        ));
    }

    Ok(unsafe { Tensor::from_owned_ptr(inner, None, mnn_tensor_destroy, None) })
}

/// Write an image to file using MNN CV.
///
/// Requires MNN built with `-DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON`.
///
/// # Arguments
/// * `path` - Output file path
/// * `tensor` - Image tensor (uint8 type)
///
/// # Returns
/// Ok(()) on success, or an error on failure.
///
/// # Example
/// ```no_run
/// use mnn_oxide::{imread, imwrite, ImreadFlags};
///
/// let img = imread("input.jpg", ImreadFlags::Color)?;
/// imwrite("output.png", &img)?;
/// # Ok::<(), mnn_oxide::MnnError>(())
/// ```
pub fn imwrite<P: AsRef<std::path::Path>>(path: P, tensor: &Tensor) -> MnnResult<()> {
    let path_cstr = std::ffi::CString::new(path.as_ref().to_str().ok_or(MnnError::InvalidPath)?)
        .map_err(|_| MnnError::InvalidPath)?;

    let result = unsafe { mnn_imwrite(path_cstr.as_ptr(), tensor.as_ptr(), std::ptr::null()) };

    if result != 0 {
        return Err(native_error(
            "MNN image write failed",
            MnnError::internal(format!("MNN imwrite failed with error code: {result}")),
        ));
    }

    Ok(())
}

/// Resize an image tensor.
///
/// Requires MNN built with `-DMNN_BUILD_OPENCV=ON`.
///
/// # Arguments
/// * `src` - Source image tensor
/// * `dst_width` - Destination width
/// * `dst_height` - Destination height
/// * `filter` - Interpolation filter
///
/// # Returns
/// A new resized tensor on success, or an error on failure.
///
/// # Example
/// ```no_run
/// use mnn_oxide::{imread, resize, ResizeFilter, ImreadFlags};
///
/// let img = imread("image.jpg", ImreadFlags::Color)?;
/// let resized = resize(&img, 224, 224, ResizeFilter::Bilinear)?;
/// # Ok::<(), mnn_oxide::MnnError>(())
/// ```
pub fn resize(
    src: &Tensor,
    dst_width: i32,
    dst_height: i32,
    filter: ResizeFilter,
) -> MnnResult<Tensor> {
    let filter_code = match filter {
        ResizeFilter::Nearest => 0,
        ResizeFilter::Bilinear => 1,
        ResizeFilter::Bicubic => 2,
    };

    let inner = unsafe { mnn_resize(src.as_ptr(), dst_width, dst_height, filter_code) };

    if inner.is_null() {
        return Err(native_error(
            "Failed to resize image",
            MnnError::internal("Failed to resize image (MNN resize returned null)"),
        ));
    }

    // Note: mnn_resize returns a tensor from VARP's getTensor()
    // We need to clone it to get an owned tensor
    Ok(unsafe { Tensor::from_owned_ptr(inner, None, mnn_tensor_destroy, None) })
}

impl Drop for ImageProcess {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                mnn_image_process_destroy(self.inner);
            }
        }
    }
}

impl std::fmt::Debug for ImageProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageProcess").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_clone_null_pointer_returns_actionable_error() {
        let result = unsafe { Matrix::from_owned_ptr(std::ptr::null_mut()) };
        match result {
            Err(MnnError::Internal(message)) => {
                assert!(message.contains("Failed to clone matrix"));
                assert!(message.contains("returned null"));
            }
            _ => panic!("expected an internal error for a null matrix pointer"),
        }
    }

    #[test]
    fn test_image_format_conversion() {
        assert_eq!(ImageFormat::Rgba.to_mnn(), MNN_IMAGE_FORMAT_RGBA);
        assert_eq!(ImageFormat::Bgr.to_mnn(), MNN_IMAGE_FORMAT_BGR);
        assert_eq!(ImageFormat::YuvNv21.to_mnn(), MNN_IMAGE_FORMAT_YUV_NV21);
    }

    #[test]
    fn test_filter_conversion() {
        assert_eq!(Filter::Nearest.to_mnn(), MNN_FILTER_NEAREST);
        assert_eq!(Filter::Bilinear.to_mnn(), MNN_FILTER_BILINEAR);
        assert_eq!(Filter::Bicubic.to_mnn(), MNN_FILTER_BICUBIC);
    }

    #[test]
    fn test_wrap_conversion() {
        assert_eq!(Wrap::ClampToEdge.to_mnn(), MNN_WRAP_CLAMP_TO_EDGE);
        assert_eq!(Wrap::Zero.to_mnn(), MNN_WRAP_ZERO);
        assert_eq!(Wrap::Repeat.to_mnn(), MNN_WRAP_REPEAT);
    }

    #[test]
    fn test_image_channels() {
        assert_eq!(ImageFormat::Rgba.channels(), 4);
        assert_eq!(ImageFormat::Rgb.channels(), 3);
        assert_eq!(ImageFormat::Gray.channels(), 1);
    }

    #[test]
    fn image_buffer_length_checks_packed_and_yuv_layouts() {
        assert_eq!(
            checked_image_buffer_len(ImageFormat::Rgb, 2, 3, 0).unwrap(),
            18
        );
        assert_eq!(
            checked_image_buffer_len(ImageFormat::Rgb, 2, 3, 8).unwrap(),
            24
        );
        assert!(checked_image_buffer_len(ImageFormat::Rgb, 2, 3, 5).is_err());
        assert_eq!(
            checked_image_buffer_len(ImageFormat::YuvNv21, 4, 3, 0).unwrap(),
            20
        );
        assert_eq!(
            checked_image_buffer_len(ImageFormat::YuvNv12, 3, 3, 0).unwrap(),
            17
        );
        assert!(checked_image_buffer_len(ImageFormat::YuvNv12, 3, 3, 3).is_err());
        assert_eq!(
            checked_image_buffer_len(ImageFormat::YuvNv12, 3, 3, 4).unwrap(),
            20
        );
        assert_eq!(
            checked_image_buffer_len(ImageFormat::YuvI420, 4, 3, 0).unwrap(),
            20
        );
        assert!(checked_image_buffer_len(ImageFormat::YuvI420, 4, 3, 8).is_err());
        assert!(checked_image_buffer_len(ImageFormat::Gray, 0, 3, 0).is_err());
        assert!(checked_image_buffer_len(ImageFormat::Gray, i32::MAX, i32::MAX, 0).is_err());
        assert_eq!(checked_image_tensor_len(3, 2, 4).unwrap(), 24);
        assert!(checked_image_tensor_len(0, 2, 4).is_err());
        assert!(checked_image_tensor_len(i32::MAX, i32::MAX, 4).is_err());
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity();
        assert!((m.get(0, 0) - 1.0).abs() < f32::EPSILON);
        assert!((m.get(1, 1) - 1.0).abs() < f32::EPSILON);
        assert!((m.get(2, 2) - 1.0).abs() < f32::EPSILON);
        assert!((m.get(0, 3) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matrix_scale() {
        let m = Matrix::scale(2.0, 3.0);
        assert!((m.get(0, 0) - 2.0).abs() < f32::EPSILON);
        assert!((m.get(1, 1) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matrix_translate() {
        let m = Matrix::translate(10.0, 20.0);
        assert!((m.get(0, 2) - 10.0).abs() < f32::EPSILON);
        assert!((m.get(1, 2) - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_image_config_default() {
        let config = ImageConfig::default();
        assert_eq!(config.source_format, ImageFormat::Rgba);
        assert_eq!(config.filter, Filter::Bilinear);
        assert_eq!(config.wrap, Wrap::ClampToEdge);
    }

    #[test]
    fn test_image_config_builder() {
        let config = ImageConfig::new()
            .with_source_format(ImageFormat::Bgr)
            .with_filter(Filter::Bicubic)
            .with_mean([1.0, 2.0, 3.0, 4.0])
            .with_normal([0.5, 0.5, 0.5, 1.0]);

        assert_eq!(config.source_format, ImageFormat::Bgr);
        assert_eq!(config.filter, Filter::Bicubic);
        assert_eq!(config.mean, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(config.normal, [0.5, 0.5, 0.5, 1.0]);
    }
}
