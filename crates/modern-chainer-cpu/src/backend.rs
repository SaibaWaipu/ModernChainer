use bytemuck::Pod;
use modern_chainer_core::{Backend, Device, Error, Result, Storage, Tensor};

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuOps;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend {
    ops: CpuOps,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for CpuBackend {
    type Storage = CpuStorage;
    type Ops = CpuOps;

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn ops(&self) -> &Self::Ops {
        &self.ops
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Storage> {
        Ok(CpuStorage::Bytes(vec![0; bytes]))
    }

    fn alloc_zeroed(&self, bytes: usize) -> Result<Self::Storage> {
        Ok(CpuStorage::Bytes(vec![0; bytes]))
    }

    fn copy_from_host(&self, dst: &mut Self::Storage, src: &[u8]) -> Result<()> {
        let bytes = dst.as_bytes_mut();
        if src.len() > bytes.len() {
            return Err(Error::invalid("copy_from_host buffer overflow"));
        }
        bytes[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host(&self, src: &Self::Storage, dst: &mut [u8]) -> Result<()> {
        let bytes = src.as_bytes();
        let len = bytes.len().min(dst.len());
        dst[..len].copy_from_slice(&bytes[..len]);
        Ok(())
    }
}

/// Host storage.  `F32` keeps a vector allocated by kernels without an extra
/// byte copy; `Bytes` is used for uploads and other dtypes.
pub enum CpuStorage {
    Bytes(Vec<u8>),
    F32(Vec<f32>),
}

impl CpuStorage {
    pub fn from_f32_vec(values: Vec<f32>) -> Self {
        CpuStorage::F32(values)
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            CpuStorage::Bytes(bytes) => bytes,
            CpuStorage::F32(values) => bytemuck::cast_slice(values),
        }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            CpuStorage::Bytes(bytes) => bytes,
            CpuStorage::F32(values) => bytemuck::cast_slice_mut(values),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        self.as_bytes()
    }

    pub fn as_slice<T: Pod>(&self) -> Result<&[T]> {
        bytemuck::try_cast_slice(self.as_bytes())
            .map_err(|e| Error::Backend(format!("cpu storage cast failed: {e}")))
    }

    pub fn as_mut_slice<T: Pod>(&mut self) -> Result<&mut [T]> {
        bytemuck::try_cast_slice_mut(self.as_bytes_mut())
            .map_err(|e| Error::Backend(format!("cpu storage cast failed: {e}")))
    }
}

impl Storage for CpuStorage {
    fn len_bytes(&self) -> usize {
        match self {
            CpuStorage::Bytes(bytes) => bytes.len(),
            CpuStorage::F32(values) => std::mem::size_of_val(values.as_slice()),
        }
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_bytes().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_bytes_mut().as_mut_ptr()
    }
}

/// Typed host access for CPU tensors.
pub trait CpuTensorExt {
    fn as_slice<T: Pod>(&self) -> Result<&[T]>;
    fn to_vec_f32(&self) -> Result<Vec<f32>>;
    fn scalar_f64(&self) -> Result<f64>;
}

impl CpuTensorExt for Tensor<CpuBackend> {
    /// Zero-copy view of a contiguous CPU tensor.
    fn as_slice<T: Pod>(&self) -> Result<&[T]> {
        if !self.is_contiguous() {
            return Err(Error::invalid("as_slice requires a contiguous tensor"));
        }
        let bytes = self.storage().as_slice::<u8>()?;
        let elem = std::mem::size_of::<T>();
        let start = self.offset() * elem;
        let len = self.numel() * elem;
        if start + len > bytes.len() {
            return Err(Error::invalid("tensor view exceeds storage"));
        }
        bytemuck::try_cast_slice(&bytes[start..start + len])
            .map_err(|e| Error::Backend(format!("cpu tensor cast failed: {e}")))
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.as_slice::<f32>()?.to_vec())
    }

    fn scalar_f64(&self) -> Result<f64> {
        if !self.is_scalar() {
            return Err(Error::invalid(format!(
                "expected scalar tensor, got {}",
                self.shape()
            )));
        }
        match self.dtype() {
            modern_chainer_core::DType::F32 => Ok(self.as_slice::<f32>()?[0] as f64),
            modern_chainer_core::DType::F64 => Ok(self.as_slice::<f64>()?[0]),
            modern_chainer_core::DType::I64 => Ok(self.as_slice::<i64>()?[0] as f64),
            modern_chainer_core::DType::I32 => Ok(self.as_slice::<i32>()?[0] as f64),
            other => Err(Error::UnsupportedDType(other)),
        }
    }
}

pub(crate) fn f32_slice(tensor: &Tensor<CpuBackend>) -> Result<&[f32]> {
    if tensor.dtype() != modern_chainer_core::DType::F32 {
        return Err(Error::UnsupportedDType(tensor.dtype()));
    }
    tensor.as_slice::<f32>()
}

pub(crate) fn make_f32(
    reference: &Tensor<CpuBackend>,
    data: Vec<f32>,
    shape: impl Into<modern_chainer_core::Shape>,
) -> Result<Tensor<CpuBackend>> {
    let shape = shape.into();
    let storage = CpuStorage::from_f32_vec(data);
    Tensor::from_storage(
        *reference.backend(),
        storage,
        modern_chainer_core::DType::F32,
        shape.clone(),
        shape.contiguous_strides(),
        0,
    )
}
