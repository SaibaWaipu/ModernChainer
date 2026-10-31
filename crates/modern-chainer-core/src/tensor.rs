use std::sync::Arc;

use bytemuck::Pod;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::shape::{Shape, Strides};
use crate::storage::{Backend, Storage};

/// A typed, strided, immutable tensor handle.
///
/// `Tensor` itself is cheap to clone: the storage is reference counted and the
/// backend is a small handle.  Mutation happens through kernels that allocate a
/// fresh `Tensor`.
pub struct Tensor<B: Backend> {
    storage: Arc<B::Storage>,
    dtype: DType,
    shape: Shape,
    strides: Strides,
    offset: usize,
    backend: B,
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            backend: self.backend.clone(),
        }
    }
}

impl<B: Backend> Tensor<B> {
    pub fn from_storage(
        backend: B,
        storage: B::Storage,
        dtype: DType,
        shape: impl Into<Shape>,
        strides: impl Into<Strides>,
        offset: usize,
    ) -> Result<Self> {
        let shape = shape.into();
        let strides = strides.into();
        if strides.rank() != shape.rank() {
            return Err(Error::invalid(format!(
                "strides rank {} does not match shape {shape}",
                strides.rank()
            )));
        }
        let needed = offset
            .checked_add(shape.numel())
            .and_then(|n| n.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| Error::invalid("tensor size overflow"))?;
        if needed > storage.len_bytes() {
            return Err(Error::invalid(format!(
                "tensor needs {needed} bytes but storage has {}",
                storage.len_bytes()
            )));
        }
        Ok(Self {
            storage: Arc::new(storage),
            dtype,
            strides,
            shape,
            offset,
            backend,
        })
    }

    pub fn from_bytes(
        backend: B,
        bytes: &[u8],
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.numel() * dtype.size_in_bytes();
        if bytes.len() != expected {
            return Err(Error::invalid(format!(
                "expected {expected} bytes for {shape} {dtype}, got {}",
                bytes.len()
            )));
        }
        let mut storage = backend.alloc_for(bytes.len(), dtype)?;
        backend.copy_from_host(&mut storage, bytes)?;
        let strides = shape.contiguous_strides();
        Self::from_storage(backend, storage, dtype, shape, strides, 0)
    }

    pub fn from_slice<T: Pod>(
        backend: B,
        data: &[T],
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        if std::mem::size_of::<T>() != dtype.size_in_bytes() {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                actual: match std::mem::size_of::<T>() {
                    2 => DType::F16,
                    4 => DType::F32,
                    8 => DType::F64,
                    n => {
                        return Err(Error::invalid(format!("no dtype with element size {n}")));
                    }
                },
            });
        }
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::invalid(format!(
                "expected {} elements for {shape}, got {}",
                shape.numel(),
                data.len()
            )));
        }
        Self::from_bytes(backend, bytemuck::cast_slice(data), shape, dtype)
    }

    pub fn from_f32(backend: B, data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        Self::from_slice(backend, data, shape, DType::F32)
    }

    pub fn zeros(backend: B, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let bytes = shape.numel() * dtype.size_in_bytes();
        let storage = backend.alloc_zeroed_for(bytes, dtype)?;
        let strides = shape.contiguous_strides();
        Self::from_storage(backend, storage, dtype, shape, strides, 0)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn device(&self) -> Device {
        self.backend.device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn storage(&self) -> &Arc<B::Storage> {
        &self.storage
    }

    /// Mutable access to storage when this tensor is the sole owner.
    pub fn storage_mut(&mut self) -> Option<&mut B::Storage> {
        Arc::get_mut(&mut self.storage)
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous_for(&self.shape)
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Copy the (contiguous) contents to host memory.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if !self.is_contiguous() {
            return Err(Error::invalid("to_bytes requires a contiguous tensor"));
        }
        let mut whole = vec![0u8; self.storage.len_bytes()];
        self.backend.copy_to_host(&self.storage, &mut whole)?;
        let start = self.offset * self.dtype.size_in_bytes();
        let len = self.numel() * self.dtype.size_in_bytes();
        Ok(whole[start..start + len].to_vec())
    }

    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if !self.is_contiguous() {
            return Err(Error::invalid("reshape requires a contiguous tensor"));
        }
        if shape.numel() != self.numel() {
            return Err(Error::shape_mismatch(
                "reshape element count",
                self.shape.clone(),
                shape,
            ));
        }
        let strides = shape.contiguous_strides();
        Ok(Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            shape,
            strides,
            offset: self.offset,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> std::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device())
            .field("offset", &self.offset)
            .finish()
    }
}
