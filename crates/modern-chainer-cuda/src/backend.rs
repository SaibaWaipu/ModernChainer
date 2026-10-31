use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext as DriverContext, CudaSlice, CudaStream};
use modern_chainer_core::{Backend, DType, Device, Error, Result, Storage};

use crate::kernels::Kernels;
use crate::ops::CudaOps;

pub(crate) struct CudaRuntime {
    #[allow(dead_code)]
    pub(crate) ctx: Arc<DriverContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    pub(crate) blaslt: Arc<CudaBlasLT>,
    pub(crate) kernels: Kernels,
    pub(crate) ordinal: u32,
    pub(crate) name: String,
}

impl CudaRuntime {
    pub(crate) fn new(ordinal: u32) -> Result<Arc<Self>> {
        let ctx = DriverContext::new(ordinal as usize)
            .map_err(|error| Error::Backend(format!("cuda context: {error}")))?;
        // A single non-default stream is used for all backend work.  Event
        // tracking across streams is unnecessary and would inject event records
        // into CUDA graph captures.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx
            .new_stream()
            .map_err(|error| Error::Backend(format!("cuda stream: {error}")))?;
        let blas = Arc::new(
            CudaBlas::new(stream.clone())
                .map_err(|error| Error::Backend(format!("cublas: {error}")))?,
        );
        let blaslt = Arc::new(
            CudaBlasLT::new(stream.clone())
                .map_err(|error| Error::Backend(format!("cublaslt: {error}")))?,
        );
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|error| Error::Backend(format!("compute capability: {error}")))?;
        let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
        let kernels = Kernels::compile(&ctx, arch)?;
        let name = ctx.name().unwrap_or_else(|_| format!("cuda:{ordinal}"));
        Ok(Arc::new(Self {
            ctx,
            stream,
            blas,
            blaslt,
            kernels,
            ordinal,
            name,
        }))
    }

    pub(crate) fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
    }
}

pub enum CudaStorage {
    F32(CudaSlice<f32>),
    I64(CudaSlice<i64>),
    U8(CudaSlice<u8>),
}

impl CudaStorage {
    pub fn f32(&self) -> Result<&CudaSlice<f32>> {
        match self {
            CudaStorage::F32(slice) => Ok(slice),
            _ => Err(Error::Backend("cuda storage is not f32".to_string())),
        }
    }

    pub fn f32_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        match self {
            CudaStorage::F32(slice) => Ok(slice),
            _ => Err(Error::Backend("cuda storage is not f32".to_string())),
        }
    }

    pub fn i64(&self) -> Result<&CudaSlice<i64>> {
        match self {
            CudaStorage::I64(slice) => Ok(slice),
            _ => Err(Error::Backend("cuda storage is not i64".to_string())),
        }
    }
}

impl Storage for CudaStorage {
    fn len_bytes(&self) -> usize {
        match self {
            CudaStorage::F32(slice) => slice.len() * 4,
            CudaStorage::I64(slice) => slice.len() * 8,
            CudaStorage::U8(slice) => slice.len(),
        }
    }

    fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) runtime: Arc<CudaRuntime>,
    pub(crate) ops: CudaOps,
}

impl CudaBackend {
    pub fn new(ordinal: u32) -> Result<Self> {
        let runtime = CudaRuntime::new(ordinal)?;
        Ok(Self {
            ops: CudaOps {
                runtime: Arc::clone(&runtime),
            },
            runtime,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.runtime.ordinal
    }

    pub fn name(&self) -> &str {
        &self.runtime.name
    }

    pub fn synchronize(&self) -> Result<()> {
        self.runtime
            .stream
            .synchronize()
            .map_err(|error| Error::Backend(format!("cuda synchronize: {error}")))
    }

    pub fn device_count() -> Result<usize> {
        let count = DriverContext::device_count()
            .map_err(|error| Error::Backend(format!("cuda device count: {error}")))?;
        Ok(count.max(0) as usize)
    }
}

impl Backend for CudaBackend {
    type Storage = CudaStorage;
    type Ops = CudaOps;

    fn device(&self) -> Device {
        Device::Cuda(self.runtime.ordinal)
    }

    fn ops(&self) -> &Self::Ops {
        &self.ops
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Storage> {
        let slice = self
            .runtime
            .stream
            .alloc_zeros::<u8>(bytes)
            .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))?;
        Ok(CudaStorage::U8(slice))
    }

    fn alloc_zeroed(&self, bytes: usize) -> Result<Self::Storage> {
        self.alloc(bytes)
    }

    fn alloc_for(&self, bytes: usize, dtype: DType) -> Result<Self::Storage> {
        match dtype {
            DType::F32 => {
                if !bytes.is_multiple_of(4) {
                    return Err(Error::invalid("f32 allocation size is not a multiple of 4"));
                }
                Ok(CudaStorage::F32(self.runtime.alloc_f32(bytes / 4)?))
            }
            DType::I64 => {
                if !bytes.is_multiple_of(8) {
                    return Err(Error::invalid("i64 allocation size is not a multiple of 8"));
                }
                let slice = self
                    .runtime
                    .stream
                    .alloc_zeros::<i64>(bytes / 8)
                    .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))?;
                Ok(CudaStorage::I64(slice))
            }
            DType::U8 | DType::Bool => self.alloc(bytes),
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn alloc_zeroed_for(&self, bytes: usize, dtype: DType) -> Result<Self::Storage> {
        self.alloc_for(bytes, dtype)
    }

    fn copy_from_host(&self, dst: &mut Self::Storage, src: &[u8]) -> Result<()> {
        match dst {
            CudaStorage::F32(slice) => {
                let (chunks, rest) = src.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err(Error::invalid("f32 upload size is not a multiple of 4"));
                }
                let host: Vec<f32> = chunks
                    .iter()
                    .map(|chunk| f32::from_ne_bytes(*chunk))
                    .collect();
                self.runtime
                    .stream
                    .memcpy_htod(&host, slice)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::I64(slice) => {
                let (chunks, rest) = src.as_chunks::<8>();
                if !rest.is_empty() {
                    return Err(Error::invalid("i64 upload size is not a multiple of 8"));
                }
                let host: Vec<i64> = chunks
                    .iter()
                    .map(|chunk| i64::from_ne_bytes(*chunk))
                    .collect();
                self.runtime
                    .stream
                    .memcpy_htod(&host, slice)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::U8(slice) => {
                self.runtime
                    .stream
                    .memcpy_htod(src, slice)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
        }
        Ok(())
    }

    fn copy_to_host(&self, src: &Self::Storage, dst: &mut [u8]) -> Result<()> {
        match src {
            CudaStorage::F32(slice) => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(slice)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let bytes = bytemuck::cast_slice::<f32, u8>(&host);
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
            CudaStorage::I64(slice) => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(slice)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let bytes = bytemuck::cast_slice::<i64, u8>(&host);
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
            CudaStorage::U8(slice) => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(slice)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let len = host.len().min(dst.len());
                dst[..len].copy_from_slice(&host[..len]);
            }
        }
        Ok(())
    }
}
