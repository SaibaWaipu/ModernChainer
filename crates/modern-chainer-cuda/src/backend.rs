use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::cublas::CudaBlas;
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext as DriverContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use modern_chainer_core::{Backend, DType, Device, Error, Result, Storage};

use crate::kernels::Kernels;
use crate::ops::CudaOps;

const POOL_LIMIT: usize = 128;

#[doc(hidden)]
pub struct CudaPool {
    f32: Mutex<Vec<CudaSlice<f32>>>,
    i64: Mutex<Vec<CudaSlice<i64>>>,
    u8: Mutex<Vec<CudaSlice<u8>>>,
    f16: Mutex<Vec<CudaSlice<f16>>>,
    bf16: Mutex<Vec<CudaSlice<bf16>>>,
}

impl CudaPool {
    fn new() -> Self {
        Self {
            f32: Mutex::new(Vec::new()),
            i64: Mutex::new(Vec::new()),
            u8: Mutex::new(Vec::new()),
            f16: Mutex::new(Vec::new()),
            bf16: Mutex::new(Vec::new()),
        }
    }

    fn take_f32(&self, min: usize) -> Option<CudaSlice<f32>> {
        let mut pool = self.f32.lock().expect("cuda pool poisoned");
        let index = pool.iter().position(|slice| slice.len() == min)?;
        Some(pool.swap_remove(index))
    }

    fn put_f32(&self, slice: CudaSlice<f32>) {
        let mut pool = self.f32.lock().expect("cuda pool poisoned");
        if pool.len() < POOL_LIMIT {
            pool.push(slice);
        }
    }

    fn take_f16(&self, min: usize) -> Option<CudaSlice<f16>> {
        let mut pool = self.f16.lock().expect("cuda pool poisoned");
        let index = pool.iter().position(|slice| slice.len() == min)?;
        Some(pool.swap_remove(index))
    }

    fn put_f16(&self, slice: CudaSlice<f16>) {
        let mut pool = self.f16.lock().expect("cuda pool poisoned");
        if pool.len() < POOL_LIMIT {
            pool.push(slice);
        }
    }

    fn take_bf16(&self, min: usize) -> Option<CudaSlice<bf16>> {
        let mut pool = self.bf16.lock().expect("cuda pool poisoned");
        let index = pool.iter().position(|slice| slice.len() == min)?;
        Some(pool.swap_remove(index))
    }

    fn put_bf16(&self, slice: CudaSlice<bf16>) {
        let mut pool = self.bf16.lock().expect("cuda pool poisoned");
        if pool.len() < POOL_LIMIT {
            pool.push(slice);
        }
    }

    fn take_i64(&self, min: usize) -> Option<CudaSlice<i64>> {
        let mut pool = self.i64.lock().expect("cuda pool poisoned");
        let index = pool.iter().position(|slice| slice.len() == min)?;
        Some(pool.swap_remove(index))
    }

    fn put_i64(&self, slice: CudaSlice<i64>) {
        let mut pool = self.i64.lock().expect("cuda pool poisoned");
        if pool.len() < POOL_LIMIT {
            pool.push(slice);
        }
    }

    fn take_u8(&self, min: usize) -> Option<CudaSlice<u8>> {
        let mut pool = self.u8.lock().expect("cuda pool poisoned");
        let index = pool.iter().position(|slice| slice.len() == min)?;
        Some(pool.swap_remove(index))
    }

    fn put_u8(&self, slice: CudaSlice<u8>) {
        let mut pool = self.u8.lock().expect("cuda pool poisoned");
        if pool.len() < POOL_LIMIT {
            pool.push(slice);
        }
    }

    fn pooled(&self) -> usize {
        self.f32.lock().expect("cuda pool poisoned").len()
            + self.i64.lock().expect("cuda pool poisoned").len()
            + self.u8.lock().expect("cuda pool poisoned").len()
            + self.f16.lock().expect("cuda pool poisoned").len()
            + self.bf16.lock().expect("cuda pool poisoned").len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeType {
    F32,
    Tf32,
    F16,
    Bf16,
}

impl ComputeType {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => ComputeType::F32,
            1 => ComputeType::Tf32,
            2 => ComputeType::F16,
            3 => ComputeType::Bf16,
            _ => ComputeType::Tf32,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            ComputeType::F32 => 0,
            ComputeType::Tf32 => 1,
            ComputeType::F16 => 2,
            ComputeType::Bf16 => 3,
        }
    }
}

pub(crate) struct CudaRuntime {
    #[allow(dead_code)]
    pub(crate) ctx: Arc<DriverContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    pub(crate) blaslt: Arc<CudaBlasLT>,
    pub(crate) kernels: Kernels,
    pub(crate) pool: Arc<CudaPool>,
    pub(crate) compute: AtomicU8,
    pub(crate) ordinal: u32,
    pub(crate) name: String,
}

impl CudaRuntime {
    pub(crate) fn compute_type(&self) -> ComputeType {
        ComputeType::from_u8(self.compute.load(Ordering::Relaxed))
    }

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
            pool: Arc::new(CudaPool::new()),
            compute: AtomicU8::new(ComputeType::Tf32.as_u8()),
            ordinal,
            name,
        }))
    }

    fn alloc_f32(&self, len: usize, zeroed: bool) -> Result<CudaSlice<f32>> {
        if let Some(mut slice) = self.pool.take_f32(len) {
            if zeroed {
                self.stream
                    .memset_zeros(&mut slice)
                    .map_err(|error| Error::Backend(format!("cuda memset: {error}")))?;
            }
            Ok(slice)
        } else {
            self.stream
                .alloc_zeros::<f32>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        }
    }

    fn alloc_i64(&self, len: usize, zeroed: bool) -> Result<CudaSlice<i64>> {
        if let Some(mut slice) = self.pool.take_i64(len) {
            if zeroed {
                self.stream
                    .memset_zeros(&mut slice)
                    .map_err(|error| Error::Backend(format!("cuda memset: {error}")))?;
            }
            Ok(slice)
        } else {
            self.stream
                .alloc_zeros::<i64>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        }
    }

    fn alloc_f16(&self, len: usize, zeroed: bool) -> Result<CudaSlice<f16>> {
        if let Some(mut slice) = self.pool.take_f16(len) {
            if zeroed {
                self.stream
                    .memset_zeros(&mut slice)
                    .map_err(|error| Error::Backend(format!("cuda memset: {error}")))?;
            }
            Ok(slice)
        } else {
            self.stream
                .alloc_zeros::<f16>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        }
    }

    fn alloc_bf16(&self, len: usize, zeroed: bool) -> Result<CudaSlice<bf16>> {
        if let Some(mut slice) = self.pool.take_bf16(len) {
            if zeroed {
                self.stream
                    .memset_zeros(&mut slice)
                    .map_err(|error| Error::Backend(format!("cuda memset: {error}")))?;
            }
            Ok(slice)
        } else {
            self.stream
                .alloc_zeros::<bf16>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        }
    }

    fn alloc_u8(&self, len: usize, zeroed: bool) -> Result<CudaSlice<u8>> {
        if let Some(mut slice) = self.pool.take_u8(len) {
            if zeroed {
                self.stream
                    .memset_zeros(&mut slice)
                    .map_err(|error| Error::Backend(format!("cuda memset: {error}")))?;
            }
            Ok(slice)
        } else {
            self.stream
                .alloc_zeros::<u8>(len)
                .map_err(|error| Error::Backend(format!("cuda alloc: {error}")))
        }
    }
}

pub enum CudaStorage {
    F32 {
        slice: Option<CudaSlice<f32>>,
        pool: Arc<CudaPool>,
    },
    I64 {
        slice: Option<CudaSlice<i64>>,
        pool: Arc<CudaPool>,
    },
    U8 {
        slice: Option<CudaSlice<u8>>,
        pool: Arc<CudaPool>,
    },
    F16 {
        slice: Option<CudaSlice<f16>>,
        pool: Arc<CudaPool>,
    },
    BF16 {
        slice: Option<CudaSlice<bf16>>,
        pool: Arc<CudaPool>,
    },
}

impl CudaStorage {
    fn from_f32(slice: CudaSlice<f32>, pool: Arc<CudaPool>) -> Self {
        Self::F32 {
            slice: Some(slice),
            pool,
        }
    }

    fn from_i64(slice: CudaSlice<i64>, pool: Arc<CudaPool>) -> Self {
        Self::I64 {
            slice: Some(slice),
            pool,
        }
    }

    fn from_u8(slice: CudaSlice<u8>, pool: Arc<CudaPool>) -> Self {
        Self::U8 {
            slice: Some(slice),
            pool,
        }
    }

    fn from_f16(slice: CudaSlice<f16>, pool: Arc<CudaPool>) -> Self {
        Self::F16 {
            slice: Some(slice),
            pool,
        }
    }

    fn from_bf16(slice: CudaSlice<bf16>, pool: Arc<CudaPool>) -> Self {
        Self::BF16 {
            slice: Some(slice),
            pool,
        }
    }

    pub fn f32(&self) -> Result<&CudaSlice<f32>> {
        match self {
            CudaStorage::F32 { slice, .. } => slice
                .as_ref()
                .ok_or_else(|| Error::Backend("cuda f32 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not f32".to_string())),
        }
    }

    pub fn f32_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        match self {
            CudaStorage::F32 { slice, .. } => slice
                .as_mut()
                .ok_or_else(|| Error::Backend("cuda f32 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not f32".to_string())),
        }
    }

    pub fn u8(&self) -> Result<&CudaSlice<u8>> {
        match self {
            CudaStorage::U8 { slice, .. } => slice
                .as_ref()
                .ok_or_else(|| Error::Backend("cuda u8 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not u8".to_string())),
        }
    }

    pub fn f16(&self) -> Result<&CudaSlice<f16>> {
        match self {
            CudaStorage::F16 { slice, .. } => slice
                .as_ref()
                .ok_or_else(|| Error::Backend("cuda f16 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not f16".to_string())),
        }
    }

    pub fn bf16(&self) -> Result<&CudaSlice<bf16>> {
        match self {
            CudaStorage::BF16 { slice, .. } => slice
                .as_ref()
                .ok_or_else(|| Error::Backend("cuda bf16 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not bf16".to_string())),
        }
    }

    pub fn f16_mut(&mut self) -> Result<&mut CudaSlice<f16>> {
        match self {
            CudaStorage::F16 { slice, .. } => slice
                .as_mut()
                .ok_or_else(|| Error::Backend("cuda f16 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not f16".to_string())),
        }
    }

    pub fn bf16_mut(&mut self) -> Result<&mut CudaSlice<bf16>> {
        match self {
            CudaStorage::BF16 { slice, .. } => slice
                .as_mut()
                .ok_or_else(|| Error::Backend("cuda bf16 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not bf16".to_string())),
        }
    }

    pub fn i64(&self) -> Result<&CudaSlice<i64>> {
        match self {
            CudaStorage::I64 { slice, .. } => slice
                .as_ref()
                .ok_or_else(|| Error::Backend("cuda i64 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not i64".to_string())),
        }
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        match self {
            CudaStorage::F32 { slice, pool } => {
                if let Some(slice) = slice.take() {
                    pool.put_f32(slice);
                }
            }
            CudaStorage::I64 { slice, pool } => {
                if let Some(slice) = slice.take() {
                    pool.put_i64(slice);
                }
            }
            CudaStorage::U8 { slice, pool } => {
                if let Some(slice) = slice.take() {
                    pool.put_u8(slice);
                }
            }
            CudaStorage::F16 { slice, pool } => {
                if let Some(slice) = slice.take() {
                    pool.put_f16(slice);
                }
            }
            CudaStorage::BF16 { slice, pool } => {
                if let Some(slice) = slice.take() {
                    pool.put_bf16(slice);
                }
            }
        }
    }
}

impl Storage for CudaStorage {
    fn len_bytes(&self) -> usize {
        match self {
            CudaStorage::F32 { slice, .. } => slice.as_ref().map_or(0, |slice| slice.len() * 4),
            CudaStorage::I64 { slice, .. } => slice.as_ref().map_or(0, |slice| slice.len() * 8),
            CudaStorage::U8 { slice, .. } => slice.as_ref().map_or(0, CudaSlice::len),
            CudaStorage::F16 { slice, .. } => slice.as_ref().map_or(0, |slice| slice.len() * 2),
            CudaStorage::BF16 { slice, .. } => slice.as_ref().map_or(0, |slice| slice.len() * 2),
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

    pub fn pooled_blocks(&self) -> usize {
        self.runtime.pool.pooled()
    }

    pub fn set_compute_type(&self, compute: ComputeType) {
        self.runtime
            .compute
            .store(compute.as_u8(), Ordering::Relaxed);
    }

    pub fn compute_type(&self) -> ComputeType {
        ComputeType::from_u8(self.runtime.compute.load(Ordering::Relaxed))
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
        let slice = self.runtime.alloc_u8(bytes, true)?;
        Ok(CudaStorage::from_u8(slice, Arc::clone(&self.runtime.pool)))
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
                let slice = self.runtime.alloc_f32(bytes / 4, false)?;
                Ok(CudaStorage::from_f32(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::I64 => {
                if !bytes.is_multiple_of(8) {
                    return Err(Error::invalid("i64 allocation size is not a multiple of 8"));
                }
                let slice = self.runtime.alloc_i64(bytes / 8, false)?;
                Ok(CudaStorage::from_i64(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::U8 | DType::Bool => {
                let slice = self.runtime.alloc_u8(bytes, false)?;
                Ok(CudaStorage::from_u8(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::F16 => {
                let slice = self.runtime.alloc_f16(bytes / 2, false)?;
                Ok(CudaStorage::from_f16(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::BF16 => {
                let slice = self.runtime.alloc_bf16(bytes / 2, false)?;
                Ok(CudaStorage::from_bf16(
                    slice,
                    Arc::clone(&self.runtime.pool),
                ))
            }
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn alloc_zeroed_for(&self, bytes: usize, dtype: DType) -> Result<Self::Storage> {
        match dtype {
            DType::F32 => {
                let slice = self.runtime.alloc_f32(bytes / 4, true)?;
                Ok(CudaStorage::from_f32(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::I64 => {
                let slice = self.runtime.alloc_i64(bytes / 8, true)?;
                Ok(CudaStorage::from_i64(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::U8 | DType::Bool => {
                let slice = self.runtime.alloc_u8(bytes, true)?;
                Ok(CudaStorage::from_u8(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::F16 => {
                let slice = self.runtime.alloc_f16(bytes / 2, true)?;
                Ok(CudaStorage::from_f16(slice, Arc::clone(&self.runtime.pool)))
            }
            DType::BF16 => {
                let slice = self.runtime.alloc_bf16(bytes / 2, true)?;
                Ok(CudaStorage::from_bf16(
                    slice,
                    Arc::clone(&self.runtime.pool),
                ))
            }
            other => Err(Error::UnsupportedDType(other)),
        }
    }

    fn copy_from_host(&self, dst: &mut Self::Storage, src: &[u8]) -> Result<()> {
        match dst {
            CudaStorage::F32 { .. } => {
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
                    .memcpy_htod(&host, dst.f32_mut()?)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::I64 { .. } => {
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
                    .memcpy_htod(&host, dst.i64_mut()?)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::U8 { .. } => {
                self.runtime
                    .stream
                    .memcpy_htod(src, dst.u8_mut()?)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::F16 { .. } => {
                let (chunks, rest) = src.as_chunks::<2>();
                if !rest.is_empty() {
                    return Err(Error::invalid("f16 upload size is not a multiple of 2"));
                }
                let host: Vec<f16> = chunks
                    .iter()
                    .map(|chunk| f16::from_bits(u16::from_ne_bytes(*chunk)))
                    .collect();
                self.runtime
                    .stream
                    .memcpy_htod(&host, dst.f16_mut()?)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
            CudaStorage::BF16 { .. } => {
                let (chunks, rest) = src.as_chunks::<2>();
                if !rest.is_empty() {
                    return Err(Error::invalid("bf16 upload size is not a multiple of 2"));
                }
                let host: Vec<bf16> = chunks
                    .iter()
                    .map(|chunk| bf16::from_bits(u16::from_ne_bytes(*chunk)))
                    .collect();
                self.runtime
                    .stream
                    .memcpy_htod(&host, dst.bf16_mut()?)
                    .map_err(|error| Error::Backend(format!("htod: {error}")))?;
            }
        }
        Ok(())
    }

    fn copy_to_host(&self, src: &Self::Storage, dst: &mut [u8]) -> Result<()> {
        match src {
            CudaStorage::F32 { .. } => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(src.f32()?)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let bytes = bytemuck::cast_slice::<f32, u8>(&host);
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
            CudaStorage::I64 { .. } => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(src.i64()?)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let bytes = bytemuck::cast_slice::<i64, u8>(&host);
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
            CudaStorage::U8 { .. } => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(src.u8()?)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let len = host.len().min(dst.len());
                dst[..len].copy_from_slice(&host[..len]);
            }
            CudaStorage::F16 { .. } => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(src.f16()?)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let mut bytes = Vec::with_capacity(host.len() * 2);
                for value in host {
                    bytes.extend_from_slice(&value.to_bits().to_ne_bytes());
                }
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
            CudaStorage::BF16 { .. } => {
                let host = self
                    .runtime
                    .stream
                    .clone_dtoh(src.bf16()?)
                    .map_err(|error| Error::Backend(format!("dtoh: {error}")))?;
                let mut bytes = Vec::with_capacity(host.len() * 2);
                for value in host {
                    bytes.extend_from_slice(&value.to_bits().to_ne_bytes());
                }
                let len = bytes.len().min(dst.len());
                dst[..len].copy_from_slice(&bytes[..len]);
            }
        }
        Ok(())
    }
}

impl CudaStorage {
    fn i64_mut(&mut self) -> Result<&mut CudaSlice<i64>> {
        match self {
            CudaStorage::I64 { slice, .. } => slice
                .as_mut()
                .ok_or_else(|| Error::Backend("cuda i64 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not i64".to_string())),
        }
    }

    fn u8_mut(&mut self) -> Result<&mut CudaSlice<u8>> {
        match self {
            CudaStorage::U8 { slice, .. } => slice
                .as_mut()
                .ok_or_else(|| Error::Backend("cuda u8 storage is released".to_string())),
            _ => Err(Error::Backend("cuda storage is not u8".to_string())),
        }
    }
}
