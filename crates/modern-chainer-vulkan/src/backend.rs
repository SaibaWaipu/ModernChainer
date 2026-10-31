use std::sync::Arc;

use modern_chainer_core::{Backend, Device, Error, Result, Storage};

use crate::gpu::GpuContext;
pub use crate::ops::VulkanOps;

pub struct VulkanStorage {
    pub(crate) buffer: wgpu::Buffer,
    len: usize,
}

impl Storage for VulkanStorage {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

#[derive(Clone)]
pub struct VulkanBackend {
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) ops: VulkanOps,
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
        let ctx = GpuContext::new()?;
        Ok(Self::from_context(ctx))
    }

    pub fn with_adapter(index: usize) -> Result<Self> {
        let ctx = GpuContext::with_adapter(index)?;
        Ok(Self::from_context(ctx))
    }

    pub(crate) fn from_context(ctx: Arc<GpuContext>) -> Self {
        Self {
            ops: VulkanOps {
                ctx: Arc::clone(&ctx),
            },
            ctx,
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.ctx.info.name
    }

    pub fn adapter_index(&self) -> usize {
        self.ctx.adapter_index
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Backend(format!("poll failed: {e}")))?;
        Ok(())
    }

    pub(crate) fn create_buffer(&self, bytes: usize) -> Result<wgpu::Buffer> {
        let size = aligned(bytes);
        Ok(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modern-chainer tensor"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    pub(crate) fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<u8>> {
        let size = aligned(len.max(4));
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modern-chainer readback"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("modern-chainer readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, Some(size as u64));
        self.ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Backend(format!("poll failed: {e}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Backend(format!("map channel closed: {e}")))?
            .map_err(|e| Error::Backend(format!("buffer map failed: {e}")))?;

        let view = slice
            .get_mapped_range()
            .map_err(|e| Error::Backend(format!("get_mapped_range failed: {e}")))?;
        let mut out = vec![0u8; len];
        out.copy_from_slice(&view[..len]);
        drop(view);
        staging.unmap();
        Ok(out)
    }
}

impl Backend for VulkanBackend {
    type Storage = VulkanStorage;
    type Ops = VulkanOps;

    fn device(&self) -> Device {
        Device::Vulkan(self.ctx.adapter_index as u32)
    }

    fn ops(&self) -> &Self::Ops {
        &self.ops
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Storage> {
        Ok(VulkanStorage {
            buffer: self.create_buffer(bytes)?,
            len: aligned(bytes),
        })
    }

    fn alloc_zeroed(&self, bytes: usize) -> Result<Self::Storage> {
        let storage = self.alloc(bytes)?;
        let zeros = vec![0u8; aligned(bytes)];
        self.ctx.queue.write_buffer(&storage.buffer, 0, &zeros);
        Ok(storage)
    }

    fn copy_from_host(&self, dst: &mut Self::Storage, src: &[u8]) -> Result<()> {
        if src.len() > dst.len {
            return Err(Error::invalid("copy_from_host buffer overflow"));
        }
        let mut padded = src.to_vec();
        padded.resize(dst.len, 0);
        self.ctx.queue.write_buffer(&dst.buffer, 0, &padded);
        Ok(())
    }

    fn copy_to_host(&self, src: &Self::Storage, dst: &mut [u8]) -> Result<()> {
        let len = dst.len().min(src.len);
        let bytes = self.read_buffer(&src.buffer, src.len)?;
        dst[..len].copy_from_slice(&bytes[..len]);
        Ok(())
    }
}

pub(crate) fn aligned(bytes: usize) -> usize {
    bytes.max(4).div_ceil(4) * 4
}
