use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use modern_chainer_core::{Error, Result};

pub(crate) struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub adapter_index: usize,
    pub pipelines: Mutex<HashMap<String, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
}

impl GpuContext {
    pub fn new() -> Result<Arc<Self>> {
        Self::with_adapter(0)
    }

    pub fn with_adapter(index: usize) -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters.get(index).cloned().ok_or_else(|| {
            Error::Backend(format!(
                "Vulkan adapter index {index} out of range (found {})",
                adapters.len()
            ))
        })?;
        Self::from_adapter(adapter, index)
    }

    fn from_adapter(adapter: wgpu::Adapter, adapter_index: usize) -> Result<Arc<Self>> {
        let info = adapter.get_info();
        let supported = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = limits
            .max_storage_buffers_per_shader_stage
            .max(8)
            .min(supported.max_storage_buffers_per_shader_stage);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("modern-chainer"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| Error::Backend(format!("failed to create Vulkan device: {e}")))?;

        Ok(Arc::new(Self {
            device,
            queue,
            info,
            adapter_index,
            pipelines: Mutex::new(HashMap::new()),
        }))
    }
}
