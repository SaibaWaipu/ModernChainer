use std::sync::Arc;

use modern_chainer_core::{Error, Result};

pub(crate) struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn new() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| Error::Backend(format!("no Vulkan adapter: {e}")))?;

        let info = adapter.get_info();
        let limits = wgpu::Limits::downlevel_defaults();
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
        }))
    }
}
