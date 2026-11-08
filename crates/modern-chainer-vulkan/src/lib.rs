//! Vulkan compute backend.
//!
//! Implemented on `wgpu` (Vulkan on Linux): NVIDIA / AMD / Intel / Mesa lavapipe.
//! Covers elementwise, exp/log/sigmoid, silu/gelu, matmul/linear (incl. fused
//! bias+activation), transpose, fused Adam, softmax CE, softmax, layer norm,
//! embedding. Attention / paged ops and RMSNorm/RoPE/cast/dropout go through a
//! CPU round-trip fallback (correct but slower); see `docs/backends.md`.

#[cfg(feature = "wgpu")]
mod backend;
#[cfg(feature = "wgpu")]
mod gpu;
#[cfg(feature = "wgpu")]
mod ops;
#[cfg(feature = "wgpu")]
mod shaders;

#[cfg(feature = "wgpu")]
pub use backend::{VulkanBackend, VulkanOps, VulkanStorage};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub available: bool,
    pub adapters: Vec<AdapterInfo>,
    pub reason: &'static str,
}

#[cfg(feature = "wgpu")]
pub fn probe() -> Capability {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let infos: Vec<AdapterInfo> = adapters
        .iter()
        .map(|adapter| {
            let info = adapter.get_info();
            AdapterInfo {
                name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
            }
        })
        .collect();
    let available = !infos.is_empty();
    Capability {
        available,
        adapters: infos,
        reason: if available {
            "ok"
        } else {
            "no Vulkan adapter found"
        },
    }
}

#[cfg(not(feature = "wgpu"))]
pub fn probe() -> Capability {
    Capability {
        available: false,
        adapters: Vec::new(),
        reason: "Vulkan backend compiled without the `wgpu` feature",
    }
}

pub fn is_available() -> bool {
    probe().available
}
