//! CUDA backend.
//!
//! The backend uses `cudarc` with dynamic library loading.  NVRTC compiles the
//! elementwise/optimizer kernels at runtime (no `nvcc` at build time) and
//! cuBLAS handles GEMM and the fused linear path.  Reduction-heavy and
//! attention ops currently round-trip through the CPU backend; see
//! `docs/backends.md`.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod graph;
#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod ops;

#[cfg(feature = "cuda")]
pub use backend::{CudaBackend, CudaStorage};
pub use graph::{CudaGraphRunner, GraphAdam, GraphMatmul};
#[cfg(feature = "cuda")]
pub use ops::CudaOps;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub available: bool,
    pub device_count: usize,
    pub devices: Vec<String>,
    pub reason: String,
}

/// Whether the CUDA backend was compiled in.
pub const fn is_compiled() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
pub fn probe() -> Capability {
    static PROBE: std::sync::OnceLock<Capability> = std::sync::OnceLock::new();
    PROBE.get_or_init(probe_impl).clone()
}

#[cfg(feature = "cuda")]
fn probe_impl() -> Capability {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(probe_inner);
    std::panic::set_hook(previous);
    result.unwrap_or_else(|_| Capability {
        available: false,
        device_count: 0,
        devices: Vec::new(),
        reason: "CUDA runtime library is unavailable (set LD_LIBRARY_PATH for libnvrtc/libcublas)"
            .to_string(),
    })
}

#[cfg(feature = "cuda")]
fn probe_inner() -> Capability {
    use cudarc::driver::CudaContext;
    let count = CudaBackend::device_count().unwrap_or(0);
    let mut devices = Vec::with_capacity(count);
    for ordinal in 0..count {
        match CudaContext::new(ordinal) {
            Ok(ctx) => devices.push(ctx.name().unwrap_or_else(|_| format!("cuda:{ordinal}"))),
            Err(error) => {
                return Capability {
                    available: false,
                    device_count: count,
                    devices,
                    reason: format!("device {ordinal}: {error}"),
                };
            }
        }
    }
    match CudaBackend::new(0) {
        Ok(backend) => Capability {
            available: true,
            device_count: count,
            devices,
            reason: format!("ok: {}", backend.name()),
        },
        Err(error) => Capability {
            available: false,
            device_count: count,
            devices,
            reason: error.to_string(),
        },
    }
}

#[cfg(not(feature = "cuda"))]
pub fn probe() -> Capability {
    Capability {
        available: false,
        device_count: 0,
        devices: Vec::new(),
        reason: "CUDA backend compiled without the `cuda` feature".to_string(),
    }
}

pub fn is_available() -> bool {
    probe().available
}

pub fn device_count() -> usize {
    probe().device_count
}
